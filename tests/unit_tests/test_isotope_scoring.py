"""Tests for MS1 isotope envelope scoring.

This module tests isotope distribution calculation, envelope detection,
and scoring functions for MS1 isotope patterns.
"""

import numpy as np
import pytest

from alphapeptfast.constants import ISOTOPE_MASS_DIFFERENCE
from alphapeptfast.scoring import (
    MS1IsotopeScorer,
    calculate_isotope_distribution,
    calculate_isotope_mz_values,
    find_isotope_envelope,
    score_isotope_envelope,
)
from alphapeptfast.scoring.isotope_scoring import (
    normalize_intensities,
    pearson_correlation,
)


# =============================================================================
# Tests for Theoretical Isotope Distribution
# =============================================================================


def test_isotope_distribution_basic():
    """Test that isotope distribution has reasonable values."""
    # Small peptide (~600 Da)
    intensities = calculate_isotope_distribution(600.0, n_peaks=5)

    # M+0 should be 1.0 (normalized)
    assert intensities[0] == pytest.approx(1.0, abs=1e-6)

    # M+1 should be significant but less than M+0
    assert 0.2 < intensities[1] < 0.5

    # M+2 should be smaller than M+1
    assert intensities[2] < intensities[1]

    # Intensities should decrease monotonically
    for i in range(len(intensities) - 1):
        assert intensities[i] >= intensities[i + 1]


def test_isotope_distribution_mass_scaling():
    """Test that isotope intensity increases with peptide mass."""
    # Small peptide
    small_intensities = calculate_isotope_distribution(500.0, n_peaks=5)

    # Medium peptide
    medium_intensities = calculate_isotope_distribution(1500.0, n_peaks=5)

    # Large peptide
    large_intensities = calculate_isotope_distribution(3000.0, n_peaks=5)

    # M+1 intensity should increase with mass (more carbons = more C13)
    assert small_intensities[1] < medium_intensities[1]
    assert medium_intensities[1] < large_intensities[1]

    # Same for M+2
    assert small_intensities[2] < medium_intensities[2]
    assert medium_intensities[2] < large_intensities[2]


def test_isotope_distribution_normalized():
    """Test that M+0 is always normalized to 1.0."""
    for mass in [500.0, 1000.0, 2000.0, 5000.0]:
        intensities = calculate_isotope_distribution(mass, n_peaks=5)
        assert intensities[0] == pytest.approx(1.0, abs=1e-6)


def test_isotope_distribution_n_peaks():
    """Test that we can calculate different numbers of peaks."""
    # 3 peaks
    intensities_3 = calculate_isotope_distribution(1000.0, n_peaks=3)
    assert len(intensities_3) == 3

    # 5 peaks
    intensities_5 = calculate_isotope_distribution(1000.0, n_peaks=5)
    assert len(intensities_5) == 5

    # 7 peaks
    intensities_7 = calculate_isotope_distribution(1000.0, n_peaks=7)
    assert len(intensities_7) == 7

    # First 3 should match
    np.testing.assert_array_almost_equal(intensities_3, intensities_5[:3])


# =============================================================================
# Tests for Isotope m/z Calculation
# =============================================================================


def test_isotope_mz_spacing_charge_1():
    """Test isotope m/z spacing for charge 1."""
    mz_values = calculate_isotope_mz_values(1000.0, charge=1, n_peaks=5)

    # Check spacing
    for i in range(len(mz_values) - 1):
        spacing = mz_values[i + 1] - mz_values[i]
        assert spacing == pytest.approx(ISOTOPE_MASS_DIFFERENCE, abs=1e-4)


def test_isotope_mz_spacing_charge_2():
    """Test isotope m/z spacing for charge 2."""
    mz_values = calculate_isotope_mz_values(650.5, charge=2, n_peaks=5)

    expected_spacing = ISOTOPE_MASS_DIFFERENCE / 2.0  # ~0.5017 m/z

    # Check spacing
    for i in range(len(mz_values) - 1):
        spacing = mz_values[i + 1] - mz_values[i]
        assert spacing == pytest.approx(expected_spacing, abs=1e-4)


def test_isotope_mz_spacing_charge_3():
    """Test isotope m/z spacing for charge 3."""
    mz_values = calculate_isotope_mz_values(433.7, charge=3, n_peaks=5)

    expected_spacing = ISOTOPE_MASS_DIFFERENCE / 3.0  # ~0.3344 m/z

    # Check spacing
    for i in range(len(mz_values) - 1):
        spacing = mz_values[i + 1] - mz_values[i]
        assert spacing == pytest.approx(expected_spacing, abs=1e-4)


def test_isotope_mz_first_peak():
    """Test that first peak is the monoisotopic m/z."""
    monoisotopic_mz = 650.5
    mz_values = calculate_isotope_mz_values(monoisotopic_mz, charge=2, n_peaks=5)

    assert mz_values[0] == pytest.approx(monoisotopic_mz, abs=1e-6)


# =============================================================================
# Tests for Isotope Envelope Detection
# =============================================================================


def test_find_isotope_envelope_perfect_match():
    """Test finding isotope envelope with perfect synthetic data."""
    # Create synthetic MS1 spectrum with perfect isotope envelope
    # Charge 2, monoisotopic at 650.5 m/z
    charge = 2
    monoisotopic_mz = 650.5
    spacing = ISOTOPE_MASS_DIFFERENCE / charge

    # MUST BE SORTED for binary search!
    spectrum_mz = np.array([
        650.500,  # M+0
        651.002,  # M+1 (650.5 + 0.502)
        651.504,  # M+2
        652.006,  # M+3
        652.508,  # M+4
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        1000.0,  # M+0 (highest)
        500.0,   # M+1
        150.0,   # M+2
        40.0,    # M+3
        10.0,    # M+4
    ], dtype=np.float32)

    # Find envelope
    obs_mz, obs_int, errors = find_isotope_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, charge,
        tolerance_ppm=10.0, n_peaks=5
    )

    # Should find all 5 peaks
    assert np.sum(obs_mz > 0) == 5

    # m/z values should match
    np.testing.assert_array_almost_equal(obs_mz, spectrum_mz, decimal=3)

    # Intensities should match
    np.testing.assert_array_almost_equal(obs_int, spectrum_intensity, decimal=1)

    # Mass errors should be near zero (< 2.5 ppm due to float precision)
    assert np.all(np.abs(errors) < 2.5)  # < 2.5 ppm error


def test_find_isotope_envelope_missing_peaks():
    """Test finding envelope when some peaks are missing."""
    charge = 2
    monoisotopic_mz = 650.5

    # Only M+0, M+1, M+3 present (M+2 and M+4 missing)
    spectrum_mz = np.array([
        650.500,  # M+0
        651.002,  # M+1
        652.006,  # M+3 (M+2 is missing!)
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        1000.0,
        500.0,
        40.0,
    ], dtype=np.float32)

    # Find envelope
    obs_mz, obs_int, errors = find_isotope_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, charge,
        tolerance_ppm=10.0, n_peaks=5
    )

    # Should find 3 peaks
    assert np.sum(obs_mz > 0) == 3

    # M+0, M+1, M+3 should be found
    assert obs_mz[0] > 0  # M+0
    assert obs_mz[1] > 0  # M+1
    assert obs_mz[2] == 0  # M+2 missing!
    assert obs_mz[3] > 0  # M+3
    assert obs_mz[4] == 0  # M+4 missing!


def test_find_isotope_envelope_with_noise():
    """Test finding envelope with interfering noise peaks."""
    charge = 2
    monoisotopic_mz = 650.5

    # Real isotope peaks + noise peaks
    spectrum_mz = np.array([
        640.0,    # Noise
        650.500,  # M+0
        650.8,    # Noise (close but outside tolerance)
        651.002,  # M+1
        651.504,  # M+2
        651.6,    # Noise
        652.006,  # M+3
        652.508,  # M+4
        660.0,    # Noise
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        50.0,     # Noise
        1000.0,   # M+0
        30.0,     # Noise
        500.0,    # M+1
        150.0,    # M+2
        20.0,     # Noise
        40.0,     # M+3
        10.0,     # M+4
        40.0,     # Noise
    ], dtype=np.float32)

    # Find envelope
    obs_mz, obs_int, errors = find_isotope_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, charge,
        tolerance_ppm=10.0, n_peaks=5
    )

    # Should find all 5 real peaks, ignoring noise
    assert np.sum(obs_mz > 0) == 5

    # Should match correct peaks (not noise)
    expected_mz = np.array([650.500, 651.002, 651.504, 652.006, 652.508])
    np.testing.assert_array_almost_equal(obs_mz, expected_mz, decimal=3)


def test_find_isotope_envelope_empty_spectrum():
    """Test finding envelope in empty spectrum."""
    # Empty spectrum
    spectrum_mz = np.array([], dtype=np.float64)
    spectrum_intensity = np.array([], dtype=np.float32)

    obs_mz, obs_int, errors = find_isotope_envelope(
        spectrum_mz, spectrum_intensity,
        650.5, charge=2,
        tolerance_ppm=10.0, n_peaks=5
    )

    # Should find 0 peaks
    assert np.sum(obs_mz > 0) == 0


# =============================================================================
# Tests for Normalization and Correlation
# =============================================================================


def test_normalize_intensities():
    """Test intensity normalization."""
    intensities = np.array([100.0, 50.0, 25.0, 10.0], dtype=np.float64)
    normalized = normalize_intensities(intensities)

    # Max should be 1.0
    assert np.max(normalized) == pytest.approx(1.0, abs=1e-6)

    # Ratios should be preserved
    assert normalized[1] / normalized[0] == pytest.approx(0.5, abs=1e-6)
    assert normalized[2] / normalized[0] == pytest.approx(0.25, abs=1e-6)


def test_normalize_intensities_zeros():
    """Test normalization of near-zero intensities."""
    intensities = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    normalized = normalize_intensities(intensities)

    # Should return zeros unchanged
    np.testing.assert_array_equal(normalized, intensities)


def test_pearson_correlation_perfect():
    """Test Pearson correlation with perfect correlation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)  # y = 2*x

    corr = pearson_correlation(x, y)
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_pearson_correlation_negative():
    """Test Pearson correlation with negative correlation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)  # Reversed

    corr = pearson_correlation(x, y)
    assert corr == pytest.approx(-1.0, abs=1e-6)


def test_pearson_correlation_uncorrelated():
    """Test Pearson correlation with uncorrelated data."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)  # Constant

    corr = pearson_correlation(x, y)
    assert corr == pytest.approx(0.0, abs=1e-6)


def test_pearson_correlation_too_few_points():
    """Test that correlation returns 0 with too few points."""
    x = np.array([1.0], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)

    corr = pearson_correlation(x, y)
    assert corr == 0.0


# =============================================================================
# Tests for Isotope Envelope Scoring
# =============================================================================


def test_score_isotope_envelope_perfect():
    """Test scoring with perfect isotope envelope."""
    # Perfect match: all peaks found, perfect intensities
    observed_mz = np.array([650.5, 651.0, 651.5, 652.0, 652.5], dtype=np.float64)
    observed_intensity = np.array([1000.0, 600.0, 180.0, 40.0, 8.0], dtype=np.float32)
    mass_errors = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    theoretical_intensity = np.array([1.0, 0.6, 0.18, 0.04, 0.008], dtype=np.float64)

    result = score_isotope_envelope(
        observed_mz, observed_intensity, mass_errors, theoretical_intensity
    )

    # All peaks found
    assert result['n_found'] == 5
    assert result['peak_coverage'] == pytest.approx(1.0, abs=1e-6)

    # Near-perfect mass accuracy
    assert result['mean_mass_error'] == pytest.approx(0.0, abs=1e-6)

    # Perfect correlation
    assert result['intensity_correlation'] == pytest.approx(1.0, abs=1e-2)

    # High combined score
    assert result['combined_score'] > 0.9


def test_score_isotope_envelope_partial():
    """Test scoring with partial isotope envelope (missing peaks)."""
    # Only 3 out of 5 peaks found
    observed_mz = np.array([650.5, 651.0, 0.0, 0.0, 652.5], dtype=np.float64)
    observed_intensity = np.array([1000.0, 600.0, 0.0, 0.0, 8.0], dtype=np.float32)
    mass_errors = np.array([0.5, 0.3, 0.0, 0.0, 0.8], dtype=np.float64)
    theoretical_intensity = np.array([1.0, 0.6, 0.18, 0.04, 0.008], dtype=np.float64)

    result = score_isotope_envelope(
        observed_mz, observed_intensity, mass_errors, theoretical_intensity
    )

    # 3 peaks found
    assert result['n_found'] == 3
    assert result['peak_coverage'] == pytest.approx(0.6, abs=1e-2)  # 3/5

    # Combined score should be lower than perfect (not all peaks)
    assert 0.3 < result['combined_score'] < 0.9


def test_score_isotope_envelope_no_peaks():
    """Test scoring with no isotope peaks found."""
    # All zeros (no peaks found)
    observed_mz = np.zeros(5, dtype=np.float64)
    observed_intensity = np.zeros(5, dtype=np.float32)
    mass_errors = np.zeros(5, dtype=np.float64)
    theoretical_intensity = np.array([1.0, 0.6, 0.18, 0.04, 0.008], dtype=np.float64)

    result = score_isotope_envelope(
        observed_mz, observed_intensity, mass_errors, theoretical_intensity
    )

    # No peaks found
    assert result['n_found'] == 0
    assert result['peak_coverage'] == 0.0
    assert result['combined_score'] == 0.0


def test_score_isotope_envelope_bad_mass_accuracy():
    """Test scoring with poor mass accuracy."""
    # All peaks found but with large mass errors
    observed_mz = np.array([650.5, 651.0, 651.5, 652.0, 652.5], dtype=np.float64)
    observed_intensity = np.array([1000.0, 600.0, 180.0, 40.0, 8.0], dtype=np.float32)
    mass_errors = np.array([15.0, 12.0, 18.0, 20.0, 22.0], dtype=np.float64)  # Bad!
    theoretical_intensity = np.array([1.0, 0.6, 0.18, 0.04, 0.008], dtype=np.float64)

    result = score_isotope_envelope(
        observed_mz, observed_intensity, mass_errors, theoretical_intensity
    )

    # All peaks found
    assert result['n_found'] == 5

    # But poor mass accuracy
    assert result['mean_mass_error'] > 10.0

    # Combined score should be lower due to poor mass accuracy
    assert result['combined_score'] < 0.75


# =============================================================================
# Tests for MS1IsotopeScorer Class
# =============================================================================


def test_ms1_isotope_scorer_initialization():
    """Test MS1IsotopeScorer initialization."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0, n_isotope_peaks=5)

    assert scorer.tolerance_ppm == 10.0
    assert scorer.n_isotope_peaks == 5


def test_ms1_isotope_scorer_score_envelope():
    """Test scoring a single envelope with MS1IsotopeScorer."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0)

    # Create synthetic MS1 spectrum
    charge = 2
    monoisotopic_mz = 650.5
    precursor_mass = (monoisotopic_mz * charge) - (charge * 1.007276)  # Remove protons

    spacing = ISOTOPE_MASS_DIFFERENCE / charge

    spectrum_mz = np.array([
        650.500,
        651.002,
        651.504,
        652.006,
        652.508,
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        1000.0,
        500.0,
        150.0,
        40.0,
        10.0,
    ], dtype=np.float32)

    result = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, charge, precursor_mass
    )

    # Should find all peaks
    assert result['n_found'] == 5
    assert result['peak_coverage'] == pytest.approx(1.0, abs=1e-2)

    # Should have good correlation
    assert result['intensity_correlation'] > 0.8

    # Should have good combined score
    assert result['combined_score'] > 0.7


def test_ms1_isotope_scorer_empty_spectrum():
    """Test scoring with empty spectrum."""
    scorer = MS1IsotopeScorer()

    spectrum_mz = np.array([], dtype=np.float64)
    spectrum_intensity = np.array([], dtype=np.float32)

    result = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        650.5, precursor_charge=2, precursor_mass=1299.0
    )

    # Should return zeros
    assert result['n_found'] == 0
    assert result['combined_score'] == 0.0


def test_ms1_isotope_scorer_batch_score():
    """Test batch scoring multiple precursors."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0)

    # Create synthetic MS1 spectrum with TWO isotope envelopes
    # Precursor 1: 650.5 m/z, charge 2
    # Precursor 2: 500.3 m/z, charge 2
    # MUST BE SORTED for binary search!
    spectrum_mz = np.array([
        # Precursor 2 envelope (lower m/z, comes first when sorted)
        500.300, 500.802, 501.304, 501.806, 502.308,
        # Precursor 1 envelope
        650.500, 651.002, 651.504, 652.006, 652.508,
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        # Precursor 2 (sorted order)
        800.0, 400.0, 120.0, 30.0, 7.0,
        # Precursor 1
        1000.0, 500.0, 150.0, 40.0, 10.0,
    ], dtype=np.float32)

    # Batch score both precursors
    precursor_mz = np.array([650.5, 500.3], dtype=np.float64)
    precursor_charge = np.array([2, 2], dtype=np.int32)
    precursor_mass = np.array([1299.0, 998.6], dtype=np.float64)

    result = scorer.batch_score(
        spectrum_mz, spectrum_intensity,
        precursor_mz, precursor_charge, precursor_mass
    )

    # Should score both precursors
    assert len(result['combined_score']) == 2

    # Both should have good scores (all peaks present)
    assert result['combined_score'][0] > 0.7
    assert result['combined_score'][1] > 0.7

    # Both should find all 5 peaks
    assert result['n_found'][0] == 5
    assert result['n_found'][1] == 5


def test_ms1_isotope_scorer_charge_state_validation():
    """Test that isotope spacing validates charge state."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0)

    # Create envelope with CHARGE 2 spacing
    charge = 2
    monoisotopic_mz = 650.5
    spacing = ISOTOPE_MASS_DIFFERENCE / 2  # ~0.502 m/z

    spectrum_mz = np.array([
        650.500,
        651.002,  # +0.502 (charge 2)
        651.504,
        652.006,
        652.508,
    ], dtype=np.float64)

    spectrum_intensity = np.array([1000.0, 500.0, 150.0, 40.0, 10.0], dtype=np.float32)

    # Score with CORRECT charge state (2)
    precursor_mass = (monoisotopic_mz * 2) - (2 * 1.007276)
    result_correct = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, precursor_charge=2, precursor_mass=precursor_mass
    )

    # Score with WRONG charge state (3) - should fail to find peaks!
    # Because it will look for spacing of ~0.334 m/z instead of 0.502
    result_wrong = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, precursor_charge=3, precursor_mass=precursor_mass
    )

    # Correct charge should find all 5 peaks
    assert result_correct['n_found'] == 5

    # Wrong charge should find fewer peaks (or none)
    assert result_wrong['n_found'] < result_correct['n_found']


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow_high_quality():
    """Test complete workflow with high-quality isotope envelope."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0)

    # Realistic peptide: PEPTIDE (mass ~799 Da, charge 2, m/z ~400.5)
    charge = 2
    monoisotopic_mz = 400.5
    precursor_mass = 799.0

    # Create realistic MS1 spectrum
    # Good isotope envelope + some noise
    spacing = ISOTOPE_MASS_DIFFERENCE / charge

    spectrum_mz = np.array([
        395.0,    # Noise
        400.500,  # M+0
        401.002,  # M+1
        401.504,  # M+2
        402.006,  # M+3
        402.508,  # M+4
        405.0,    # Noise
        410.3,    # Noise
    ], dtype=np.float64)

    # Realistic intensities (decreasing envelope)
    spectrum_intensity = np.array([
        50.0,     # Noise
        10000.0,  # M+0 (base peak)
        4000.0,   # M+1 (~40% for 800 Da peptide)
        800.0,    # M+2
        120.0,    # M+3
        15.0,     # M+4
        30.0,     # Noise
        25.0,     # Noise
    ], dtype=np.float32)

    result = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, precursor_charge=charge, precursor_mass=precursor_mass
    )

    # Should find all 5 isotope peaks
    assert result['n_found'] == 5

    # Should have 100% coverage
    assert result['peak_coverage'] == pytest.approx(1.0, abs=1e-2)

    # Should have good mass accuracy (< 2.5 ppm)
    assert result['mean_mass_error'] < 2.5

    # Should have high intensity correlation
    assert result['intensity_correlation'] > 0.7

    # Should have high combined score
    assert result['combined_score'] > 0.7


def test_full_workflow_poor_quality():
    """Test complete workflow with poor-quality isotope envelope."""
    scorer = MS1IsotopeScorer(tolerance_ppm=10.0)

    charge = 2
    monoisotopic_mz = 400.5
    precursor_mass = 799.0

    # Poor spectrum: only 2 isotope peaks, lots of noise
    spectrum_mz = np.array([
        395.0,    # Noise
        400.500,  # M+0 (only this found!)
        401.002,  # M+1 (only this found!)
        # M+2, M+3, M+4 missing!
        405.0,    # Noise
        410.3,    # Noise
    ], dtype=np.float64)

    spectrum_intensity = np.array([
        500.0,    # Noise (high!)
        1000.0,   # M+0
        400.0,    # M+1
        300.0,    # Noise (high!)
        200.0,    # Noise
    ], dtype=np.float32)

    result = scorer.score_envelope(
        spectrum_mz, spectrum_intensity,
        monoisotopic_mz, precursor_charge=charge, precursor_mass=precursor_mass
    )

    # Should find only 2 peaks
    assert result['n_found'] == 2

    # Low coverage
    assert result['peak_coverage'] < 0.5

    # Lower combined score (but still decent due to good correlation of found peaks)
    assert result['combined_score'] < 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

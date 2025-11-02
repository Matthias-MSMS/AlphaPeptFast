"""Comprehensive unit tests for candidate matching and feature extraction.

Tests the core PSM scoring pipeline:
- Binary search matching of theoretical fragments to spectrum
- Feature extraction (33 production features)
- RT coelution filtering
- Ion series analysis
- Edge cases and failure modes
"""

import numpy as np
import pytest

from alphapeptfast.search.candidate_matching import (
    match_candidates_batch,
    extract_features,
    MatchResults,
)


class TestCandidateMatching:
    """Test binary search matching of candidates to spectrum."""

    def test_single_candidate_perfect_match(self):
        """Test matching single candidate with all fragments present."""
        # Theoretical fragments (1 candidate, 5 fragments)
        theo_mz = np.array([[100.0, 200.0, 300.0, 400.0, 500.0]], dtype=np.float64)
        theo_type = np.array([[0, 0, 1, 1, 1]], dtype=np.uint8)  # b, b, y, y, y
        theo_pos = np.array([[1, 2, 1, 2, 3]], dtype=np.uint8)
        theo_charge = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
        frags_per_cand = np.array([5], dtype=np.int32)

        # Observed spectrum (all present, perfect match)
        obs_mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        obs_intensity = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=np.float32)
        obs_rt = np.array([600.0, 600.0, 600.0, 600.0, 600.0], dtype=np.float32)
        prec_rt = 600.0

        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            mz_tolerance_ppm=10.0, rt_tolerance_sec=10.0
        )

        match_counts = results[0]
        match_intensities = results[1]
        match_mz_errors = results[2]

        # Should match all 5 fragments
        assert match_counts[0] == 5
        assert np.allclose(match_intensities[0, :5], [1000, 2000, 3000, 4000, 5000])
        assert np.allclose(match_mz_errors[0, :5], 0.0, atol=1e-6)  # Perfect match

    def test_multiple_candidates_batch(self):
        """Test matching multiple candidates in parallel."""
        # 3 candidates, each with 3 fragments
        theo_mz = np.array([
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [110.0, 210.0, 310.0],
        ], dtype=np.float64)
        theo_type = np.zeros((3, 3), dtype=np.uint8)
        theo_pos = np.array([[1, 2, 3]] * 3, dtype=np.uint8)
        theo_charge = np.ones((3, 3), dtype=np.uint8)
        frags_per_cand = np.array([3, 3, 3], dtype=np.int32)

        # Spectrum has some matching fragments
        obs_mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64)
        obs_intensity = np.ones(5, dtype=np.float32) * 1000.0
        obs_rt = np.ones(5, dtype=np.float32) * 600.0
        prec_rt = 600.0

        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt
        )

        match_counts = results[0]

        # Candidate 0 should match: 100, 200, 300 → 3 matches
        # Candidate 1 should match: 150, 250 → 2 matches (350 not in spectrum)
        # Candidate 2 should match: none → 0 matches (110, 210, 310 not in spectrum)
        assert match_counts[0] == 3
        assert match_counts[1] == 2
        assert match_counts[2] == 0

    def test_mz_tolerance_filtering(self):
        """Test that m/z tolerance is enforced correctly."""
        theo_mz = np.array([[100.0, 200.0]], dtype=np.float64)
        theo_type = np.zeros((1, 2), dtype=np.uint8)
        theo_pos = np.array([[1, 2]], dtype=np.uint8)
        theo_charge = np.ones((1, 2), dtype=np.uint8)
        frags_per_cand = np.array([2], dtype=np.int32)

        # Peaks slightly off: 100.001 (10 PPM), 200.005 (25 PPM)
        obs_mz = np.array([100.001, 200.005], dtype=np.float64)
        obs_intensity = np.ones(2, dtype=np.float32)
        obs_rt = np.ones(2, dtype=np.float32) * 600.0
        prec_rt = 600.0

        # Strict tolerance (5 PPM) - should match only first
        results_strict = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            mz_tolerance_ppm=5.0
        )
        assert results_strict[0][0] == 0  # Neither within 5 PPM of 100 or 200

        # Lenient tolerance (30 PPM) - should match both
        results_lenient = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            mz_tolerance_ppm=30.0
        )
        assert results_lenient[0][0] == 2  # Both within tolerance

    def test_rt_coelution_filtering(self):
        """Test that RT tolerance filters non-coeluting fragments."""
        theo_mz = np.array([[100.0, 200.0, 300.0]], dtype=np.float64)
        theo_type = np.zeros((1, 3), dtype=np.uint8)
        theo_pos = np.array([[1, 2, 3]], dtype=np.uint8)
        theo_charge = np.ones((1, 3), dtype=np.uint8)
        frags_per_cand = np.array([3], dtype=np.int32)

        # Peaks at different RTs
        obs_mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        obs_intensity = np.ones(3, dtype=np.float32)
        obs_rt = np.array([600.0, 605.0, 620.0], dtype=np.float32)  # 0s, 5s, 20s diff
        prec_rt = 600.0

        # Strict RT tolerance (5 sec) - should match first 2 only
        results_strict = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            rt_tolerance_sec=5.0
        )
        assert results_strict[0][0] == 2  # 100 and 200 within 5s

        # Lenient tolerance (30 sec) - should match all 3
        results_lenient = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            rt_tolerance_sec=30.0
        )
        assert results_lenient[0][0] == 3  # All within 30s

    def test_max_matches_overflow_protection(self):
        """Test that max_matches prevents array overflow."""
        # 1 candidate with 100 fragments
        n_frags = 100
        theo_mz = np.arange(100, 100 + n_frags, dtype=np.float64).reshape(1, -1)
        theo_type = np.zeros((1, n_frags), dtype=np.uint8)
        theo_pos = np.arange(1, n_frags + 1, dtype=np.uint8).reshape(1, -1)
        theo_charge = np.ones((1, n_frags), dtype=np.uint8)
        frags_per_cand = np.array([n_frags], dtype=np.int32)

        # All peaks present
        obs_mz = np.arange(100, 100 + n_frags, dtype=np.float64)
        obs_intensity = np.ones(n_frags, dtype=np.float32)
        obs_rt = np.ones(n_frags, dtype=np.float32) * 600.0
        prec_rt = 600.0

        # Limit to 50 matches
        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            max_matches_per_candidate=50
        )

        match_counts = results[0]
        # Should cap at 50, not crash
        assert match_counts[0] == 50

    def test_ppm_error_calculation(self):
        """Test that PPM errors are calculated correctly (signed)."""
        theo_mz = np.array([[100.0, 200.0]], dtype=np.float64)
        theo_type = np.zeros((1, 2), dtype=np.uint8)
        theo_pos = np.array([[1, 2]], dtype=np.uint8)
        theo_charge = np.ones((1, 2), dtype=np.uint8)
        frags_per_cand = np.array([2], dtype=np.int32)

        # Obs: 100.001 (+10 PPM), 199.998 (-10 PPM)
        obs_mz = np.array([100.001, 199.998], dtype=np.float64)
        obs_intensity = np.ones(2, dtype=np.float32)
        obs_rt = np.ones(2, dtype=np.float32) * 600.0
        prec_rt = 600.0

        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt,
            mz_tolerance_ppm=20.0
        )

        match_mz_errors = results[2]

        # PPM error for 100.001: (100.001 - 100.0) / 100.0 * 1e6 = 10 PPM
        # PPM error for 199.998: (199.998 - 200.0) / 200.0 * 1e6 = -10 PPM
        assert np.isclose(match_mz_errors[0, 0], 10.0, atol=0.1)
        assert np.isclose(match_mz_errors[0, 1], -10.0, atol=0.1)

    def test_empty_spectrum(self):
        """Test matching against empty spectrum."""
        theo_mz = np.array([[100.0, 200.0]], dtype=np.float64)
        theo_type = np.zeros((1, 2), dtype=np.uint8)
        theo_pos = np.array([[1, 2]], dtype=np.uint8)
        theo_charge = np.ones((1, 2), dtype=np.uint8)
        frags_per_cand = np.array([2], dtype=np.int32)

        # Empty spectrum
        obs_mz = np.array([], dtype=np.float64)
        obs_intensity = np.array([], dtype=np.float32)
        obs_rt = np.array([], dtype=np.float32)
        prec_rt = 600.0

        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt
        )

        match_counts = results[0]
        # Should return 0 matches, not crash
        assert match_counts[0] == 0

    def test_empty_candidate(self):
        """Test candidate with zero fragments."""
        # Empty candidate (0 fragments)
        theo_mz = np.zeros((1, 0), dtype=np.float64)
        theo_type = np.zeros((1, 0), dtype=np.uint8)
        theo_pos = np.zeros((1, 0), dtype=np.uint8)
        theo_charge = np.zeros((1, 0), dtype=np.uint8)
        frags_per_cand = np.array([0], dtype=np.int32)

        obs_mz = np.array([100.0, 200.0], dtype=np.float64)
        obs_intensity = np.ones(2, dtype=np.float32)
        obs_rt = np.ones(2, dtype=np.float32) * 600.0
        prec_rt = 600.0

        results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt
        )

        match_counts = results[0]
        # Should return 0 matches for empty candidate
        assert match_counts[0] == 0


class TestFeatureExtraction:
    """Test feature extraction from matching results."""

    def test_basic_features_calculation(self):
        """Test that basic features are calculated correctly."""
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=10,
            match_intensities=np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=np.float32),
            match_mz_errors=np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0], dtype=np.float32),
            match_rt_diffs=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float32),
            match_types=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8),  # 5 b, 5 y
            match_positions=np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.uint8),
            match_charges=np.ones(10, dtype=np.uint8),
            n_theoretical_fragments=20,
        )

        # Fragment matching features
        assert features['match_count'] == 10.0
        assert features['coverage'] == 0.5  # 10/20
        assert features['total_intensity'] == 55000.0
        assert features['mean_intensity'] == 5500.0
        assert features['max_intensity'] == 10000.0

        # RT features
        assert features['mean_rt_diff'] == 2.75  # Mean of 0.5 to 5.0
        assert features['min_rt_diff'] == 0.5
        assert features['max_rt_diff'] == 5.0

        # Ion series
        assert features['n_b_ions'] == 5.0
        assert features['n_y_ions'] == 5.0
        assert features['y_to_b_ratio'] == 1.0

        # Precursor
        assert features['precursor_intensity'] == 1e6
        assert features['precursor_charge'] == 2.0

    def test_zero_matches_features(self):
        """Test feature extraction when no fragments matched."""
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=0,
            match_intensities=np.array([], dtype=np.float32),
            match_mz_errors=np.array([], dtype=np.float32),
            match_rt_diffs=np.array([], dtype=np.float32),
            match_types=np.array([], dtype=np.uint8),
            match_positions=np.array([], dtype=np.uint8),
            match_charges=np.array([], dtype=np.uint8),
            n_theoretical_fragments=20,
        )

        # Should return minimal features
        assert features['match_count'] == 0.0
        assert features['coverage'] == 0.0
        assert features['total_intensity'] == 0.0
        assert features['mean_intensity'] == 0.0
        assert features['mean_rt_diff'] == 0.0
        assert features['n_b_ions'] == 0.0
        assert features['n_y_ions'] == 0.0
        assert features['matched_fragments_string'] == ""

        # Precursor features should still be present
        assert features['precursor_intensity'] == 1e6
        assert features['precursor_charge'] == 2.0

    def test_b_series_continuity(self):
        """Test b-ion series continuity calculation."""
        # b1, b2, b3, b5 (gap at b4) → continuity = 3 (b1-b2-b3)
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=4,
            match_intensities=np.ones(4, dtype=np.float32),
            match_mz_errors=np.zeros(4, dtype=np.float32),
            match_rt_diffs=np.zeros(4, dtype=np.float32),
            match_types=np.array([0, 0, 0, 0], dtype=np.uint8),  # All b ions
            match_positions=np.array([1, 2, 3, 5], dtype=np.uint8),  # Gap at 4
            match_charges=np.ones(4, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # Longest consecutive b series: b1-b2-b3 (3 ions)
        assert features['b_series_continuity'] == 3.0
        assert features['y_series_continuity'] == 0.0  # No y ions
        assert features['max_continuity'] == 3.0

    def test_y_series_continuity(self):
        """Test y-ion series continuity calculation."""
        # y1, y2, y3, y4, y6 (gap at y5) → continuity = 4 (y1-y2-y3-y4)
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=5,
            match_intensities=np.ones(5, dtype=np.float32),
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.zeros(5, dtype=np.float32),
            match_types=np.array([1, 1, 1, 1, 1], dtype=np.uint8),  # All y ions
            match_positions=np.array([1, 2, 3, 4, 6], dtype=np.uint8),  # Gap at 5
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        assert features['b_series_continuity'] == 0.0  # No b ions
        assert features['y_series_continuity'] == 4.0
        assert features['max_continuity'] == 4.0

    def test_high_low_mid_mass_ions(self):
        """Test position distribution features."""
        # PEPTIDE = 7 letters
        # High: >70% = positions 6, 7
        # Low: <30% = positions 1, 2
        # Mid: 30-70% = positions 3, 4, 5
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=6,
            match_intensities=np.ones(6, dtype=np.float32),
            match_mz_errors=np.zeros(6, dtype=np.float32),
            match_rt_diffs=np.zeros(6, dtype=np.float32),
            match_types=np.zeros(6, dtype=np.uint8),
            match_positions=np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8),
            match_charges=np.ones(6, dtype=np.uint8),
            n_theoretical_fragments=12,
        )

        # PEPTIDE length = 7
        # threshold_low = int(0.3 * 7) = 2
        # threshold_high = int(0.7 * 7) = 4
        # Low: positions < 2 → position 1 → 1 ion
        # High: positions > 4 → positions 5, 6 → 2 ions
        # Mid: rest → positions 2, 3, 4 → 3 ions
        assert features['n_low_mass_ions'] == 1.0
        assert features['n_high_mass_ions'] == 2.0
        assert features['n_mid_mass_ions'] == 3.0

    def test_matched_fragments_string(self):
        """Test matched fragments string encoding."""
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=3,
            match_intensities=np.array([1000.5, 2500.3, 1800.7], dtype=np.float32),
            match_mz_errors=np.zeros(3, dtype=np.float32),
            match_rt_diffs=np.zeros(3, dtype=np.float32),
            match_types=np.array([0, 1, 1], dtype=np.uint8),  # b, y, y
            match_positions=np.array([2, 5, 7], dtype=np.uint8),
            match_charges=np.array([1, 1, 2], dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # Format: "b2_1:1000.5|y5_1:2500.3|y7_2:1800.7"
        matched_str = features['matched_fragments_string']
        assert 'b2_1:1000.5' in matched_str
        assert 'y5_1:2500.3' in matched_str
        assert 'y7_2:1800.7' in matched_str

    def test_intensity_snr(self):
        """Test signal-to-noise ratio calculation."""
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=5,
            match_intensities=np.array([100, 200, 300, 400, 1000], dtype=np.float32),
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.zeros(5, dtype=np.float32),
            match_types=np.zeros(5, dtype=np.uint8),
            match_positions=np.arange(1, 6, dtype=np.uint8),
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # SNR = max / mean = 1000 / 400 = 2.5
        assert features['intensity_snr'] == 2.5

    def test_match_efficiency(self):
        """Test match efficiency calculation."""
        features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=8,
            match_intensities=np.ones(8, dtype=np.float32),
            match_mz_errors=np.zeros(8, dtype=np.float32),
            match_rt_diffs=np.zeros(8, dtype=np.float32),
            match_types=np.zeros(8, dtype=np.uint8),
            match_positions=np.arange(1, 9, dtype=np.uint8),
            match_charges=np.ones(8, dtype=np.uint8),
            n_theoretical_fragments=20,
        )

        # Efficiency = 8 / 20 = 0.4
        assert features['match_efficiency'] == 0.4


class TestMatchResultsNamedTuple:
    """Test MatchResults named tuple structure."""

    def test_match_results_structure(self):
        """Test that MatchResults is properly structured."""
        # Create a dummy result
        result = MatchResults(
            match_counts=np.array([5]),
            match_intensities=np.zeros((1, 50)),
            match_mz_errors=np.zeros((1, 50)),
            match_rt_diffs=np.zeros((1, 50)),
            match_types=np.zeros((1, 50), dtype=np.uint8),
            match_positions=np.zeros((1, 50), dtype=np.uint8),
            match_charges=np.zeros((1, 50), dtype=np.uint8),
        )

        # Check attributes accessible
        assert result.match_counts[0] == 5
        assert result.match_intensities.shape == (1, 50)
        assert result.match_types.dtype == np.uint8

    def test_match_results_from_function(self):
        """Test creating MatchResults from function output."""
        theo_mz = np.array([[100.0]], dtype=np.float64)
        theo_type = np.zeros((1, 1), dtype=np.uint8)
        theo_pos = np.array([[1]], dtype=np.uint8)
        theo_charge = np.ones((1, 1), dtype=np.uint8)
        frags_per_cand = np.array([1], dtype=np.int32)

        obs_mz = np.array([100.0], dtype=np.float64)
        obs_intensity = np.array([1000.0], dtype=np.float32)
        obs_rt = np.array([600.0], dtype=np.float32)
        prec_rt = 600.0

        raw_results = match_candidates_batch(
            theo_mz, theo_type, theo_pos, theo_charge, frags_per_cand,
            obs_mz, obs_intensity, obs_rt, prec_rt
        )

        # Can wrap in MatchResults
        results = MatchResults(*raw_results)
        assert results.match_counts[0] == 1


class TestExtendedFeatureExtraction:
    """Test extract_features_extended with advanced scoring."""

    def test_extended_features_without_scorers(self):
        """Test that extended features default to 0 without scorers."""
        from alphapeptfast.search.candidate_matching import extract_features_extended

        features = extract_features_extended(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            precursor_mz=650.5,
            precursor_mass=1299.0,
            match_count=5,
            match_intensities=np.ones(5, dtype=np.float32) * 1000,
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.ones(5, dtype=np.float32),
            match_types=np.zeros(5, dtype=np.uint8),
            match_positions=np.arange(1, 6, dtype=np.uint8),
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # Should have all 37 features
        assert 'fragment_intensity_correlation' in features
        assert 'ms1_isotope_score' in features
        assert 'ms2_isotope_fraction' in features
        assert 'ms2_isotope_recommended_weight' in features

        # New features should default to 0
        assert features['fragment_intensity_correlation'] == 0.0
        assert features['ms1_isotope_score'] == 0.0
        assert features['ms2_isotope_fraction'] == 0.0
        assert features['ms2_isotope_recommended_weight'] == 0.0

        # Baseline features should still be calculated
        assert features['match_count'] == 5.0
        assert features['coverage'] == 0.5
        assert features['precursor_intensity'] == 1e6

    def test_extended_features_backward_compatible(self):
        """Test that basic features match extract_features()."""
        from alphapeptfast.search.candidate_matching import extract_features_extended, extract_features

        # Run both functions
        basic_features = extract_features(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            match_count=5,
            match_intensities=np.ones(5, dtype=np.float32) * 1000,
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.ones(5, dtype=np.float32),
            match_types=np.zeros(5, dtype=np.uint8),
            match_positions=np.arange(1, 6, dtype=np.uint8),
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        extended_features = extract_features_extended(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            precursor_mz=650.5,
            precursor_mass=1299.0,
            match_count=5,
            match_intensities=np.ones(5, dtype=np.float32) * 1000,
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.ones(5, dtype=np.float32),
            match_types=np.zeros(5, dtype=np.uint8),
            match_positions=np.arange(1, 6, dtype=np.uint8),
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # All baseline features should match
        for key in basic_features:
            if key != 'matched_fragments_string':  # Skip string comparison
                assert basic_features[key] == extended_features[key], f"Mismatch for {key}"

    def test_extended_features_count(self):
        """Test that we get exactly 37 features."""
        from alphapeptfast.search.candidate_matching import extract_features_extended

        features = extract_features_extended(
            peptide='PEPTIDE',
            charge=2,
            precursor_intensity=1e6,
            precursor_mz=650.5,
            precursor_mass=1299.0,
            match_count=5,
            match_intensities=np.ones(5, dtype=np.float32),
            match_mz_errors=np.zeros(5, dtype=np.float32),
            match_rt_diffs=np.ones(5, dtype=np.float32),
            match_types=np.zeros(5, dtype=np.uint8),
            match_positions=np.arange(1, 6, dtype=np.uint8),
            match_charges=np.ones(5, dtype=np.uint8),
            n_theoretical_fragments=10,
        )

        # Count: 30 baseline (12 fragment + 5 RT + 10 ion + 2 precursor + 1 string)
        # + 4 new advanced features = 34 total
        assert len(features) == 34, f"Expected 34 features, got {len(features)}"

        # Check all expected keys are present
        expected_keys = [
            # Fragment matching (12)
            'match_count', 'coverage', 'total_intensity', 'mean_intensity',
            'max_intensity', 'median_intensity', 'intensity_std',
            'mean_abs_ppm_error', 'ppm_error_std', 'max_abs_ppm_error',
            'intensity_snr', 'match_efficiency',
            # RT (5)
            'mean_rt_diff', 'std_rt_diff', 'max_rt_diff', 'min_rt_diff', 'median_rt_diff',
            # Ion series (10)
            'n_b_ions', 'n_y_ions', 'y_to_b_ratio',
            'b_series_continuity', 'y_series_continuity', 'max_continuity',
            'n_high_mass_ions', 'n_low_mass_ions', 'n_mid_mass_ions',
            'mean_fragment_spacing',
            # Precursor (2)
            'precursor_intensity', 'precursor_charge',
            # String (1)
            'matched_fragments_string',
            # NEW advanced features (4)
            'fragment_intensity_correlation',
            'ms1_isotope_score',
            'ms2_isotope_fraction',
            'ms2_isotope_recommended_weight',
        ]

        # Should be exactly 34 features
        assert len(expected_keys) == 34
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

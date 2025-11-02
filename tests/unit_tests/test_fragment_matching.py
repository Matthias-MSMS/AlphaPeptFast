"""Tests for fragment matching algorithms.

Tests the core spectrum matching algorithms including binary search, PPM
tolerance matching, RT coelution filtering, and ion mirroring.
"""

import numpy as np
import pytest

from alphapeptfast.search.fragment_matching import (
    binary_search_mz,
    match_fragments_to_spectrum,
    match_fragments_with_coelution,
    calculate_complementary_mz,
    calculate_match_statistics,
)
from alphapeptfast.fragments.generator import (
    encode_peptide_to_ord,
    generate_by_ions,
)
from alphapeptfast.constants import PROTON_MASS


class TestBinarySearchMZ:
    """Test binary search for m/z matching."""

    def test_exact_match(self):
        """Test binary search finds exact match."""
        spectrum_mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        # Search for exact match
        idx = binary_search_mz(spectrum_mz, target_mz=200.0, tol_ppm=10.0)

        assert idx == 1
        assert spectrum_mz[idx] == 200.0

    def test_closest_match_within_tolerance(self):
        """Test binary search finds closest match within PPM tolerance."""
        spectrum_mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        # Search slightly off target (within tolerance)
        # 200.002 vs 200.0 = 10 ppm error
        idx = binary_search_mz(spectrum_mz, target_mz=200.002, tol_ppm=20.0)

        assert idx == 1  # Should match 200.0
        assert abs(spectrum_mz[idx] - 200.002) < 0.01

    def test_no_match_outside_tolerance(self):
        """Test binary search returns -1 when no match within tolerance."""
        spectrum_mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        # Search far from any peak (>50 ppm away)
        idx = binary_search_mz(spectrum_mz, target_mz=250.0, tol_ppm=10.0)

        assert idx == -1

    def test_empty_spectrum(self):
        """Test binary search handles empty spectrum."""
        spectrum_mz = np.array([], dtype=np.float32)

        idx = binary_search_mz(spectrum_mz, target_mz=200.0, tol_ppm=10.0)

        assert idx == -1

    def test_edge_cases_first_last(self):
        """Test binary search at edges of spectrum."""
        spectrum_mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        # First element
        idx = binary_search_mz(spectrum_mz, target_mz=100.0, tol_ppm=10.0)
        assert idx == 0

        # Last element
        idx = binary_search_mz(spectrum_mz, target_mz=400.0, tol_ppm=10.0)
        assert idx == 3

    def test_ppm_tolerance_calculation(self):
        """Test PPM tolerance is calculated correctly."""
        spectrum_mz = np.array([500.0, 500.01, 500.02], dtype=np.float32)

        # At m/z 500, 20 ppm = 0.01 Da
        # 500.01 is 20 ppm from 500.0
        idx = binary_search_mz(spectrum_mz, target_mz=500.0, tol_ppm=20.0)

        # Should find 500.0 as closest
        assert idx == 0

    def test_multiple_matches_returns_closest(self):
        """Test that when multiple peaks match, closest is returned."""
        # Two peaks within tolerance
        spectrum_mz = np.array([199.98, 200.02, 300.0], dtype=np.float32)

        # Target at 200.0, both 199.98 and 200.02 are within 200 ppm
        idx = binary_search_mz(spectrum_mz, target_mz=200.0, tol_ppm=200.0)

        # Should return closest (199.98 is 100 ppm away, 200.02 is 100 ppm away)
        # Both equally close, should return first found
        assert idx in [0, 1]

    def test_high_resolution_data(self):
        """Test with high-resolution Orbitrap-like data."""
        # Simulate dense high-res spectrum
        spectrum_mz = np.array([
            199.9995, 200.0000, 200.0005, 200.0010
        ], dtype=np.float32)

        # 5 ppm at m/z 200 = 0.001 Da
        idx = binary_search_mz(spectrum_mz, target_mz=200.0000, tol_ppm=5.0)

        assert idx == 1  # Exact match


class TestMatchFragmentsToSpectrum:
    """Test basic fragment matching without RT."""

    def test_simple_matching(self):
        """Test basic fragment matching."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),  # b and y
            fragment_charges=(1,)
        )

        # Create synthetic spectrum with some matching peaks
        # CRITICAL: spectrum_mz MUST be sorted for binary search!
        spectrum_mz = np.sort(np.array([theo_mz[0], theo_mz[5], theo_mz[10]], dtype=np.float32))
        spectrum_intensity = np.array([100.0, 200.0, 150.0], dtype=np.float32)

        # Match fragments
        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=10.0
        )

        # Should find 3 matches
        assert len(match_idx) == 3
        assert len(obs_mz) == 3
        assert len(obs_int) == 3
        assert len(ppm_errors) == 3

    def test_no_matches(self):
        """Test when no fragments match."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Create spectrum with no matching peaks
        spectrum_mz = np.array([1500.0, 1600.0, 1700.0], dtype=np.float32)
        spectrum_intensity = np.array([100.0, 200.0, 150.0], dtype=np.float32)

        # Match fragments
        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=10.0
        )

        # Should find 0 matches
        assert len(match_idx) == 0

    def test_intensity_filtering(self):
        """Test minimum intensity threshold."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Create spectrum with varying intensities
        spectrum_mz = np.array([theo_mz[0], theo_mz[1], theo_mz[2]], dtype=np.float32)
        spectrum_intensity = np.array([50.0, 150.0, 250.0], dtype=np.float32)

        # Match with intensity threshold
        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=10.0,
            min_intensity=100.0
        )

        # Should only match peaks >= 100.0
        assert len(match_idx) == 2
        assert np.all(obs_int >= 100.0)

    def test_ppm_error_calculation(self):
        """Test that PPM errors are calculated correctly."""
        # Simple synthetic case
        theo_mz = np.array([200.0], dtype=np.float64)
        theo_type = np.array([0], dtype=np.int8)
        theo_pos = np.array([1], dtype=np.int8)
        theo_charge = np.array([1], dtype=np.int8)

        # Observed peak 10 ppm higher
        spectrum_mz = np.array([200.002], dtype=np.float32)  # 10 ppm at m/z 200
        spectrum_intensity = np.array([100.0], dtype=np.float32)

        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=20.0
        )

        assert len(match_idx) == 1
        assert abs(ppm_errors[0] - 10.0) < 0.5  # ~10 ppm error

    def test_coverage_calculation(self):
        """Test fragment coverage calculation."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        n_theoretical = len(theo_mz)

        # Match 50% of fragments
        n_to_match = n_theoretical // 2
        spectrum_mz = np.array(theo_mz[:n_to_match], dtype=np.float32)
        spectrum_intensity = np.ones(n_to_match, dtype=np.float32) * 100.0

        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=10.0
        )

        coverage = len(match_idx) / n_theoretical
        assert 0.4 < coverage < 0.6  # Approximately 50%


class TestMatchFragmentsWithCoelution:
    """Test fragment matching with RT coelution filtering."""

    def test_rt_coelution_filtering(self):
        """Test that RT coelution filter works correctly."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Create features with different RTs
        spectrum_mz = np.array([theo_mz[0], theo_mz[1], theo_mz[2]], dtype=np.float32)
        spectrum_intensity = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        # CRITICAL: RT in SECONDS, not minutes!
        spectrum_rt = np.array([450.0, 455.0, 600.0], dtype=np.float32)  # seconds

        precursor_rt = 452.0  # seconds
        precursor_mass = 800.0

        # Match with 5 sec RT tolerance
        match_idx, obs_mz, obs_int, mass_shifts, rt_deltas = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=precursor_rt,
            precursor_mass=precursor_mass,
            mz_tol_ppm=10.0,
            rt_tol_sec=5.0
        )

        # Should only match first two (within 5 sec), not third (148 sec away)
        assert len(match_idx) == 2
        assert np.all(rt_deltas <= 5.0)

    def test_rt_in_seconds_not_minutes(self):
        """CRITICAL: Verify RT is interpreted as seconds, not minutes."""
        # Create theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEP")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),
            fragment_charges=(1,)
        )

        # Create feature
        spectrum_mz = np.array([theo_mz[0]], dtype=np.float32)
        spectrum_intensity = np.array([100.0], dtype=np.float32)

        # REALISTIC RT VALUES IN SECONDS
        # Typical DIA run: 30 min = 1800 sec
        # Feature at 25 min = 1500 sec
        spectrum_rt = np.array([1500.0], dtype=np.float32)  # 25 minutes in seconds
        precursor_rt = 1505.0  # 5 seconds later

        # With 10 sec tolerance, should match
        match_idx, _, _, _, rt_deltas = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=precursor_rt,
            precursor_mass=400.0,
            rt_tol_sec=10.0
        )

        assert len(match_idx) == 1
        assert rt_deltas[0] == 5.0  # 5 seconds difference

        # With 3 sec tolerance, should NOT match
        match_idx2, _, _, _, _ = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=precursor_rt,
            precursor_mass=400.0,
            rt_tol_sec=3.0
        )

        assert len(match_idx2) == 0  # Too far away

    def test_rt_typical_dia_cycle_time(self):
        """Test RT filtering with realistic DIA cycle times."""
        # Typical DIA cycle time: 0.5-2 seconds
        # Features should co-elute within ~5 seconds

        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Precursor elutes at 25 minutes = 1500 seconds
        precursor_rt = 1500.0

        # Create features across one DIA cycle (~1 sec)
        n_features = 5
        spectrum_mz = np.array(theo_mz[:n_features], dtype=np.float32)
        spectrum_intensity = np.ones(n_features, dtype=np.float32) * 100.0
        # Features spread across 1 second (realistic for DIA cycle)
        spectrum_rt = np.array([1499.8, 1500.0, 1500.2, 1500.5, 1500.8], dtype=np.float32)

        # With 2 sec tolerance, all should match
        match_idx, _, _, _, rt_deltas = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=precursor_rt,
            precursor_mass=800.0,
            rt_tol_sec=2.0
        )

        assert len(match_idx) == n_features
        assert np.all(rt_deltas <= 1.0)  # All within 1 second

    def test_ion_mirror_disabled(self):
        """Test matching with ion mirroring disabled."""
        peptide_ord = encode_peptide_to_ord("PEP")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),
            fragment_charges=(1,)
        )

        spectrum_mz = np.array([theo_mz[0]], dtype=np.float32)
        spectrum_intensity = np.array([100.0], dtype=np.float32)
        spectrum_rt = np.array([450.0], dtype=np.float32)

        match_idx, obs_mz, obs_int, mass_shifts, rt_deltas = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=450.0,
            precursor_mass=400.0,
            enable_ion_mirror=False
        )

        # Mass shifts should be 0 when disabled
        assert len(match_idx) == 1
        assert mass_shifts[0] == 0.0


class TestComplementaryMZ:
    """Test complementary m/z calculation for ion mirroring."""

    def test_simple_complementary_calculation(self):
        """Test basic complementary fragment calculation."""
        # Peptide with mass 1000 Da
        # Fragment at m/z 300 (charge 1+) = 300 Da neutral
        # Complementary should be 1000 - 300 = 700 Da
        # At charge 1+: (700 + 1.007276) / 1 = 701.007276

        comp_mz = calculate_complementary_mz(
            precursor_mass=1000.0,
            fragment_mz=300.0,
            fragment_charge=1,
            complementary_charge=1
        )

        # Expected: (1000 - 300 + PROTON_MASS) / 1
        expected_neutral_comp = 1000.0 - (300.0 - PROTON_MASS)
        expected_mz = (expected_neutral_comp + PROTON_MASS) / 1

        assert abs(comp_mz - expected_mz) < 0.01

    def test_complementary_with_different_charges(self):
        """Test complementary calculation with different charge states."""
        # Fragment at charge 2+
        comp_mz = calculate_complementary_mz(
            precursor_mass=1000.0,
            fragment_mz=200.0,  # Charge 2+
            fragment_charge=2,
            complementary_charge=1
        )

        # Convert fragment m/z to neutral: 200*2 - 2*PROTON = ~398
        # Complementary neutral: 1000 - 398 = ~602
        # Complementary m/z at charge 1: 602 + PROTON = ~603

        assert 600 < comp_mz < 610

    def test_complementary_symmetry(self):
        """Test that b and y fragments are complementary."""
        precursor_mass = 800.0

        # b-fragment at 300 Da
        b_mz = 300.0

        # Calculate complementary (should be y-fragment)
        y_mz = calculate_complementary_mz(
            precursor_mass=precursor_mass,
            fragment_mz=b_mz,
            fragment_charge=1,
            complementary_charge=1
        )

        # Now calculate complementary of y (should give back ~b)
        b_mz_back = calculate_complementary_mz(
            precursor_mass=precursor_mass,
            fragment_mz=y_mz,
            fragment_charge=1,
            complementary_charge=1
        )

        # Should be approximately symmetric
        assert abs(b_mz - b_mz_back) < 1.0  # Within 1 Da


class TestMatchStatistics:
    """Test match statistics calculation."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        intensities = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
        theoretical_count = 10

        coverage, total, mean = calculate_match_statistics(intensities, theoretical_count)

        assert coverage == 0.4  # 4/10
        assert total == 1000.0  # Sum
        assert mean == 250.0  # Average

    def test_perfect_coverage(self):
        """Test statistics with 100% coverage."""
        intensities = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        theoretical_count = 3

        coverage, total, mean = calculate_match_statistics(intensities, theoretical_count)

        assert coverage == 1.0
        assert total == 600.0
        assert mean == 200.0

    def test_no_matches(self):
        """Test statistics with no matches."""
        intensities = np.array([], dtype=np.float32)
        theoretical_count = 10

        coverage, total, mean = calculate_match_statistics(intensities, theoretical_count)

        assert coverage == 0.0
        assert total == 0.0
        assert mean == 0.0

    def test_zero_theoretical_fragments(self):
        """Test statistics when theoretical count is zero."""
        intensities = np.array([100.0], dtype=np.float32)
        theoretical_count = 0

        coverage, total, mean = calculate_match_statistics(intensities, theoretical_count)

        assert coverage == 0.0  # Defined as 0 when denominator is 0


class TestPerformance:
    """Performance tests for fragment matching."""

    def test_binary_search_performance(self):
        """Test binary search is fast enough for production."""
        # Create large spectrum (typical Orbitrap MS2: 200-1000 peaks)
        n_peaks = 500
        spectrum_mz = np.sort(np.random.uniform(200, 2000, n_peaks).astype(np.float32))

        # Perform many searches (warm up Numba JIT)
        for _ in range(100):
            binary_search_mz(spectrum_mz, target_mz=500.0, tol_ppm=10.0)

        # Measure time for 1000 searches
        import time
        start = time.perf_counter()
        for _ in range(1000):
            binary_search_mz(spectrum_mz, target_mz=500.0, tol_ppm=10.0)
        elapsed = time.perf_counter() - start

        # Should be <1ms for 1000 searches (>1M ops/sec target)
        ops_per_sec = 1000 / elapsed
        assert ops_per_sec > 100000  # At least 100k ops/sec

    def test_fragment_matching_performance(self):
        """Test fragment matching throughput."""
        # Create typical peptide
        peptide_ord = encode_peptide_to_ord("PEPTIDEPEPTIDE")  # 14 residues
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1, 2)
        )

        # Create realistic spectrum
        n_peaks = 300
        spectrum_mz = np.sort(np.random.uniform(200, 2000, n_peaks).astype(np.float32))
        spectrum_intensity = np.random.uniform(100, 10000, n_peaks).astype(np.float32)

        # Warm up
        for _ in range(10):
            match_fragments_to_spectrum(
                theo_mz, theo_type, theo_pos, theo_charge,
                spectrum_mz, spectrum_intensity,
                mz_tol_ppm=10.0
            )

        # Measure time for 100 matches
        import time
        start = time.perf_counter()
        for _ in range(100):
            match_fragments_to_spectrum(
                theo_mz, theo_type, theo_pos, theo_charge,
                spectrum_mz, spectrum_intensity,
                mz_tol_ppm=10.0
            )
        elapsed = time.perf_counter() - start

        # Target: >10k peptides/sec = <0.1ms per peptide
        # 100 peptides should take <10ms
        peptides_per_sec = 100 / elapsed
        assert peptides_per_sec > 1000  # At least 1k peptides/sec


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_matching_workflow(self):
        """Test complete fragment matching workflow."""
        # Generate theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDER")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1, 2)
        )

        # Create synthetic spectrum with 50% matches
        n_to_match = len(theo_mz) // 2
        spectrum_mz = np.array(theo_mz[:n_to_match], dtype=np.float32)
        spectrum_intensity = np.random.uniform(100, 1000, n_to_match).astype(np.float32)

        # Match fragments
        match_idx, obs_mz, obs_int, ppm_errors = match_fragments_to_spectrum(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity,
            mz_tol_ppm=10.0
        )

        # Calculate statistics
        coverage, total_int, mean_int = calculate_match_statistics(
            obs_int, len(theo_mz)
        )

        # Verify results
        assert len(match_idx) > 0
        assert 0.0 < coverage <= 1.0
        assert total_int > 0
        assert mean_int > 0

    def test_dia_feature_matching_workflow(self):
        """Test DIA feature-based matching workflow."""
        # Generate theoretical fragments
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Create features with RT (typical DIA)
        n_features = len(theo_mz) // 2
        spectrum_mz = np.array(theo_mz[:n_features], dtype=np.float32)
        spectrum_intensity = np.random.uniform(1000, 10000, n_features).astype(np.float32)
        # All features co-elute at 25 min = 1500 sec ± 2 sec
        precursor_rt = 1500.0
        spectrum_rt = np.random.uniform(1498, 1502, n_features).astype(np.float32)

        # Match with RT filtering
        match_idx, obs_mz, obs_int, mass_shifts, rt_deltas = match_fragments_with_coelution(
            theo_mz, theo_type, theo_pos, theo_charge,
            spectrum_mz, spectrum_intensity, spectrum_rt,
            precursor_rt=precursor_rt,
            precursor_mass=800.0,
            mz_tol_ppm=10.0,
            rt_tol_sec=5.0
        )

        # Verify all matches are within RT tolerance
        assert len(match_idx) > 0
        assert np.all(rt_deltas <= 5.0)

        # Calculate coverage
        coverage, _, _ = calculate_match_statistics(obs_int, len(theo_mz))
        assert coverage > 0.0

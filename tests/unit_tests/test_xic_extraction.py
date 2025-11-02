"""Tests for ultra-fast XIC extraction.

Tests the XIC extraction algorithms including binary search, parallel extraction,
mass error tracking, and correlation-based scoring.
"""

import numpy as np
import pytest

from alphapeptfast.xic import (
    binary_search_mz_range,
    build_xics_ultrafast,
    build_xics_with_mass_matrix,
    calculate_mass_error_features,
    score_xic_correlation,
    score_peptide_with_mass_errors,
    UltraFastXICExtractor,
)


class TestBinarySearchMZRange:
    """Test binary search for m/z range finding."""

    def test_exact_match(self):
        """Test finding exact m/z match."""
        mz_array = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        start, end = binary_search_mz_range(mz_array, target_mz=200.0, ppm_tolerance=1.0)

        assert start == 1
        assert end == 2

    def test_multiple_matches(self):
        """Test finding multiple peaks within tolerance."""
        mz_array = np.array([100.0, 200.0, 200.1, 200.2, 300.0], dtype=np.float32)

        # Large tolerance to capture all 200.x masses
        start, end = binary_search_mz_range(mz_array, target_mz=200.1, ppm_tolerance=500.0)

        assert end - start == 3  # Indices 1, 2, 3

    def test_no_matches(self):
        """Test when no peaks match."""
        mz_array = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        start, end = binary_search_mz_range(mz_array, target_mz=250.0, ppm_tolerance=1.0)

        assert start == end  # Empty range

    def test_empty_array(self):
        """Test with empty m/z array."""
        mz_array = np.array([], dtype=np.float32)

        start, end = binary_search_mz_range(mz_array, target_mz=200.0, ppm_tolerance=10.0)

        assert start == 0
        assert end == 0

    def test_ppm_tolerance_scaling(self):
        """Test that PPM tolerance scales with m/z."""
        # At m/z 500, 30 ppm = 0.015 Da
        # At m/z 1000, 30 ppm = 0.03 Da
        mz_array = np.array([500.0, 500.01, 1000.0, 1000.02], dtype=np.float32)

        # Search at 500 with 30 ppm (should capture 500.0 and 500.01)
        start1, end1 = binary_search_mz_range(mz_array, target_mz=500.0, ppm_tolerance=30.0)
        assert end1 - start1 == 2  # 500.0 and 500.01

        # Search at 1000 with 30 ppm (should capture 1000.0 and 1000.02)
        start2, end2 = binary_search_mz_range(mz_array, target_mz=1000.0, ppm_tolerance=30.0)
        assert end2 - start2 == 2  # 1000.0 and 1000.02


class TestBuildXICsUltrafast:
    """Test basic XIC extraction without mass tracking."""

    def test_simple_xic_extraction(self):
        """Test basic XIC extraction."""
        # Create simple data: 3 scans, 2 peaks per scan
        # IMPORTANT: m/z array must be sorted for binary search!
        mz_array = np.array([100.0, 100.0, 100.0, 200.0, 200.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 15.0, 12.0, 20.0, 25.0, 22.0], dtype=np.float32)
        scan_array = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

        # Search for 2 fragments (m/z 100 and 200)
        fragment_mzs = np.array([[100.0, 200.0]], dtype=np.float64)  # 1 peptide, 2 fragments

        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=3, ppm_tolerance=10.0
        )

        # Check shape
        assert xic.shape == (1, 2, 3)  # 1 peptide, 2 fragments, 3 scans

        # Check values
        assert xic[0, 0, 0] == 10.0  # Frag 0 (m/z 100), scan 0
        assert xic[0, 0, 1] == 15.0  # Frag 0, scan 1
        assert xic[0, 0, 2] == 12.0  # Frag 0, scan 2
        assert xic[0, 1, 0] == 20.0  # Frag 1 (m/z 200), scan 0

    def test_no_matches(self):
        """Test when fragments don't match any peaks."""
        mz_array = np.array([100.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)

        # Search for non-existent m/z
        fragment_mzs = np.array([[500.0, 600.0]], dtype=np.float64)

        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=10.0
        )

        # Should be all zeros
        assert np.all(xic == 0)

    def test_zero_padding_skipped(self):
        """Test that zero m/z values are skipped (padding)."""
        mz_array = np.array([100.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)

        # Use 0.0 as padding
        fragment_mzs = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)

        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=10.0
        )

        # Only first fragment should have data
        assert xic[0, 0, 0] == 10.0
        assert xic[0, 1, 0] == 0.0
        assert xic[0, 2, 0] == 0.0

    def test_multiple_peptides(self):
        """Test extracting XICs for multiple peptides."""
        # Simple data
        mz_array = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        scan_array = np.array([0, 0, 0], dtype=np.int32)

        # 3 peptides, each with different fragments
        fragment_mzs = np.array([
            [100.0, 0.0],
            [200.0, 0.0],
            [300.0, 0.0]
        ], dtype=np.float64)

        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=10.0
        )

        assert xic.shape == (3, 2, 1)
        assert xic[0, 0, 0] == 10.0
        assert xic[1, 0, 0] == 20.0
        assert xic[2, 0, 0] == 30.0

    def test_summing_multiple_peaks(self):
        """Test that multiple peaks in same scan are summed."""
        # Two peaks with same m/z in same scan
        mz_array = np.array([100.0, 100.001], dtype=np.float32)  # Very close m/z
        intensity_array = np.array([10.0, 5.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)

        fragment_mzs = np.array([[100.0]], dtype=np.float64)

        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=20.0  # Large tolerance
        )

        # Should sum both peaks
        assert xic[0, 0, 0] == 15.0


class TestBuildXICsWithMassMatrix:
    """Test XIC extraction with mass error tracking."""

    def test_mass_matrix_extraction(self):
        """Test that mass matrices are created correctly."""
        mz_array = np.array([100.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)

        fragment_mzs = np.array([[100.0, 200.0]], dtype=np.float64)

        xic, mass_sum, mass_count = build_xics_with_mass_matrix(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=10.0
        )

        # Check shapes
        assert xic.shape == (1, 2, 1)
        assert mass_sum.shape == (1, 2, 1)
        assert mass_count.shape == (1, 2, 1)

        # Check values
        assert xic[0, 0, 0] == 10.0
        assert xic[0, 1, 0] == 20.0

        # Mass sum should be observed_mz * intensity
        assert mass_sum[0, 0, 0] == 100.0 * 10.0
        assert mass_sum[0, 1, 0] == 200.0 * 20.0

        # Mass count should be number of peaks
        assert mass_count[0, 0, 0] == 1
        assert mass_count[0, 1, 0] == 1

    def test_weighted_average_mz_calculation(self):
        """Test that weighted average m/z can be calculated."""
        # Two peaks with slightly different m/z
        mz_array = np.array([100.0, 100.01], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)  # Different intensities
        scan_array = np.array([0, 0], dtype=np.int32)

        fragment_mzs = np.array([[100.005]], dtype=np.float64)  # Target in middle

        xic, mass_sum, mass_count = build_xics_with_mass_matrix(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1, ppm_tolerance=100.0  # Large tolerance
        )

        # Calculate weighted average m/z
        total_intensity = xic[0, 0, 0]
        weighted_avg_mz = mass_sum[0, 0, 0] / total_intensity

        # Should be weighted toward higher intensity peak (100.01)
        assert weighted_avg_mz > 100.005


class TestCalculateMassErrorFeatures:
    """Test mass error feature calculation."""

    def test_simple_mass_error(self):
        """Test basic mass error calculation."""
        # Create simple XIC with known mass error
        n_scans = 3
        xic_matrix = np.zeros((1, 1, n_scans), dtype=np.float32)
        xic_matrix[0, 0, :] = [10.0, 20.0, 15.0]

        # Theoretical m/z = 100.0
        # Observed m/z = 100.002 (20 ppm error at m/z 100)
        theoretical_mzs = np.array([[100.0]], dtype=np.float64)
        mass_sum_matrix = np.zeros((1, 1, n_scans), dtype=np.float64)
        mass_sum_matrix[0, 0, :] = [100.002 * 10.0, 100.002 * 20.0, 100.002 * 15.0]
        mass_count_matrix = np.ones((1, 1, n_scans), dtype=np.int32)

        mean_errors, error_stds, consistencies = calculate_mass_error_features(
            mass_sum_matrix, mass_count_matrix, xic_matrix, theoretical_mzs
        )

        # Mean error should be ~20 ppm
        assert abs(mean_errors[0, 0] - 20.0) < 0.5

        # Std should be very small (all same error)
        assert error_stds[0, 0] < 1.0

        # Consistency should be high (low std)
        assert consistencies[0] > 0.9

    def test_no_signal(self):
        """Test with no signal (all zeros)."""
        xic_matrix = np.zeros((1, 1, 3), dtype=np.float32)
        mass_sum_matrix = np.zeros((1, 1, 3), dtype=np.float64)
        mass_count_matrix = np.zeros((1, 1, 3), dtype=np.int32)
        theoretical_mzs = np.array([[100.0]], dtype=np.float64)

        mean_errors, error_stds, consistencies = calculate_mass_error_features(
            mass_sum_matrix, mass_count_matrix, xic_matrix, theoretical_mzs
        )

        # Should all be zero
        assert mean_errors[0, 0] == 0.0
        assert error_stds[0, 0] == 0.0
        assert consistencies[0] == 0.0


class TestScoreXICCorrelation:
    """Test XIC correlation scoring."""

    def test_perfect_correlation(self):
        """Test with perfectly correlated fragments."""
        # Create XIC where all fragments have same shape
        n_scans = 10
        xic = np.zeros((3, n_scans), dtype=np.float32)

        # All fragments follow same pattern
        pattern = np.array([0, 0, 10, 50, 100, 50, 10, 0, 0, 0], dtype=np.float32)
        for i in range(3):
            xic[i, :] = pattern * (i + 1)  # Different scales but same shape

        score = score_xic_correlation(xic, min_intensity=10.0)

        # Should have high correlation (close to 1.0)
        assert score > 0.9

    def test_uncorrelated_fragments(self):
        """Test with uncorrelated fragments."""
        n_scans = 10
        xic = np.zeros((3, n_scans), dtype=np.float32)

        # Random patterns
        np.random.seed(42)
        for i in range(3):
            xic[i, :] = np.random.uniform(100, 1000, n_scans).astype(np.float32)

        score = score_xic_correlation(xic, min_intensity=50.0)

        # Should have lower correlation
        assert score < 0.8

    def test_insufficient_fragments(self):
        """Test that <3 fragments returns 0."""
        xic = np.ones((2, 10), dtype=np.float32) * 100

        score = score_xic_correlation(xic, min_intensity=10.0)

        # Requires at least 3 fragments
        assert score == 0.0

    def test_low_intensity(self):
        """Test that low intensity returns 0."""
        xic = np.ones((5, 10), dtype=np.float32) * 10.0  # Low intensity

        score = score_xic_correlation(xic, min_intensity=100.0)

        assert score == 0.0


class TestScorePeptideWithMassErrors:
    """Test combined scoring with mass errors."""

    def test_good_peptide(self):
        """Test scoring of good quality peptide."""
        # Good XIC (correlated fragments)
        n_scans = 10
        xic = np.zeros((5, n_scans), dtype=np.float32)
        pattern = np.array([0, 10, 50, 100, 50, 10, 0, 0, 0, 0], dtype=np.float32)
        for i in range(5):
            xic[i, :] = pattern * (i + 1) * 100

        # Good mass errors (low error, low std)
        mean_mass_errors = np.array([2.0, -1.0, 3.0, -2.0, 1.0], dtype=np.float32)  # All < 5 ppm
        mass_error_stds = np.array([1.0, 1.5, 0.8, 1.2, 1.0], dtype=np.float32)  # All < 3 ppm
        mass_consistency = 0.9

        xic_score, mass_score, combined = score_peptide_with_mass_errors(
            xic, mean_mass_errors, mass_error_stds, mass_consistency
        )

        # All scores should be good
        assert xic_score > 0.5
        assert mass_score > 0.5
        assert combined > 0.5

    def test_bad_xic_good_mass(self):
        """Test peptide with bad XIC but good mass errors."""
        # Bad XIC (random/uncorrelated)
        np.random.seed(123)
        xic = np.random.uniform(0, 100, (5, 10)).astype(np.float32)

        # Good mass errors
        mean_mass_errors = np.ones(5, dtype=np.float32) * 2.0
        mass_error_stds = np.ones(5, dtype=np.float32) * 1.0
        mass_consistency = 0.9

        xic_score, mass_score, combined = score_peptide_with_mass_errors(
            xic, mean_mass_errors, mass_error_stds, mass_consistency
        )

        # Good mass errors contribute 30%, so combined can be moderate
        # XIC score should still be lower than mass score
        assert xic_score < mass_score


class TestUltraFastXICExtractor:
    """Test main XICExtractor class."""

    def test_extractor_initialization(self):
        """Test extractor can be initialized."""
        extractor = UltraFastXICExtractor(
            ppm_tolerance=20.0,
            min_intensity=100.0,
            track_mass_errors=True
        )

        assert extractor.ppm_tolerance == 20.0
        assert extractor.min_intensity == 100.0
        assert extractor.track_mass_errors is True

    def test_extract_xics_without_mass_tracking(self):
        """Test XIC extraction without mass tracking."""
        extractor = UltraFastXICExtractor(track_mass_errors=False)

        mz_array = np.array([100.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)
        fragment_mzs = np.array([[100.0, 200.0]], dtype=np.float64)

        result = extractor.extract_xics(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1
        )

        assert 'xic_matrix' in result
        assert 'mass_sum_matrix' not in result

    def test_extract_xics_with_mass_tracking(self):
        """Test XIC extraction with mass tracking."""
        extractor = UltraFastXICExtractor(track_mass_errors=True)

        mz_array = np.array([100.0, 200.0], dtype=np.float32)
        intensity_array = np.array([10.0, 20.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)
        fragment_mzs = np.array([[100.0, 200.0]], dtype=np.float64)

        result = extractor.extract_xics(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=1
        )

        assert 'xic_matrix' in result
        assert 'mass_sum_matrix' in result
        assert 'mass_count_matrix' in result

    def test_score_peptides_without_mass(self):
        """Test peptide scoring without mass errors."""
        extractor = UltraFastXICExtractor(track_mass_errors=False)

        # Create simple XIC
        xic_matrix = np.ones((2, 5, 10), dtype=np.float32) * 100
        fragment_mzs = np.ones((2, 5), dtype=np.float64) * 100

        scores = extractor.score_peptides(xic_matrix, fragment_mzs)

        assert 'xic_scores' in scores
        assert 'combined_scores' in scores
        assert 'mass_scores' not in scores

    def test_score_peptides_with_mass(self):
        """Test peptide scoring with mass errors."""
        extractor = UltraFastXICExtractor(track_mass_errors=True)

        # Create data
        n_peptides, n_fragments, n_scans = 2, 5, 10
        xic_matrix = np.ones((n_peptides, n_fragments, n_scans), dtype=np.float32) * 100
        fragment_mzs = np.ones((n_peptides, n_fragments), dtype=np.float64) * 100

        # Mass matrices
        mass_sum_matrix = xic_matrix.astype(np.float64) * 100  # observed_mz * intensity
        mass_count_matrix = np.ones((n_peptides, n_fragments, n_scans), dtype=np.int32)

        scores = extractor.score_peptides(
            xic_matrix, fragment_mzs,
            mass_sum_matrix, mass_count_matrix
        )

        assert 'xic_scores' in scores
        assert 'mass_scores' in scores
        assert 'combined_scores' in scores
        assert 'mean_mass_errors' in scores


class TestPerformance:
    """Performance tests for XIC extraction."""

    def test_binary_search_performance(self):
        """Test binary search is fast."""
        # Large m/z array
        n_peaks = 100000
        mz_array = np.sort(np.random.uniform(200, 2000, n_peaks).astype(np.float32))

        # Warm up
        for _ in range(10):
            binary_search_mz_range(mz_array, target_mz=1000.0, ppm_tolerance=20.0)

        # Time many searches
        import time
        n_searches = 10000
        start = time.perf_counter()
        for _ in range(n_searches):
            binary_search_mz_range(mz_array, target_mz=1000.0, ppm_tolerance=20.0)
        elapsed = time.perf_counter() - start

        # Should be fast (>100k searches/sec target)
        searches_per_sec = n_searches / elapsed
        assert searches_per_sec > 50000

    def test_xic_extraction_performance(self):
        """Test XIC extraction throughput."""
        # Realistic DIA data size
        n_scans = 1000
        n_peaks_per_scan = 200
        n_total_peaks = n_scans * n_peaks_per_scan

        # Generate synthetic data
        np.random.seed(42)
        mz_array = np.sort(np.random.uniform(200, 2000, n_total_peaks).astype(np.float32))
        intensity_array = np.random.uniform(100, 10000, n_total_peaks).astype(np.float32)
        scan_array = np.repeat(np.arange(n_scans), n_peaks_per_scan).astype(np.int32)

        # 100 peptides, 10 fragments each
        fragment_mzs = np.random.uniform(200, 2000, (100, 10)).astype(np.float64)

        # Warm up Numba
        build_xics_ultrafast(
            mz_array[:1000], intensity_array[:1000], scan_array[:1000],
            fragment_mzs[:10], n_scans=100, ppm_tolerance=20.0
        )

        # Time extraction
        import time
        start = time.perf_counter()
        xic = build_xics_ultrafast(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=n_scans, ppm_tolerance=20.0
        )
        elapsed = time.perf_counter() - start

        # Should be fast (<1 second for 100k peptide-fragment combinations)
        assert elapsed < 2.0

        # Check output
        assert xic.shape == (100, 10, n_scans)


class TestIntegration:
    """Integration tests for complete XIC workflows."""

    def test_complete_workflow(self):
        """Test complete XIC extraction and scoring workflow."""
        # Create synthetic DIA data
        n_scans = 100
        n_peaks_per_scan = 50
        n_total_peaks = n_scans * n_peaks_per_scan

        np.random.seed(42)
        mz_array = np.sort(np.random.uniform(200, 2000, n_total_peaks).astype(np.float32))
        intensity_array = np.random.uniform(100, 10000, n_total_peaks).astype(np.float32)
        scan_array = np.repeat(np.arange(n_scans), n_peaks_per_scan).astype(np.int32)

        # 10 peptides, 8 fragments each
        fragment_mzs = np.random.uniform(200, 2000, (10, 8)).astype(np.float64)

        # Create extractor
        extractor = UltraFastXICExtractor(ppm_tolerance=20.0, track_mass_errors=True)

        # Extract XICs
        result = extractor.extract_xics(
            mz_array, intensity_array, scan_array,
            fragment_mzs, n_scans=n_scans
        )

        # Score peptides
        scores = extractor.score_peptides(
            result['xic_matrix'], fragment_mzs,
            result['mass_sum_matrix'], result['mass_count_matrix']
        )

        # Verify results
        assert result['xic_matrix'].shape == (10, 8, n_scans)
        assert len(scores['combined_scores']) == 10
        assert np.all(scores['combined_scores'] >= 0)
        assert np.all(scores['combined_scores'] <= 1)

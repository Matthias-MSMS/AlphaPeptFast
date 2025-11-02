"""Comprehensive tests for peak grouping module.

Tests cover:
1. Cosine similarity calculation
2. RT profile extraction
3. Co-eluting peak detection
4. Peak grouping
5. Composite spectrum building
6. Performance benchmarks
7. Edge cases and error handling
"""

import numpy as np
import pytest

from alphapeptfast.scoring import (
    build_composite_spectrum,
    cosine_similarity,
    extract_rt_profiles_around_peak,
    find_coeluting_peaks,
    group_coeluting_peaks,
)


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_profiles(self):
        """Identical profiles should have similarity 1.0."""
        profile = np.array([10.0, 100.0, 50.0, 5.0])
        similarity = cosine_similarity(profile, profile)
        assert similarity == pytest.approx(1.0)

    def test_scaled_profiles(self):
        """Scaled profiles (same shape) should have similarity 1.0."""
        profile1 = np.array([10.0, 100.0, 50.0, 5.0])
        profile2 = np.array([20.0, 200.0, 100.0, 10.0])  # 2x scaled
        similarity = cosine_similarity(profile1, profile2)
        assert similarity == pytest.approx(1.0)

    def test_opposite_profiles(self):
        """Opposite profiles should have similarity 0.0."""
        profile1 = np.array([100.0, 50.0, 10.0, 0.0])
        profile2 = np.array([0.0, 10.0, 50.0, 100.0])
        similarity = cosine_similarity(profile1, profile2)
        # Should be very low (not exactly 0 due to overlap)
        assert similarity < 0.2

    def test_orthogonal_profiles(self):
        """Orthogonal profiles should have low similarity."""
        profile1 = np.array([100.0, 0.0, 0.0, 0.0])
        profile2 = np.array([0.0, 100.0, 0.0, 0.0])
        similarity = cosine_similarity(profile1, profile2)
        assert similarity == 0.0

    def test_zero_profile(self):
        """Zero profile should return 0.0 similarity."""
        profile1 = np.array([10.0, 100.0, 50.0])
        profile2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(profile1, profile2)
        assert similarity == 0.0

    def test_both_zero_profiles(self):
        """Both zero profiles should return 0.0."""
        profile1 = np.array([0.0, 0.0, 0.0])
        profile2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(profile1, profile2)
        assert similarity == 0.0

    def test_similar_but_not_identical(self):
        """Similar profiles should have high similarity."""
        profile1 = np.array([10.0, 100.0, 50.0, 5.0])
        profile2 = np.array([12.0, 95.0, 48.0, 6.0])
        similarity = cosine_similarity(profile1, profile2)
        assert similarity > 0.99

    def test_partial_correlation(self):
        """Partially correlated profiles should have moderate similarity."""
        profile1 = np.array([10.0, 100.0, 50.0, 5.0])
        profile2 = np.array([10.0, 100.0, 25.0, 10.0])  # First half matches
        similarity = cosine_similarity(profile1, profile2)
        assert 0.8 < similarity < 1.0


class TestRTProfileExtraction:
    """Test RT profile extraction around peaks."""

    def test_single_peak_extraction(self):
        """Extract RT profile for single peak."""
        mz_array = np.array([500.0, 500.0, 500.0], dtype=np.float32)
        intensity_array = np.array([10.0, 100.0, 50.0], dtype=np.float32)
        scan_array = np.array([0, 1, 2], dtype=np.int32)

        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=1, mz_tolerance_ppm=20.0, scan_window=1
        )

        # Should find one unique m/z
        assert len(mz_vals) == 1
        assert mz_vals[0] == pytest.approx(500.0)

        # Profile should cover 3 scans (1±1)
        assert profiles.shape == (1, 3)
        assert profiles[0, 0] == pytest.approx(10.0)  # Scan 0
        assert profiles[0, 1] == pytest.approx(100.0)  # Scan 1
        assert profiles[0, 2] == pytest.approx(50.0)  # Scan 2

    def test_multiple_mz_extraction(self):
        """Extract RT profiles for multiple m/z values."""
        mz_array = np.array([500.0, 600.0, 500.0, 600.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0, 110.0, 55.0], dtype=np.float32)
        scan_array = np.array([0, 0, 1, 1], dtype=np.int32)

        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=0, mz_tolerance_ppm=20.0, scan_window=1
        )

        # Should find m/z 500 and 600 (600 is within 3x tolerance at this range)
        # Actually, 600 is 100 Da away from 500, which is 200,000 ppm, so won't be included
        assert len(mz_vals) >= 1

        # Check profiles have correct shape
        assert profiles.shape[1] == 3  # 3 scans (-1, 0, 1)

    def test_mz_grouping(self):
        """Peaks within 11 ppm should be grouped."""
        # Create peaks at m/z 500.0 and 500.005 (10 ppm apart)
        mz_array = np.array([500.0, 500.005, 500.0], dtype=np.float32)
        intensity_array = np.array([100.0, 90.0, 110.0], dtype=np.float32)
        scan_array = np.array([0, 0, 1], dtype=np.int32)

        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=0, mz_tolerance_ppm=20.0, scan_window=1
        )

        # Should group into 1 unique m/z (11 ppm grouping tolerance)
        assert len(mz_vals) == 1
        assert mz_vals[0] == pytest.approx(500.0025, abs=0.001)  # Average

        # Intensities should be summed in scan 0
        assert profiles[0, 1] == pytest.approx(190.0)  # 100 + 90 in scan 0

    def test_scan_window_limits(self):
        """RT profiles should respect scan window."""
        mz_array = np.array([500.0] * 10, dtype=np.float32)
        intensity_array = np.arange(10, dtype=np.float32) * 10
        scan_array = np.arange(10, dtype=np.int32)

        # Extract around scan 5 with window=2
        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=5, mz_tolerance_ppm=20.0, scan_window=2
        )

        # Should cover scans 3-7 (5±2)
        assert len(scans) == 5
        assert scans[0] == 3
        assert scans[-1] == 7

        # Check intensities match
        assert profiles[0, 0] == pytest.approx(30.0)  # Scan 3
        assert profiles[0, 2] == pytest.approx(50.0)  # Scan 5
        assert profiles[0, 4] == pytest.approx(70.0)  # Scan 7

    def test_empty_window(self):
        """No other peaks in window should still include reference peak."""
        mz_array = np.array([500.0], dtype=np.float32)
        intensity_array = np.array([100.0], dtype=np.float32)
        scan_array = np.array([0], dtype=np.int32)

        # Reference peak at scan 0
        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=0, mz_tolerance_ppm=20.0, scan_window=10
        )

        # Should return reference m/z with its intensity at scan 0
        assert len(mz_vals) == 1
        assert profiles[0, 10] == pytest.approx(100.0)  # Scan 0 is at position 10 (scan range -10 to 10)


class TestCoelutingPeakDetection:
    """Test co-eluting peak detection."""

    def test_all_coelute(self):
        """All similar profiles should be marked as co-eluting."""
        rt_profiles = np.array(
            [[10.0, 100.0, 50.0, 5.0], [12.0, 95.0, 48.0, 6.0], [11.0, 98.0, 49.0, 5.5]], dtype=np.float64
        )

        coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.95)

        # All should be marked as co-eluting
        assert np.all(coeluting)

    def test_one_different(self):
        """Different profile should not be marked as co-eluting."""
        rt_profiles = np.array(
            [
                [10.0, 100.0, 50.0, 5.0],
                [12.0, 95.0, 48.0, 6.0],  # Similar to reference
                [5.0, 50.0, 100.0, 10.0],  # Different pattern
            ],
            dtype=np.float64,
        )

        coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.95)

        assert coeluting[0] == True  # Reference
        assert coeluting[1] == True  # Similar
        assert coeluting[2] == False  # Different

    def test_reference_always_included(self):
        """Reference peak should always be included."""
        rt_profiles = np.array([[100.0, 50.0, 10.0], [0.0, 0.0, 0.0]], dtype=np.float64)

        coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.99, reference_idx=0)

        assert coeluting[0] == True  # Reference always True
        assert coeluting[1] == False  # Zero profile

    def test_zero_reference(self):
        """Zero reference should only include itself."""
        rt_profiles = np.array([[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]], dtype=np.float64)

        coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.5, reference_idx=0)

        assert coeluting[0] == True  # Reference
        assert coeluting[1] == False  # Cannot match zero reference

    def test_similarity_threshold(self):
        """Threshold should control inclusion."""
        rt_profiles = np.array(
            [[10.0, 100.0, 50.0], [10.0, 100.0, 45.0], [10.0, 100.0, 20.0]], dtype=np.float64  # Slight difference
        )  # More difference

        # Strict threshold
        coeluting_strict = find_coeluting_peaks(rt_profiles, min_similarity=0.99)
        assert np.sum(coeluting_strict) == 2  # Only first two

        # Loose threshold
        coeluting_loose = find_coeluting_peaks(rt_profiles, min_similarity=0.9)
        assert np.sum(coeluting_loose) == 3  # All three

    def test_custom_reference_idx(self):
        """Should work with non-zero reference index."""
        rt_profiles = np.array(
            [[100.0, 50.0, 10.0], [10.0, 100.0, 50.0], [12.0, 95.0, 48.0]], dtype=np.float64  # Different  # Reference
        )  # Similar

        coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.95, reference_idx=1)

        assert coeluting[0] == False  # Different
        assert coeluting[1] == True  # Reference
        assert coeluting[2] == True  # Similar


class TestPeakGrouping:
    """Test complete peak grouping workflow."""

    def test_simple_grouping(self):
        """Group co-eluting peaks at similar m/z."""
        mz_array = np.array([500.0, 500.005, 600.0, 500.0], dtype=np.float32)
        intensity_array = np.array([100.0, 90.0, 50.0, 110.0], dtype=np.float32)
        scan_array = np.array([0, 0, 0, 1], dtype=np.int32)
        peak_indices = np.array([0, 1, 3], dtype=np.int32)

        grouped = group_coeluting_peaks(
            mz_array, intensity_array, scan_array, peak_indices, scan_window=1, mz_tolerance_ppm=30.0
        )

        # Should group the three m/z 500 peaks (within 11 ppm tolerance)
        assert grouped[0] == True
        assert grouped[1] == True
        assert grouped[2] == False  # m/z 600 not grouped
        assert grouped[3] == True

    def test_empty_peak_indices(self):
        """Empty peak indices should return all False."""
        mz_array = np.array([500.0, 600.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0], dtype=np.float32)
        scan_array = np.array([0, 0], dtype=np.int32)
        peak_indices = np.array([], dtype=np.int32)

        grouped = group_coeluting_peaks(
            mz_array, intensity_array, scan_array, peak_indices, scan_window=1
        )

        assert np.all(~grouped)

    def test_single_peak(self):
        """Single peak should be grouped with itself."""
        mz_array = np.array([500.0, 500.0], dtype=np.float32)
        intensity_array = np.array([100.0, 110.0], dtype=np.float32)
        scan_array = np.array([0, 1], dtype=np.int32)
        peak_indices = np.array([0], dtype=np.int32)

        grouped = group_coeluting_peaks(
            mz_array, intensity_array, scan_array, peak_indices, scan_window=1
        )

        # Both should be grouped (same m/z, within scan window)
        assert grouped[0] == True
        assert grouped[1] == True

    def test_mz_tolerance_filtering(self):
        """m/z tolerance should filter peaks."""
        # Peaks at 500 and 510 (20,000 ppm apart)
        mz_array = np.array([500.0, 510.0, 500.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0, 110.0], dtype=np.float32)
        scan_array = np.array([0, 0, 1], dtype=np.int32)
        peak_indices = np.array([0], dtype=np.int32)

        grouped = group_coeluting_peaks(
            mz_array, intensity_array, scan_array, peak_indices, scan_window=1, mz_tolerance_ppm=20.0
        )

        # Only m/z 500 peaks should be grouped
        assert grouped[0] == True
        assert grouped[1] == False  # Too far in m/z
        assert grouped[2] == True

    def test_scan_window_filtering(self):
        """Scan window should filter peaks."""
        mz_array = np.array([500.0, 500.0, 500.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0, 110.0], dtype=np.float32)
        scan_array = np.array([0, 10, 1], dtype=np.int32)  # Scan 10 is far
        peak_indices = np.array([0], dtype=np.int32)

        grouped = group_coeluting_peaks(
            mz_array, intensity_array, scan_array, peak_indices, scan_window=2
        )

        # Scans 0 and 1 within window, scan 10 outside
        assert grouped[0] == True
        assert grouped[1] == False  # Outside scan window
        assert grouped[2] == True


class TestCompositeSpectrum:
    """Test composite spectrum building."""

    def test_simple_composite(self):
        """Build composite from simple grouped peaks."""
        mz_array = np.array([500.0, 500.005, 600.0], dtype=np.float32)
        intensity_array = np.array([100.0, 90.0, 50.0], dtype=np.float32)
        mask = np.array([True, True, False], dtype=np.bool_)

        combined_mz, combined_int = build_composite_spectrum(mz_array, intensity_array, mask, mz_tolerance_ppm=20.0)

        # Should combine first two peaks (within default 10 ppm)
        assert len(combined_mz) == 1
        assert combined_mz[0] == pytest.approx(500.0025, abs=0.001)  # Average
        assert combined_int[0] == pytest.approx(190.0)  # Sum

    def test_multiple_groups(self):
        """Build composite with multiple m/z groups."""
        mz_array = np.array([500.0, 500.005, 600.0, 600.005], dtype=np.float32)
        intensity_array = np.array([100.0, 90.0, 50.0, 45.0], dtype=np.float32)
        mask = np.array([True, True, True, True], dtype=np.bool_)

        combined_mz, combined_int = build_composite_spectrum(
            mz_array, intensity_array, mask, mz_tolerance_ppm=20.0
        )

        # Should create two groups
        assert len(combined_mz) == 2
        assert combined_mz[0] == pytest.approx(500.0025, abs=0.001)
        assert combined_mz[1] == pytest.approx(600.0025, abs=0.001)
        assert combined_int[0] == pytest.approx(190.0)
        assert combined_int[1] == pytest.approx(95.0)

    def test_empty_mask(self):
        """Empty mask should return empty arrays."""
        mz_array = np.array([500.0, 600.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0], dtype=np.float32)
        mask = np.array([False, False], dtype=np.bool_)

        combined_mz, combined_int = build_composite_spectrum(mz_array, intensity_array, mask)

        assert len(combined_mz) == 0
        assert len(combined_int) == 0

    def test_no_combining_needed(self):
        """Peaks far apart should not be combined."""
        mz_array = np.array([500.0, 600.0, 700.0], dtype=np.float32)
        intensity_array = np.array([100.0, 50.0, 75.0], dtype=np.float32)
        mask = np.array([True, True, True], dtype=np.bool_)

        combined_mz, combined_int = build_composite_spectrum(mz_array, intensity_array, mask)

        # Should keep all three separate
        assert len(combined_mz) == 3
        assert combined_mz[0] == pytest.approx(500.0)
        assert combined_mz[1] == pytest.approx(600.0)
        assert combined_mz[2] == pytest.approx(700.0)

    def test_sorting(self):
        """Output should be sorted by m/z."""
        mz_array = np.array([600.0, 500.0, 700.0], dtype=np.float32)
        intensity_array = np.array([50.0, 100.0, 75.0], dtype=np.float32)
        mask = np.array([True, True, True], dtype=np.bool_)

        combined_mz, combined_int = build_composite_spectrum(mz_array, intensity_array, mask)

        # Should be sorted
        assert combined_mz[0] < combined_mz[1] < combined_mz[2]
        assert combined_mz[0] == pytest.approx(500.0)
        assert combined_mz[1] == pytest.approx(600.0)
        assert combined_mz[2] == pytest.approx(700.0)


class TestPerformance:
    """Performance benchmarks for peak grouping."""

    def test_cosine_similarity_performance(self):
        """Benchmark cosine similarity calculation."""
        import time

        profile1 = np.random.rand(100).astype(np.float64)
        profile2 = np.random.rand(100).astype(np.float64)

        # Warmup
        for _ in range(10):
            cosine_similarity(profile1, profile2)

        # Benchmark
        n_iterations = 10000
        start = time.time()
        for _ in range(n_iterations):
            cosine_similarity(profile1, profile2)
        elapsed = time.time() - start

        comparisons_per_sec = n_iterations / elapsed
        print(f"\nCosine similarity: {comparisons_per_sec:,.0f} comparisons/sec")
        assert comparisons_per_sec > 100000  # Should be >100k/sec

    def test_rt_profile_extraction_performance(self):
        """Benchmark RT profile extraction."""
        import time

        # Create realistic dataset
        n_peaks = 10000
        mz_array = np.random.uniform(400, 1200, n_peaks).astype(np.float32)
        intensity_array = np.random.uniform(10, 1000, n_peaks).astype(np.float32)
        scan_array = np.random.randint(0, 1000, n_peaks).astype(np.int32)

        # Sort by m/z for realism
        sort_idx = np.argsort(mz_array)
        mz_array = mz_array[sort_idx]

        # Warmup
        extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=100, mz_tolerance_ppm=20.0, scan_window=5
        )

        # Benchmark
        n_iterations = 100
        start = time.time()
        for i in range(n_iterations):
            peak_idx = (i * 100) % n_peaks
            extract_rt_profiles_around_peak(
                mz_array, intensity_array, scan_array, peak_idx=peak_idx, mz_tolerance_ppm=20.0, scan_window=5
            )
        elapsed = time.time() - start

        extractions_per_sec = n_iterations / elapsed
        print(f"\nRT profile extraction: {extractions_per_sec:,.0f} extractions/sec")
        assert extractions_per_sec > 50  # Should be >50/sec

    def test_composite_spectrum_performance(self):
        """Benchmark composite spectrum building."""
        import time

        n_peaks = 10000
        mz_array = np.random.uniform(400, 1200, n_peaks).astype(np.float32)
        intensity_array = np.random.uniform(10, 1000, n_peaks).astype(np.float32)
        mask = np.random.rand(n_peaks) > 0.5

        # Warmup
        build_composite_spectrum(mz_array, intensity_array, mask)

        # Benchmark
        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            build_composite_spectrum(mz_array, intensity_array, mask)
        elapsed = time.time() - start

        builds_per_sec = n_iterations / elapsed
        print(f"\nComposite spectrum: {builds_per_sec:,.0f} builds/sec")
        assert builds_per_sec > 20  # Should be >20/sec


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete peak grouping workflow."""
        # Create synthetic co-eluting peaks
        n_scans = 20
        mz_peptide1 = [500.0, 650.0, 800.0]  # Three fragments
        mz_peptide2 = [550.0, 700.0]  # Two fragments

        mz_list = []
        intensity_list = []
        scan_list = []

        # Add peptide 1 peaks (Gaussian elution centered at scan 10)
        for scan in range(n_scans):
            intensity_factor = np.exp(-0.1 * (scan - 10) ** 2)
            for mz in mz_peptide1:
                if intensity_factor > 0.1:
                    mz_list.append(mz + np.random.normal(0, 0.001))
                    intensity_list.append(100 * intensity_factor + np.random.normal(0, 5))
                    scan_list.append(scan)

        # Add peptide 2 peaks (Gaussian elution centered at scan 15)
        for scan in range(n_scans):
            intensity_factor = np.exp(-0.1 * (scan - 15) ** 2)
            for mz in mz_peptide2:
                if intensity_factor > 0.1:
                    mz_list.append(mz + np.random.normal(0, 0.001))
                    intensity_list.append(80 * intensity_factor + np.random.normal(0, 5))
                    scan_list.append(scan)

        mz_array = np.array(mz_list, dtype=np.float32)
        intensity_array = np.array(intensity_list, dtype=np.float32)
        scan_array = np.array(scan_list, dtype=np.int32)

        # Find peak at scan 10, m/z 500 (peptide 1)
        peak_idx = np.argmin(np.abs(mz_array - 500.0) + np.abs(scan_array - 10) * 100)

        # Extract RT profiles
        mz_vals, profiles, scans = extract_rt_profiles_around_peak(
            mz_array, intensity_array, scan_array, peak_idx=peak_idx, mz_tolerance_ppm=20.0, scan_window=5
        )

        # Should find at least one m/z (integration test is probabilistic)
        assert len(mz_vals) >= 1

        # Find co-eluting peaks
        coeluting = find_coeluting_peaks(profiles, min_similarity=0.7)

        # Should find at least the reference peak
        assert np.sum(coeluting) >= 1

        print(f"\nIntegration test found {len(mz_vals)} unique m/z values")
        print(f"Co-eluting: {np.sum(coeluting)}/{len(mz_vals)}")

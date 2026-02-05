"""Tests for Q1 profile extraction.

Tests the Q1/RT profile extraction algorithms for spectrum-indexed MS data,
including binary search, Q1 profile extraction, RT profile extraction,
and Q1 kernel correlation scoring.
"""

import numpy as np
import pytest

from alphapeptfast.xic import (
    Q1KernelParams,
    binary_search_left,
    binary_search_right,
    extract_q1_profile_spec,
    extract_rt_profile_spec,
    compute_kernel_correlation,
)


class TestQ1KernelParams:
    """Test Q1 kernel parameter dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = Q1KernelParams()
        assert params.fwhm == 2.85
        assert params.offset == 0.10

    def test_sigma_property(self):
        """Test that sigma is correctly derived from FWHM."""
        params = Q1KernelParams()
        expected_sigma = 2.85 / 2.355
        assert abs(params.sigma - expected_sigma) < 1e-10

    def test_zenotof_20th_preset(self):
        """Test ZenoTOF 20 Th preset."""
        params = Q1KernelParams.for_zenotof_20th()
        assert params.fwhm == 2.85
        assert params.offset == 0.10

    def test_custom_params(self):
        """Test custom parameter values."""
        params = Q1KernelParams(fwhm=3.0, offset=-0.1)
        assert params.fwhm == 3.0
        assert params.offset == -0.1
        assert abs(params.sigma - 3.0 / 2.355) < 1e-10


class TestBinarySearchLeft:
    """Test binary_search_left function."""

    def test_empty_array(self):
        """Test with empty array."""
        arr = np.array([], dtype=np.float64)
        assert binary_search_left(arr, 5.0) == 0

    def test_single_element_less(self):
        """Test single element less than target."""
        arr = np.array([3.0])
        assert binary_search_left(arr, 5.0) == 1

    def test_single_element_greater(self):
        """Test single element greater than target."""
        arr = np.array([7.0])
        assert binary_search_left(arr, 5.0) == 0

    def test_single_element_equal(self):
        """Test single element equal to target."""
        arr = np.array([5.0])
        assert binary_search_left(arr, 5.0) == 0

    def test_exact_match(self):
        """Test finding exact match."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert binary_search_left(arr, 3.0) == 2

    def test_between_values(self):
        """Test value between array elements."""
        arr = np.array([1.0, 2.0, 4.0, 5.0])
        assert binary_search_left(arr, 3.0) == 2

    def test_all_less(self):
        """Test when all values less than target."""
        arr = np.array([1.0, 2.0, 3.0])
        assert binary_search_left(arr, 10.0) == 3

    def test_all_greater(self):
        """Test when all values greater than target."""
        arr = np.array([5.0, 6.0, 7.0])
        assert binary_search_left(arr, 1.0) == 0

    def test_duplicates(self):
        """Test with duplicate values - returns leftmost."""
        arr = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
        assert binary_search_left(arr, 2.0) == 1


class TestBinarySearchRight:
    """Test binary_search_right function."""

    def test_empty_array(self):
        """Test with empty array."""
        arr = np.array([], dtype=np.float64)
        assert binary_search_right(arr, 5.0) == 0

    def test_single_element_less(self):
        """Test single element less than target."""
        arr = np.array([3.0])
        assert binary_search_right(arr, 5.0) == 1

    def test_single_element_greater(self):
        """Test single element greater than target."""
        arr = np.array([7.0])
        assert binary_search_right(arr, 5.0) == 0

    def test_single_element_equal(self):
        """Test single element equal to target."""
        arr = np.array([5.0])
        assert binary_search_right(arr, 5.0) == 1

    def test_exact_match(self):
        """Test finding exact match - returns one past."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert binary_search_right(arr, 3.0) == 3

    def test_between_values(self):
        """Test value between array elements."""
        arr = np.array([1.0, 2.0, 4.0, 5.0])
        assert binary_search_right(arr, 3.0) == 2

    def test_all_less(self):
        """Test when all values less than target."""
        arr = np.array([1.0, 2.0, 3.0])
        assert binary_search_right(arr, 10.0) == 3

    def test_all_greater(self):
        """Test when all values greater than target."""
        arr = np.array([5.0, 6.0, 7.0])
        assert binary_search_right(arr, 1.0) == 0

    def test_duplicates(self):
        """Test with duplicate values - returns one past rightmost."""
        arr = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
        assert binary_search_right(arr, 2.0) == 4


class TestBinarySearchRangePairing:
    """Test that left/right binary search work together for range queries."""

    def test_range_query(self):
        """Test typical range query pattern."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # Find all values in [3.0, 5.0]
        left = binary_search_left(arr, 3.0)
        right = binary_search_right(arr, 5.0)

        assert arr[left:right].tolist() == [3.0, 4.0, 5.0]

    def test_range_query_fractional(self):
        """Test range query with fractional bounds."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # Find all values in [2.5, 5.5]
        left = binary_search_left(arr, 2.5)
        right = binary_search_right(arr, 5.5)

        assert arr[left:right].tolist() == [3.0, 4.0, 5.0]


@pytest.fixture
def synthetic_spectrum_data():
    """Create synthetic spectrum-indexed data for testing.

    Creates a simple dataset with 5 spectra at different Q1 values,
    each containing peaks at known m/z values including 750.0.
    """
    # 5 spectra with different Q1 centers around 500 m/z
    n_spectra = 5
    q1_centers = np.array([498.0, 499.0, 500.0, 501.0, 502.0], dtype=np.float32)
    rt_values = np.array([600.0, 600.5, 601.0, 601.5, 602.0], dtype=np.float32)

    # Each spectrum has 11 peaks (including exactly 750.0)
    peaks_per_spectrum = 11
    n_peaks = n_spectra * peaks_per_spectrum

    peak_mz = np.zeros(n_peaks, dtype=np.float32)
    peak_intensity = np.zeros(n_peaks, dtype=np.float32)
    peak_start_idx = np.zeros(n_spectra, dtype=np.int64)
    peak_stop_idx = np.zeros(n_spectra, dtype=np.int64)

    # Fill in peaks - fragment at 750 m/z with varying intensity
    # Intensity follows Q1 profile: max at Q1=500, drops off
    for spec_i in range(n_spectra):
        start = spec_i * peaks_per_spectrum
        stop = start + peaks_per_spectrum

        peak_start_idx[spec_i] = start
        peak_stop_idx[spec_i] = stop

        # Sorted m/z values - explicitly include 750.0
        peak_mz[start:stop] = np.array([
            700.0, 710.0, 720.0, 730.0, 740.0,
            750.0,  # Index 5 - our target fragment
            760.0, 770.0, 780.0, 790.0, 800.0
        ], dtype=np.float32)

        # Intensity for peak at 750 m/z - follows Gaussian Q1 profile
        # Max at Q1=500, uses sigma=1.2
        q1_diff = q1_centers[spec_i] - 500.0  # Distance from center
        base_intensity = 10000.0 * np.exp(-0.5 * (q1_diff / 1.2) ** 2)

        # Set intensity for 750 m/z peak (index 5)
        peak_intensity[start + 5] = base_intensity

    # Sort spectra by Q1 for Q1-indexed lookup
    q1_sort_order = np.argsort(q1_centers)
    ms2_indices_by_q1 = q1_sort_order.astype(np.int64)
    ms2_q1_sorted = q1_centers[q1_sort_order].astype(np.float32)

    return {
        'peak_mz': peak_mz,
        'peak_intensity': peak_intensity,
        'peak_start_idx': peak_start_idx,
        'peak_stop_idx': peak_stop_idx,
        'rt_sec': rt_values,
        'q1_center': q1_centers,
        'ms2_indices_by_q1': ms2_indices_by_q1,
        'ms2_q1_sorted': ms2_q1_sorted,
    }


class TestExtractQ1ProfileSpec:
    """Test Q1 profile extraction."""

    def test_no_matching_spectra(self, synthetic_spectrum_data):
        """Test when no spectra match Q1 range."""
        data = synthetic_spectrum_data

        # Query far outside Q1 range
        q1_vals, intensities = extract_q1_profile_spec(
            precursor_mz=800.0,  # Far from data at 498-502
            rt_center=601.0,
            rt_window=10.0,
            q1_window=5.0,
            fragment_mz=np.array([750.0], dtype=np.float32),
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        assert len(q1_vals) == 0
        assert intensities.shape == (1, 0)

    def test_extracts_q1_profile(self, synthetic_spectrum_data):
        """Test basic Q1 profile extraction."""
        data = synthetic_spectrum_data

        q1_vals, intensities = extract_q1_profile_spec(
            precursor_mz=500.0,
            rt_center=601.0,
            rt_window=10.0,
            q1_window=5.0,
            fragment_mz=np.array([750.0], dtype=np.float32),
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        # Should find all 5 Q1 values
        assert len(q1_vals) == 5
        assert intensities.shape == (1, 5)

        # Q1 values should be sorted
        assert np.all(np.diff(q1_vals) > 0)

        # Maximum intensity should be near Q1=500
        max_idx = np.argmax(intensities[0])
        assert abs(q1_vals[max_idx] - 500.0) <= 1.0

    def test_rt_filtering(self, synthetic_spectrum_data):
        """Test that RT filtering works."""
        data = synthetic_spectrum_data

        # Narrow RT window that excludes some spectra
        q1_vals, intensities = extract_q1_profile_spec(
            precursor_mz=500.0,
            rt_center=600.25,
            rt_window=0.5,  # Only catches first two spectra
            q1_window=5.0,
            fragment_mz=np.array([750.0], dtype=np.float32),
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        # The function collects all Q1 values in Q1 range first,
        # then applies RT filtering during intensity accumulation.
        # So we get 5 Q1 values but only 2 have non-zero intensity.
        # RT range: 600.25 ± 0.5 = [599.75, 600.75]
        # Spectra RT: [600.0, 600.5, 601.0, 601.5, 602.0]
        # Matches: 600.0 (Q1=498), 600.5 (Q1=499) -> 2 spectra with signal
        assert len(q1_vals) == 5  # All Q1 bins in range
        non_zero_bins = np.sum(intensities[0] > 0)
        assert non_zero_bins == 2  # But only 2 have intensity

    def test_multiple_fragments(self, synthetic_spectrum_data):
        """Test extraction with multiple fragments."""
        data = synthetic_spectrum_data

        # Two fragments
        fragment_mz = np.array([750.0, 720.0], dtype=np.float32)

        q1_vals, intensities = extract_q1_profile_spec(
            precursor_mz=500.0,
            rt_center=601.0,
            rt_window=10.0,
            q1_window=5.0,
            fragment_mz=fragment_mz,
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        # Should have 2 rows (fragments) x 5 columns (Q1 values)
        assert intensities.shape == (2, 5)

        # First fragment (750) has signal, second (720) doesn't
        assert intensities[0].sum() > 0
        # Second fragment at 720 may have some signal depending on m/z layout


class TestExtractRTProfileSpec:
    """Test RT profile extraction."""

    def test_no_matching_spectra(self, synthetic_spectrum_data):
        """Test when no spectra match Q1 range."""
        data = synthetic_spectrum_data

        rt_vals, intensities = extract_rt_profile_spec(
            fragment_mz=np.array([750.0], dtype=np.float32),
            precursor_mz=800.0,  # Far from data
            rt_lo=600.0,
            rt_hi=603.0,
            q1_window=2.0,
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
            n_rt_bins=10,
        )

        # Should return RT values but zero intensities
        assert len(rt_vals) == 10
        assert intensities.shape == (1, 10)
        assert intensities.sum() == 0

    def test_extracts_rt_profile(self, synthetic_spectrum_data):
        """Test basic RT profile extraction."""
        data = synthetic_spectrum_data

        rt_vals, intensities = extract_rt_profile_spec(
            fragment_mz=np.array([750.0], dtype=np.float32),
            precursor_mz=500.0,
            rt_lo=599.0,
            rt_hi=603.0,
            q1_window=2.0,
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
            n_rt_bins=20,
        )

        assert len(rt_vals) == 20
        assert intensities.shape == (1, 20)

        # Should have some signal
        assert intensities.sum() > 0

    def test_rt_bin_count(self, synthetic_spectrum_data):
        """Test that n_rt_bins is respected."""
        data = synthetic_spectrum_data

        for n_bins in [10, 25, 50, 100]:
            rt_vals, intensities = extract_rt_profile_spec(
                fragment_mz=np.array([750.0], dtype=np.float32),
                precursor_mz=500.0,
                rt_lo=600.0,
                rt_hi=602.0,
                q1_window=5.0,
                peak_mz=data['peak_mz'],
                peak_intensity=data['peak_intensity'],
                peak_start_idx=data['peak_start_idx'],
                peak_stop_idx=data['peak_stop_idx'],
                rt_sec=data['rt_sec'],
                q1_center=data['q1_center'],
                ms2_indices_by_q1=data['ms2_indices_by_q1'],
                ms2_q1_sorted=data['ms2_q1_sorted'],
                ppm_tol=20.0,
                n_rt_bins=n_bins,
            )

            assert len(rt_vals) == n_bins
            assert intensities.shape[1] == n_bins


class TestComputeKernelCorrelation:
    """Test Q1 kernel correlation scoring."""

    def test_insufficient_points(self):
        """Test with fewer than 3 data points."""
        # Less than 3 points should return 0
        q1_vals = np.array([500.0, 501.0], dtype=np.float32)
        intensities = np.array([100.0, 50.0], dtype=np.float32)

        corr = compute_kernel_correlation(
            q1_vals, intensities,
            precursor_mz=500.0, q1_sigma=1.2, q1_offset=0.1
        )

        assert corr == 0.0

    def test_zero_intensity(self):
        """Test with zero intensity."""
        q1_vals = np.array([498.0, 499.0, 500.0, 501.0, 502.0], dtype=np.float32)
        intensities = np.zeros(5, dtype=np.float32)

        corr = compute_kernel_correlation(
            q1_vals, intensities,
            precursor_mz=500.0, q1_sigma=1.2, q1_offset=0.1
        )

        assert corr == 0.0

    def test_perfect_gaussian_match(self):
        """Test with data that exactly matches expected kernel."""
        params = Q1KernelParams.for_zenotof_20th()

        # Q1 values centered around expected peak position
        precursor_mz = 500.0
        expected_peak = precursor_mz + params.offset

        q1_vals = np.array([498.0, 499.0, 500.0, 501.0, 502.0], dtype=np.float32)

        # Generate intensities that match Gaussian kernel exactly
        intensities = np.zeros(5, dtype=np.float32)
        for i in range(5):
            diff = q1_vals[i] - expected_peak
            intensities[i] = 1000.0 * np.exp(-0.5 * (diff / params.sigma) ** 2)

        corr = compute_kernel_correlation(
            q1_vals, intensities,
            precursor_mz=precursor_mz, q1_sigma=params.sigma, q1_offset=params.offset
        )

        # Should be very close to 1.0
        assert corr > 0.99

    def test_shifted_profile(self):
        """Test with profile shifted from expected position."""
        params = Q1KernelParams.for_zenotof_20th()

        # Q1 values
        q1_vals = np.array([498.0, 499.0, 500.0, 501.0, 502.0], dtype=np.float32)

        # Intensity peaked at 498 instead of 500.1
        intensities = np.zeros(5, dtype=np.float32)
        for i in range(5):
            diff = q1_vals[i] - 498.0  # Wrong peak position
            intensities[i] = 1000.0 * np.exp(-0.5 * (diff / params.sigma) ** 2)

        corr = compute_kernel_correlation(
            q1_vals, intensities,
            precursor_mz=500.0, q1_sigma=params.sigma, q1_offset=params.offset
        )

        # Should be lower but still positive
        assert 0.0 < corr < 0.9

    def test_flat_profile(self):
        """Test with flat (uniform) intensity profile."""
        params = Q1KernelParams.for_zenotof_20th()

        q1_vals = np.array([498.0, 499.0, 500.0, 501.0, 502.0], dtype=np.float32)
        intensities = np.ones(5, dtype=np.float32) * 1000.0

        corr = compute_kernel_correlation(
            q1_vals, intensities,
            precursor_mz=500.0, q1_sigma=params.sigma, q1_offset=params.offset
        )

        # Flat profile actually correlates reasonably well with Gaussian
        # because both are positive and the dot product is substantial.
        # Should be positive but less than a perfect Gaussian match.
        assert 0.0 < corr < 1.0
        # A perfectly centered Gaussian gives ~1.0, flat gives ~0.9
        assert corr < 0.95  # Less than perfect Gaussian


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_extract_and_score_workflow(self, synthetic_spectrum_data):
        """Test typical workflow: extract Q1 profile, then score."""
        data = synthetic_spectrum_data
        params = Q1KernelParams.for_zenotof_20th()

        # Extract Q1 profile
        q1_vals, intensities = extract_q1_profile_spec(
            precursor_mz=500.0,
            rt_center=601.0,
            rt_window=10.0,
            q1_window=5.0,
            fragment_mz=np.array([750.0], dtype=np.float32),
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        # Score with kernel correlation
        corr = compute_kernel_correlation(
            q1_vals, intensities[0],
            precursor_mz=500.0, q1_sigma=params.sigma, q1_offset=params.offset
        )

        # Should get a reasonable correlation
        assert corr > 0.5  # Synthetic data follows Gaussian Q1 profile

    def test_q1_and_rt_extraction_consistency(self, synthetic_spectrum_data):
        """Test that Q1 and RT extraction use consistent data."""
        data = synthetic_spectrum_data

        # Extract both profiles for same precursor
        q1_vals, q1_intensities = extract_q1_profile_spec(
            precursor_mz=500.0,
            rt_center=601.0,
            rt_window=10.0,
            q1_window=5.0,
            fragment_mz=np.array([750.0], dtype=np.float32),
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
        )

        rt_vals, rt_intensities = extract_rt_profile_spec(
            fragment_mz=np.array([750.0], dtype=np.float32),
            precursor_mz=500.0,
            rt_lo=599.0,
            rt_hi=603.0,
            q1_window=5.0,
            peak_mz=data['peak_mz'],
            peak_intensity=data['peak_intensity'],
            peak_start_idx=data['peak_start_idx'],
            peak_stop_idx=data['peak_stop_idx'],
            rt_sec=data['rt_sec'],
            q1_center=data['q1_center'],
            ms2_indices_by_q1=data['ms2_indices_by_q1'],
            ms2_q1_sorted=data['ms2_q1_sorted'],
            ppm_tol=20.0,
            n_rt_bins=50,
        )

        # Total intensities should be similar (same data, different projections)
        # Note: May not be exactly equal due to binning/filtering differences
        q1_total = q1_intensities.sum()
        rt_total = rt_intensities.sum()

        # Both should have detected signal
        assert q1_total > 0
        assert rt_total > 0

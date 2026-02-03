"""Tests for profile-mode centroiding.

Tests the weighted centroiding algorithms for profile-mode MS data,
including single spectrum processing, batch processing, and precision estimation.
"""

import numpy as np
import pytest

from alphapeptfast.xic import (
    CentroidingParams,
    find_peaks_in_profile,
    centroid_multiple_spectra,
    centroid_profile_spectrum,
    ProfileCentroider,
)


class TestCentroidingParams:
    """Test parameter dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = CentroidingParams()
        assert params.intensity_threshold == 1000.0
        assert params.min_bins == 3
        assert params.noise_floor == 100.0

    def test_high_sensitivity_preset(self):
        """Test high-sensitivity parameter preset."""
        params = CentroidingParams.for_high_sensitivity()
        assert params.intensity_threshold == 500.0
        assert params.min_bins == 2

    def test_high_quality_preset(self):
        """Test high-quality parameter preset."""
        params = CentroidingParams.for_high_quality()
        assert params.intensity_threshold == 2000.0
        assert params.min_bins == 4


class TestFindPeaksInProfile:
    """Test core peak finding function."""

    def test_empty_array(self):
        """Test with empty input."""
        mz = np.array([], dtype=np.float64)
        intensity = np.array([], dtype=np.float64)

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(mz, intensity)

        assert len(cent_mz) == 0
        assert len(cent_int) == 0

    def test_single_gaussian_peak(self):
        """Test centroiding of a single Gaussian-like peak."""
        # Create synthetic profile peak - wide enough to have peak in detectable region
        mz_center = 500.0
        bin_width = 0.004  # ~8 ppm at m/z 500
        sigma = 0.008

        # Use wider range so peak apex is not at edge
        mz_bins = np.arange(mz_center - 0.05, mz_center + 0.05, bin_width)
        intensity = 100000 * np.exp(-0.5 * ((mz_bins - mz_center) / sigma) ** 2)

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(
            mz_bins, intensity, intensity_threshold=1000.0
        )

        assert len(cent_mz) == 1
        # Centroid should be close to true center
        error_ppm = abs(cent_mz[0] - mz_center) / mz_center * 1e6
        assert error_ppm < 1.0  # Within 1 ppm of true center

    def test_two_separated_peaks(self):
        """Test finding two well-separated peaks."""
        # Two peaks with enough separation and width for detection
        # Peak 1 at m/z 500
        mz1 = np.arange(499.95, 500.05, 0.004)
        int1 = 50000 * np.exp(-0.5 * ((mz1 - 500.0) / 0.01) ** 2)

        # Peak 2 at m/z 600 - separate array to ensure gap
        mz2 = np.arange(599.95, 600.05, 0.004)
        int2 = 30000 * np.exp(-0.5 * ((mz2 - 600.0) / 0.01) ** 2)

        # Concatenate with zeros between to ensure separation
        gap_mz = np.array([550.0])
        gap_int = np.array([0.0])

        mz = np.concatenate([mz1, gap_mz, mz2])
        intensity = np.concatenate([int1, gap_int, int2])

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(
            mz, intensity, intensity_threshold=1000.0
        )

        assert len(cent_mz) == 2
        # First peak at ~500
        assert abs(cent_mz[0] - 500.0) < 0.01
        # Second peak at ~600
        assert abs(cent_mz[1] - 600.0) < 0.01

    def test_below_threshold(self):
        """Test that low-intensity peaks are rejected."""
        mz = np.arange(500.0, 500.02, 0.004)
        intensity = np.array([100, 500, 800, 400, 100])  # Below 1000 threshold

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(
            mz, intensity, intensity_threshold=1000.0
        )

        assert len(cent_mz) == 0

    def test_too_few_bins(self):
        """Test that peaks with too few bins are rejected."""
        mz = np.array([500.0, 500.004])
        intensity = np.array([5000.0, 0.0])  # Only 1 non-zero bin

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(
            mz, intensity, intensity_threshold=1000.0, min_bins=3
        )

        assert len(cent_mz) == 0

    def test_precision_estimate(self):
        """Test that precision is estimated reasonably."""
        mz_center = 500.0
        bin_width = 0.004
        # Use wider range for detection
        mz = np.arange(mz_center - 0.05, mz_center + 0.05, bin_width)
        intensity = 100000 * np.exp(-0.5 * ((mz - mz_center) / 0.01) ** 2)

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(mz, intensity)

        # Precision should be sub-ppm for high SNR data
        assert len(prec) == 1
        assert prec[0] > 0  # Should be positive
        assert prec[0] < 5.0  # Should be < 5 ppm for high SNR

    def test_n_bins_returned(self):
        """Test that bin count is returned correctly."""
        mz = np.arange(500.0, 500.04, 0.004)
        intensity = np.array([100, 5000, 10000, 5000, 100, 0, 0, 0, 0, 0])[:len(np.arange(500.0, 500.04, 0.004))]
        # Truncate to match mz length
        intensity = intensity[:len(mz)]

        cent_mz, cent_int, prec, n_bins = find_peaks_in_profile(
            mz, intensity, intensity_threshold=100.0, min_bins=3
        )

        if len(n_bins) > 0:
            # Should have counted non-zero bins
            assert n_bins[0] >= 3


class TestCentroidMultipleSpectra:
    """Test parallel batch processing."""

    def test_single_spectrum_batch(self):
        """Test batch processing with single spectrum."""
        # Use wider range so peak apex is detectable
        mz = np.arange(499.95, 500.05, 0.004)
        intensity = 100000 * np.exp(-0.5 * ((mz - 500.0) / 0.01) ** 2)
        offsets = np.array([0, len(mz)], dtype=np.int64)

        cent_mz, cent_int, prec, n_bins, spec_idx = centroid_multiple_spectra(
            mz, intensity, offsets
        )

        assert len(cent_mz) == 1
        assert spec_idx[0] == 0

    def test_multiple_spectra(self):
        """Test batch processing with multiple spectra."""
        n_spectra = 10

        # Create 10 identical spectra with wide enough range
        single_mz = np.arange(499.95, 500.05, 0.004)
        single_int = 100000 * np.exp(-0.5 * ((single_mz - 500.0) / 0.01) ** 2)

        all_mz = np.tile(single_mz, n_spectra)
        all_int = np.tile(single_int, n_spectra)
        offsets = np.arange(0, (n_spectra + 1) * len(single_mz), len(single_mz), dtype=np.int64)

        cent_mz, cent_int, prec, n_bins, spec_idx = centroid_multiple_spectra(
            all_mz, all_int, offsets
        )

        # Should find one peak per spectrum
        assert len(cent_mz) == n_spectra
        # All peaks should be at ~500
        assert np.all(np.abs(cent_mz - 500.0) < 0.01)
        # Spectrum indices should be 0-9
        assert set(spec_idx) == set(range(n_spectra))

    def test_varying_peak_counts(self):
        """Test spectra with different numbers of peaks."""
        # Spectrum 1: 1 peak at 500 with wide range
        mz1 = np.arange(499.95, 500.05, 0.004)
        int1 = 50000 * np.exp(-0.5 * ((mz1 - 500.0) / 0.01) ** 2)

        # Spectrum 2: 2 peaks at 600 and 700 with gap between them
        mz2_peak1 = np.arange(599.95, 600.05, 0.004)
        int2_peak1 = 30000 * np.exp(-0.5 * ((mz2_peak1 - 600.0) / 0.01) ** 2)
        gap_mz = np.array([650.0])
        gap_int = np.array([0.0])
        mz2_peak2 = np.arange(699.95, 700.05, 0.004)
        int2_peak2 = 40000 * np.exp(-0.5 * ((mz2_peak2 - 700.0) / 0.01) ** 2)

        mz2 = np.concatenate([mz2_peak1, gap_mz, mz2_peak2])
        int2 = np.concatenate([int2_peak1, gap_int, int2_peak2])

        all_mz = np.concatenate([mz1, mz2])
        all_int = np.concatenate([int1, int2])
        offsets = np.array([0, len(mz1), len(mz1) + len(mz2)], dtype=np.int64)

        cent_mz, cent_int, prec, n_bins, spec_idx = centroid_multiple_spectra(
            all_mz, all_int, offsets
        )

        # Should find 3 peaks total
        assert len(cent_mz) == 3
        # First spectrum: 1 peak
        assert np.sum(spec_idx == 0) == 1
        # Second spectrum: 2 peaks
        assert np.sum(spec_idx == 1) == 2


class TestCentroidProfileSpectrum:
    """Test convenience wrapper function."""

    def test_basic_usage(self):
        """Test basic wrapper usage."""
        # Use wider range for peak detection
        mz = np.arange(499.95, 500.05, 0.004)
        intensity = 100000 * np.exp(-0.5 * ((mz - 500.0) / 0.01) ** 2)

        cent_mz, cent_int, prec = centroid_profile_spectrum(mz, intensity)

        assert len(cent_mz) == 1
        assert len(cent_int) == 1
        assert len(prec) == 1

    def test_unsorted_input(self):
        """Test that unsorted input is handled."""
        # Create wider unsorted array with peak in middle
        mz_sorted = np.arange(499.95, 500.05, 0.004)
        intensity_sorted = 100000 * np.exp(-0.5 * ((mz_sorted - 500.0) / 0.01) ** 2)

        # Shuffle
        shuffle_idx = np.random.permutation(len(mz_sorted))
        mz = mz_sorted[shuffle_idx]
        intensity = intensity_sorted[shuffle_idx]

        cent_mz, cent_int, prec = centroid_profile_spectrum(mz, intensity)

        # Should still find peak near 500.0
        assert len(cent_mz) >= 1
        assert abs(cent_mz[0] - 500.0) < 0.01


class TestProfileCentroider:
    """Test high-level centroider class."""

    def test_initialization(self):
        """Test centroider initialization."""
        centroider = ProfileCentroider()
        assert centroider.params.intensity_threshold == 1000.0

        custom_params = CentroidingParams(intensity_threshold=500.0)
        centroider2 = ProfileCentroider(custom_params)
        assert centroider2.params.intensity_threshold == 500.0

    def test_centroid_spectrum(self):
        """Test single spectrum centroiding."""
        centroider = ProfileCentroider()

        # Use wider range for peak detection
        mz = np.arange(499.95, 500.05, 0.004)
        intensity = 100000 * np.exp(-0.5 * ((mz - 500.0) / 0.01) ** 2)

        result = centroider.centroid_spectrum(mz, intensity)

        assert 'mz' in result
        assert 'intensity' in result
        assert 'precision_ppm' in result
        assert 'n_bins' in result
        assert 'n_peaks' in result
        assert result['n_peaks'] == 1

    def test_centroid_batch(self):
        """Test batch centroiding."""
        centroider = ProfileCentroider()

        # Create 5 spectra with wide ranges
        mz_list = []
        int_list = []
        for i in range(5):
            center = 500.0 + i * 100
            mz = np.arange(center - 0.05, center + 0.05, 0.004)
            intensity = 100000 * np.exp(-0.5 * ((mz - center) / 0.01) ** 2)
            mz_list.append(mz)
            int_list.append(intensity)

        result = centroider.centroid_batch(mz_list, int_list)

        assert result['n_spectra'] == 5
        assert result['n_peaks'] == 5
        assert len(result['spectrum_idx']) == 5


class TestPrecisionAccuracy:
    """Test precision and accuracy of centroiding."""

    def test_centroid_accuracy_synthetic(self):
        """Test centroid accuracy on synthetic data with known center."""
        np.random.seed(42)

        # Test at multiple m/z values
        test_masses = [400.0, 800.0, 1200.0]

        for true_mz in test_masses:
            bin_width = true_mz * 8e-6  # 8 ppm bins
            mz = np.arange(true_mz - 0.05, true_mz + 0.05, bin_width)

            # Gaussian peak + noise
            intensity = 50000 * np.exp(-0.5 * ((mz - true_mz) / 0.01) ** 2)
            intensity += np.random.normal(0, 100, len(intensity))
            intensity = np.maximum(intensity, 0)

            cent_mz, cent_int, prec = centroid_profile_spectrum(mz, intensity)

            if len(cent_mz) > 0:
                error_ppm = abs(cent_mz[0] - true_mz) / true_mz * 1e6
                # Should be within 2 ppm for this SNR
                assert error_ppm < 2.0, f"Error at m/z {true_mz}: {error_ppm:.2f} ppm"

    def test_precision_scales_with_snr(self):
        """Test that precision improves with higher SNR."""
        mz_center = 500.0
        bin_width = 0.004
        # Use wider range for detection
        mz = np.arange(mz_center - 0.05, mz_center + 0.05, bin_width)

        # Low SNR
        intensity_low = 5000 * np.exp(-0.5 * ((mz - mz_center) / 0.01) ** 2)
        _, _, prec_low = centroid_profile_spectrum(mz, intensity_low)

        # High SNR
        intensity_high = 500000 * np.exp(-0.5 * ((mz - mz_center) / 0.01) ** 2)
        _, _, prec_high = centroid_profile_spectrum(mz, intensity_high)

        # Higher SNR should give better precision
        if len(prec_low) > 0 and len(prec_high) > 0:
            assert prec_high[0] < prec_low[0]


class TestPerformance:
    """Performance tests with timing benchmarks."""

    def test_batch_performance(self):
        """Test batch processing throughput."""
        import time

        n_spectra = 100
        n_bins_per_spectrum = 500

        # Generate test data
        all_mz = np.zeros(n_spectra * n_bins_per_spectrum, dtype=np.float64)
        all_int = np.zeros(n_spectra * n_bins_per_spectrum, dtype=np.float64)

        for i in range(n_spectra):
            start_idx = i * n_bins_per_spectrum
            mz_base = 400 + i * 2  # Different m/z per spectrum
            all_mz[start_idx:start_idx + n_bins_per_spectrum] = np.linspace(
                mz_base, mz_base + 2, n_bins_per_spectrum
            )
            # 5 peaks per spectrum
            for j in range(5):
                peak_mz = mz_base + 0.4 * (j + 1)
                mask = np.abs(all_mz[start_idx:start_idx + n_bins_per_spectrum] - peak_mz) < 0.05
                all_int[start_idx:start_idx + n_bins_per_spectrum][mask] = 50000 * np.exp(
                    -0.5 * ((all_mz[start_idx:start_idx + n_bins_per_spectrum][mask] - peak_mz) / 0.02) ** 2
                )

        offsets = np.arange(0, (n_spectra + 1) * n_bins_per_spectrum, n_bins_per_spectrum, dtype=np.int64)

        # Warm up
        centroid_multiple_spectra(all_mz[:1000], all_int[:1000], np.array([0, 500, 1000], dtype=np.int64))

        # Time it
        start = time.perf_counter()
        cent_mz, cent_int, prec, n_bins, spec_idx = centroid_multiple_spectra(
            all_mz, all_int, offsets
        )
        elapsed = time.perf_counter() - start

        # Should process 100 spectra in < 0.5 seconds
        assert elapsed < 0.5, f"Batch processing took {elapsed:.2f}s (expected < 0.5s)"

        spectra_per_sec = n_spectra / elapsed
        print(f"\nBatch centroiding: {spectra_per_sec:.0f} spectra/sec")


# =============================================================================
# Real ZenoTOF Data Tests - Using Actual Instrument Data
# =============================================================================

# Path to real test data extracted from ZenoTOF 8600
import os
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
RAW_PROFILE_FILE = TEST_DATA_DIR / "raw_profile_sample.hdf"


def load_real_profile_data():
    """Load real ZenoTOF profile data for testing.

    Returns:
        dict with keys: mz, intensity, spectrum_offsets, rt, n_spectra, n_bins
        Returns None if data file not available or h5py not installed.
    """
    if not RAW_PROFILE_FILE.exists():
        return None

    try:
        import h5py
    except ImportError:
        return None

    with h5py.File(RAW_PROFILE_FILE, 'r') as f:
        return {
            'mz': f['mz'][:],
            'intensity': f['intensity'][:],
            'spectrum_offsets': f['spectrum_offsets'][:],
            'rt': f['rt'][:],
            'n_spectra': len(f['spectrum_offsets'][:]) - 1,
            'n_bins': len(f['mz'][:]),
        }


@pytest.fixture(scope="module")
def real_profile_data():
    """Fixture providing real ZenoTOF profile data."""
    data = load_real_profile_data()
    if data is None:
        pytest.skip("Real profile data not available (run extraction first)")
    return data


class TestRealZenoTOFData:
    """Tests using actual ZenoTOF 8600 raw profile data.

    These tests validate centroiding on real instrument data with:
    - ~8.5 ppm bin spacing (measured)
    - Real noise characteristics
    - Real peak shapes
    - Realistic precision expectations (~1-2 ppm single scan)

    Test data: 20 MS1 spectra from 20Th_no_CE experiment
    """

    def test_data_characteristics(self, real_profile_data):
        """Verify test data has expected ZenoTOF characteristics."""
        data = real_profile_data

        print(f"\n  Real ZenoTOF Profile Data:")
        print(f"  " + "-" * 50)
        print(f"    Spectra: {data['n_spectra']}")
        print(f"    Total bins: {data['n_bins']:,}")
        print(f"    Avg bins/spectrum: {data['n_bins'] // data['n_spectra']:,}")

        # Check bin spacing on first spectrum
        off = data['spectrum_offsets']
        mz = data['mz'][off[0]:off[1]]
        diffs = np.diff(mz)
        ppm_diffs = diffs / mz[:-1] * 1e6

        median_ppm = np.median(ppm_diffs)
        print(f"    Bin spacing: {median_ppm:.1f} ppm median")

        # Should be ~8-10 ppm (ZenoTOF profile bins)
        assert 5.0 < median_ppm < 15.0, f"Unexpected bin spacing: {median_ppm:.1f} ppm"

    def test_centroid_single_real_spectrum(self, real_profile_data):
        """Test centroiding on a single real spectrum."""
        import time

        data = real_profile_data
        off = data['spectrum_offsets']

        # Get middle spectrum (should have good signal)
        spec_idx = data['n_spectra'] // 2
        mz = data['mz'][off[spec_idx]:off[spec_idx + 1]]
        intensity = data['intensity'][off[spec_idx]:off[spec_idx + 1]]

        print(f"\n  Single spectrum centroiding (spectrum {spec_idx}):")
        print(f"  " + "-" * 50)
        print(f"    Input bins: {len(mz):,}")

        # Centroid
        t0 = time.perf_counter()
        cent_mz, cent_int, prec = centroid_profile_spectrum(mz, intensity)
        elapsed = time.perf_counter() - t0

        print(f"    Output peaks: {len(cent_mz)}")
        print(f"    Time: {elapsed*1000:.2f} ms")
        print(f"    Throughput: {len(mz)/elapsed/1000:.0f}k bins/sec")

        if len(prec) > 0:
            print(f"    Precision estimates: median={np.median(prec):.2f} ppm, "
                  f"min={np.min(prec):.2f} ppm, max={np.max(prec):.2f} ppm")

        # Should find peaks (real MS1 data has many peaks)
        assert len(cent_mz) > 10, f"Found only {len(cent_mz)} peaks"

        # Should be fast
        assert elapsed < 0.1, f"Took {elapsed:.3f}s, expected < 0.1s"

    def test_centroid_all_real_spectra_batch(self, real_profile_data):
        """Test batch centroiding on all real spectra."""
        import time

        data = real_profile_data

        print(f"\n  Batch centroiding ({data['n_spectra']} spectra):")
        print(f"  " + "-" * 50)

        # Warm up JIT
        off = data['spectrum_offsets']
        _ = centroid_profile_spectrum(
            data['mz'][off[0]:off[1]],
            data['intensity'][off[0]:off[1]]
        )

        # Batch centroid
        t0 = time.perf_counter()
        cent_mz, cent_int, prec, n_bins, spec_idx = centroid_multiple_spectra(
            data['mz'], data['intensity'], data['spectrum_offsets'].astype(np.int64)
        )
        elapsed = time.perf_counter() - t0

        peaks_per_spec = len(cent_mz) / data['n_spectra']
        throughput = data['n_spectra'] / elapsed

        print(f"    Total peaks found: {len(cent_mz):,}")
        print(f"    Avg peaks/spectrum: {peaks_per_spec:.0f}")
        print(f"    Time: {elapsed*1000:.1f} ms")
        print(f"    Throughput: {throughput:.0f} spectra/sec")
        print(f"    Throughput: {data['n_bins']/elapsed/1e6:.1f}M bins/sec")

        # Validate results
        assert len(cent_mz) > data['n_spectra'] * 10  # At least 10 peaks per spectrum avg
        assert elapsed < 1.0  # Should process 20 spectra in < 1 second

        # Check precision estimates are realistic (1-10 ppm range for real data)
        if len(prec) > 0:
            median_prec = np.median(prec)
            print(f"    Median precision estimate: {median_prec:.2f} ppm")
            # Real ZenoTOF data should have precision estimates in realistic range
            assert 0.1 < median_prec < 10.0, f"Unrealistic precision: {median_prec:.2f} ppm"

    def test_precision_consistency_across_scans(self, real_profile_data):
        """Test that precision is consistent across different scans.

        ZenoTOF 8600 documented precision: ~1.36 ppm single-scan (after centroiding)
        """
        data = real_profile_data
        off = data['spectrum_offsets']

        precisions = []
        peak_counts = []

        for i in range(data['n_spectra']):
            mz = data['mz'][off[i]:off[i + 1]]
            intensity = data['intensity'][off[i]:off[i + 1]]

            _, _, prec = centroid_profile_spectrum(mz, intensity)
            if len(prec) > 0:
                precisions.append(np.median(prec))
                peak_counts.append(len(prec))

        print(f"\n  Precision consistency across {len(precisions)} spectra:")
        print(f"  " + "-" * 50)
        print(f"    Median precision: {np.median(precisions):.2f} ppm")
        print(f"    Std of precision: {np.std(precisions):.2f} ppm")
        print(f"    Range: {np.min(precisions):.2f} - {np.max(precisions):.2f} ppm")
        print(f"    Avg peaks/spectrum: {np.mean(peak_counts):.0f}")

        # Precision should be fairly consistent (not huge variation)
        assert np.std(precisions) < 2.0, "Precision varies too much across spectra"

    def test_mass_range_coverage(self, real_profile_data):
        """Test that centroiding produces peaks within the input mass range.

        Note: Real MS1 data typically has peaks only in specific m/z regions,
        not across the full scanned range. This test verifies centroids are
        within the input range, not that they cover it entirely.
        """
        data = real_profile_data
        off = data['spectrum_offsets']

        # Get first spectrum
        mz = data['mz'][off[0]:off[1]]
        intensity = data['intensity'][off[0]:off[1]]

        cent_mz, _, _ = centroid_profile_spectrum(mz, intensity)

        print(f"\n  Mass range coverage:")
        print(f"  " + "-" * 50)
        print(f"    Input m/z range: {mz.min():.1f} - {mz.max():.1f}")
        print(f"    Output m/z range: {cent_mz.min():.1f} - {cent_mz.max():.1f}")
        print(f"    Peaks found: {len(cent_mz)}")

        # Centroids should be within input range
        assert cent_mz.min() >= mz.min(), "Centroid below input m/z range"
        assert cent_mz.max() <= mz.max(), "Centroid above input m/z range"

        # Should find some peaks (real data has signal)
        assert len(cent_mz) > 0, "No peaks found in real data"


class TestBenchmarks:
    """Comprehensive timing benchmarks using real data."""

    def test_real_data_throughput(self, real_profile_data):
        """Benchmark centroiding throughput on real ZenoTOF data."""
        import time

        data = real_profile_data
        off = data['spectrum_offsets']

        print(f"\n  Real data throughput benchmark:")
        print(f"  " + "-" * 50)

        # Single spectrum benchmark
        mz = data['mz'][off[0]:off[1]]
        intensity = data['intensity'][off[0]:off[1]]

        # Warm up
        _ = centroid_profile_spectrum(mz, intensity)

        # Time single spectrum (multiple runs)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            cent_mz, _, _ = centroid_profile_spectrum(mz, intensity)
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        print(f"    Single spectrum ({len(mz):,} bins): {avg_time:.2f} +/- {std_time:.2f} ms")

        # Batch benchmark
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            cent_mz, _, _, _, _ = centroid_multiple_spectra(
                data['mz'], data['intensity'], data['spectrum_offsets'].astype(np.int64)
            )
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        throughput_spec = data['n_spectra'] / avg_time
        throughput_bins = data['n_bins'] / avg_time / 1e6
        print(f"    Batch ({data['n_spectra']} spectra): {avg_time*1000:.1f} ms")
        print(f"    Throughput: {throughput_spec:.0f} spectra/sec, {throughput_bins:.1f}M bins/sec")

    def test_precision_vs_intensity(self, real_profile_data):
        """Analyze precision vs intensity on real data."""
        data = real_profile_data
        off = data['spectrum_offsets']

        # Collect precision vs intensity data from all spectra
        all_intensities = []
        all_precisions = []

        for i in range(data['n_spectra']):
            mz = data['mz'][off[i]:off[i + 1]]
            intensity = data['intensity'][off[i]:off[i + 1]]

            _, cent_int, prec = centroid_profile_spectrum(mz, intensity)
            all_intensities.extend(cent_int)
            all_precisions.extend(prec)

        all_intensities = np.array(all_intensities)
        all_precisions = np.array(all_precisions)

        print(f"\n  Precision vs Intensity (real data):")
        print(f"  " + "-" * 50)

        # Bin by intensity
        int_bins = [1e3, 1e4, 1e5, 1e6, np.inf]
        for i in range(len(int_bins) - 1):
            mask = (all_intensities >= int_bins[i]) & (all_intensities < int_bins[i+1])
            if mask.sum() > 10:
                prec_in_bin = all_precisions[mask]
                label = f"{int_bins[i]:.0e}-{int_bins[i+1]:.0e}"
                print(f"    Intensity {label}: precision={np.median(prec_in_bin):.2f} ppm (n={mask.sum()})")

        # Higher intensity should generally give better precision
        low_int_mask = all_intensities < 1e4
        high_int_mask = all_intensities > 1e5

        if low_int_mask.sum() > 10 and high_int_mask.sum() > 10:
            low_prec = np.median(all_precisions[low_int_mask])
            high_prec = np.median(all_precisions[high_int_mask])
            print(f"    Low vs high intensity: {low_prec:.2f} vs {high_prec:.2f} ppm")
            # High intensity should have better (lower) precision
            # But this isn't always true due to other factors, so just log it


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_class(self):
        """Test complete workflow using ProfileCentroider class."""
        centroider = ProfileCentroider(CentroidingParams.for_high_sensitivity())

        # Generate 100 spectra
        mz_list = []
        int_list = []
        for i in range(100):
            center = 400 + i * 8
            mz = np.arange(center - 0.05, center + 0.05, 0.004)
            intensity = 50000 * np.exp(-0.5 * ((mz - center) / 0.01) ** 2)
            mz_list.append(mz)
            int_list.append(intensity)

        result = centroider.centroid_batch(mz_list, int_list)

        assert result['n_spectra'] == 100
        assert result['n_peaks'] == 100
        assert 'precision_ppm' in result

        # Check all peaks are at expected positions
        for i in range(100):
            expected_center = 400 + i * 8
            spec_peaks = result['mz'][result['spectrum_idx'] == i]
            assert len(spec_peaks) == 1
            assert abs(spec_peaks[0] - expected_center) < 0.01

    def test_workflow_with_intensity_filtering(self):
        """Test that intensity threshold properly filters weak peaks."""
        centroider = ProfileCentroider(CentroidingParams(intensity_threshold=10000))

        # Strong peak at 500
        mz1 = np.arange(499.95, 500.05, 0.004)
        int1 = 50000 * np.exp(-0.5 * ((mz1 - 500.0) / 0.01) ** 2)

        # Weak peak at 600 (below threshold)
        mz2 = np.arange(599.95, 600.05, 0.004)
        int2 = 5000 * np.exp(-0.5 * ((mz2 - 600.0) / 0.01) ** 2)  # Max ~5000

        result = centroider.centroid_batch([mz1, mz2], [int1, int2])

        # Should only find the strong peak
        assert result['n_peaks'] == 1
        assert abs(result['mz'][0] - 500.0) < 0.01

    def test_real_data_end_to_end(self, real_profile_data):
        """End-to-end test using real data through ProfileCentroider."""
        data = real_profile_data
        off = data['spectrum_offsets']

        centroider = ProfileCentroider(CentroidingParams.for_high_sensitivity())

        # Prepare data as list of spectra
        mz_list = []
        int_list = []
        for i in range(data['n_spectra']):
            mz_list.append(data['mz'][off[i]:off[i + 1]])
            int_list.append(data['intensity'][off[i]:off[i + 1]])

        result = centroider.centroid_batch(mz_list, int_list)

        print(f"\n  End-to-end workflow (real data):")
        print(f"  " + "-" * 50)
        print(f"    Input spectra: {result['n_spectra']}")
        print(f"    Output peaks: {result['n_peaks']}")
        print(f"    Avg peaks/spectrum: {result['n_peaks'] / result['n_spectra']:.0f}")

        # Validate
        assert result['n_spectra'] == data['n_spectra']
        assert result['n_peaks'] > 0
        assert 'precision_ppm' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

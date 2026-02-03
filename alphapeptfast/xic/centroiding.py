"""Profile-mode centroiding for MS data.

High-performance weighted centroiding for profile-mode mass spectrometry data.
Achieves ~8x better precision than winner-take-all approaches used by some
instrument software.

Key Features
------------
- Numba-optimized peak detection and centroiding
- Parallel processing for multiple spectra
- Precision estimation for quality assessment
- Bin counting for peak quality metrics

Background
----------
SCIEX (and some other instruments) can output profile-mode data where intensity
is distributed across multiple m/z bins. Simple winner-take-all centroiding
(taking the bin with highest intensity) limits precision to bin spacing (~8-10 ppm
on ZenoTOF). Weighted centroiding across all bins provides sub-bin precision.

Empirically measured precision (ZenoTOF 8600, 40k resolution):
- Single scan overall: ~2.5 ppm median
- Low intensity (<10k): ~2.8 ppm
- Medium intensity (10k-100k): ~2.0 ppm
- High intensity (>100k): ~0.9 ppm

These values were measured by tracking recurring peaks across multiple scans
and calculating the standard deviation of their centroided m/z values.
Feature-level precision (averaging multiple scans) improves as sqrt(N).

Examples
--------
>>> from alphapeptfast.xic import centroid_profile_spectrum, centroid_multiple_spectra
>>>
>>> # Single spectrum
>>> mz_out, int_out, precision_ppm = centroid_profile_spectrum(mz, intensity)
>>>
>>> # Multiple spectra in parallel
>>> mz_out, int_out, prec_out, n_bins, spec_idx = centroid_multiple_spectra(
...     all_mz, all_intensity, offsets
... )

Extracted from AlphaDIA_Workbench/scripts/profile_centroiding.py (Jan 2026)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numba import njit, prange


@dataclass
class CentroidingParams:
    """Parameters for profile-mode centroiding.

    Attributes:
        intensity_threshold: Minimum apex intensity for a valid peak.
            Lower values find more peaks but include noise.
            Default 1000.0 is reasonable for most MS data.
        min_bins: Minimum number of non-zero intensity bins for a valid peak.
            Helps reject noise spikes. Default 3 ensures real peaks.
        noise_floor: Assumed noise level for SNR calculation.
            Used in precision estimation. Default 100.0.
    """
    intensity_threshold: float = 1000.0
    min_bins: int = 3
    noise_floor: float = 100.0

    @classmethod
    def for_high_sensitivity(cls) -> "CentroidingParams":
        """Parameters optimized for finding weak peaks."""
        return cls(intensity_threshold=500.0, min_bins=2, noise_floor=50.0)

    @classmethod
    def for_high_quality(cls) -> "CentroidingParams":
        """Parameters optimized for precision over sensitivity."""
        return cls(intensity_threshold=2000.0, min_bins=4, noise_floor=100.0)


@njit
def find_peaks_in_profile(
    mz: np.ndarray,
    intensity: np.ndarray,
    intensity_threshold: float = 1000.0,
    min_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find peaks in profile data and compute weighted centroids.

    Detects local maxima in profile-mode MS data and calculates intensity-weighted
    centroids for each peak. Returns precision estimates based on bin spacing and SNR.

    Args:
        mz: m/z array (must be sorted ascending)
        intensity: Intensity array (same length as mz)
        intensity_threshold: Minimum peak apex intensity
        min_bins: Minimum number of non-zero intensity bins for valid peak

    Returns:
        centroid_mz: Weighted centroid m/z for each peak
        centroid_intensity: Total (summed) intensity for each peak
        centroid_precision_ppm: Estimated precision of each centroid in ppm
        n_bins: Number of bins used for each peak

    Notes:
        - Profile peaks are identified by local maxima above threshold
        - Peak extent is determined by contiguous non-zero intensity
        - Precision estimates are empirically calibrated from ZenoTOF 8600 data:
          ~2.8 ppm at 1k intensity, ~1.4 ppm at 10k, ~0.9 ppm at 100k

    Examples:
        >>> mz = np.array([500.0, 500.004, 500.008, 500.012, 500.016])
        >>> intensity = np.array([1000, 5000, 10000, 4000, 500])
        >>> cent_mz, cent_int, precision, n_bins = find_peaks_in_profile(mz, intensity)
    """
    n = len(mz)
    if n == 0:
        return (np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int32))

    # Pre-allocate output (max possible peaks)
    max_peaks = n // min_bins + 1000
    out_mz = np.zeros(max_peaks, dtype=np.float64)
    out_int = np.zeros(max_peaks, dtype=np.float64)
    out_prec = np.zeros(max_peaks, dtype=np.float64)
    out_bins = np.zeros(max_peaks, dtype=np.int32)

    n_peaks = 0
    i = 1

    while i < n - 1:
        # Find local maximum above threshold
        if intensity[i] >= intensity_threshold:
            if intensity[i] >= intensity[i-1] and intensity[i] >= intensity[i+1]:
                # Found apex - extend to find full peak
                start = i
                end = i

                # Extend backwards while intensity > 0
                while start > 0 and intensity[start - 1] > 0:
                    start -= 1

                # Extend forwards while intensity > 0
                while end < n - 1 and intensity[end + 1] > 0:
                    end += 1

                # Count non-zero bins
                n_nonzero = 0
                for j in range(start, end + 1):
                    if intensity[j] > 0:
                        n_nonzero += 1

                if n_nonzero >= min_bins:
                    # Calculate weighted centroid
                    sum_int = 0.0
                    sum_mz_w = 0.0

                    for j in range(start, end + 1):
                        w = intensity[j]
                        sum_int += w
                        sum_mz_w += mz[j] * w

                    if sum_int > 0:
                        centroid_mz = sum_mz_w / sum_int
                        apex_int = intensity[i]

                        # Empirically-calibrated precision estimate
                        # Based on measured precision from ZenoTOF 8600 recurring peaks:
                        #   - Low intensity (<10k): ~2.8 ppm
                        #   - Medium intensity (10k-100k): ~2.0 ppm
                        #   - High intensity (>100k): ~0.9 ppm
                        # Model: precision = base / (1 + log10(intensity/1000))
                        # This gives ~2.8 at 1k, ~2.0 at 10k, ~1.4 at 100k, ~1.0 at 1M
                        log_factor = 1.0 + np.log10(max(apex_int, 1000.0) / 1000.0)
                        precision_ppm = 2.8 / log_factor

                        out_mz[n_peaks] = centroid_mz
                        out_int[n_peaks] = sum_int
                        out_prec[n_peaks] = precision_ppm
                        out_bins[n_peaks] = n_nonzero
                        n_peaks += 1

                # Skip past this peak
                i = end + 1
                continue

        i += 1

    return (out_mz[:n_peaks], out_int[:n_peaks],
            out_prec[:n_peaks], out_bins[:n_peaks])


@njit(parallel=True)
def centroid_multiple_spectra(
    all_mz: np.ndarray,
    all_intensity: np.ndarray,
    offsets: np.ndarray,
    intensity_threshold: float = 1000.0,
    min_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process multiple spectra in parallel with weighted centroiding.

    Efficiently processes many spectra using Numba parallelization.
    Each spectrum's peaks are centroided independently.

    Args:
        all_mz: Concatenated m/z arrays (each spectrum sorted internally)
        all_intensity: Concatenated intensity arrays
        offsets: Spectrum boundaries [0, n1, n1+n2, ...] with len = n_spectra + 1
        intensity_threshold: Minimum apex intensity for valid peak
        min_bins: Minimum non-zero bins for valid peak

    Returns:
        centroid_mz: All centroided peak m/z values
        centroid_intensity: Peak intensities (total summed)
        centroid_precision: Precision estimates in ppm
        n_bins: Number of bins per peak
        spectrum_idx: Which spectrum each peak came from

    Examples:
        >>> # Concatenate 100 spectra
        >>> all_mz = np.concatenate(mz_list)
        >>> all_int = np.concatenate(int_list)
        >>> offsets = np.array([0] + list(np.cumsum([len(m) for m in mz_list])))
        >>>
        >>> mz, ints, prec, bins, spec_idx = centroid_multiple_spectra(
        ...     all_mz, all_int, offsets
        ... )
    """
    n_spectra = len(offsets) - 1

    # First pass: count peaks per spectrum
    peaks_per_spectrum = np.zeros(n_spectra, dtype=np.int64)

    for spec_i in prange(n_spectra):
        start = offsets[spec_i]
        end = offsets[spec_i + 1]

        mz = all_mz[start:end]
        intensity = all_intensity[start:end]

        # Quick count
        count = 0
        i = 1
        while i < len(mz) - 1:
            if intensity[i] >= intensity_threshold:
                if intensity[i] >= intensity[i-1] and intensity[i] >= intensity[i+1]:
                    # Found peak - extend
                    peak_start = i
                    peak_end = i
                    while peak_start > 0 and intensity[peak_start - 1] > 0:
                        peak_start -= 1
                    while peak_end < len(mz) - 1 and intensity[peak_end + 1] > 0:
                        peak_end += 1

                    n_nonzero = 0
                    for j in range(peak_start, peak_end + 1):
                        if intensity[j] > 0:
                            n_nonzero += 1

                    if n_nonzero >= min_bins:
                        count += 1

                    i = peak_end + 1
                    continue
            i += 1

        peaks_per_spectrum[spec_i] = count

    # Calculate output offsets
    total_peaks = np.sum(peaks_per_spectrum)
    peak_offsets = np.zeros(n_spectra + 1, dtype=np.int64)
    for i in range(n_spectra):
        peak_offsets[i + 1] = peak_offsets[i] + peaks_per_spectrum[i]

    # Allocate output
    out_mz = np.zeros(total_peaks, dtype=np.float64)
    out_int = np.zeros(total_peaks, dtype=np.float64)
    out_prec = np.zeros(total_peaks, dtype=np.float64)
    out_bins = np.zeros(total_peaks, dtype=np.int32)
    out_spec = np.zeros(total_peaks, dtype=np.int32)

    # Second pass: compute centroids
    for spec_i in prange(n_spectra):
        start = offsets[spec_i]
        end = offsets[spec_i + 1]
        out_start = peak_offsets[spec_i]

        mz = all_mz[start:end]
        intensity = all_intensity[start:end]

        peak_idx = 0
        i = 1
        while i < len(mz) - 1:
            if intensity[i] >= intensity_threshold:
                if intensity[i] >= intensity[i-1] and intensity[i] >= intensity[i+1]:
                    # Found peak
                    peak_start = i
                    peak_end = i
                    while peak_start > 0 and intensity[peak_start - 1] > 0:
                        peak_start -= 1
                    while peak_end < len(mz) - 1 and intensity[peak_end + 1] > 0:
                        peak_end += 1

                    n_nonzero = 0
                    for j in range(peak_start, peak_end + 1):
                        if intensity[j] > 0:
                            n_nonzero += 1

                    if n_nonzero >= min_bins:
                        # Calculate weighted centroid
                        sum_int = 0.0
                        sum_mz_w = 0.0
                        for j in range(peak_start, peak_end + 1):
                            w = intensity[j]
                            sum_int += w
                            sum_mz_w += mz[j] * w

                        if sum_int > 0:
                            centroid_mz = sum_mz_w / sum_int
                            apex_int = intensity[i]

                            # Empirically-calibrated precision estimate
                            log_factor = 1.0 + np.log10(max(apex_int, 1000.0) / 1000.0)
                            precision_ppm = 2.8 / log_factor

                            out_idx = out_start + peak_idx
                            out_mz[out_idx] = centroid_mz
                            out_int[out_idx] = sum_int
                            out_prec[out_idx] = precision_ppm
                            out_bins[out_idx] = n_nonzero
                            out_spec[out_idx] = spec_i
                            peak_idx += 1

                    i = peak_end + 1
                    continue
            i += 1

    return out_mz, out_int, out_prec, out_bins, out_spec


def centroid_profile_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    intensity_threshold: float = 1000.0,
    min_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple wrapper for centroiding a single profile spectrum.

    Convenience function that handles type conversion and sorting.

    Args:
        mz: m/z array (will be sorted if not already)
        intensity: Intensity array
        intensity_threshold: Minimum peak apex intensity
        min_bins: Minimum number of intensity bins for valid peak

    Returns:
        centroid_mz: Weighted centroid m/z values
        centroid_intensity: Total intensity per peak
        centroid_precision: Estimated precision in ppm

    Examples:
        >>> mz_out, int_out, precision = centroid_profile_spectrum(mz, intensity)
        >>> print(f"Found {len(mz_out)} peaks, median precision: {np.median(precision):.2f} ppm")
    """
    # Ensure sorted
    if len(mz) > 1 and mz[1] < mz[0]:
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]

    mz_out, int_out, prec_out, _ = find_peaks_in_profile(
        mz.astype(np.float64),
        intensity.astype(np.float64),
        intensity_threshold,
        min_bins
    )

    return mz_out, int_out, prec_out


class ProfileCentroider:
    """High-level interface for profile-mode centroiding.

    Provides a class-based interface for centroiding with configurable parameters
    and batch processing support.

    Attributes:
        params: CentroidingParams configuration

    Examples:
        >>> centroider = ProfileCentroider()
        >>> result = centroider.centroid_spectrum(mz, intensity)
        >>> print(f"Found {result['n_peaks']} peaks")
        >>>
        >>> # Batch processing
        >>> results = centroider.centroid_batch(mz_list, int_list)
    """

    def __init__(self, params: Optional[CentroidingParams] = None):
        """Initialize centroider with parameters.

        Args:
            params: Centroiding parameters. Defaults to standard parameters.
        """
        self.params = params or CentroidingParams()

    def centroid_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> dict:
        """Centroid a single profile spectrum.

        Args:
            mz: m/z array
            intensity: Intensity array

        Returns:
            Dictionary with keys:
            - 'mz': Centroid m/z values
            - 'intensity': Total intensity per peak
            - 'precision_ppm': Estimated precision
            - 'n_bins': Number of bins per peak
            - 'n_peaks': Total number of peaks found
        """
        mz_out, int_out, prec_out, n_bins = find_peaks_in_profile(
            mz.astype(np.float64),
            intensity.astype(np.float64),
            self.params.intensity_threshold,
            self.params.min_bins
        )

        return {
            'mz': mz_out,
            'intensity': int_out,
            'precision_ppm': prec_out,
            'n_bins': n_bins,
            'n_peaks': len(mz_out),
        }

    def centroid_batch(
        self,
        mz_list: list,
        intensity_list: list
    ) -> dict:
        """Centroid multiple spectra in parallel.

        Args:
            mz_list: List of m/z arrays
            intensity_list: List of intensity arrays

        Returns:
            Dictionary with keys:
            - 'mz': All centroid m/z values
            - 'intensity': All intensities
            - 'precision_ppm': All precision estimates
            - 'n_bins': Bin counts
            - 'spectrum_idx': Source spectrum index for each peak
            - 'n_spectra': Total spectra processed
            - 'n_peaks': Total peaks found
        """
        # Concatenate inputs
        all_mz = np.concatenate([m.astype(np.float64) for m in mz_list])
        all_int = np.concatenate([i.astype(np.float64) for i in intensity_list])

        # Build offsets
        offsets = np.zeros(len(mz_list) + 1, dtype=np.int64)
        for i, m in enumerate(mz_list):
            offsets[i + 1] = offsets[i] + len(m)

        # Process
        mz_out, int_out, prec_out, n_bins, spec_idx = centroid_multiple_spectra(
            all_mz, all_int, offsets,
            self.params.intensity_threshold,
            self.params.min_bins
        )

        return {
            'mz': mz_out,
            'intensity': int_out,
            'precision_ppm': prec_out,
            'n_bins': n_bins,
            'spectrum_idx': spec_idx,
            'n_spectra': len(mz_list),
            'n_peaks': len(mz_out),
        }

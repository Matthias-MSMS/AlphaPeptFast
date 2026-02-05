"""Fast Q1 profile and RT profile extraction for spectrum-indexed MS data.

High-performance extraction of Q1 profiles and RT chromatograms using
spectrum-indexed arrays and Numba JIT compilation. Achieves 10-50x speedup
over per-spectrum HDF5 reads.

Key Features
------------
- Binary search on Q1-sorted spectrum index for O(log n) lookup
- Binary search per spectrum for fragment m/z matching
- Q1 kernel correlation scoring with calibratable parameters
- RT binning for chromatogram generation

Memory Requirements
-------------------
This module is designed for full-memory operation. Loading spectrum-indexed
data typically requires 15-20GB RAM for large datasets (~2.5B peaks):
- peak_mz: float32 array (~10GB)
- peak_intensity: float32 array (~10GB)
- Index arrays: ~0.5GB

This is a deliberate tradeoff: 10-50x faster extraction vs memory footprint.
For memory-constrained environments, use RT-segment streaming instead.

Q1 Kernel Parameters
--------------------
Q1 transmission profiles are instrument-specific AND window-width-specific.
Default parameters are calibrated for ZenoTOF 8600 with 20 Th windows:
- FWHM: 2.85 Da
- Offset: +0.10 Da (from precursor m/z)

Use calibrate_q1_offset.py to measure offset for your instrument/method.
FWHM may need empirical adjustment for different Q1 window widths.

Data Format: Spectrum-Indexed Arrays
------------------------------------
The functions expect pre-structured "spectrum-indexed" data:
- peak_mz, peak_intensity: Flat arrays of all peaks
- peak_start_idx, peak_stop_idx: Boundaries into peak arrays per spectrum
- rt_sec, q1_center: Per-spectrum metadata
- ms2_indices_by_q1, ms2_q1_sorted: Q1-sorted index for fast lookup

This format avoids expensive global sorting (2.5B elements) while enabling
O(log n) lookups via binary search.

Examples
--------
>>> from alphapeptfast.xic import extract_q1_profile_spec, Q1KernelParams
>>>
>>> # Extract Q1 profile for a precursor
>>> q1_vals, intensities = extract_q1_profile_spec(
...     precursor_mz=500.0,
...     rt_center=600.0,  # seconds
...     rt_window=30.0,
...     q1_window=5.0,
...     fragment_mz=fragment_mz_array,
...     peak_mz=data['peak_mz'],
...     peak_intensity=data['peak_intensity'],
...     peak_start_idx=data['peak_start_idx'],
...     peak_stop_idx=data['peak_stop_idx'],
...     rt_sec=data['rt_sec'],
...     q1_center=data['q1_center'],
...     ms2_indices_by_q1=data['ms2_indices_by_q1'],
...     ms2_q1_sorted=data['ms2_q1_sorted'],
...     ppm_tol=20.0,
... )
>>>
>>> # Score with kernel correlation
>>> params = Q1KernelParams.for_zenotof_20th()
>>> corr = compute_kernel_correlation(
...     q1_vals, intensities[0], precursor_mz, params.sigma, params.offset
... )

Extracted from AlphaDIA_Workbench/scripts/fast_q1_extraction.py (Feb 2026)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numba import njit


@dataclass
class Q1KernelParams:
    """Q1 transmission kernel parameters.

    These are instrument-specific AND window-width-specific.
    Must be calibrated from data (see calibrate_q1_offset.py).

    Attributes:
        fwhm: Full width at half maximum in Da (default 2.85 for ZenoTOF)
        offset: Center offset from precursor m/z in Da (default 0.10)

    Examples:
        >>> params = Q1KernelParams()
        >>> params.sigma  # Gaussian sigma derived from FWHM
        1.210...
        >>>
        >>> # Use preset for ZenoTOF
        >>> params = Q1KernelParams.for_zenotof_20th()
    """
    fwhm: float = 2.85  # Da
    offset: float = 0.10  # Da

    @property
    def sigma(self) -> float:
        """Gaussian sigma derived from FWHM."""
        return self.fwhm / 2.355

    @classmethod
    def for_zenotof_20th(cls) -> "Q1KernelParams":
        """Default parameters for ZenoTOF 8600 with 20 Th windows."""
        return cls(fwhm=2.85, offset=0.10)


@njit(cache=True)
def binary_search_left(arr: np.ndarray, value: float) -> int:
    """Binary search for leftmost position where arr[i] >= value.

    Args:
        arr: Sorted array (ascending order)
        value: Target value to search for

    Returns:
        Index of leftmost position where arr[i] >= value,
        or len(arr) if all values are less than target.

    Notes:
        - Array must be sorted in ascending order
        - Returns insertion point for value to maintain sorted order
        - O(log n) time complexity

    Examples:
        >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> binary_search_left(arr, 3.0)
        2
        >>> binary_search_left(arr, 2.5)
        2
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def binary_search_right(arr: np.ndarray, value: float) -> int:
    """Binary search for rightmost position where arr[i] <= value.

    Args:
        arr: Sorted array (ascending order)
        value: Target value to search for

    Returns:
        Index of first position where arr[i] > value,
        or len(arr) if all values are <= target.

    Notes:
        - Array must be sorted in ascending order
        - Returns one past the rightmost matching position
        - O(log n) time complexity

    Examples:
        >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> binary_search_right(arr, 3.0)
        3
        >>> binary_search_right(arr, 3.5)
        3
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def extract_q1_profile_spec(
    precursor_mz: float,
    rt_center: float,
    rt_window: float,
    q1_window: float,
    fragment_mz: np.ndarray,
    # Spectrum-indexed arrays
    peak_mz: np.ndarray,
    peak_intensity: np.ndarray,
    peak_start_idx: np.ndarray,
    peak_stop_idx: np.ndarray,
    rt_sec: np.ndarray,
    q1_center: np.ndarray,
    # Q1-sorted spectrum index
    ms2_indices_by_q1: np.ndarray,
    ms2_q1_sorted: np.ndarray,
    ppm_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Q1 profile using spectrum-indexed arrays.

    Uses binary search on Q1-sorted spectrum index, then iterates through
    matching spectra and uses binary search for fragment m/z within each spectrum.

    Args:
        precursor_mz: Precursor m/z value
        rt_center: RT center in seconds
        rt_window: RT window half-width in seconds (±)
        q1_window: Q1 window half-width in Da (±)
        fragment_mz: Array of fragment m/z values to extract
        peak_mz: Flat array of all peak m/z values
        peak_intensity: Flat array of all peak intensities
        peak_start_idx: Start index into peak arrays per spectrum
        peak_stop_idx: Stop index into peak arrays per spectrum
        rt_sec: RT in seconds per spectrum
        q1_center: Q1 center m/z per spectrum
        ms2_indices_by_q1: Spectrum indices sorted by Q1
        ms2_q1_sorted: Q1 values in sorted order
        ppm_tol: Mass tolerance in PPM

    Returns:
        q1_values: Unique Q1 centers found in range (sorted)
        intensities: (n_fragments, n_q1) intensity matrix

    Notes:
        - RT is in seconds (not minutes)
        - Returns empty arrays if no spectra match Q1/RT criteria
        - Q1 values are rounded to 2 decimal places for binning
    """
    n_fragments = len(fragment_mz)

    # RT bounds
    rt_lo = rt_center - rt_window
    rt_hi = rt_center + rt_window

    # Q1 bounds
    q1_lo = precursor_mz - q1_window
    q1_hi = precursor_mz + q1_window

    # Find spectrum range for Q1 window using binary search
    spec_start = binary_search_left(ms2_q1_sorted, q1_lo)
    spec_stop = binary_search_right(ms2_q1_sorted, q1_hi)

    if spec_stop <= spec_start:
        return np.zeros(0, dtype=np.float32), np.zeros((n_fragments, 0), dtype=np.float32)

    # Collect unique Q1 values in range
    q1_in_range = ms2_q1_sorted[spec_start:spec_stop]
    unique_q1_set = set()
    for i in range(len(q1_in_range)):
        unique_q1_set.add(round(q1_in_range[i] * 100) / 100)  # Round to 2 decimals

    # Convert to sorted array
    n_q1 = len(unique_q1_set)
    q1_values = np.empty(n_q1, dtype=np.float32)
    i = 0
    for v in sorted(unique_q1_set):
        q1_values[i] = v
        i += 1

    # Pre-compute m/z tolerances
    frag_tol_lo = np.empty(n_fragments, dtype=np.float32)
    frag_tol_hi = np.empty(n_fragments, dtype=np.float32)
    for f_idx in range(n_fragments):
        tol_da = fragment_mz[f_idx] * ppm_tol * 1e-6
        frag_tol_lo[f_idx] = fragment_mz[f_idx] - tol_da
        frag_tol_hi[f_idx] = fragment_mz[f_idx] + tol_da

    # Output intensities: (n_fragments, n_q1)
    intensities = np.zeros((n_fragments, n_q1), dtype=np.float32)

    # Iterate through matching spectra
    for si in range(spec_start, spec_stop):
        spec_idx = ms2_indices_by_q1[si]

        # Check RT filter
        spec_rt = rt_sec[spec_idx]
        if spec_rt < rt_lo or spec_rt > rt_hi:
            continue

        # Find Q1 bin index
        # NOTE: Must cast to float32 to match q1_values array type,
        # otherwise float64 rounding gives 374.15 while float32 gives 374.149993
        # and binary search fails to find the correct bin
        spec_q1 = q1_center[spec_idx]
        q1_rounded = np.float32(round(spec_q1 * 100) / 100)
        q1_idx = binary_search_left(q1_values, q1_rounded)
        if q1_idx >= n_q1:
            continue

        # Get peaks for this spectrum
        p_start = peak_start_idx[spec_idx]
        p_stop = peak_stop_idx[spec_idx]

        if p_stop <= p_start:
            continue

        # Extract each fragment using binary search
        for f_idx in range(n_fragments):
            mz_lo = frag_tol_lo[f_idx]
            mz_hi = frag_tol_hi[f_idx]

            # Binary search for m/z range within spectrum
            left = binary_search_left(peak_mz[p_start:p_stop], mz_lo) + p_start
            right = binary_search_right(peak_mz[p_start:p_stop], mz_hi) + p_start

            # Sum intensities in range
            for pi in range(left, right):
                intensities[f_idx, q1_idx] += peak_intensity[pi]

    return q1_values, intensities


@njit(cache=True)
def extract_rt_profile_spec(
    fragment_mz: np.ndarray,
    precursor_mz: float,
    rt_lo: float,
    rt_hi: float,
    q1_window: float,
    # Spectrum-indexed arrays
    peak_mz: np.ndarray,
    peak_intensity: np.ndarray,
    peak_start_idx: np.ndarray,
    peak_stop_idx: np.ndarray,
    rt_sec: np.ndarray,
    q1_center: np.ndarray,
    # Q1-sorted spectrum index
    ms2_indices_by_q1: np.ndarray,
    ms2_q1_sorted: np.ndarray,
    ppm_tol: float,
    n_rt_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract RT profile (XIC) using spectrum-indexed arrays.

    Args:
        fragment_mz: Array of fragment m/z values to extract
        precursor_mz: Precursor m/z value
        rt_lo: RT lower bound in seconds
        rt_hi: RT upper bound in seconds
        q1_window: Q1 window half-width in Da (±)
        peak_mz: Flat array of all peak m/z values
        peak_intensity: Flat array of all peak intensities
        peak_start_idx: Start index into peak arrays per spectrum
        peak_stop_idx: Stop index into peak arrays per spectrum
        rt_sec: RT in seconds per spectrum
        q1_center: Q1 center m/z per spectrum
        ms2_indices_by_q1: Spectrum indices sorted by Q1
        ms2_q1_sorted: Q1 values in sorted order
        ppm_tol: Mass tolerance in PPM
        n_rt_bins: Number of RT bins (default 50)

    Returns:
        rt_values: RT bin centers in seconds
        intensities: (n_fragments, n_rt_bins) intensity matrix

    Notes:
        - RT is in seconds (not minutes)
        - Spectra are binned into n_rt_bins between rt_lo and rt_hi
    """
    n_fragments = len(fragment_mz)

    # RT bins
    rt_values = np.linspace(rt_lo, rt_hi, n_rt_bins)
    rt_bin_width = (rt_hi - rt_lo) / (n_rt_bins - 1) if n_rt_bins > 1 else 1.0

    # Q1 bounds (narrow window for XIC)
    q1_lo_bound = precursor_mz - q1_window
    q1_hi_bound = precursor_mz + q1_window

    # Find spectrum range for Q1 window
    spec_start = binary_search_left(ms2_q1_sorted, q1_lo_bound)
    spec_stop = binary_search_right(ms2_q1_sorted, q1_hi_bound)

    intensities = np.zeros((n_fragments, n_rt_bins), dtype=np.float32)

    if spec_stop <= spec_start:
        return rt_values, intensities

    # Pre-compute m/z tolerances
    frag_tol_lo = np.empty(n_fragments, dtype=np.float32)
    frag_tol_hi = np.empty(n_fragments, dtype=np.float32)
    for f_idx in range(n_fragments):
        tol_da = fragment_mz[f_idx] * ppm_tol * 1e-6
        frag_tol_lo[f_idx] = fragment_mz[f_idx] - tol_da
        frag_tol_hi[f_idx] = fragment_mz[f_idx] + tol_da

    # Iterate through matching spectra
    for si in range(spec_start, spec_stop):
        spec_idx = ms2_indices_by_q1[si]

        # Check RT filter
        spec_rt = rt_sec[spec_idx]
        if spec_rt < rt_lo or spec_rt > rt_hi:
            continue

        # Find RT bin
        rt_bin = int((spec_rt - rt_lo) / rt_bin_width)
        if rt_bin < 0:
            rt_bin = 0
        elif rt_bin >= n_rt_bins:
            rt_bin = n_rt_bins - 1

        # Get peaks for this spectrum
        p_start = peak_start_idx[spec_idx]
        p_stop = peak_stop_idx[spec_idx]

        if p_stop <= p_start:
            continue

        # Extract each fragment using binary search
        for f_idx in range(n_fragments):
            mz_lo = frag_tol_lo[f_idx]
            mz_hi = frag_tol_hi[f_idx]

            # Binary search for m/z range within spectrum
            left = binary_search_left(peak_mz[p_start:p_stop], mz_lo) + p_start
            right = binary_search_right(peak_mz[p_start:p_stop], mz_hi) + p_start

            # Sum intensities in range
            for pi in range(left, right):
                intensities[f_idx, rt_bin] += peak_intensity[pi]

    return rt_values, intensities


@njit
def compute_kernel_correlation(
    q1_vals: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    q1_sigma: float,
    q1_offset: float,
) -> float:
    """Compute correlation between Q1 profile and expected Gaussian kernel.

    Measures how well the observed Q1 intensity profile matches the expected
    Gaussian transmission profile of the quadrupole.

    Args:
        q1_vals: Q1 center values (m/z)
        intensities: Intensity values at each Q1 position
        precursor_mz: Precursor m/z value
        q1_sigma: Gaussian sigma of Q1 transmission (Da), from Q1KernelParams.sigma
        q1_offset: Q1 center offset from precursor m/z (Da), from Q1KernelParams.offset

    Returns:
        Correlation coefficient [0, 1] between observed and expected profiles.
        Returns 0.0 if insufficient data points (<3) or zero intensity.

    Notes:
        - Higher correlation indicates better match to expected Q1 shape
        - Typical good peptides have correlation > 0.8
        - Use Q1KernelParams to get calibrated sigma and offset values

    Examples:
        >>> params = Q1KernelParams.for_zenotof_20th()
        >>> corr = compute_kernel_correlation(
        ...     q1_vals, intensities, precursor_mz,
        ...     params.sigma, params.offset
        ... )
    """
    if len(q1_vals) < 3:
        return 0.0

    # Expected peak position
    q1_peak = precursor_mz + q1_offset

    # Generate expected Gaussian kernel
    kernel = np.zeros(len(q1_vals), dtype=np.float32)
    for i in range(len(q1_vals)):
        diff = q1_vals[i] - q1_peak
        kernel[i] = np.exp(-0.5 * (diff / q1_sigma) ** 2)

    # Normalize
    kernel_norm = np.sqrt(np.sum(kernel ** 2))
    int_norm = np.sqrt(np.sum(intensities ** 2))

    if kernel_norm < 1e-10 or int_norm < 1e-10:
        return 0.0

    # Correlation
    corr = np.sum(kernel * intensities) / (kernel_norm * int_norm)
    return corr

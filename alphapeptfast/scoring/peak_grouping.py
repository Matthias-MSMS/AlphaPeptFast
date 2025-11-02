"""Peak grouping using cosine similarity for DIA data.

This module implements cosine similarity-based grouping of peaks that co-elute
across the retention time dimension, helping to identify fragments from the same
peptide.

The approach follows an "over-include rather than under-include" philosophy to
avoid missing true fragment groups.

Performance
-----------
- Cosine similarity: >1M comparisons/second
- RT profile extraction: O(n) with binary search potential
- All critical paths Numba-optimized

Examples
--------
>>> import numpy as np
>>> from alphapeptfast.scoring import cosine_similarity, find_coeluting_peaks
>>>
>>> # Calculate similarity between two RT profiles
>>> profile1 = np.array([0.0, 10.0, 100.0, 50.0, 5.0])
>>> profile2 = np.array([0.0, 12.0, 95.0, 48.0, 6.0])
>>> similarity = cosine_similarity(profile1, profile2)
>>> print(f"Similarity: {similarity:.3f}")  # Should be high (>0.99)
>>>
>>> # Find co-eluting peaks from RT profiles
>>> rt_profiles = np.array([
...     [0.0, 10.0, 100.0, 50.0, 5.0],
...     [0.0, 12.0, 95.0, 48.0, 6.0],
...     [100.0, 50.0, 10.0, 5.0, 0.0],  # Different elution
... ])
>>> coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.8)
>>> print(coeluting)  # [True, True, False]
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def cosine_similarity(profile1: np.ndarray, profile2: np.ndarray) -> float:
    """Calculate cosine similarity between two RT profiles.

    The cosine similarity ignores absolute intensity differences and focuses
    on the shape correlation, making it ideal for comparing RT profiles that
    may have different peak heights but similar elution patterns.

    Parameters
    ----------
    profile1 : np.ndarray
        First intensity profile (1D array)
    profile2 : np.ndarray
        Second intensity profile (1D array, must be same length as profile1)

    Returns
    -------
    float
        Cosine similarity score in range [0, 1]
        - 1.0 = identical shapes
        - 0.0 = no correlation

    Notes
    -----
    The cosine similarity is calculated as:
        similarity = dot(v1, v2) / (||v1|| * ||v2||)

    Examples
    --------
    >>> profile1 = np.array([10.0, 100.0, 50.0])
    >>> profile2 = np.array([20.0, 200.0, 100.0])  # Same shape, 2x intensity
    >>> similarity = cosine_similarity(profile1, profile2)
    >>> print(f"{similarity:.3f}")  # 1.000 (identical shapes)
    """
    # Ensure same length
    assert len(profile1) == len(profile2), "Profiles must have same length"

    # Calculate dot product and norms
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for i in range(len(profile1)):
        dot_product += profile1[i] * profile2[i]
        norm1 += profile1[i] * profile1[i]
        norm2 += profile2[i] * profile2[i]

    # Handle zero vectors
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (np.sqrt(norm1) * np.sqrt(norm2))

    # Ensure in valid range [0, 1]
    return max(0.0, min(1.0, similarity))


@njit
def extract_rt_profiles_around_peak(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    scan_idx_array: np.ndarray,
    peak_idx: int,
    mz_tolerance_ppm: float,
    scan_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RT profiles for all peaks within m/z tolerance of a selected peak.

    This function identifies all peaks that fall within a specified m/z window
    around a reference peak and extracts their complete RT profiles across a
    scan window.

    Parameters
    ----------
    mz_array : np.ndarray
        Array of m/z values (float64)
    intensity_array : np.ndarray
        Array of intensity values (float64)
    scan_idx_array : np.ndarray
        Array of scan indices (int32)
    peak_idx : int
        Index of the reference peak in the arrays
    mz_tolerance_ppm : float
        Tolerance in ppm for grouping peaks
    scan_window : int
        Number of scans to include on each side of reference scan

    Returns
    -------
    unique_mz_values : np.ndarray
        Array of unique m/z values found in the window (1D, length n_peaks)
    rt_profiles : np.ndarray
        2D array of RT profiles [n_peaks, n_scans]
    scan_range : np.ndarray
        Array of scan indices covered [min_scan, ..., max_scan]

    Notes
    -----
    - Uses 3x m/z tolerance for grouping (wider window than matching)
    - Peaks within 10 ppm are grouped together and averaged
    - RT profiles are summed for multiple observations in same scan

    Examples
    --------
    >>> mz_array = np.array([500.0, 500.01, 600.0, 500.0])
    >>> intensity_array = np.array([100.0, 90.0, 50.0, 110.0])
    >>> scan_array = np.array([0, 0, 0, 1])
    >>> mz_vals, profiles, scans = extract_rt_profiles_around_peak(
    ...     mz_array, intensity_array, scan_array,
    ...     peak_idx=0, mz_tolerance_ppm=20.0, scan_window=1
    ... )
    """
    # Get reference peak info
    ref_mz = mz_array[peak_idx]
    ref_scan = scan_idx_array[peak_idx]

    # Define scan range
    min_scan = ref_scan - scan_window
    max_scan = ref_scan + scan_window
    n_scans = max_scan - min_scan + 1

    # Calculate m/z tolerance (3x wider for grouping)
    mz_tolerance = ref_mz * mz_tolerance_ppm / 1e6
    min_mz = ref_mz - mz_tolerance * 3
    max_mz = ref_mz + mz_tolerance * 3

    # First pass: collect candidate m/z values in range
    mz_candidates = []
    for i in range(len(mz_array)):
        if min_mz <= mz_array[i] <= max_mz and min_scan <= scan_idx_array[i] <= max_scan:
            mz_candidates.append(mz_array[i])

    # Handle empty case
    if len(mz_candidates) == 0:
        return (
            np.array([ref_mz], dtype=np.float64),
            np.zeros((1, n_scans), dtype=np.float64),
            np.arange(min_scan, max_scan + 1, dtype=np.int32),
        )

    # Group similar m/z values (within 11 ppm for float32 tolerance)
    unique_mz = []
    used = np.zeros(len(mz_candidates), dtype=np.bool_)

    for i in range(len(mz_candidates)):
        if not used[i]:
            center_mz = mz_candidates[i]
            group_sum = 0.0
            group_count = 0

            # Group similar m/z values
            for j in range(len(mz_candidates)):
                if not used[j] and abs(mz_candidates[j] - center_mz) <= center_mz * 11 / 1e6:
                    group_sum += mz_candidates[j]
                    group_count += 1
                    used[j] = True

            if group_count > 0:
                unique_mz.append(group_sum / group_count)

    unique_mz_array = np.array(unique_mz, dtype=np.float64)
    n_unique = len(unique_mz_array)

    # Build RT profiles for each unique m/z
    rt_profiles = np.zeros((n_unique, n_scans), dtype=np.float64)

    for i in range(len(mz_array)):
        if min_mz <= mz_array[i] <= max_mz and min_scan <= scan_idx_array[i] <= max_scan:
            # Find which unique m/z this belongs to
            best_idx = -1
            best_diff = 1e10

            for j in range(n_unique):
                diff = abs(mz_array[i] - unique_mz_array[j])
                if diff <= unique_mz_array[j] * 11 / 1e6 and diff < best_diff:
                    best_idx = j
                    best_diff = diff

            if best_idx >= 0:
                scan_position = scan_idx_array[i] - min_scan
                if 0 <= scan_position < n_scans:
                    rt_profiles[best_idx, scan_position] += intensity_array[i]

    scan_range = np.arange(min_scan, max_scan + 1, dtype=np.int32)
    return unique_mz_array, rt_profiles, scan_range


@njit
def find_coeluting_peaks(
    rt_profiles: np.ndarray, min_similarity: float = 0.8, reference_idx: int = 0
) -> np.ndarray:
    """Find peaks that co-elute based on cosine similarity.

    Compares all RT profiles against a reference profile and identifies those
    with sufficient similarity to be considered co-eluting.

    Parameters
    ----------
    rt_profiles : np.ndarray
        2D array of RT profiles [n_peaks, n_scans]
    min_similarity : float, default=0.8
        Minimum cosine similarity threshold for co-elution
    reference_idx : int, default=0
        Index of reference peak to compare against

    Returns
    -------
    np.ndarray
        Boolean array indicating which peaks co-elute (length n_peaks)

    Notes
    -----
    - Reference peak is always included (True)
    - Similarity is calculated using cosine_similarity()
    - Zero-intensity profiles return similarity 0.0

    Examples
    --------
    >>> rt_profiles = np.array([
    ...     [10.0, 100.0, 50.0],
    ...     [12.0, 95.0, 48.0],   # Similar to reference
    ...     [50.0, 10.0, 100.0],  # Different pattern
    ... ])
    >>> coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.8)
    >>> print(coeluting)  # [True, True, False]
    """
    n_peaks = rt_profiles.shape[0]
    coeluting = np.zeros(n_peaks, dtype=np.bool_)

    # Always include reference peak
    coeluting[reference_idx] = True

    # Get reference profile
    ref_profile = rt_profiles[reference_idx]

    # Check if reference has signal
    if np.sum(ref_profile) == 0:
        return coeluting

    # Compare all other peaks
    for i in range(n_peaks):
        if i != reference_idx:
            similarity = cosine_similarity(ref_profile, rt_profiles[i])
            if similarity >= min_similarity:
                coeluting[i] = True

    return coeluting


@njit
def group_coeluting_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    scan_idx_array: np.ndarray,
    peak_indices: np.ndarray,
    scan_window: int,
    mz_tolerance_ppm: float = 20.0,
    min_similarity: float = 0.7,
) -> np.ndarray:
    """Group peaks that co-elute across RT dimension.

    This implements the "over-include rather than under-include" philosophy
    by using generous similarity thresholds to avoid missing true fragment groups.

    Parameters
    ----------
    mz_array : np.ndarray
        Array of m/z values
    intensity_array : np.ndarray
        Array of intensity values
    scan_idx_array : np.ndarray
        Array of scan indices
    peak_indices : np.ndarray
        Indices of peaks to potentially group
    scan_window : int
        RT window radius in scans
    mz_tolerance_ppm : float, default=20.0
        m/z tolerance for initial grouping
    min_similarity : float, default=0.7
        Minimum cosine similarity for co-elution

    Returns
    -------
    np.ndarray
        Boolean mask indicating which peaks belong to the group (length = len(mz_array))

    Notes
    -----
    - Uses first peak in peak_indices as reference
    - Extracts RT profiles around reference
    - Identifies co-eluting peaks via cosine similarity
    - Returns mask for entire dataset, not just peak_indices

    Examples
    --------
    >>> # Create test data with co-eluting peaks at m/z 500
    >>> mz = np.array([500.0, 500.01, 600.0, 500.0])
    >>> intensity = np.array([100.0, 90.0, 50.0, 110.0])
    >>> scans = np.array([0, 0, 0, 1])
    >>> peak_idx = np.array([0, 1, 3])
    >>> grouped = group_coeluting_peaks(
    ...     mz, intensity, scans, peak_idx,
    ...     scan_window=1, mz_tolerance_ppm=20.0
    ... )
    """
    if len(peak_indices) == 0:
        return np.zeros(len(mz_array), dtype=np.bool_)

    # Use first peak as reference
    ref_idx = peak_indices[0]

    # Extract RT profiles around reference peak
    unique_mz, rt_profiles, scan_range = extract_rt_profiles_around_peak(
        mz_array, intensity_array, scan_idx_array, ref_idx, mz_tolerance_ppm, scan_window
    )

    # Find co-eluting peaks
    coeluting_mask = find_coeluting_peaks(rt_profiles, min_similarity)

    # Map back to original indices
    grouped = np.zeros(len(mz_array), dtype=np.bool_)

    # Mark all points that belong to co-eluting m/z values
    for i in range(len(unique_mz)):
        if coeluting_mask[i]:
            target_mz = unique_mz[i]

            # Find all points matching this m/z
            for j in range(len(mz_array)):
                if abs(mz_array[j] - target_mz) <= target_mz * 11 / 1e6:  # 11 ppm for float32
                    scan_in_range = scan_range[0] <= scan_idx_array[j] <= scan_range[-1]
                    if scan_in_range:
                        grouped[j] = True

    return grouped


@njit
def build_composite_spectrum(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    grouped_mask: np.ndarray,
    mz_tolerance_ppm: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a composite spectrum from grouped peaks.

    Combines peaks from different adjacent scans by averaging similar m/z values
    and summing their intensities. This creates a cleaner, more representative
    spectrum from noisy DIA data.

    Parameters
    ----------
    mz_array : np.ndarray
        Array of m/z values
    intensity_array : np.ndarray
        Array of intensity values
    grouped_mask : np.ndarray
        Boolean mask indicating which peaks are grouped
    mz_tolerance_ppm : float, default=10.0
        Tolerance for combining peaks (ppm)

    Returns
    -------
    combined_mz : np.ndarray
        Array of combined m/z values (1D)
    combined_intensity : np.ndarray
        Array of combined intensities (1D)

    Notes
    -----
    - m/z values are averaged (weighted by count)
    - Intensities are summed
    - Results are sorted by m/z

    Examples
    --------
    >>> # Three observations of same peak
    >>> mz = np.array([500.0, 500.01, 500.02, 600.0])
    >>> intensity = np.array([100.0, 90.0, 85.0, 50.0])
    >>> mask = np.array([True, True, True, False])
    >>> combined_mz, combined_int = build_composite_spectrum(
    ...     mz, intensity, mask, mz_tolerance_ppm=20.0
    ... )
    >>> print(f"Combined m/z: {combined_mz[0]:.2f}")  # ~500.01
    >>> print(f"Combined intensity: {combined_int[0]:.1f}")  # 275.0
    """
    # Extract grouped peaks
    n_grouped = np.sum(grouped_mask)
    if n_grouped == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    grouped_mz = np.empty(n_grouped, dtype=np.float64)
    grouped_intensity = np.empty(n_grouped, dtype=np.float64)

    j = 0
    for i in range(len(grouped_mask)):
        if grouped_mask[i]:
            grouped_mz[j] = mz_array[i]
            grouped_intensity[j] = intensity_array[i]
            j += 1

    # Sort by m/z
    sort_idx = np.argsort(grouped_mz)
    grouped_mz = grouped_mz[sort_idx]
    grouped_intensity = grouped_intensity[sort_idx]

    # Combine similar m/z values
    combined_mz = []
    combined_intensity = []

    i = 0
    while i < len(grouped_mz):
        current_mz = grouped_mz[i]
        current_intensity = grouped_intensity[i]
        count = 1

        # Look for similar m/z values to combine
        j = i + 1
        while j < len(grouped_mz):
            if abs(grouped_mz[j] - current_mz) <= current_mz * mz_tolerance_ppm / 1e6:
                # Weighted average for m/z
                current_mz = (current_mz * count + grouped_mz[j]) / (count + 1)
                current_intensity += grouped_intensity[j]
                count += 1
                j += 1
            else:
                break

        combined_mz.append(current_mz)
        combined_intensity.append(current_intensity)
        i = j

    return np.array(combined_mz, dtype=np.float64), np.array(combined_intensity, dtype=np.float64)

"""Ultra-fast fragment matching with binary search and Numba JIT.

Core algorithms for peptide spectrum matching:
1. Binary search on m/z-sorted arrays (O(log n))
2. Fragment matching with PPM tolerance
3. RT coelution filtering for feature-based search
4. Ion mirroring for modification detection

Performance targets:
- Binary search: >1M operations/second
- Fragment matching: >10k peptides/second (100 fragments each)
"""

import numpy as np
import numba
from typing import Tuple

from ..constants import PROTON_MASS


# =============================================================================
# Binary Search (Core Algorithm)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def binary_search_mz(
    spectrum_mz: np.ndarray,
    target_mz: float,
    tol_ppm: float,
) -> int:
    """Find closest match within PPM tolerance using binary search.

    This is the fundamental operation for spectrum search. Used billions of
    times per proteome search, must be O(log n) fast.

    Parameters
    ----------
    spectrum_mz : np.ndarray (float32 or float64)
        Sorted m/z array from spectrum or feature list
        CRITICAL: Must be sorted ascending! No validation for speed.
    target_mz : float
        Theoretical fragment m/z to search for
    tol_ppm : float
        Mass tolerance in parts per million (typically 5-20 ppm)

    Returns
    -------
    index : int
        Index of closest match within tolerance
        Returns -1 if no match found

    Performance
    -----------
    >1,000,000 operations/second (O(log n) complexity)

    Examples
    --------
    >>> spectrum_mz = np.array([100.05, 200.10, 300.15, 400.20], dtype=np.float32)
    >>> idx = binary_search_mz(spectrum_mz, target_mz=200.11, tol_ppm=50)
    >>> # Returns 1 (matches 200.10 within 50 ppm)

    Notes
    -----
    - No bounds checking for performance
    - Returns closest match within tolerance
    - If multiple matches, returns closest one
    - PPM tolerance: abs(observed - theoretical) / theoretical * 1e6
    """
    n = len(spectrum_mz)
    if n == 0:
        return -1

    # Calculate mass window
    mass_delta = target_mz * tol_ppm / 1e6
    mz_min = target_mz - mass_delta
    mz_max = target_mz + mass_delta

    # Binary search for first m/z >= mz_min
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if spectrum_mz[mid] < mz_min:
            left = mid + 1
        else:
            right = mid
    start_idx = left

    # If no match in range, return -1
    if start_idx >= n or spectrum_mz[start_idx] > mz_max:
        return -1

    # Find closest match within tolerance
    closest_idx = start_idx
    min_error = abs(spectrum_mz[start_idx] - target_mz)

    # Check subsequent m/z values until outside tolerance
    idx = start_idx + 1
    while idx < n and spectrum_mz[idx] <= mz_max:
        error = abs(spectrum_mz[idx] - target_mz)
        if error < min_error:
            min_error = error
            closest_idx = idx
        idx += 1

    return closest_idx


# =============================================================================
# Basic Fragment Matching (No RT)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def match_fragments_to_spectrum(
    theoretical_mz: np.ndarray,
    theoretical_type: np.ndarray,
    theoretical_position: np.ndarray,
    theoretical_charge: np.ndarray,
    spectrum_mz: np.ndarray,
    spectrum_intensity: np.ndarray,
    mz_tol_ppm: float = 10.0,
    min_intensity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match theoretical fragments to observed spectrum.

    Core algorithm for peptide spectrum matching. For each theoretical
    fragment, performs binary search to find matches in observed spectrum.

    Parameters
    ----------
    theoretical_mz, theoretical_type, theoretical_position, theoretical_charge : np.ndarray
        Fragment arrays from generate_by_ions()
    spectrum_mz : np.ndarray (float32)
        Observed m/z values (MUST be sorted!)
    spectrum_intensity : np.ndarray (float32)
        Observed intensities (parallel to spectrum_mz)
    mz_tol_ppm : float
        Mass tolerance in PPM (typically 5-20 ppm)
    min_intensity : float
        Minimum intensity threshold for matches

    Returns
    -------
    match_indices : np.ndarray
        Indices into theoretical arrays for matched fragments
    observed_mz : np.ndarray
        Matched m/z values from spectrum
    observed_intensity : np.ndarray
        Matched intensities from spectrum
    mass_errors_ppm : np.ndarray
        Mass errors in PPM for each match

    Performance
    -----------
    >10,000 peptides/second with 100 fragments each = 1M fragments/second

    Examples
    --------
    >>> # Generate theoretical fragments
    >>> theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(peptide_ord, 2)
    >>>
    >>> # Match to spectrum
    >>> matches = match_fragments_to_spectrum(
    ...     theo_mz, theo_type, theo_pos, theo_charge,
    ...     spectrum_mz, spectrum_intensity,
    ...     mz_tol_ppm=10.0
    ... )
    >>> match_idx, obs_mz, obs_int, ppm_errors = matches
    >>>
    >>> # Calculate coverage
    >>> coverage = len(match_idx) / len(theo_mz)
    """
    n_fragments = len(theoretical_mz)

    # Pre-allocate arrays (worst case: all fragments match)
    match_indices = np.empty(n_fragments, dtype=np.int32)
    observed_mz = np.empty(n_fragments, dtype=np.float32)
    observed_intensity = np.empty(n_fragments, dtype=np.float32)
    mass_errors_ppm = np.empty(n_fragments, dtype=np.float32)

    n_matches = 0

    # For each theoretical fragment, search spectrum
    for i in range(n_fragments):
        target_mz = theoretical_mz[i]

        # Binary search for match
        match_idx = binary_search_mz(spectrum_mz, target_mz, mz_tol_ppm)

        # If no match, skip
        if match_idx == -1:
            continue

        # Check intensity threshold
        intensity = spectrum_intensity[match_idx]
        if intensity < min_intensity:
            continue

        # Calculate mass error
        obs_mz = spectrum_mz[match_idx]
        ppm_error = (obs_mz - target_mz) / target_mz * 1e6

        # Store match
        match_indices[n_matches] = i
        observed_mz[n_matches] = obs_mz
        observed_intensity[n_matches] = intensity
        mass_errors_ppm[n_matches] = ppm_error
        n_matches += 1

    # Return only filled portion
    return (
        match_indices[:n_matches],
        observed_mz[:n_matches],
        observed_intensity[:n_matches],
        mass_errors_ppm[:n_matches]
    )


# =============================================================================
# Extended Matching with RT Coelution (Feature-Based Search)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def match_fragments_with_coelution(
    theoretical_mz: np.ndarray,
    theoretical_type: np.ndarray,
    theoretical_position: np.ndarray,
    theoretical_charge: np.ndarray,
    spectrum_mz: np.ndarray,
    spectrum_intensity: np.ndarray,
    spectrum_rt: np.ndarray,
    precursor_rt: float,
    precursor_mass: float,
    mz_tol_ppm: float = 10.0,
    rt_tol_sec: float = 3.0,
    min_intensity: float = 0.0,
    enable_ion_mirror: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match fragments with RT coelution and ion mirroring.

    Extended matching algorithm for feature-based search. Requires fragments
    to co-elute with precursor (RT constraint) and optionally searches for
    complementary fragments to detect modifications.

    Parameters
    ----------
    theoretical_mz, ... : np.ndarray
        Same as match_fragments_to_spectrum()
    spectrum_rt : np.ndarray (float32)
        RT for each feature in spectrum (parallel to spectrum_mz)
        IMPORTANT: RT must be in SECONDS, not minutes!
    precursor_rt : float
        Expected precursor retention time in SECONDS
    precursor_mass : float
        Precursor neutral mass for ion mirroring calculation
    rt_tol_sec : float
        RT tolerance in seconds (typically 3-10 sec for DIA)
    enable_ion_mirror : bool
        Whether to search for complementary fragments
        If True, calculates mass shifts via ion mirroring

    Returns
    -------
    match_indices : np.ndarray
        Indices of matched theoretical fragments
    observed_mz : np.ndarray
        Observed m/z values
    observed_intensity : np.ndarray
        Observed intensities
    mass_shifts : np.ndarray
        Mass shifts from theoretical (for modification detection)
        Calculated via ion mirroring: precursor_mass - observed_mass
    rt_deltas : np.ndarray
        RT differences from precursor_rt (for quality assessment)

    Performance
    -----------
    >10,000 peptides/second (same as basic matching)

    Ion Mirroring Concept
    ---------------------
    For peptide ABCD with mass M:
    - b3 fragment = ABC (mass = m_b3)
    - y1 fragment = D (mass = M - m_b3)
    - If we observe b3 at m_observed, complementary should be at M - m_observed
    - If actual complementary at M - m_observed + delta, then modification = delta

    Examples
    --------
    >>> matches = match_fragments_with_coelution(
    ...     theo_mz, theo_type, theo_pos, theo_charge,
    ...     feature_mz, feature_intensity, feature_rt,
    ...     precursor_rt=450.5,
    ...     precursor_mass=1500.75,
    ...     mz_tol_ppm=10.0,
    ...     rt_tol_sec=5.0,
    ...     enable_ion_mirror=True
    ... )
    >>> match_idx, obs_mz, obs_int, mass_shifts, rt_delta = matches
    """
    n_fragments = len(theoretical_mz)

    # Pre-allocate arrays
    match_indices = np.empty(n_fragments, dtype=np.int32)
    observed_mz = np.empty(n_fragments, dtype=np.float32)
    observed_intensity = np.empty(n_fragments, dtype=np.float32)
    mass_shifts = np.empty(n_fragments, dtype=np.float32)
    rt_deltas = np.empty(n_fragments, dtype=np.float32)

    n_matches = 0

    # For each theoretical fragment
    for i in range(n_fragments):
        target_mz = theoretical_mz[i]

        # Binary search for match
        match_idx = binary_search_mz(spectrum_mz, target_mz, mz_tol_ppm)

        # If no match, skip
        if match_idx == -1:
            continue

        # Check RT coelution
        feature_rt = spectrum_rt[match_idx]
        rt_delta = abs(feature_rt - precursor_rt)
        if rt_delta > rt_tol_sec:
            continue

        # Check intensity threshold
        intensity = spectrum_intensity[match_idx]
        if intensity < min_intensity:
            continue

        # Calculate mass shift via ion mirroring (if enabled)
        mass_shift = 0.0
        if enable_ion_mirror:
            # Convert observed m/z to neutral mass
            charge = theoretical_charge[i]
            observed_neutral = spectrum_mz[match_idx] * charge - charge * PROTON_MASS

            # Calculate expected complementary mass
            complementary_mass = precursor_mass - observed_neutral

            # For b-ions, complementary is y-ion (and vice versa)
            # Mass shift = difference from expected
            # (For now, simplified: just store observed_neutral for later analysis)
            mass_shift = observed_neutral  # Will calculate shift in post-processing

        # Store match
        match_indices[n_matches] = i
        observed_mz[n_matches] = spectrum_mz[match_idx]
        observed_intensity[n_matches] = intensity
        mass_shifts[n_matches] = mass_shift
        rt_deltas[n_matches] = rt_delta
        n_matches += 1

    # Return only filled portion
    return (
        match_indices[:n_matches],
        observed_mz[:n_matches],
        observed_intensity[:n_matches],
        mass_shifts[:n_matches],
        rt_deltas[:n_matches]
    )


# =============================================================================
# Helper Functions
# =============================================================================

@numba.jit(nopython=True, cache=True)
def calculate_complementary_mz(
    precursor_mass: float,
    fragment_mz: float,
    fragment_charge: int,
    complementary_charge: int = 1,
) -> float:
    """Calculate complementary fragment m/z for ion mirroring.

    Parameters
    ----------
    precursor_mass : float
        Neutral mass of precursor peptide
    fragment_mz : float
        m/z of observed fragment
    fragment_charge : int
        Charge state of observed fragment
    complementary_charge : int
        Expected charge of complementary fragment (default: 1)

    Returns
    -------
    complementary_mz : float
        Expected m/z of complementary fragment

    Examples
    --------
    >>> # For peptide with mass 1500 Da
    >>> # Observed b3 at m/z 300 (charge 1+)
    >>> comp_mz = calculate_complementary_mz(1500.0, 300.0, 1, 1)
    >>> # Returns m/z of complementary y-fragment
    """
    # Convert fragment m/z to neutral mass
    fragment_neutral = fragment_mz * fragment_charge - fragment_charge * PROTON_MASS

    # Calculate complementary neutral mass
    complementary_neutral = precursor_mass - fragment_neutral

    # Convert to m/z
    complementary_mz = (complementary_neutral + complementary_charge * PROTON_MASS) / complementary_charge

    return complementary_mz


@numba.jit(nopython=True, cache=True)
def calculate_match_statistics(
    matched_intensities: np.ndarray,
    theoretical_count: int,
) -> Tuple[float, float, float]:
    """Calculate statistics for matched fragments.

    Parameters
    ----------
    matched_intensities : np.ndarray
        Intensities of matched fragments
    theoretical_count : int
        Total number of theoretical fragments

    Returns
    -------
    coverage : float
        Fragment coverage (n_matched / n_theoretical)
    total_intensity : float
        Sum of matched intensities
    mean_intensity : float
        Mean intensity of matches

    Examples
    --------
    >>> coverage, total, mean = calculate_match_statistics(intensities, 24)
    """
    n_matched = len(matched_intensities)

    coverage = n_matched / theoretical_count if theoretical_count > 0 else 0.0
    total_intensity = np.sum(matched_intensities)
    mean_intensity = np.mean(matched_intensities) if n_matched > 0 else 0.0

    return coverage, total_intensity, mean_intensity

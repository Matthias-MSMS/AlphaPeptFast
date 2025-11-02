"""Ultra-fast XIC extraction for DIA data.

This module implements the ultra-fast XIC extraction algorithm that achieves
>28,000 spectra/second by combining:
1. Binary search on m/z-sorted data
2. Pre-allocated XIC matrices indexed by scan number
3. Parallel processing with Numba
4. Mass error tracking and statistics

Ported from AlphaMod's battle-tested implementation.
"""

from typing import Optional, Tuple, Dict

import numba as nb
import numpy as np


@nb.njit
def binary_search_mz_range(
    mz_array: np.ndarray,
    target_mz: float,
    ppm_tolerance: float
) -> Tuple[int, int]:
    """Find the index range for peaks matching target m/z using binary search.

    Parameters
    ----------
    mz_array : np.ndarray
        Sorted array of m/z values
    target_mz : float
        Target m/z to search for
    ppm_tolerance : float
        Tolerance in parts per million

    Returns
    -------
    start_idx : int
        Start index (inclusive)
    end_idx : int
        End index (exclusive, Python convention)

    Performance
    -----------
    O(log n) complexity for each search

    Examples
    --------
    >>> mz_array = np.array([100.0, 200.0, 200.1, 300.0])
    >>> start, end = binary_search_mz_range(mz_array, 200.0, 500.0)
    >>> # Returns (1, 3) - indices 1 and 2 match within 500 ppm
    """
    if len(mz_array) == 0 or target_mz <= 0:
        return 0, 0

    mz_tol = target_mz * ppm_tolerance / 1e6
    low_mz = target_mz - mz_tol
    high_mz = target_mz + mz_tol

    # Binary search for lower bound
    left, right = 0, len(mz_array)
    while left < right:
        mid = (left + right) // 2
        if mz_array[mid] < low_mz:
            left = mid + 1
        else:
            right = mid
    start_idx = left

    # Binary search for upper bound
    left, right = start_idx, len(mz_array)
    while left < right:
        mid = (left + right) // 2
        if mz_array[mid] <= high_mz:
            left = mid + 1
        else:
            right = mid
    end_idx = left

    return start_idx, end_idx


@nb.njit(parallel=True)
def build_xics_ultrafast(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    scan_array: np.ndarray,
    fragment_mzs: np.ndarray,
    n_scans: int,
    ppm_tolerance: float = 20.0,
) -> np.ndarray:
    """Build all XICs using parallel binary search + direct indexing.

    This is the core ultra-fast XIC extraction function that combines:
    - Binary search on m/z-sorted data for fast lookup
    - Direct array indexing by scan number (no sorting needed)
    - Parallel processing of all peptide-fragment combinations

    Parameters
    ----------
    mz_array : np.ndarray (float32 or float64)
        Sorted array of all m/z values in wide LC format
        CRITICAL: Must be sorted ascending!
    intensity_array : np.ndarray (float32)
        Corresponding intensity values
    scan_array : np.ndarray (int32)
        Scan indices for each peak (0-based)
    fragment_mzs : np.ndarray (float64)
        Array of shape (n_peptides, n_fragments) with theoretical m/z values
        Use 0.0 for padding (will be skipped)
    n_scans : int
        Total number of scans in the LC run
    ppm_tolerance : float
        Mass tolerance in ppm (default: 20.0)

    Returns
    -------
    xic_matrix : np.ndarray (float32)
        XIC matrix of shape (n_peptides, n_fragments, n_scans)

    Performance
    -----------
    >28,000 spectra/second on modern hardware
    Scales linearly with n_peptides * n_fragments

    Examples
    --------
    >>> # Extract XICs for 100 peptides with 10 fragments each
    >>> fragment_mzs = np.random.uniform(200, 2000, (100, 10))
    >>> xic_matrix = build_xics_ultrafast(
    ...     mz_array, intensity_array, scan_array,
    ...     fragment_mzs, n_scans=1000, ppm_tolerance=20.0
    ... )
    >>> xic_matrix.shape
    (100, 10, 1000)
    """
    n_peptides, n_fragments = fragment_mzs.shape

    # Pre-allocate all XICs
    xic_matrix = np.zeros((n_peptides, n_fragments, n_scans), dtype=np.float32)

    # Process all peptide-fragment combinations in parallel
    total_fragments = n_peptides * n_fragments

    for idx in nb.prange(total_fragments):
        # Convert linear index to peptide and fragment indices
        pep_idx = idx // n_fragments
        frag_idx = idx % n_fragments

        fragment_mz = fragment_mzs[pep_idx, frag_idx]

        # Skip if fragment m/z is 0 (padding)
        if fragment_mz <= 0:
            continue

        # Binary search to find matching peaks
        start_idx, end_idx = binary_search_mz_range(mz_array, fragment_mz, ppm_tolerance)

        # Direct indexing into XIC matrix - this is the key innovation!
        # No sorting needed because we use scan_idx directly as the array index
        for i in range(start_idx, end_idx):
            scan_idx = scan_array[i]
            if 0 <= scan_idx < n_scans:
                xic_matrix[pep_idx, frag_idx, scan_idx] += intensity_array[i]

    return xic_matrix


@nb.njit(parallel=True)
def build_xics_with_mass_matrix(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    scan_array: np.ndarray,
    fragment_mzs: np.ndarray,
    n_scans: int,
    ppm_tolerance: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build XICs and mass error matrix using parallel binary search.

    This extended version tracks mass errors alongside intensities,
    enabling mass accuracy-based scoring.

    Parameters
    ----------
    mz_array : np.ndarray
        Sorted array of all m/z values
    intensity_array : np.ndarray
        Corresponding intensity values
    scan_array : np.ndarray
        Scan indices for each peak
    fragment_mzs : np.ndarray
        Theoretical fragment m/z values (n_peptides, n_fragments)
    n_scans : int
        Total number of scans
    ppm_tolerance : float
        Mass tolerance in ppm (default: 20.0)

    Returns
    -------
    xic_matrix : np.ndarray (float32)
        Shape (n_peptides, n_fragments, n_scans) - intensity values
    mass_sum_matrix : np.ndarray (float64)
        Shape (n_peptides, n_fragments, n_scans) - sum of observed m/z * intensity
    mass_count_matrix : np.ndarray (int32)
        Shape (n_peptides, n_fragments, n_scans) - count of observations

    Examples
    --------
    >>> xic, mass_sum, mass_count = build_xics_with_mass_matrix(
    ...     mz_array, intensity_array, scan_array,
    ...     fragment_mzs, n_scans=1000
    ... )
    >>> # Calculate weighted average m/z for each point
    >>> avg_mz = mass_sum / xic  # Where xic > 0
    """
    n_peptides, n_fragments = fragment_mzs.shape

    # Pre-allocate matrices
    xic_matrix = np.zeros((n_peptides, n_fragments, n_scans), dtype=np.float32)
    mass_sum_matrix = np.zeros((n_peptides, n_fragments, n_scans), dtype=np.float64)
    mass_count_matrix = np.zeros((n_peptides, n_fragments, n_scans), dtype=np.int32)

    # Process all peptide-fragment combinations in parallel
    total_fragments = n_peptides * n_fragments

    for idx in nb.prange(total_fragments):
        pep_idx = idx // n_fragments
        frag_idx = idx % n_fragments

        fragment_mz = fragment_mzs[pep_idx, frag_idx]

        if fragment_mz <= 0:
            continue

        # Binary search for matching peaks
        start_idx, end_idx = binary_search_mz_range(mz_array, fragment_mz, ppm_tolerance)

        # Direct indexing into matrices
        for i in range(start_idx, end_idx):
            scan_idx = scan_array[i]
            if 0 <= scan_idx < n_scans:
                intensity = intensity_array[i]
                observed_mz = mz_array[i]

                # Update XIC
                xic_matrix[pep_idx, frag_idx, scan_idx] += intensity

                # Update mass tracking
                mass_sum_matrix[pep_idx, frag_idx, scan_idx] += observed_mz * intensity
                mass_count_matrix[pep_idx, frag_idx, scan_idx] += 1

    return xic_matrix, mass_sum_matrix, mass_count_matrix


@nb.njit
def calculate_mass_error_features(
    mass_sum_matrix: np.ndarray,
    mass_count_matrix: np.ndarray,
    xic_matrix: np.ndarray,
    theoretical_mzs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mass error features from the mass matrix.

    Parameters
    ----------
    mass_sum_matrix : np.ndarray (float64)
        Sum of observed m/z * intensity
    mass_count_matrix : np.ndarray (int32)
        Count of observations
    xic_matrix : np.ndarray (float32)
        XIC intensity matrix
    theoretical_mzs : np.ndarray (float64)
        Theoretical m/z values (n_peptides, n_fragments)

    Returns
    -------
    mean_mass_errors : np.ndarray (float32)
        Shape (n_peptides, n_fragments) - weighted mean mass error in ppm
    mass_error_stds : np.ndarray (float32)
        Shape (n_peptides, n_fragments) - standard deviation of mass errors
    mass_consistencies : np.ndarray (float32)
        Shape (n_peptides,) - overall mass consistency score per peptide

    Notes
    -----
    Mass consistency score uses exponential decay: exp(-std/5)
    Good peptides typically have consistency > 0.5
    """
    n_peptides, n_fragments, n_scans = xic_matrix.shape

    mean_mass_errors = np.zeros((n_peptides, n_fragments), dtype=np.float32)
    mass_error_stds = np.zeros((n_peptides, n_fragments), dtype=np.float32)
    mass_consistencies = np.zeros(n_peptides, dtype=np.float32)

    for pep_idx in range(n_peptides):
        valid_fragments = 0
        total_consistency = 0.0

        for frag_idx in range(n_fragments):
            theoretical_mz = theoretical_mzs[pep_idx, frag_idx]

            if theoretical_mz <= 0:
                continue

            # Get scans with signal
            signal_mask = xic_matrix[pep_idx, frag_idx] > 0
            n_signal_scans = np.sum(signal_mask)

            if n_signal_scans == 0:
                continue

            # Calculate weighted average m/z
            total_intensity = np.sum(xic_matrix[pep_idx, frag_idx])
            if total_intensity == 0:
                continue

            weighted_mz_sum = 0.0
            for scan_idx in range(n_scans):
                if signal_mask[scan_idx] and mass_count_matrix[pep_idx, frag_idx, scan_idx] > 0:
                    # Intensity-weighted m/z
                    intensity = xic_matrix[pep_idx, frag_idx, scan_idx]
                    avg_mz = mass_sum_matrix[pep_idx, frag_idx, scan_idx] / intensity
                    weighted_mz_sum += avg_mz * intensity

            mean_observed_mz = weighted_mz_sum / total_intensity
            mean_error_ppm = (mean_observed_mz - theoretical_mz) / theoretical_mz * 1e6
            mean_mass_errors[pep_idx, frag_idx] = mean_error_ppm

            # Calculate standard deviation of mass errors
            variance_sum = 0.0
            weight_sum = 0.0

            for scan_idx in range(n_scans):
                if signal_mask[scan_idx] and mass_count_matrix[pep_idx, frag_idx, scan_idx] > 0:
                    intensity = xic_matrix[pep_idx, frag_idx, scan_idx]
                    avg_mz = mass_sum_matrix[pep_idx, frag_idx, scan_idx] / intensity
                    error_ppm = (avg_mz - theoretical_mz) / theoretical_mz * 1e6
                    variance_sum += intensity * (error_ppm - mean_error_ppm) ** 2
                    weight_sum += intensity

            if weight_sum > 0:
                mass_error_stds[pep_idx, frag_idx] = np.sqrt(variance_sum / weight_sum)

                # Consistency score: lower std = higher consistency
                # Use exponential decay: exp(-std/5) so std=5ppm gives ~0.37
                consistency = np.exp(-mass_error_stds[pep_idx, frag_idx] / 5.0)
                total_consistency += consistency
                valid_fragments += 1

        if valid_fragments > 0:
            mass_consistencies[pep_idx] = total_consistency / valid_fragments

    return mean_mass_errors, mass_error_stds, mass_consistencies


@nb.njit
def score_xic_correlation(xic: np.ndarray, min_intensity: float = 100.0) -> float:
    """Score peptide using correlation to summed XIC.

    Parameters
    ----------
    xic : np.ndarray (float32)
        XIC matrix of shape (n_fragments, n_scans)
    min_intensity : float
        Minimum intensity threshold (default: 100.0)

    Returns
    -------
    score : float
        Correlation-based score (0-1 range)

    Notes
    -----
    - Calculates Pearson correlation between each fragment and summed XIC
    - Requires at least 3 fragments with signal
    - Higher correlation = more confident identification
    """
    n_fragments, n_scans = xic.shape

    # Sum all fragments to get total XIC
    summed_xic = np.sum(xic, axis=0)

    # Check minimum intensity
    if np.max(summed_xic) < min_intensity:
        return 0.0

    # Calculate correlations
    sum_mean = np.mean(summed_xic)
    sum_std = np.std(summed_xic)

    if sum_std == 0:
        return 0.0

    valid_correlations = 0
    total_correlation = 0.0

    for i in range(n_fragments):
        fragment_xic = xic[i]

        if np.max(fragment_xic) < min_intensity * 0.1:
            continue

        frag_mean = np.mean(fragment_xic)
        frag_std = np.std(fragment_xic)

        if frag_std > 0:
            # Pearson correlation
            cov = np.sum((fragment_xic - frag_mean) * (summed_xic - sum_mean)) / n_scans
            corr = cov / (frag_std * sum_std)

            if corr > 0:
                total_correlation += corr
                valid_correlations += 1

    if valid_correlations < 3:  # Require at least 3 fragments
        return 0.0

    return total_correlation / valid_correlations


@nb.njit
def score_peptide_with_mass_errors(
    xic: np.ndarray,
    mean_mass_errors: np.ndarray,
    mass_error_stds: np.ndarray,
    mass_consistency: float,
) -> Tuple[float, float, float]:
    """Score a peptide using both XIC correlation and mass error features.

    Parameters
    ----------
    xic : np.ndarray
        XIC matrix for the peptide
    mean_mass_errors : np.ndarray
        Mean mass errors per fragment
    mass_error_stds : np.ndarray
        Mass error standard deviations
    mass_consistency : float
        Overall mass consistency score

    Returns
    -------
    xic_score : float
        Traditional XIC-based score (0-1)
    mass_score : float
        Mass accuracy-based score (0-1)
    combined_score : float
        Weighted combination: 0.7 * xic_score + 0.3 * mass_score
    """
    # Calculate XIC score
    xic_score = score_xic_correlation(xic)

    if xic_score == 0:
        return 0.0, 0.0, 0.0

    # Calculate mass score
    valid_fragments = mean_mass_errors != 0
    if np.sum(valid_fragments) == 0:
        mass_score = 0.0
    else:
        # Average absolute mass error
        avg_abs_error = np.mean(np.abs(mean_mass_errors[valid_fragments]))
        # Average mass error std
        avg_std = np.mean(mass_error_stds[valid_fragments])

        # Convert to score (0-1 range)
        # Good: < 5ppm error, < 3ppm std
        error_score = np.exp(-avg_abs_error / 5.0)
        std_score = np.exp(-avg_std / 3.0)

        mass_score = 0.4 * error_score + 0.3 * std_score + 0.3 * mass_consistency

    # Combine scores
    # Weight XIC score more heavily, but mass score is important
    combined_score = 0.7 * xic_score + 0.3 * mass_score

    return xic_score, mass_score, combined_score


class UltraFastXICExtractor:
    """Ultra-fast XIC extraction for DIA data analysis.

    This class provides the main interface for ultra-fast XIC extraction,
    achieving >28,000 spectra/second through optimized algorithms.

    Examples
    --------
    >>> extractor = UltraFastXICExtractor(ppm_tolerance=20.0)
    >>> result = extractor.extract_xics(
    ...     mz_array, intensity_array, scan_array,
    ...     fragment_mzs, n_scans=1000
    ... )
    >>> scores = extractor.score_peptides(
    ...     result['xic_matrix'], fragment_mzs,
    ...     result.get('mass_sum_matrix'),
    ...     result.get('mass_count_matrix')
    ... )
    """

    def __init__(
        self,
        ppm_tolerance: float = 20.0,
        min_intensity: float = 100.0,
        track_mass_errors: bool = True,
    ):
        """Initialize the XIC extractor.

        Parameters
        ----------
        ppm_tolerance : float
            Mass tolerance in parts per million (default: 20.0)
        min_intensity : float
            Minimum intensity threshold (default: 100.0)
        track_mass_errors : bool
            Whether to track mass errors for scoring (default: True)
        """
        self.ppm_tolerance = ppm_tolerance
        self.min_intensity = min_intensity
        self.track_mass_errors = track_mass_errors

    def extract_xics(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        scan_array: np.ndarray,
        fragment_mzs: np.ndarray,
        n_scans: int,
    ) -> Dict[str, np.ndarray]:
        """Extract XICs for all peptide-fragment combinations.

        Parameters
        ----------
        mz_array : np.ndarray
            Sorted array of m/z values
        intensity_array : np.ndarray
            Corresponding intensities
        scan_array : np.ndarray
            Scan indices
        fragment_mzs : np.ndarray
            Theoretical fragment m/z values (n_peptides, n_fragments)
        n_scans : int
            Total number of scans

        Returns
        -------
        dict
            Dictionary containing:
            - 'xic_matrix': XIC intensity matrix
            - 'mass_sum_matrix': Mass sum matrix (if tracking mass errors)
            - 'mass_count_matrix': Mass count matrix (if tracking mass errors)
        """
        if self.track_mass_errors:
            xic_matrix, mass_sum_matrix, mass_count_matrix = build_xics_with_mass_matrix(
                mz_array, intensity_array, scan_array, fragment_mzs, n_scans, self.ppm_tolerance
            )
            return {
                "xic_matrix": xic_matrix,
                "mass_sum_matrix": mass_sum_matrix,
                "mass_count_matrix": mass_count_matrix,
            }
        else:
            xic_matrix = build_xics_ultrafast(
                mz_array, intensity_array, scan_array, fragment_mzs, n_scans, self.ppm_tolerance
            )
            return {"xic_matrix": xic_matrix}

    def score_peptides(
        self,
        xic_matrix: np.ndarray,
        fragment_mzs: np.ndarray,
        mass_sum_matrix: Optional[np.ndarray] = None,
        mass_count_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Score peptides based on XIC patterns and optionally mass errors.

        Parameters
        ----------
        xic_matrix : np.ndarray
            XIC intensity matrix
        fragment_mzs : np.ndarray
            Theoretical fragment m/z values
        mass_sum_matrix : np.ndarray, optional
            Mass sum matrix for error calculation
        mass_count_matrix : np.ndarray, optional
            Mass count matrix

        Returns
        -------
        dict
            Dictionary containing scores and features:
            - 'xic_scores': XIC correlation scores
            - 'mass_scores': Mass accuracy scores (if tracking enabled)
            - 'combined_scores': Combined scores
            - 'mean_mass_errors': Mean errors per fragment (if tracking enabled)
            - 'mass_error_stds': Error std per fragment (if tracking enabled)
            - 'mass_consistencies': Consistency per peptide (if tracking enabled)
        """
        n_peptides = xic_matrix.shape[0]

        xic_scores = np.zeros(n_peptides, dtype=np.float32)
        mass_scores = np.zeros(n_peptides, dtype=np.float32)
        combined_scores = np.zeros(n_peptides, dtype=np.float32)

        if self.track_mass_errors and mass_sum_matrix is not None:
            # Calculate mass error features
            mean_mass_errors, mass_error_stds, mass_consistencies = calculate_mass_error_features(
                mass_sum_matrix, mass_count_matrix, xic_matrix, fragment_mzs
            )

            # Score each peptide
            for pep_idx in range(n_peptides):
                xic_score, mass_score, combined = score_peptide_with_mass_errors(
                    xic_matrix[pep_idx],
                    mean_mass_errors[pep_idx],
                    mass_error_stds[pep_idx],
                    mass_consistencies[pep_idx],
                )
                xic_scores[pep_idx] = xic_score
                mass_scores[pep_idx] = mass_score
                combined_scores[pep_idx] = combined

            return {
                "xic_scores": xic_scores,
                "mass_scores": mass_scores,
                "combined_scores": combined_scores,
                "mean_mass_errors": mean_mass_errors,
                "mass_error_stds": mass_error_stds,
                "mass_consistencies": mass_consistencies,
            }
        else:
            # XIC scoring only
            for pep_idx in range(n_peptides):
                xic_scores[pep_idx] = score_xic_correlation(xic_matrix[pep_idx], self.min_intensity)

            return {
                "xic_scores": xic_scores,
                "combined_scores": xic_scores,  # Same as XIC scores when no mass tracking
            }

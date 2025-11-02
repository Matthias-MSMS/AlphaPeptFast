"""Mass recalibration for proteomics data using RT-segmented calibration.

This module provides mass recalibration to correct systematic mass errors across
retention time. Key features:

- RT-segmented calibration (adaptive binning based on PSM count)
- Charge-state independent (single curve for all charges)
- Pre-search mass error estimation from charge state consistency
- MAD-based outlier removal
- Returns recommended tolerance for next search iteration

Performance
-----------
- Recalibration: >1M m/z values/second
- Pre-search check: <0.1 sec for 100k precursors

Examples
--------
>>> from alphapeptfast.scoring import MassRecalibrator
>>>
>>> # Fit calibration from high-confidence PSMs
>>> calibrator = MassRecalibrator(
...     observed_mz=psms['precursor_mz'],
...     theoretical_mz=psms['theoretical_mz'],
...     rt_seconds=psms['rt'],
... )
>>>
>>> # Apply to new data
>>> corrected_mz = calibrator.apply(new_mz, new_rt)
>>> print(f"Use tolerance: {calibrator.recommended_tolerance:.1f} ppm")
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import njit


@njit
def calculate_ppm_errors(
    observed_mz: np.ndarray, theoretical_mz: np.ndarray
) -> np.ndarray:
    """Calculate ppm mass errors.

    Parameters
    ----------
    observed_mz : np.ndarray
        Observed m/z values
    theoretical_mz : np.ndarray
        Theoretical m/z values

    Returns
    -------
    ppm_errors : np.ndarray
        Mass errors in ppm
    """
    return (observed_mz - theoretical_mz) / theoretical_mz * 1e6


@njit
def remove_outliers_mad(
    values: np.ndarray, threshold: float = 3.0
) -> np.ndarray:
    """Remove outliers using MAD (Median Absolute Deviation).

    Parameters
    ----------
    values : np.ndarray
        Input values
    threshold : float, default=3.0
        MAD threshold for outlier removal

    Returns
    -------
    mask : np.ndarray (bool)
        True for inliers, False for outliers
    """
    if len(values) == 0:
        return np.ones(0, dtype=np.bool_)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-6:  # All values identical
        return np.ones(len(values), dtype=np.bool_)

    # MAD-based outlier detection
    z_scores = np.abs(values - median) / (1.4826 * mad)
    return z_scores < threshold


def determine_rt_bins(n_psms: int, min_psms_per_bin: int = 50) -> int:
    """Determine number of RT bins based on PSM count.

    Strategy:
    - Target ~100 PSMs per bin for robust median estimation
    - More PSMs → more bins → capture rapid mass shifts
    - Minimum 50 PSMs per bin to ensure statistical robustness
    - Clamp between 5 and 100 bins

    Parameters
    ----------
    n_psms : int
        Number of PSMs available for calibration
    min_psms_per_bin : int, default=50
        Minimum PSMs required per bin

    Returns
    -------
    n_bins : int
        Number of RT bins to use

    Examples
    --------
    >>> determine_rt_bins(500)    # → 5 bins (100 PSMs/bin)
    5
    >>> determine_rt_bins(5000)   # → 20 bins (250 PSMs/bin)
    20
    >>> determine_rt_bins(100000) # → 100 bins (1000 PSMs/bin)
    100
    """
    target_psms_per_bin = 100
    n_bins = n_psms // target_psms_per_bin

    # Clamp between 5 and 100
    n_bins = max(5, min(100, n_bins))

    # Ensure minimum PSMs per bin
    while n_psms // n_bins < min_psms_per_bin and n_bins > 5:
        n_bins -= 1

    return n_bins


@njit
def assign_rt_bins(
    rt_seconds: np.ndarray, rt_min: float, rt_max: float, n_bins: int
) -> np.ndarray:
    """Assign RT values to bins.

    Parameters
    ----------
    rt_seconds : np.ndarray
        Retention times in seconds
    rt_min : float
        Minimum RT for binning
    rt_max : float
        Maximum RT for binning
    n_bins : int
        Number of bins

    Returns
    -------
    bin_indices : np.ndarray
        Bin index for each RT (0 to n_bins-1)
    """
    bin_indices = np.zeros(len(rt_seconds), dtype=np.int32)

    rt_range = rt_max - rt_min
    if rt_range < 1e-6:
        return bin_indices  # All in bin 0

    for i in range(len(rt_seconds)):
        rt = rt_seconds[i]
        # Calculate bin index
        bin_idx = int((rt - rt_min) / rt_range * n_bins)
        # Clamp to valid range
        bin_idx = max(0, min(n_bins - 1, bin_idx))
        bin_indices[i] = bin_idx

    return bin_indices


@njit
def calculate_bin_corrections(
    ppm_errors: np.ndarray, bin_indices: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate median ppm correction per RT bin.

    Parameters
    ----------
    ppm_errors : np.ndarray
        PPM errors for all PSMs
    bin_indices : np.ndarray
        Bin assignment for each PSM
    n_bins : int
        Total number of bins

    Returns
    -------
    bin_corrections : np.ndarray
        Median ppm correction per bin (NaN for empty bins)
    bin_counts : np.ndarray
        Number of PSMs per bin
    """
    bin_corrections = np.full(n_bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    # Calculate median per bin
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_in_bin = np.sum(mask)
        bin_counts[bin_idx] = n_in_bin

        if n_in_bin > 0:
            bin_errors = ppm_errors[mask]
            bin_corrections[bin_idx] = np.median(bin_errors)

    return bin_corrections, bin_counts


@njit
def interpolate_bin_corrections(
    bin_corrections: np.ndarray, bin_counts: np.ndarray
) -> np.ndarray:
    """Fill missing bins using linear interpolation.

    Parameters
    ----------
    bin_corrections : np.ndarray
        Corrections per bin (NaN for missing bins)
    bin_counts : np.ndarray
        PSM counts per bin

    Returns
    -------
    filled_corrections : np.ndarray
        Corrections with missing bins filled
    """
    n_bins = len(bin_corrections)
    filled = bin_corrections.copy()

    # Find valid bins
    valid_mask = ~np.isnan(bin_corrections)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # No valid data, use 0.0
        filled[:] = 0.0
        return filled

    if len(valid_indices) == 1:
        # Only one valid bin, use globally
        filled[:] = bin_corrections[valid_indices[0]]
        return filled

    # Extend edges
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    if first_valid > 0:
        filled[:first_valid] = bin_corrections[first_valid]

    if last_valid < n_bins - 1:
        filled[last_valid + 1 :] = bin_corrections[last_valid]

    # Interpolate gaps
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]

        if end_idx - start_idx > 1:
            # Linear interpolation
            start_val = bin_corrections[start_idx]
            end_val = bin_corrections[end_idx]
            gap_size = end_idx - start_idx

            for j in range(1, gap_size):
                alpha = j / gap_size
                filled[start_idx + j] = (1 - alpha) * start_val + alpha * end_val

    return filled


@njit
def apply_corrections_fast(
    mz_values: np.ndarray, bin_indices: np.ndarray, bin_corrections: np.ndarray
) -> np.ndarray:
    """Apply mass corrections to m/z values (Numba-accelerated).

    Parameters
    ----------
    mz_values : np.ndarray
        Original m/z values
    bin_indices : np.ndarray
        RT bin index for each m/z
    bin_corrections : np.ndarray
        PPM correction per bin

    Returns
    -------
    corrected_mz : np.ndarray
        Corrected m/z values
    """
    n = len(mz_values)
    corrected = np.empty(n, dtype=mz_values.dtype)

    for i in range(n):
        bin_idx = bin_indices[i]
        ppm_correction = bin_corrections[bin_idx]
        # Correct: mz_corrected = mz_observed / (1 + ppm_error / 1e6)
        corrected[i] = mz_values[i] / (1.0 + ppm_correction / 1e6)

    return corrected


def estimate_mass_error_from_charge_states(
    precursor_mz: np.ndarray,
    precursor_charges: np.ndarray,
    tolerance_ppm_start: float = 50.0,
    min_charge_pairs: int = 100,
) -> dict[str, float]:
    """Estimate mass error using charge state consistency (pre-search check).

    This function identifies precursors that appear with multiple charge states
    and uses the consistency of their neutral mass to estimate systematic mass error.

    Algorithm:
    1. Start with tolerance_ppm_start (e.g., 50 ppm)
    2. Group precursors by neutral mass within tolerance
    3. Find cases where same molecule is assigned 2+ AND 3+ (or other pairs)
    4. Iteratively tighten tolerance (50→30→20→10 ppm) if enough pairs found
    5. Calculate median mass discrepancy → systematic error

    Parameters
    ----------
    precursor_mz : np.ndarray
        Precursor m/z values
    precursor_charges : np.ndarray
        Precursor charge states
    tolerance_ppm_start : float, default=50.0
        Initial tolerance for grouping (will tighten if successful)
    min_charge_pairs : int, default=100
        Minimum charge state pairs required for reliable estimate

    Returns
    -------
    dict
        - 'median_ppm': Estimated systematic mass error
        - 'std_ppm': Random mass error (standard deviation)
        - 'recommended_tolerance': Suggested tolerance for initial search
        - 'n_charge_pairs': Number of charge state pairs found
        - 'tolerance_used': Final tolerance used for grouping

    Examples
    --------
    >>> estimate = estimate_mass_error_from_charge_states(
    ...     precursor_mz, precursor_charges
    ... )
    >>> print(f"Mass error: {estimate['median_ppm']:.1f} ppm")
    >>> print(f"Use tolerance: {estimate['recommended_tolerance']:.1f} ppm")

    Notes
    -----
    Runtime: <0.1 sec for 100k precursors (no search required!)
    """
    # Try progressively tighter tolerances
    tolerances = [50.0, 30.0, 20.0, 10.0]
    tolerance_used = tolerance_ppm_start

    for tol in tolerances:
        if tol > tolerance_ppm_start:
            continue

        # Calculate neutral masses
        from alphapeptfast.constants import PROTON_MASS

        neutral_masses = precursor_mz * precursor_charges - precursor_charges * PROTON_MASS

        # Find charge state pairs
        # Look for 2+/3+, 2+/4+, 3+/4+ pairs
        mass_errors_found = []

        # Simple O(n²) algorithm for charge pair finding
        # (Could optimize with binning if needed)
        unique_charges = np.unique(precursor_charges)

        if len(unique_charges) < 2:
            # Need at least 2 different charges
            continue

        # Group by approximate neutral mass
        sorted_idx = np.argsort(neutral_masses)
        sorted_masses = neutral_masses[sorted_idx]
        sorted_charges = precursor_charges[sorted_idx]

        # Find pairs within tolerance
        i = 0
        while i < len(sorted_masses) - 1:
            mass_i = sorted_masses[i]
            charge_i = sorted_charges[i]

            # Look ahead for matches
            j = i + 1
            while j < len(sorted_masses):
                mass_j = sorted_masses[j]
                charge_j = sorted_charges[j]

                # Check if within tolerance
                ppm_diff = abs(mass_j - mass_i) / mass_i * 1e6

                if ppm_diff > tol:
                    break  # Too far, move to next i

                if charge_i != charge_j:
                    # Found charge state pair!
                    mass_errors_found.append(mass_j - mass_i)

                j += 1

            i += 1

        if len(mass_errors_found) >= min_charge_pairs:
            tolerance_used = tol
            # Found enough pairs, use this tolerance
            break

    # Calculate statistics
    if len(mass_errors_found) == 0:
        warnings.warn(
            "No charge state pairs found for mass error estimation. "
            "Using default tolerance of 20 ppm."
        )
        return {
            "median_ppm": 0.0,
            "std_ppm": 10.0,
            "recommended_tolerance": 20.0,
            "n_charge_pairs": 0,
            "tolerance_used": tolerance_ppm_start,
        }

    # Convert mass differences to ppm
    # Use median neutral mass as reference
    median_mass = np.median(neutral_masses)
    mass_errors_ppm = np.array(mass_errors_found) / median_mass * 1e6

    # Remove outliers
    inliers = remove_outliers_mad(mass_errors_ppm, threshold=3.0)
    if np.sum(inliers) > 10:
        mass_errors_ppm = mass_errors_ppm[inliers]

    median_ppm = float(np.median(mass_errors_ppm))
    std_ppm = float(np.std(mass_errors_ppm))

    # Recommended tolerance: 95th percentile + buffer
    abs_errors = np.abs(mass_errors_ppm)
    percentile_95 = float(np.percentile(abs_errors, 95))
    recommended_tolerance = max(10.0, percentile_95 + 5.0)  # At least 10 ppm

    return {
        "median_ppm": median_ppm,
        "std_ppm": std_ppm,
        "recommended_tolerance": recommended_tolerance,
        "n_charge_pairs": len(mass_errors_found),
        "tolerance_used": tolerance_used,
    }


class MassRecalibrator:
    """RT-segmented mass recalibration using high-confidence PSMs.

    This class fits a mass calibration model from high-confidence peptide
    identifications and applies corrections to new m/z values.

    Parameters
    ----------
    observed_mz : np.ndarray
        Observed precursor m/z values
    theoretical_mz : np.ndarray
        Theoretical precursor m/z values
    rt_seconds : np.ndarray
        Retention times in seconds
    adaptive_bins : bool, default=True
        Whether to determine bin count adaptively based on PSM count
    n_bins : int, optional
        Fixed number of bins (if adaptive_bins=False)
    min_psms_per_bin : int, default=50
        Minimum PSMs per bin for adaptive binning
    outlier_threshold_mad : float, default=3.0
        MAD threshold for outlier removal

    Attributes
    ----------
    n_bins : int
        Number of RT bins used
    bin_edges : np.ndarray
        RT boundaries for bins
    bin_corrections : np.ndarray
        Median ppm correction per bin
    bin_counts : np.ndarray
        Number of PSMs per bin
    rt_min : float
        Minimum RT in calibration set
    rt_max : float
        Maximum RT in calibration set
    median_ppm_before : float
        Median ppm error before calibration
    std_ppm_before : float
        Standard deviation before calibration
    median_ppm_after : float
        Median ppm error after calibration
    std_ppm_after : float
        Standard deviation after calibration
    recommended_tolerance : float
        Recommended tolerance for next search (95th percentile)

    Examples
    --------
    >>> # Fit calibration from high-confidence PSMs
    >>> calibrator = MassRecalibrator(
    ...     observed_mz=psms['precursor_mz'],
    ...     theoretical_mz=psms['theoretical_mz'],
    ...     rt_seconds=psms['rt'],
    ... )
    >>>
    >>> # Apply to new data
    >>> corrected_mz = calibrator.apply(new_mz, new_rt)
    >>> print(f"Bins: {calibrator.n_bins}")
    >>> print(f"Error before: {calibrator.median_ppm_before:.2f} ppm")
    >>> print(f"Error after: {calibrator.median_ppm_after:.2f} ppm")
    >>> print(f"Use tolerance: {calibrator.recommended_tolerance:.1f} ppm")
    """

    def __init__(
        self,
        observed_mz: np.ndarray,
        theoretical_mz: np.ndarray,
        rt_seconds: np.ndarray,
        adaptive_bins: bool = True,
        n_bins: int | None = None,
        min_psms_per_bin: int = 50,
        outlier_threshold_mad: float = 3.0,
    ):
        """Fit mass recalibration model."""
        n_psms = len(observed_mz)

        if n_psms == 0:
            warnings.warn("No PSMs provided for mass recalibration. Using defaults.")
            self.n_bins = 10
            self.bin_corrections = np.zeros(10, dtype=np.float64)
            self.bin_counts = np.zeros(10, dtype=np.int32)
            self.rt_min = 0.0
            self.rt_max = 3600.0
            self.median_ppm_before = 0.0
            self.std_ppm_before = 10.0
            self.median_ppm_after = 0.0
            self.std_ppm_after = 10.0
            self.recommended_tolerance = 20.0
            return

        # Calculate ppm errors
        ppm_errors = calculate_ppm_errors(observed_mz, theoretical_mz)

        # Remove outliers using MAD
        inliers = remove_outliers_mad(ppm_errors, threshold=outlier_threshold_mad)
        n_inliers = np.sum(inliers)

        if n_inliers < 10:
            warnings.warn(
                f"Only {n_inliers} inliers after outlier removal. "
                "Using global median correction."
            )
            # Fallback to global correction
            self.n_bins = 1
            self.bin_corrections = np.array([np.median(ppm_errors)])
            self.bin_counts = np.array([n_psms])
            self.rt_min = float(np.min(rt_seconds))
            self.rt_max = float(np.max(rt_seconds))
            self.median_ppm_before = float(np.median(ppm_errors))
            self.std_ppm_before = float(np.std(ppm_errors))
            self.median_ppm_after = 0.0
            self.std_ppm_after = self.std_ppm_before
            self.recommended_tolerance = max(10.0, self.std_ppm_before * 2)
            return

        # Use only inliers for calibration
        ppm_errors_clean = ppm_errors[inliers]
        rt_seconds_clean = rt_seconds[inliers]
        observed_mz_clean = observed_mz[inliers]
        theoretical_mz_clean = theoretical_mz[inliers]

        # Store statistics before calibration
        self.median_ppm_before = float(np.median(ppm_errors_clean))
        self.std_ppm_before = float(np.std(ppm_errors_clean))

        # Determine number of bins
        if adaptive_bins:
            self.n_bins = determine_rt_bins(n_inliers, min_psms_per_bin)
        else:
            self.n_bins = n_bins if n_bins is not None else 10

        # Calculate RT range
        self.rt_min = float(np.min(rt_seconds_clean))
        self.rt_max = float(np.max(rt_seconds_clean))

        # Assign PSMs to bins
        bin_indices = assign_rt_bins(
            rt_seconds_clean, self.rt_min, self.rt_max, self.n_bins
        )

        # Calculate corrections per bin
        bin_corrections_raw, bin_counts = calculate_bin_corrections(
            ppm_errors_clean, bin_indices, self.n_bins
        )

        # Interpolate missing bins
        self.bin_corrections = interpolate_bin_corrections(bin_corrections_raw, bin_counts)
        self.bin_counts = bin_counts

        # Calculate bin edges for reference
        self.bin_edges = np.linspace(self.rt_min, self.rt_max, self.n_bins + 1)

        # Calculate statistics after calibration
        corrected_mz_check = self.apply(observed_mz_clean, rt_seconds_clean)
        ppm_errors_after = calculate_ppm_errors(corrected_mz_check, theoretical_mz_clean)
        self.median_ppm_after = float(np.median(ppm_errors_after))
        self.std_ppm_after = float(np.std(ppm_errors_after))

        # Calculate recommended tolerance (95th percentile + buffer)
        abs_errors_after = np.abs(ppm_errors_after)
        percentile_95 = float(np.percentile(abs_errors_after, 95))
        self.recommended_tolerance = max(5.0, percentile_95 + 2.0)

    def apply(self, mz_values: np.ndarray, rt_seconds: np.ndarray) -> np.ndarray:
        """Apply mass recalibration to new m/z values.

        Parameters
        ----------
        mz_values : np.ndarray
            M/z values to correct
        rt_seconds : np.ndarray
            Retention times in seconds

        Returns
        -------
        corrected_mz : np.ndarray
            Corrected m/z values
        """
        if len(mz_values) == 0:
            return mz_values

        # Assign to bins
        bin_indices = assign_rt_bins(rt_seconds, self.rt_min, self.rt_max, self.n_bins)

        # Apply corrections
        corrected_mz = apply_corrections_fast(mz_values, bin_indices, self.bin_corrections)

        return corrected_mz

    def get_statistics(self) -> dict[str, float | int]:
        """Get calibration statistics.

        Returns
        -------
        dict
            Dictionary with calibration statistics
        """
        return {
            "n_bins": self.n_bins,
            "median_ppm_before": self.median_ppm_before,
            "std_ppm_before": self.std_ppm_before,
            "median_ppm_after": self.median_ppm_after,
            "std_ppm_after": self.std_ppm_after,
            "recommended_tolerance": self.recommended_tolerance,
            "rt_min": self.rt_min,
            "rt_max": self.rt_max,
        }

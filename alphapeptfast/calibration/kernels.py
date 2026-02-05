"""Kernel learning for Q1 and RT profiles.

This module provides functions to learn kernel parameters (FWHM, offset)
from collections of intensity profiles. The same algorithm works for:
- Q1 profiles: How intensity varies as quadrupole slides over a precursor (position = m/z)
- RT profiles: Chromatographic peak shape (position = time in seconds)

Example Usage
-------------
```python
from alphapeptfast.calibration import learn_kernel_from_profiles

# Learn Q1 kernel from extracted profiles
kernel = learn_kernel_from_profiles(
    positions=[q1_vals_1, q1_vals_2, ...],    # Q1 values per profile
    intensities=[int_1, int_2, ...],           # Corresponding intensities
    expected_center=precursor_mz_array,        # Expected center for each profile
)

print(f"Learned FWHM: {kernel.fwhm:.2f} Da")
print(f"Learned offset: {kernel.offset:.3f} Da")
```

Kernel Types
------------
- **Gaussian**: Symmetric bell curve, typical for narrow Q1 windows (2-4 Th) or RT peaks
- **Trapezoid**: Flat top with Gaussian flanks, typical for wide Q1 windows (20 Th)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class LearnedKernel:
    """Learned kernel parameters for Q1 or RT profiles.

    Attributes
    ----------
    fwhm : float
        Full width at half maximum in position units (Da for Q1, seconds for RT)
    offset : float
        Center offset from expected position (positive = shifted right)
    fwhm_std : float
        Standard deviation of FWHM estimates (uncertainty)
    offset_std : float
        Standard deviation of offset estimates (uncertainty)
    n_profiles : int
        Number of profiles used to learn parameters
    kernel_type : str
        Detected kernel shape: 'gaussian', 'trapezoid', or 'unknown'
    flat_top_width : float
        Width of flat top for trapezoidal kernels (0 for Gaussian)
    """

    fwhm: float
    offset: float
    fwhm_std: float
    offset_std: float
    n_profiles: int
    kernel_type: str = "gaussian"
    flat_top_width: float = 0.0

    @property
    def sigma(self) -> float:
        """Gaussian sigma derived from FWHM."""
        return self.fwhm / 2.355

    def __repr__(self) -> str:
        return (
            f"LearnedKernel(fwhm={self.fwhm:.3f}±{self.fwhm_std:.3f}, "
            f"offset={self.offset:.3f}±{self.offset_std:.3f}, "
            f"type={self.kernel_type}, n={self.n_profiles})"
        )


def measure_profile_fwhm(
    positions: np.ndarray,
    intensities: np.ndarray,
) -> float:
    """Measure FWHM of a single profile via linear interpolation.

    Parameters
    ----------
    positions : np.ndarray
        Position values (Q1 m/z or RT seconds)
    intensities : np.ndarray
        Intensity values at each position

    Returns
    -------
    fwhm : float
        Full width at half maximum, or NaN if cannot be determined
    """
    if len(positions) < 3 or len(intensities) < 3:
        return np.nan

    max_int = intensities.max()
    if max_int <= 0:
        return np.nan

    half_max = max_int / 2.0

    # Find crossings of half-max level
    above_half = intensities >= half_max

    # Find first crossing (rising edge)
    left_idx = -1
    for i in range(len(above_half) - 1):
        if not above_half[i] and above_half[i + 1]:
            # Interpolate crossing point
            if intensities[i + 1] != intensities[i]:
                frac = (half_max - intensities[i]) / (intensities[i + 1] - intensities[i])
                left_pos = positions[i] + frac * (positions[i + 1] - positions[i])
                left_idx = i
                break
    else:
        # No rising edge found - check if starts above half-max
        if above_half[0]:
            left_pos = positions[0]
            left_idx = 0
        else:
            return np.nan

    # Find last crossing (falling edge)
    right_idx = -1
    for i in range(len(above_half) - 1, 0, -1):
        if above_half[i - 1] and not above_half[i]:
            # Interpolate crossing point
            if intensities[i] != intensities[i - 1]:
                frac = (half_max - intensities[i - 1]) / (intensities[i] - intensities[i - 1])
                right_pos = positions[i - 1] + frac * (positions[i] - positions[i - 1])
                right_idx = i
                break
    else:
        # No falling edge found - check if ends above half-max
        if above_half[-1]:
            right_pos = positions[-1]
            right_idx = len(positions) - 1
        else:
            return np.nan

    if left_idx >= 0 and right_idx >= 0 and right_pos > left_pos:
        return right_pos - left_pos
    else:
        return np.nan


def measure_profile_center(
    positions: np.ndarray,
    intensities: np.ndarray,
) -> float:
    """Compute intensity-weighted center of a profile.

    Parameters
    ----------
    positions : np.ndarray
        Position values (Q1 m/z or RT seconds)
    intensities : np.ndarray
        Intensity values at each position

    Returns
    -------
    center : float
        Intensity-weighted mean position, or NaN if no signal
    """
    total_int = intensities.sum()
    if total_int <= 0:
        return np.nan
    return (positions * intensities).sum() / total_int


def detect_flat_top(
    positions: np.ndarray,
    intensities: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """Detect flat-top width for trapezoidal profiles.

    A flat top is detected when multiple consecutive points are above
    threshold * max_intensity.

    Parameters
    ----------
    positions : np.ndarray
        Position values (Q1 m/z or RT seconds)
    intensities : np.ndarray
        Intensity values at each position
    threshold : float
        Fraction of max intensity to consider "flat" (default 0.9)

    Returns
    -------
    flat_top_width : float
        Width of flat top region, or 0 if no clear flat top
    """
    if len(positions) < 3:
        return 0.0

    max_int = intensities.max()
    if max_int <= 0:
        return 0.0

    flat_threshold = threshold * max_int
    above_threshold = intensities >= flat_threshold

    # Find contiguous regions above threshold
    in_flat = False
    flat_start = None
    flat_end = None
    max_flat_width = 0.0

    for i in range(len(above_threshold)):
        if above_threshold[i]:
            if not in_flat:
                in_flat = True
                flat_start = positions[i]
            flat_end = positions[i]
        else:
            if in_flat:
                flat_width = flat_end - flat_start
                if flat_width > max_flat_width:
                    max_flat_width = flat_width
                in_flat = False

    # Check last region
    if in_flat and flat_end is not None and flat_start is not None:
        flat_width = flat_end - flat_start
        if flat_width > max_flat_width:
            max_flat_width = flat_width

    return max_flat_width


def learn_kernel_from_profiles(
    positions: List[np.ndarray],
    intensities: List[np.ndarray],
    expected_center: np.ndarray,
    min_n_points: int = 3,
) -> LearnedKernel:
    """Learn kernel parameters from a collection of profiles.

    Parameters
    ----------
    positions : List[np.ndarray]
        List of position arrays (Q1 m/z or RT), one per profile
    intensities : List[np.ndarray]
        List of intensity arrays, one per profile
    expected_center : np.ndarray
        Expected center position for each profile (precursor m/z or apex RT)
    min_n_points : int
        Minimum number of points required for a valid profile

    Returns
    -------
    kernel : LearnedKernel
        Learned kernel parameters with uncertainty estimates

    Notes
    -----
    Algorithm:
    1. For each profile with >= min_n_points:
       - Measure FWHM via interpolation
       - Find intensity-weighted center
       - Compute offset from expected_center
    2. Aggregate across profiles using median (robust to outliers)
    3. Estimate uncertainty from IQR
    4. Detect kernel type (Gaussian vs trapezoid)
    """
    if len(positions) != len(intensities) or len(positions) != len(expected_center):
        raise ValueError("positions, intensities, and expected_center must have same length")

    fwhm_values = []
    offset_values = []
    flat_top_values = []

    for i, (pos, inten) in enumerate(zip(positions, intensities)):
        # Skip profiles with too few points
        if len(pos) < min_n_points:
            continue

        # Skip profiles with no signal
        if inten.max() <= 0:
            continue

        # Measure FWHM
        fwhm = measure_profile_fwhm(pos, inten)
        if np.isfinite(fwhm) and fwhm > 0:
            fwhm_values.append(fwhm)

        # Measure center and offset
        center = measure_profile_center(pos, inten)
        if np.isfinite(center):
            offset = center - expected_center[i]
            offset_values.append(offset)

        # Detect flat top
        flat_top = detect_flat_top(pos, inten)
        if flat_top > 0:
            flat_top_values.append(flat_top)

    # Check we have enough data
    if len(fwhm_values) < 3:
        raise ValueError(f"Insufficient valid profiles: {len(fwhm_values)} < 3")

    # Aggregate using median (robust)
    fwhm_arr = np.array(fwhm_values)
    offset_arr = np.array(offset_values)

    median_fwhm = float(np.median(fwhm_arr))
    median_offset = float(np.median(offset_arr)) if len(offset_arr) > 0 else 0.0

    # Estimate uncertainty from IQR (robust standard deviation)
    fwhm_iqr = np.percentile(fwhm_arr, 75) - np.percentile(fwhm_arr, 25)
    fwhm_std = float(fwhm_iqr / 1.35)  # IQR to std conversion for normal dist

    if len(offset_arr) > 0:
        offset_iqr = np.percentile(offset_arr, 75) - np.percentile(offset_arr, 25)
        offset_std = float(offset_iqr / 1.35)
    else:
        offset_std = 0.0

    # Detect kernel type
    # If significant flat top detected, it's a trapezoid
    if len(flat_top_values) > len(fwhm_values) * 0.3:  # >30% have flat top
        median_flat_top = float(np.median(flat_top_values))
        if median_flat_top > median_fwhm * 0.2:  # Flat top > 20% of FWHM
            kernel_type = "trapezoid"
        else:
            kernel_type = "gaussian"
            median_flat_top = 0.0
    else:
        kernel_type = "gaussian"
        median_flat_top = 0.0

    return LearnedKernel(
        fwhm=median_fwhm,
        offset=median_offset,
        fwhm_std=fwhm_std,
        offset_std=offset_std,
        n_profiles=len(fwhm_values),
        kernel_type=kernel_type,
        flat_top_width=median_flat_top,
    )

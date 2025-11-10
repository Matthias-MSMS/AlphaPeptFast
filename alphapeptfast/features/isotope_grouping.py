"""
Isotope pattern detection for MS1 features with automatic charge detection.

Detects M0, M1, M2 isotope patterns with strict mass spacing and RT co-elution.
Optimized for high-resolution mass spectrometry (Orbitrap, MR-TOF).

Author: Claude Code (ported from MSC_MS1_high_res)
Date: November 2025
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


# Physical constants (in Da)
C13_MASS_DIFF = 1.0033548  # 13C - 12C mass difference
C13_2X_MASS_DIFF = 2.0067096  # 2× 13C spacing (for M+2)
S34_MASS_DIFF = 1.9957959  # 34S - 32S mass difference (Cys/Met peptides)


class InstrumentType(Enum):
    """Instrument types with different mass accuracy characteristics."""
    ORBITRAP = "orbitrap"  # ~240K resolution, 2-5 ppm
    MR_TOF = "mr_tof"      # >1M resolution, <1 ppm
    ASTRAL = "astral"      # Orbitrap-based, similar to Orbitrap


@dataclass
class IsotopeGroupingParams:
    """Parameters for isotope pattern detection.

    Uses ppm-based tolerances for instrument-independent parameters.
    """

    # Mass spacing tolerances (ppm)
    mz_tolerance_ppm: float = 2.0

    # RT co-elution criterion
    rt_tolerance_factor: float = 1.0  # Multiple of FWHM

    # M+2 isotope detection
    detect_c13_2x: bool = True  # Look for 13C2 M+2 isotopes
    detect_s34: bool = True     # Look for 34S M+2 isotopes

    @classmethod
    def for_instrument(cls, instrument: InstrumentType) -> 'IsotopeGroupingParams':
        """Create parameters optimized for specific instrument type.

        Args:
            instrument: Instrument type enum

        Returns:
            IsotopeGroupingParams with instrument-specific defaults
        """
        if instrument == InstrumentType.MR_TOF:
            return cls(
                mz_tolerance_ppm=1.2,  # Ultra-tight for >1M resolution
                rt_tolerance_factor=1.0,
                detect_c13_2x=True,
                detect_s34=True
            )
        elif instrument in (InstrumentType.ORBITRAP, InstrumentType.ASTRAL):
            return cls(
                mz_tolerance_ppm=3.0,  # Looser for ~240K resolution
                rt_tolerance_factor=1.0,
                detect_c13_2x=True,
                detect_s34=True
            )
        else:
            raise ValueError(f"Unknown instrument type: {instrument}")


@dataclass
class IsotopeGroup:
    """Container for an isotope pattern (M0, M1, M2) with charge state."""

    # Feature indices in original array
    m0_idx: int
    m1_idx: int = -1  # -1 if not found
    m2_idx: int = -1  # -1 if not found

    # Mass values
    m0_mz: float = 0.0
    m1_mz: float = 0.0
    m2_mz: float = 0.0

    # RT values (seconds)
    m0_rt: float = 0.0
    m1_rt: float = 0.0
    m2_rt: float = 0.0

    # Intensities
    m0_intensity: float = 0.0
    m1_intensity: float = 0.0
    m2_intensity: float = 0.0

    # Quality metrics
    m0_m1_mass_error_ppm: float = np.nan
    m1_m2_mass_error_ppm: float = np.nan
    m0_m2_mass_error_ppm: float = np.nan
    m2_isotope_type: str = ''  # '13C2', '34S', or ''

    m0_m1_rt_diff: float = np.nan
    m1_m2_rt_diff: float = np.nan

    # Pattern completeness
    has_m1: bool = False
    has_m2: bool = False

    # Charge state (automatically detected from isotope spacing)
    charge: int = 0  # 1, 2, 3, or 0 if unknown


@njit
def calculate_mass_error_ppm(observed_diff: float, expected_diff: float, m0_mz: float) -> float:
    """Calculate mass spacing error in ppm.

    Args:
        observed_diff: Observed m/z difference
        expected_diff: Expected m/z difference (Da)
        m0_mz: M0 m/z value (for ppm calculation)

    Returns:
        Mass error in ppm
    """
    delta = observed_diff - expected_diff
    return (delta / m0_mz) * 1e6


@njit
def find_isotope_candidates(
    m0_idx: int,
    m0_mz: float,
    m0_rt: float,
    m0_fwhm: float,
    all_mz: np.ndarray,
    all_rt: np.ndarray,
    all_intensity: np.ndarray,
    mz_tol_ppm: float,
    rt_tol_factor: float,
    expected_spacing: float
) -> Tuple[int, float, float, float]:
    """Find the best isotope candidate (M1 or M2) for a given M0 feature.

    Uses binary search on sorted m/z array for O(log n) performance.

    Args:
        m0_idx: Index of M0 feature
        m0_mz: M0 m/z value
        m0_rt: M0 RT value (seconds)
        m0_fwhm: M0 FWHM (seconds)
        all_mz: All feature m/z values (MUST BE SORTED)
        all_rt: All feature RT values (seconds)
        all_intensity: All feature intensities
        mz_tol_ppm: Mass tolerance in ppm
        rt_tol_factor: RT tolerance as multiple of FWHM
        expected_spacing: Expected m/z spacing (Da)

    Returns:
        Tuple of (candidate_idx, mass_error_ppm, rt_diff, intensity)
        Returns (-1, nan, nan, 0.0) if no candidate found
    """
    n_features = len(all_mz)

    # Expected m/z for isotope
    expected_mz = m0_mz + expected_spacing

    # Mass tolerance in Da
    mz_tol_da = (mz_tol_ppm / 1e6) * expected_mz

    # RT tolerance in seconds
    rt_tol_sec = rt_tol_factor * m0_fwhm

    # Binary search to find m/z range
    mz_min = expected_mz - mz_tol_da
    mz_max = expected_mz + mz_tol_da

    # Find start index
    left = 0
    right = n_features
    while left < right:
        mid = (left + right) // 2
        if all_mz[mid] < mz_min:
            left = mid + 1
        else:
            right = mid
    start_idx = left

    # Find end index
    left = 0
    right = n_features
    while left < right:
        mid = (left + right) // 2
        if all_mz[mid] <= mz_max:
            left = mid + 1
        else:
            right = mid
    end_idx = left

    # Search only in the m/z window
    best_idx = -1
    best_mass_error = np.inf
    best_rt_diff = np.nan
    best_intensity = 0.0

    for i in range(start_idx, end_idx):
        if i == m0_idx:
            continue

        # Check m/z tolerance
        mz_diff = all_mz[i] - m0_mz
        mass_error_ppm = calculate_mass_error_ppm(mz_diff, expected_spacing, m0_mz)

        # Check RT co-elution
        rt_diff = all_rt[i] - m0_rt
        if abs(rt_diff) > rt_tol_sec:
            continue

        # Valid candidate - keep the one with smallest mass error
        if abs(mass_error_ppm) < abs(best_mass_error):
            best_idx = i
            best_mass_error = mass_error_ppm
            best_rt_diff = rt_diff
            best_intensity = all_intensity[i]

    return best_idx, best_mass_error, best_rt_diff, best_intensity


@njit
def detect_isotope_patterns_numba(
    mz_array: np.ndarray,
    rt_array: np.ndarray,
    fwhm_array: np.ndarray,
    intensity_array: np.ndarray,
    mz_tol_ppm: float,
    rt_tol_factor: float,
    detect_c13_2x: bool,
    detect_s34: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect isotope patterns with automatic charge state detection.

    Returns:
        Tuple of arrays for building IsotopeGroup objects:
        - m0_indices: Indices of M0 features
        - m1_indices: Indices of M1 features (-1 if not found)
        - m2_indices: Indices of M2 features (-1 if not found)
        - m0_m1_errors: Mass errors for M0→M1 (ppm)
        - m1_m2_errors: Mass errors for M1→M2 (ppm)
        - m0_m2_errors: Mass errors for M0→M+2 (ppm)
        - m2_types: Type of M+2 isotope (0='', 1='13C2', 2='34S')
        - charge_states: Inferred charge state (1, 2, 3, or 0 if unknown)
    """
    n_features = len(mz_array)

    # Pre-allocate result arrays
    m0_indices = np.zeros(n_features, dtype=np.int64)
    m1_indices = np.full(n_features, -1, dtype=np.int64)
    m2_indices = np.full(n_features, -1, dtype=np.int64)

    m0_m1_errors = np.full(n_features, np.nan, dtype=np.float64)
    m1_m2_errors = np.full(n_features, np.nan, dtype=np.float64)
    m0_m2_errors = np.full(n_features, np.nan, dtype=np.float64)
    m2_types = np.zeros(n_features, dtype=np.int64)
    charge_states = np.zeros(n_features, dtype=np.int64)

    n_groups = 0

    # For each potential M0 feature
    for i in range(n_features):
        m0_mz = mz_array[i]
        m0_rt = rt_array[i]
        m0_fwhm = fwhm_array[i]

        if m0_fwhm <= 0:
            continue

        # Try different charge states (z=2, z=3, z=1)
        best_m1_idx = -1
        best_m1_error = np.nan
        best_charge = 0
        best_m1_spacing = 0.0

        # Try z=2: M1 at +0.502 Da
        m1_idx_z2, m1_error_z2, _, _ = find_isotope_candidates(
            i, m0_mz, m0_rt, m0_fwhm,
            mz_array, rt_array, intensity_array,
            mz_tol_ppm, rt_tol_factor, C13_MASS_DIFF / 2.0
        )

        # Try z=3: M1 at +0.335 Da
        m1_idx_z3, m1_error_z3, _, _ = find_isotope_candidates(
            i, m0_mz, m0_rt, m0_fwhm,
            mz_array, rt_array, intensity_array,
            mz_tol_ppm, rt_tol_factor, C13_MASS_DIFF / 3.0
        )

        # Try z=1: M1 at +1.003 Da
        m1_idx_z1, m1_error_z1, _, _ = find_isotope_candidates(
            i, m0_mz, m0_rt, m0_fwhm,
            mz_array, rt_array, intensity_array,
            mz_tol_ppm, rt_tol_factor, C13_MASS_DIFF
        )

        # Choose best candidate (smallest absolute error)
        if m1_idx_z2 >= 0:
            best_m1_idx = m1_idx_z2
            best_m1_error = m1_error_z2
            best_charge = 2
            best_m1_spacing = C13_MASS_DIFF / 2.0

        if m1_idx_z3 >= 0 and (best_m1_idx < 0 or abs(m1_error_z3) < abs(best_m1_error)):
            best_m1_idx = m1_idx_z3
            best_m1_error = m1_error_z3
            best_charge = 3
            best_m1_spacing = C13_MASS_DIFF / 3.0

        if m1_idx_z1 >= 0 and (best_m1_idx < 0 or abs(m1_error_z1) < abs(best_m1_error)):
            best_m1_idx = m1_idx_z1
            best_m1_error = m1_error_z1
            best_charge = 1
            best_m1_spacing = C13_MASS_DIFF

        m1_idx = best_m1_idx
        m1_error = best_m1_error
        charge = best_charge

        # Look for M2 if M1 found
        m2_idx = -1
        m2_error_from_m1 = np.nan
        m2_type = 0

        if m1_idx >= 0 and charge > 0:
            m1_mz = mz_array[m1_idx]
            m1_rt = rt_array[m1_idx]
            m1_fwhm = fwhm_array[m1_idx]

            if m1_fwhm > 0:
                m2_idx, m2_error_from_m1, _, _ = find_isotope_candidates(
                    m1_idx, m1_mz, m1_rt, m1_fwhm,
                    mz_array, rt_array, intensity_array,
                    mz_tol_ppm, rt_tol_factor, best_m1_spacing
                )

        # Check for M+2 directly from M0
        m2_from_m0_idx = -1
        m2_from_m0_error = np.nan

        if detect_c13_2x or detect_s34:
            if detect_c13_2x:
                m2_c13_idx, m2_c13_error, _, _ = find_isotope_candidates(
                    i, m0_mz, m0_rt, m0_fwhm,
                    mz_array, rt_array, intensity_array,
                    mz_tol_ppm, rt_tol_factor, C13_2X_MASS_DIFF
                )

                if m2_c13_idx >= 0:
                    m2_from_m0_idx = m2_c13_idx
                    m2_from_m0_error = m2_c13_error
                    m2_type = 1  # 13C2

            if detect_s34 and m2_from_m0_idx < 0:
                m2_s34_idx, m2_s34_error, _, _ = find_isotope_candidates(
                    i, m0_mz, m0_rt, m0_fwhm,
                    mz_array, rt_array, intensity_array,
                    mz_tol_ppm, rt_tol_factor, S34_MASS_DIFF
                )

                if m2_s34_idx >= 0:
                    m2_from_m0_idx = m2_s34_idx
                    m2_from_m0_error = m2_s34_error
                    m2_type = 2  # 34S

        # Reconcile M2 candidates
        final_m2_idx = -1
        final_m2_error = np.nan

        if m2_idx >= 0:
            final_m2_idx = m2_idx
            m2_mz = mz_array[m2_idx]
            actual_spacing = m2_mz - m0_mz

            # Determine type by actual spacing
            dist_to_c13_2x = abs(actual_spacing - C13_2X_MASS_DIFF)
            dist_to_s34 = abs(actual_spacing - S34_MASS_DIFF)

            if dist_to_s34 < dist_to_c13_2x:
                final_m2_error = calculate_mass_error_ppm(actual_spacing, S34_MASS_DIFF, m0_mz)
                m2_type = 2
            else:
                final_m2_error = calculate_mass_error_ppm(actual_spacing, C13_2X_MASS_DIFF, m0_mz)
                m2_type = 1

        elif m2_from_m0_idx >= 0:
            final_m2_idx = m2_from_m0_idx
            final_m2_error = m2_from_m0_error

        # Store isotope group
        m0_indices[n_groups] = i
        m1_indices[n_groups] = m1_idx
        m2_indices[n_groups] = final_m2_idx
        m0_m1_errors[n_groups] = m1_error
        m1_m2_errors[n_groups] = m2_error_from_m1
        m0_m2_errors[n_groups] = final_m2_error
        m2_types[n_groups] = m2_type
        charge_states[n_groups] = charge

        n_groups += 1

    # Trim arrays
    return (
        m0_indices[:n_groups],
        m1_indices[:n_groups],
        m2_indices[:n_groups],
        m0_m1_errors[:n_groups],
        m1_m2_errors[:n_groups],
        m0_m2_errors[:n_groups],
        m2_types[:n_groups],
        charge_states[:n_groups]
    )


def detect_isotope_patterns(
    features: np.ndarray,
    params: IsotopeGroupingParams
) -> List[IsotopeGroup]:
    """Detect isotope patterns in a feature set.

    Args:
        features: Structured numpy array with fields:
            - mz: m/z values
            - rt: retention time (seconds)
            - fwhm_sec: FWHM (seconds)
            - intensity: intensities
        params: Isotope grouping parameters

    Returns:
        List of IsotopeGroup objects
    """
    # Extract and sort by m/z
    mz = features['mz']
    rt = features['rt']
    fwhm = features['fwhm_sec']
    intensity = features['intensity']

    sort_idx = np.argsort(mz)
    mz_sorted = mz[sort_idx]
    rt_sorted = rt[sort_idx]
    fwhm_sorted = fwhm[sort_idx]
    intensity_sorted = intensity[sort_idx]

    # Detect patterns
    (m0_indices, m1_indices, m2_indices,
     m0_m1_errors, m1_m2_errors, m0_m2_errors,
     m2_types, charge_states) = detect_isotope_patterns_numba(
        mz_sorted, rt_sorted, fwhm_sorted, intensity_sorted,
        params.mz_tolerance_ppm, params.rt_tolerance_factor,
        params.detect_c13_2x, params.detect_s34
    )

    # Build IsotopeGroup objects
    isotope_groups = []
    m2_type_map = {0: '', 1: '13C2', 2: '34S'}

    for i in range(len(m0_indices)):
        m0_idx = m0_indices[i]
        m1_idx = m1_indices[i]
        m2_idx = m2_indices[i]

        # Map back to original indices
        m0_orig_idx = sort_idx[m0_idx]
        m1_orig_idx = sort_idx[m1_idx] if m1_idx >= 0 else -1
        m2_orig_idx = sort_idx[m2_idx] if m2_idx >= 0 else -1

        group = IsotopeGroup(
            m0_idx=m0_orig_idx,
            m1_idx=m1_orig_idx,
            m2_idx=m2_orig_idx,

            m0_mz=features['mz'][m0_orig_idx],
            m1_mz=features['mz'][m1_orig_idx] if m1_idx >= 0 else 0.0,
            m2_mz=features['mz'][m2_orig_idx] if m2_idx >= 0 else 0.0,

            m0_rt=features['rt'][m0_orig_idx],
            m1_rt=features['rt'][m1_orig_idx] if m1_idx >= 0 else 0.0,
            m2_rt=features['rt'][m2_orig_idx] if m2_idx >= 0 else 0.0,

            m0_intensity=features['intensity'][m0_orig_idx],
            m1_intensity=features['intensity'][m1_orig_idx] if m1_idx >= 0 else 0.0,
            m2_intensity=features['intensity'][m2_orig_idx] if m2_idx >= 0 else 0.0,

            m0_m1_mass_error_ppm=m0_m1_errors[i],
            m1_m2_mass_error_ppm=m1_m2_errors[i],
            m0_m2_mass_error_ppm=m0_m2_errors[i],
            m2_isotope_type=m2_type_map[m2_types[i]],

            m0_m1_rt_diff=features['rt'][m1_orig_idx] - features['rt'][m0_orig_idx] if m1_idx >= 0 else np.nan,
            m1_m2_rt_diff=features['rt'][m2_orig_idx] - features['rt'][m1_orig_idx] if m1_idx >= 0 and m2_idx >= 0 else np.nan,

            has_m1=m1_idx >= 0,
            has_m2=m2_idx >= 0,
            charge=int(charge_states[i])
        )

        isotope_groups.append(group)

    return isotope_groups

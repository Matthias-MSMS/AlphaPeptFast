"""
Feature finding for MS1 spectra using intensity-weighted peak grouping.

The "argmax" algorithm groups peaks by:
1. Processing peaks in descending intensity order (greedy)
2. Collecting peaks within ppm tolerance in m/z and RT tolerance
3. Computing intensity-weighted centroid for feature m/z

This approach is simple but effective for Q1 scanning data and general MS1
feature detection. Validated to achieve ~2 ppm precision on ZenoTOF 45k data.

Key components:
- `find_features_numba`: Core greedy feature finding algorithm
- `find_features_core_anneal`: Two-phase algorithm with adaptive RT expansion
- `find_isotope_patterns`: Charge state inference from M+0/M+1/M+2 spacing
- `find_charge_pairs`: Mass accuracy validation via z=2/z=3 matching
- `FeatureFinder`: Orchestrator class combining all components

Author: Claude Code
Date: February 2026
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from ..constants import ISOTOPE_MASS_DIFFERENCE, PROTON_MASS

# Module-level constant alias for clarity in isotope code
C13_MASS_DIFF = ISOTOPE_MASS_DIFFERENCE


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class FeatureFinderParams:
    """Parameters for feature finding.

    Parameters
    ----------
    ppm_tol : float
        Mass tolerance in ppm (default: 15.0 for ZenoTOF 45k)
    rt_tol_sec : float
        RT tolerance in seconds (default: 3.0)
    intensity_threshold : float
        Minimum intensity for seed peaks (default: 1000.0)
    min_peaks : int
        Minimum peaks per feature (default: 3)
    isotope_ppm_tol : float
        PPM tolerance for isotope detection (default: 10.0)
    isotope_rt_tol_sec : float
        RT tolerance for isotope co-elution (default: 5.0)
    charge_pair_rt_tol_sec : float
        RT tolerance for charge pair matching (default: 5.0)
    charge_pair_mass_tol_da : float
        Neutral mass tolerance for charge pairs in Da (default: 0.1)
    min_charge : int
        Minimum charge state to detect (default: 2)
    max_charge : int
        Maximum charge state to detect (default: 4)
    """
    ppm_tol: float = 15.0
    rt_tol_sec: float = 3.0
    intensity_threshold: float = 1000.0
    min_peaks: int = 3

    isotope_ppm_tol: float = 10.0
    isotope_rt_tol_sec: float = 5.0

    charge_pair_rt_tol_sec: float = 5.0
    charge_pair_mass_tol_da: float = 0.1

    min_charge: int = 2
    max_charge: int = 4


# =============================================================================
# Core Feature Finding (Numba-optimized)
# =============================================================================

@njit
def find_features_numba(
    mz: np.ndarray,
    intensity: np.ndarray,
    scan: np.ndarray,
    rt: np.ndarray,
    ppm_tol: float,
    rt_tol_sec: float,
    intensity_threshold: float,
    min_peaks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Numba-optimized feature finding with pre-allocated buffers.

    Uses greedy algorithm from highest intensity peaks.
    No Python list/set - all operations use pre-allocated numpy arrays.

    Parameters
    ----------
    mz : np.ndarray
        m/z values, SORTED ascending
    intensity : np.ndarray
        Intensity values
    scan : np.ndarray
        Cycle/scan indices
    rt : np.ndarray
        RT values in seconds (per peak)
    ppm_tol : float
        Mass tolerance in ppm
    rt_tol_sec : float
        RT tolerance in seconds
    intensity_threshold : float
        Minimum intensity
    min_peaks : int
        Minimum peaks per feature

    Returns
    -------
    feat_mz : np.ndarray
        Feature m/z values (intensity-weighted centroid)
    feat_rt : np.ndarray
        Feature RT values (intensity-weighted mean)
    feat_intensity : np.ndarray
        Total feature intensity
    feat_mz_std_ppm : np.ndarray
        Mass precision in ppm (intensity-weighted std)
    feat_rt_start : np.ndarray
        RT start of feature
    feat_rt_end : np.ndarray
        RT end of feature
    feat_n_peaks : np.ndarray
        Number of peaks in feature
    feat_n_scans : np.ndarray
        Number of unique scans in feature
    n_features : int
        Total number of features found
    """
    n_peaks = len(mz)

    # Track which peaks are assigned to features
    peak_to_feature = np.full(n_peaks, -1, dtype=np.int32)

    # Pre-allocate output arrays
    max_features = n_peaks // min_peaks + 1000
    feat_mz = np.zeros(max_features, dtype=np.float64)
    feat_rt = np.zeros(max_features, dtype=np.float64)
    feat_intensity = np.zeros(max_features, dtype=np.float64)
    feat_mz_std_ppm = np.zeros(max_features, dtype=np.float64)
    feat_rt_start = np.zeros(max_features, dtype=np.float64)
    feat_rt_end = np.zeros(max_features, dtype=np.float64)
    feat_n_peaks = np.zeros(max_features, dtype=np.int32)
    feat_n_scans = np.zeros(max_features, dtype=np.int32)

    # Pre-allocate temp buffer for matched indices
    matched_buffer = np.zeros(50000, dtype=np.int64)

    # Process peaks in intensity order (greedy)
    intensity_order = np.argsort(-intensity)
    n_features = 0

    for order_idx in range(n_peaks):
        peak_idx = intensity_order[order_idx]

        # Skip if below threshold or already assigned
        if intensity[peak_idx] < intensity_threshold:
            continue
        if peak_to_feature[peak_idx] >= 0:
            continue

        # Seed peak
        seed_mz = mz[peak_idx]
        seed_rt = rt[peak_idx]

        # Calculate tolerance
        tol_da = seed_mz * ppm_tol / 1e6

        # Binary search for m/z range
        left = np.searchsorted(mz, seed_mz - tol_da)
        right = np.searchsorted(mz, seed_mz + tol_da)

        # Collect matches into buffer (NO Python list!)
        matched_count = 0
        sum_int = 0.0
        sum_mz_w = 0.0
        sum_rt_w = 0.0
        sum_mz_sq_w = 0.0
        min_rt = seed_rt
        max_rt = seed_rt

        # Track unique scans with a simple array
        unique_scans = np.zeros(10000, dtype=np.int32)
        n_unique_scans = 0

        for i in range(left, right):
            if peak_to_feature[i] >= 0:
                continue

            # Check RT tolerance
            if abs(rt[i] - seed_rt) > rt_tol_sec:
                continue

            # This peak matches
            matched_buffer[matched_count] = i
            matched_count += 1

            w = intensity[i]
            sum_int += w
            sum_mz_w += mz[i] * w
            sum_rt_w += rt[i] * w
            sum_mz_sq_w += mz[i] * mz[i] * w

            if rt[i] < min_rt:
                min_rt = rt[i]
            if rt[i] > max_rt:
                max_rt = rt[i]

            # Track unique scans
            scan_i = scan[i]
            is_new_scan = True
            for j in range(n_unique_scans):
                if unique_scans[j] == scan_i:
                    is_new_scan = False
                    break
            if is_new_scan and n_unique_scans < 10000:
                unique_scans[n_unique_scans] = scan_i
                n_unique_scans += 1

        # Check minimum requirements
        if matched_count < min_peaks:
            continue

        # Calculate feature properties
        mean_mz = sum_mz_w / sum_int
        mean_rt = sum_rt_w / sum_int

        # Mass precision in ppm (intensity-weighted std)
        variance = sum_mz_sq_w / sum_int - mean_mz * mean_mz
        if variance > 0:
            mz_std = np.sqrt(variance)
            mz_std_ppm = mz_std / mean_mz * 1e6
        else:
            mz_std_ppm = 0.0

        # Store feature
        feat_mz[n_features] = mean_mz
        feat_rt[n_features] = mean_rt
        feat_intensity[n_features] = sum_int
        feat_mz_std_ppm[n_features] = mz_std_ppm
        feat_rt_start[n_features] = min_rt
        feat_rt_end[n_features] = max_rt
        feat_n_peaks[n_features] = matched_count
        feat_n_scans[n_features] = n_unique_scans

        # Mark peaks as used
        for j in range(matched_count):
            peak_to_feature[matched_buffer[j]] = n_features

        n_features += 1

        if n_features >= max_features - 1:
            break

    return (feat_mz, feat_rt, feat_intensity, feat_mz_std_ppm,
            feat_rt_start, feat_rt_end, feat_n_peaks, feat_n_scans,
            n_features)


# =============================================================================
# Core-Anneal Feature Finding
# =============================================================================

@njit
def _collect_peaks_for_feature(
    mz: np.ndarray,
    intensity: np.ndarray,
    scan: np.ndarray,
    rt: np.ndarray,
    peak_to_feature: np.ndarray,
    seed_mz: float,
    seed_rt: float,
    ppm_tol: float,
    rt_tol_sec: float,
    matched_buffer: np.ndarray,
) -> Tuple[int, float, float, float, float, float, float, int]:
    """
    Collect peaks matching a feature seed. Returns statistics.

    Returns
    -------
    matched_count : int
        Number of peaks matched
    sum_int : float
        Total intensity
    mean_mz : float
        Intensity-weighted mean m/z
    mean_rt : float
        Intensity-weighted mean RT
    mz_std_ppm : float
        Mass precision in ppm
    min_rt : float
        Minimum RT
    max_rt : float
        Maximum RT
    n_unique_scans : int
        Number of unique scans
    """
    tol_da = seed_mz * ppm_tol / 1e6

    # Binary search for m/z range
    left = np.searchsorted(mz, seed_mz - tol_da)
    right = np.searchsorted(mz, seed_mz + tol_da)

    matched_count = 0
    sum_int = 0.0
    sum_mz_w = 0.0
    sum_rt_w = 0.0
    sum_mz_sq_w = 0.0
    min_rt = seed_rt
    max_rt = seed_rt

    # Track unique scans
    unique_scans = np.zeros(10000, dtype=np.int32)
    n_unique_scans = 0

    for i in range(left, right):
        if peak_to_feature[i] >= 0:
            continue

        # Check RT tolerance
        if abs(rt[i] - seed_rt) > rt_tol_sec:
            continue

        # This peak matches
        matched_buffer[matched_count] = i
        matched_count += 1

        w = intensity[i]
        sum_int += w
        sum_mz_w += mz[i] * w
        sum_rt_w += rt[i] * w
        sum_mz_sq_w += mz[i] * mz[i] * w

        if rt[i] < min_rt:
            min_rt = rt[i]
        if rt[i] > max_rt:
            max_rt = rt[i]

        # Track unique scans
        scan_i = scan[i]
        is_new_scan = True
        for j in range(n_unique_scans):
            if unique_scans[j] == scan_i:
                is_new_scan = False
                break
        if is_new_scan and n_unique_scans < 10000:
            unique_scans[n_unique_scans] = scan_i
            n_unique_scans += 1

    if matched_count == 0 or sum_int == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Calculate feature properties
    mean_mz = sum_mz_w / sum_int
    mean_rt = sum_rt_w / sum_int

    # Mass precision in ppm
    variance = sum_mz_sq_w / sum_int - mean_mz * mean_mz
    if variance > 0:
        mz_std = np.sqrt(variance)
        mz_std_ppm = mz_std / mean_mz * 1e6
    else:
        mz_std_ppm = 0.0

    return (matched_count, sum_int, mean_mz, mean_rt, mz_std_ppm, min_rt, max_rt, n_unique_scans)


@njit
def find_features_core_anneal(
    mz: np.ndarray,
    intensity: np.ndarray,
    scan: np.ndarray,
    rt: np.ndarray,
    ppm_tol: float,
    core_rt_tol_sec: float,
    anneal_factor: float,
    intensity_threshold: float,
    min_peaks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Two-pass feature finding with core-anneal strategy.

    Core-anneal prevents over-merging while catching all real features:
    - Core: Conservative, finds high-confidence seeds with tight RT tolerance
    - Anneal: Liberal, extends features based on actual peak width (FWHM)

    Parameters
    ----------
    mz : np.ndarray
        m/z values, SORTED ascending
    intensity : np.ndarray
        Intensity values
    scan : np.ndarray
        Cycle/scan indices
    rt : np.ndarray
        RT values in seconds (per peak)
    ppm_tol : float
        Mass tolerance in ppm
    core_rt_tol_sec : float
        Tight RT tolerance for core phase (e.g., 1.8 sec)
    anneal_factor : float
        Factor to expand RT window in anneal phase (e.g., 1.5x FWHM)
    intensity_threshold : float
        Minimum intensity
    min_peaks : int
        Minimum peaks per feature

    Returns
    -------
    Feature arrays and count (same format as find_features_numba)
    """
    n_peaks = len(mz)

    # Track which peaks are assigned to features
    peak_to_feature = np.full(n_peaks, -1, dtype=np.int32)

    # Pre-allocate output arrays
    max_features = n_peaks // min_peaks + 1000
    feat_mz = np.zeros(max_features, dtype=np.float64)
    feat_rt = np.zeros(max_features, dtype=np.float64)
    feat_intensity = np.zeros(max_features, dtype=np.float64)
    feat_mz_std_ppm = np.zeros(max_features, dtype=np.float64)
    feat_rt_start = np.zeros(max_features, dtype=np.float64)
    feat_rt_end = np.zeros(max_features, dtype=np.float64)
    feat_n_peaks = np.zeros(max_features, dtype=np.int32)
    feat_n_scans = np.zeros(max_features, dtype=np.int32)

    # Temp buffers
    matched_buffer = np.zeros(50000, dtype=np.int64)
    anneal_buffer = np.zeros(50000, dtype=np.int64)

    # Process peaks in intensity order (greedy)
    intensity_order = np.argsort(-intensity)
    n_features = 0

    for order_idx in range(n_peaks):
        peak_idx = intensity_order[order_idx]

        # Skip if below threshold or already assigned
        if intensity[peak_idx] < intensity_threshold:
            continue
        if peak_to_feature[peak_idx] >= 0:
            continue

        # ==== PHASE 1: CORE (tight RT tolerance) ====
        seed_mz = mz[peak_idx]
        seed_rt = rt[peak_idx]

        (core_count, core_int, core_mean_mz, core_mean_rt, core_mz_std,
         core_rt_min, core_rt_max, core_n_scans) = _collect_peaks_for_feature(
            mz, intensity, scan, rt, peak_to_feature,
            seed_mz, seed_rt, ppm_tol, core_rt_tol_sec, matched_buffer
        )

        # Skip if core doesn't meet requirements
        if core_count < min_peaks:
            continue

        # Calculate FWHM from core RT range (approximate)
        # FWHM ~ 2.355 * sigma, but for elution profile we use RT range
        core_fwhm = core_rt_max - core_rt_min
        if core_fwhm < 1.0:
            core_fwhm = 2.0  # Minimum FWHM fallback

        # ==== PHASE 2: ANNEAL (adaptive RT expansion) ====
        # Expand RT window based on FWHM
        anneal_rt_tol = anneal_factor * core_fwhm

        # Use refined m/z from core phase as new seed
        (anneal_count, anneal_int, anneal_mean_mz, anneal_mean_rt, anneal_mz_std,
         anneal_rt_min, anneal_rt_max, anneal_n_scans) = _collect_peaks_for_feature(
            mz, intensity, scan, rt, peak_to_feature,
            core_mean_mz, core_mean_rt, ppm_tol, anneal_rt_tol, anneal_buffer
        )

        # Use annealed results if they found more peaks
        if anneal_count >= core_count:
            final_count = anneal_count
            final_int = anneal_int
            final_mz = anneal_mean_mz
            final_rt = anneal_mean_rt
            final_mz_std = anneal_mz_std
            final_rt_min = anneal_rt_min
            final_rt_max = anneal_rt_max
            final_n_scans = anneal_n_scans
            final_buffer = anneal_buffer
        else:
            # Core was better (edge case, shouldn't happen often)
            final_count = core_count
            final_int = core_int
            final_mz = core_mean_mz
            final_rt = core_mean_rt
            final_mz_std = core_mz_std
            final_rt_min = core_rt_min
            final_rt_max = core_rt_max
            final_n_scans = core_n_scans
            final_buffer = matched_buffer

        # Store feature
        feat_mz[n_features] = final_mz
        feat_rt[n_features] = final_rt
        feat_intensity[n_features] = final_int
        feat_mz_std_ppm[n_features] = final_mz_std
        feat_rt_start[n_features] = final_rt_min
        feat_rt_end[n_features] = final_rt_max
        feat_n_peaks[n_features] = final_count
        feat_n_scans[n_features] = final_n_scans

        # Mark peaks as used
        for j in range(final_count):
            peak_to_feature[final_buffer[j]] = n_features

        n_features += 1

        if n_features >= max_features - 1:
            break

    return (feat_mz, feat_rt, feat_intensity, feat_mz_std_ppm,
            feat_rt_start, feat_rt_end, feat_n_peaks, feat_n_scans,
            n_features)


# =============================================================================
# Isotope Detection (for charge inference)
# =============================================================================

@njit
def _find_peak_at_mz(mz_sorted, mz_order, feature_rt, feature_intensity,
                     target_mz, ref_rt, ref_int, ppm_tol, rt_tol_sec):
    """Helper: find a peak at target_mz within tolerances.

    Returns
    -------
    idx : int
        Index of best matching peak (-1 if not found)
    rt_diff : float
        RT difference to matched peak
    ratio : float
        Intensity ratio to reference
    """
    tol_da = target_mz * ppm_tol / 1e6
    left = np.searchsorted(mz_sorted, target_mz - tol_da)
    right = np.searchsorted(mz_sorted, target_mz + tol_da)

    best_idx = -1
    best_rt_diff = 0.0
    best_ratio = 0.0

    for j in range(left, right):
        cand_idx = mz_order[j]
        rt_diff = abs(feature_rt[cand_idx] - ref_rt)
        if rt_diff > rt_tol_sec:
            continue
        ratio = feature_intensity[cand_idx] / ref_int
        if ratio > 1.5:  # Isotope shouldn't be much brighter
            continue
        if ratio > best_ratio:
            best_idx = cand_idx
            best_rt_diff = rt_diff
            best_ratio = ratio

    return best_idx, best_rt_diff, best_ratio


@njit
def find_isotope_patterns(
    feature_mz: np.ndarray,
    feature_rt: np.ndarray,
    feature_intensity: np.ndarray,
    n_features: int,
    ppm_tol: float,
    rt_tol_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect isotope patterns and infer charge states.

    WARNING: Isotope spacing is NOT useful for mass accuracy at moderate resolution!
    At 15 ppm on each peak, spacing error ~ 11,000 ppm (1.1%) due to error propagation.
    Use this ONLY for charge state assignment.

    Strategy to resolve z=1 vs z=2 ambiguity:
    - z=1 M+1 spacing = z=2 M+2 spacing (1.003 Da)
    - z=2 M+1 spacing = z=4 M+2 spacing (0.502 Da)

    Resolution: Check for M+2 to confirm charge state.

    Parameters
    ----------
    feature_mz : np.ndarray
        Feature m/z values
    feature_rt : np.ndarray
        Feature RT values
    feature_intensity : np.ndarray
        Feature intensities
    n_features : int
        Number of features
    ppm_tol : float
        Mass tolerance in ppm
    rt_tol_sec : float
        RT tolerance for co-elution

    Returns
    -------
    charge : np.ndarray
        Inferred charge state (0 if unknown)
    m1_idx : np.ndarray
        Index of M+1 feature (-1 if not found)
    m1_rt_diff : np.ndarray
        RT difference to M+1
    """
    charge = np.zeros(n_features, dtype=np.int32)
    m1_idx = np.full(n_features, -1, dtype=np.int32)
    m1_rt_diff = np.zeros(n_features, dtype=np.float64)

    # Sort features by m/z for binary search
    mz_order = np.argsort(feature_mz[:n_features])
    mz_sorted = feature_mz[mz_order]

    for i in range(n_features):
        m0_mz = feature_mz[i]
        m0_rt = feature_rt[i]
        m0_int = feature_intensity[i]

        best_charge = 0
        best_m1_idx = -1
        best_rt_diff = 0.0
        best_score = -1.0

        # Try charge states from HIGH to LOW (4, 3, 2, 1)
        for z in range(4, 0, -1):
            spacing = C13_MASS_DIFF / z

            # Look for M+1
            m1_mz = m0_mz + spacing
            m1_found_idx, m1_found_rt_diff, m1_ratio = _find_peak_at_mz(
                mz_sorted, mz_order, feature_rt, feature_intensity,
                m1_mz, m0_rt, m0_int, ppm_tol, rt_tol_sec
            )

            if m1_found_idx < 0:
                continue  # No M+1, skip this charge

            # Check for M+2 to confirm charge state
            m2_mz = m0_mz + 2 * spacing
            m2_found_idx, m2_found_rt_diff, m2_ratio = _find_peak_at_mz(
                mz_sorted, mz_order, feature_rt, feature_intensity,
                m2_mz, m0_rt, m0_int, ppm_tol, rt_tol_sec
            )

            # Score components
            ratio_score = 1.0 - abs(m1_ratio - 0.6)  # Optimal M+1/M0 around 0.6
            if ratio_score < 0:
                ratio_score = 0.0
            rt_score = 1.0 - (m1_found_rt_diff / rt_tol_sec)

            # M+2 confirmation bonus: strongly prefer charge states with M+2
            m2_bonus = 0.5 if m2_found_idx >= 0 else 0.0

            # Charge bias: peptides are usually z>=2, so penalize z=1 unless confirmed
            if z == 1 and m2_found_idx < 0:
                charge_penalty = -0.3  # Penalize unconfirmed z=1
            else:
                charge_penalty = 0.0

            score = ratio_score + rt_score + m2_bonus + charge_penalty

            if score > best_score:
                best_charge = z
                best_m1_idx = m1_found_idx
                best_rt_diff = m1_found_rt_diff
                best_score = score

        charge[i] = best_charge
        m1_idx[i] = best_m1_idx
        m1_rt_diff[i] = best_rt_diff

    return charge, m1_idx, m1_rt_diff


# =============================================================================
# Charge Pair Analysis (KEY for mass accuracy validation)
# =============================================================================

@njit
def find_charge_pairs(
    feature_mz: np.ndarray,
    feature_rt: np.ndarray,
    feature_intensity: np.ndarray,
    charge: np.ndarray,
    n_features: int,
    rt_tol_sec: float,
    mass_tol_da: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find charge pairs (same peptide at z=2 and z=3).

    This is the KEY validation for mass accuracy!
    Two independent m/z measurements -> true accuracy check.

    For z=2 features, look for z=3 partner with:
    - RT co-elution
    - Same neutral mass (within tolerance)

    Parameters
    ----------
    feature_mz : np.ndarray
        Feature m/z values
    feature_rt : np.ndarray
        Feature RT values
    feature_intensity : np.ndarray
        Feature intensities
    charge : np.ndarray
        Inferred charge states from isotope detection
    n_features : int
        Number of features
    rt_tol_sec : float
        RT tolerance for co-elution
    mass_tol_da : float
        Neutral mass tolerance in Da

    Returns
    -------
    has_partner : np.ndarray
        Boolean, True if z=2 feature has z=3 partner
    partner_idx : np.ndarray
        Index of z=3 partner (-1 if none)
    mass_error_da : np.ndarray
        Neutral mass difference in Da
    mass_error_ppm : np.ndarray
        Neutral mass difference in ppm
    """
    has_partner = np.zeros(n_features, dtype=np.bool_)
    partner_idx = np.full(n_features, -1, dtype=np.int32)
    mass_error_da = np.zeros(n_features, dtype=np.float64)
    mass_error_ppm = np.zeros(n_features, dtype=np.float64)

    # Get z=2 and z=3 feature indices
    z2_indices = np.where(charge[:n_features] == 2)[0]
    z3_indices = np.where(charge[:n_features] == 3)[0]

    if len(z2_indices) == 0 or len(z3_indices) == 0:
        return has_partner, partner_idx, mass_error_da, mass_error_ppm

    # Calculate neutral masses for z=3 features
    z3_neutral_mass = feature_mz[z3_indices] * 3 - 3 * PROTON_MASS
    z3_rt = feature_rt[z3_indices]

    # Sort z=3 by neutral mass for binary search
    mass_order = np.argsort(z3_neutral_mass)
    z3_mass_sorted = z3_neutral_mass[mass_order]

    # For each z=2 feature, find z=3 partner
    for i in range(len(z2_indices)):
        z2_idx = z2_indices[i]
        z2_mz = feature_mz[z2_idx]
        z2_rt = feature_rt[z2_idx]

        # Calculate neutral mass from z=2
        z2_neutral_mass = z2_mz * 2 - 2 * PROTON_MASS

        # Binary search for z=3 candidates with matching neutral mass
        left = np.searchsorted(z3_mass_sorted, z2_neutral_mass - mass_tol_da)
        right = np.searchsorted(z3_mass_sorted, z2_neutral_mass + mass_tol_da)

        best_partner_idx = -1
        best_mass_error = 1e10

        for j in range(left, right):
            z3_sorted_idx = mass_order[j]
            z3_orig_idx = z3_indices[z3_sorted_idx]

            # Check RT co-elution
            rt_diff = abs(z3_rt[z3_sorted_idx] - z2_rt)
            if rt_diff > rt_tol_sec:
                continue

            # Calculate mass error
            z3_mass = z3_neutral_mass[z3_sorted_idx]
            error_da = abs(z2_neutral_mass - z3_mass)

            if error_da < best_mass_error:
                best_mass_error = error_da
                best_partner_idx = z3_orig_idx

        if best_partner_idx >= 0:
            has_partner[z2_idx] = True
            partner_idx[z2_idx] = best_partner_idx
            mass_error_da[z2_idx] = best_mass_error
            mass_error_ppm[z2_idx] = best_mass_error / z2_neutral_mass * 1e6

    return has_partner, partner_idx, mass_error_da, mass_error_ppm


# =============================================================================
# Feature Finder Class (Orchestrator)
# =============================================================================

class FeatureFinder:
    """Feature finder for MS1 data using the argmax algorithm.

    This class orchestrates the feature finding pipeline:
    1. Find features (greedy intensity-ordered grouping)
    2. Detect isotopes (charge state inference)
    3. Find charge pairs (mass accuracy validation)
    4. Calculate statistics

    Parameters
    ----------
    params : FeatureFinderParams, optional
        Algorithm parameters. If None, uses defaults.

    Examples
    --------
    >>> finder = FeatureFinder()
    >>> features = finder.find_features(mz, intensity, scan, rt)
    >>> features = finder.detect_isotopes()
    >>> features = finder.find_charge_pairs()
    >>> stats = finder.calculate_statistics()
    """

    def __init__(self, params: Optional[FeatureFinderParams] = None):
        self.params = params or FeatureFinderParams()

        # Results storage
        self.features: Optional[Dict] = None
        self.n_features: int = 0

    def find_features(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        scan: np.ndarray,
        rt: np.ndarray,
    ) -> Dict:
        """
        Find features in wide spectrum data.

        Parameters
        ----------
        mz : np.ndarray
            m/z values (MUST be sorted!)
        intensity : np.ndarray
            Intensity values
        scan : np.ndarray
            Scan/cycle indices
        rt : np.ndarray
            RT in SECONDS (per peak)

        Returns
        -------
        features : dict
            Dictionary with feature arrays
        """
        # Run numba feature finder
        (feat_mz, feat_rt, feat_int, feat_mz_std, feat_rt_start, feat_rt_end,
         feat_n_peaks, feat_n_scans, n_features) = find_features_numba(
            mz.astype(np.float64),
            intensity.astype(np.float64),
            scan.astype(np.int32),
            rt.astype(np.float64),
            self.params.ppm_tol,
            self.params.rt_tol_sec,
            self.params.intensity_threshold,
            self.params.min_peaks,
        )

        self.n_features = n_features

        # Trim to actual size
        self.features = {
            'mz': feat_mz[:n_features],
            'rt': feat_rt[:n_features],
            'intensity': feat_int[:n_features],
            'mz_std_ppm': feat_mz_std[:n_features],
            'rt_start': feat_rt_start[:n_features],
            'rt_end': feat_rt_end[:n_features],
            'n_peaks': feat_n_peaks[:n_features],
            'n_scans': feat_n_scans[:n_features],
        }

        return self.features

    def detect_isotopes(self) -> Dict:
        """Detect isotope patterns and infer charge states.

        WARNING: Isotope spacing is NOT useful for mass accuracy!
        Use only for charge state assignment.

        Returns
        -------
        features : dict
            Features dict with charge, m1_idx, m1_rt_diff added
        """
        if self.features is None:
            raise ValueError("Run find_features first")

        charge, m1_idx, m1_rt_diff = find_isotope_patterns(
            self.features['mz'],
            self.features['rt'],
            self.features['intensity'],
            self.n_features,
            self.params.isotope_ppm_tol,
            self.params.isotope_rt_tol_sec,
        )

        # Add to features
        self.features['charge'] = charge
        self.features['m1_idx'] = m1_idx
        self.features['m1_rt_diff'] = m1_rt_diff

        return self.features

    def find_charge_pairs(self) -> Dict:
        """Find charge pairs (z=2 and z=3 of same peptide).

        This is the KEY validation for mass accuracy!

        Returns
        -------
        features : dict
            Features dict with charge pair info added
        """
        if self.features is None:
            raise ValueError("Run find_features first")
        if 'charge' not in self.features:
            raise ValueError("Run detect_isotopes first")

        has_partner, partner_idx, mass_error_da, mass_error_ppm = find_charge_pairs(
            self.features['mz'],
            self.features['rt'],
            self.features['intensity'],
            self.features['charge'],
            self.n_features,
            self.params.charge_pair_rt_tol_sec,
            self.params.charge_pair_mass_tol_da,
        )

        # Add to features
        self.features['has_charge_partner'] = has_partner
        self.features['charge_partner_idx'] = partner_idx
        self.features['charge_pair_mass_error_da'] = mass_error_da
        self.features['charge_pair_mass_error_ppm'] = mass_error_ppm

        return self.features

    def calculate_statistics(self) -> Dict:
        """Calculate mass precision and accuracy statistics.

        Returns
        -------
        stats : dict
            Dictionary with precision and accuracy metrics
        """
        if self.features is None:
            raise ValueError("Run find_features first")

        stats = {}

        # 1. Feature-internal precision (mz_std_ppm)
        mz_std = self.features['mz_std_ppm']
        valid_std = mz_std[mz_std > 0]

        if len(valid_std) > 0:
            stats['precision_median_ppm'] = float(np.median(valid_std))
            stats['precision_mean_ppm'] = float(np.mean(valid_std))
            stats['precision_90th_ppm'] = float(np.percentile(valid_std, 90))

        # 2. Charge distribution
        if 'charge' in self.features:
            stats['n_z2'] = int(np.sum(self.features['charge'] == 2))
            stats['n_z3'] = int(np.sum(self.features['charge'] == 3))
            stats['n_z4'] = int(np.sum(self.features['charge'] == 4))

        # 3. Mass accuracy from charge pairs
        if 'has_charge_partner' in self.features:
            has_partner = self.features['has_charge_partner']
            n_pairs = np.sum(has_partner)

            if n_pairs > 0:
                errors = self.features['charge_pair_mass_error_ppm'][has_partner]
                stats['accuracy_median_ppm'] = float(np.median(errors))
                stats['accuracy_mean_ppm'] = float(np.mean(errors))
                stats['accuracy_90th_ppm'] = float(np.percentile(errors, 90))
                stats['n_charge_pairs'] = int(n_pairs)

        stats['n_features'] = self.n_features
        return stats

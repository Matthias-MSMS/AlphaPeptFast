"""
Feature quality scoring for MS1 features.

Assigns quality scores (0-100) to MS1 features based on:
- Mass precision (mz_std_ppm)
- Elution shape (FWHM)
- Peak count and scan coverage
- Intensity relative to threshold
- Isotope pattern quality (for IsotopeGroup)
- Charge state consistency (for ConsolidatedFeature)

Author: Claude Code (ported from MSC_MS1_high_res with extensions)
Date: November 2025
"""

import numpy as np
from numba import njit
from typing import List, Optional

from .isotope_grouping import IsotopeGroup
from .charge_consolidation import ConsolidatedFeature


@njit
def calculate_base_quality_score(
    mz_std_ppm: float,
    fwhm: float,
    n_peaks: int,
    n_scans: int,
    intensity: float,
    intensity_threshold: float = 1000.0
) -> float:
    """Calculate base quality score for a feature (numba-optimized).

    This is the core scoring function from MSC_MS1_high_res, ported to alphapeptfast.

    Components (0-100 total):
    - Mass accuracy: 0-30 points
    - FWHM (elution shape): 0-30 points
    - Peak count: 0-20 points
    - Scan count: 0-10 points
    - Intensity: 0-10 points

    Args:
        mz_std_ppm: Mass standard deviation in ppm
        fwhm: Full Width at Half Maximum (seconds), -1 if not available
        n_peaks: Number of peaks in the feature
        n_scans: Number of scans in the feature
        intensity: Total intensity
        intensity_threshold: Minimum intensity threshold used

    Returns:
        Quality score from 0 to 100
    """
    score = 0.0

    # Mass accuracy component (0-30 points)
    if mz_std_ppm < 1.0:
        score += 30.0
    elif mz_std_ppm < 2.0:
        score += 25.0
    elif mz_std_ppm < 5.0:
        score += 20.0
    elif mz_std_ppm < 10.0:
        score += 10.0

    # FWHM component (0-30 points)
    if 1.0 <= fwhm <= 10.0:  # Ideal peak width
        score += 30.0
    elif 0.5 <= fwhm <= 20.0:  # Acceptable
        score += 20.0
    elif fwhm > 0:  # Has FWHM but not ideal
        score += 10.0

    # Peak count component (0-20 points)
    if n_peaks >= 10:
        score += 20.0
    elif n_peaks >= 5:
        score += 15.0
    elif n_peaks >= 3:
        score += 10.0

    # Scan count component (0-10 points)
    if n_scans >= 5:
        score += 10.0
    elif n_scans >= 3:
        score += 5.0

    # Intensity component (0-10 points)
    if intensity_threshold > 0:
        intensity_ratio = intensity / intensity_threshold
        if intensity_ratio > 100:
            score += 10.0
        elif intensity_ratio > 10:
            score += 7.0
        elif intensity_ratio > 1:
            score += 3.0
    else:
        # No threshold set, use absolute intensity
        if intensity > 10000:
            score += 10.0
        elif intensity > 1000:
            score += 7.0
        elif intensity > 100:
            score += 3.0

    # Cap at 100
    if score > 100.0:
        score = 100.0

    return score


def calculate_isotope_quality(group: IsotopeGroup) -> float:
    """Calculate isotope pattern quality bonus (0-20 points).

    Extends base quality score with isotope-specific components:
    - Has M1 isotope: 10 points
    - Has M2 isotope: 5 points
    - Isotope mass error < 1 ppm: 5 points

    Args:
        group: IsotopeGroup with isotope pattern detection results

    Returns:
        Isotope quality bonus (0-20 points)
    """
    score = 0.0

    # M1 presence (10 points)
    if group.has_m1:
        score += 10.0

        # Isotope mass error quality (5 points)
        if abs(group.m0_m1_mass_error_ppm) < 1.0:
            score += 5.0
        elif abs(group.m0_m1_mass_error_ppm) < 2.0:
            score += 3.0
        elif abs(group.m0_m1_mass_error_ppm) < 5.0:
            score += 1.0

    # M2 presence (5 points)
    if group.has_m2:
        score += 5.0

    return score


def calculate_isotope_group_quality(
    group: IsotopeGroup,
    mz_std_ppm: float,
    fwhm: float,
    n_peaks: int,
    n_scans: int,
    intensity_threshold: float = 1000.0
) -> float:
    """Calculate quality score for an IsotopeGroup.

    Combines base quality (100 points) with isotope quality bonus (20 points),
    then rescales to 0-100.

    Args:
        group: IsotopeGroup to score
        mz_std_ppm: Mass standard deviation in ppm
        fwhm: Full Width at Half Maximum (seconds)
        n_peaks: Number of peaks in the feature
        n_scans: Number of scans in the feature
        intensity_threshold: Minimum intensity threshold

    Returns:
        Quality score from 0 to 100
    """
    # Base quality (0-100)
    base_score = calculate_base_quality_score(
        mz_std_ppm,
        fwhm,
        n_peaks,
        n_scans,
        group.m0_intensity,
        intensity_threshold
    )

    # Isotope bonus (0-20)
    isotope_bonus = calculate_isotope_quality(group)

    # Combined score (0-120), rescaled to 0-100
    combined = base_score + isotope_bonus
    rescaled = (combined / 120.0) * 100.0

    return min(rescaled, 100.0)


def calculate_charge_consistency_bonus(
    mass_consistency_ppm: float,
    n_charge_states: int
) -> float:
    """Calculate charge state consistency bonus (0-10 points).

    Rewards features with multiple charge states that have consistent neutral mass.

    Args:
        mass_consistency_ppm: RSD of neutral mass across charge states (ppm)
        n_charge_states: Number of charge states detected

    Returns:
        Charge consistency bonus (0-10 points)
    """
    score = 0.0

    # Only apply bonus if multiple charge states
    if n_charge_states >= 2:
        # Mass consistency
        if mass_consistency_ppm < 2.0:
            score += 10.0
        elif mass_consistency_ppm < 5.0:
            score += 5.0
        elif mass_consistency_ppm < 10.0:
            score += 2.0

    return score


def calculate_consolidated_feature_quality(
    feature: ConsolidatedFeature,
    mz_std_ppm: float,
    fwhm: float,
    n_peaks: int,
    n_scans: int,
    intensity_threshold: float = 1000.0
) -> float:
    """Calculate quality score for a ConsolidatedFeature.

    Combines base quality (100 points) with isotope quality (20 points)
    and charge consistency bonus (10 points), then rescales to 0-100.

    Args:
        feature: ConsolidatedFeature to score
        mz_std_ppm: Mass standard deviation in ppm
        fwhm: Full Width at Half Maximum (seconds)
        n_peaks: Number of peaks across all charge states
        n_scans: Number of scans across all charge states
        intensity_threshold: Minimum intensity threshold

    Returns:
        Quality score from 0 to 100
    """
    # Base quality (0-100)
    base_score = calculate_base_quality_score(
        mz_std_ppm,
        fwhm,
        n_peaks,
        n_scans,
        feature.total_intensity,
        intensity_threshold
    )

    # Isotope bonus from best charge state (0-20)
    # Use the isotope group from the most intense charge state
    best_group = feature.isotope_groups_by_charge[feature.best_charge]
    isotope_bonus = calculate_isotope_quality(best_group)

    # Charge consistency bonus (0-10)
    charge_bonus = calculate_charge_consistency_bonus(
        feature.mass_consistency_ppm,
        len(feature.charge_states)
    )

    # Combined score (0-130), rescaled to 0-100
    combined = base_score + isotope_bonus + charge_bonus
    rescaled = (combined / 130.0) * 100.0

    return min(rescaled, 100.0)


def score_isotope_groups(
    groups: List[IsotopeGroup],
    mz_std_ppm: np.ndarray,
    fwhm: np.ndarray,
    n_peaks: np.ndarray,
    n_scans: np.ndarray,
    intensity_threshold: float = 1000.0
) -> np.ndarray:
    """Batch scoring for isotope groups.

    Args:
        groups: List of IsotopeGroup objects
        mz_std_ppm: Array of mass std deviations (ppm)
        fwhm: Array of FWHM values (seconds)
        n_peaks: Array of peak counts
        n_scans: Array of scan counts
        intensity_threshold: Minimum intensity threshold

    Returns:
        Array of quality scores (0-100)
    """
    scores = np.zeros(len(groups), dtype=np.float32)

    for i, group in enumerate(groups):
        scores[i] = calculate_isotope_group_quality(
            group,
            mz_std_ppm[i],
            fwhm[i],
            int(n_peaks[i]),
            int(n_scans[i]),
            intensity_threshold
        )

    return scores


def score_consolidated_features(
    features: List[ConsolidatedFeature],
    mz_std_ppm: np.ndarray,
    fwhm: np.ndarray,
    n_peaks: np.ndarray,
    n_scans: np.ndarray,
    intensity_threshold: float = 1000.0
) -> np.ndarray:
    """Batch scoring for consolidated features.

    Args:
        features: List of ConsolidatedFeature objects
        mz_std_ppm: Array of mass std deviations (ppm)
        fwhm: Array of FWHM values (seconds)
        n_peaks: Array of peak counts
        n_scans: Array of scan counts
        intensity_threshold: Minimum intensity threshold

    Returns:
        Array of quality scores (0-100)
    """
    scores = np.zeros(len(features), dtype=np.float32)

    for i, feature in enumerate(features):
        scores[i] = calculate_consolidated_feature_quality(
            feature,
            mz_std_ppm[i],
            fwhm[i],
            int(n_peaks[i]),
            int(n_scans[i]),
            intensity_threshold
        )

    return scores


def filter_by_quality(
    groups_or_features: List,
    quality_scores: np.ndarray,
    min_quality: float = 70.0
) -> List:
    """Filter features by minimum quality threshold.

    Args:
        groups_or_features: List of IsotopeGroup or ConsolidatedFeature objects
        quality_scores: Array of quality scores
        min_quality: Minimum quality threshold (0-100)

    Returns:
        Filtered list of features
    """
    mask = quality_scores >= min_quality
    return [groups_or_features[i] for i in range(len(groups_or_features)) if mask[i]]

"""
Charge state consolidation for MS1 features.

Groups features at different charge states (z=2, z=3, z=4) that represent
the same peptide using neutral mass matching with RT co-elution.

Includes parameter learning from ground truth data.

Author: Claude Code (ported and corrected from MSC_MS1_high_res)
Date: November 2025
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from alphapeptfast.constants import PROTON_MASS
from .isotope_grouping import IsotopeGroup, InstrumentType


@dataclass
class ChargeConsolidationParams:
    """Parameters for charge state consolidation.

    CORRECTED: Uses ppm-based tolerances instead of Da.
    """

    # Mass tolerance (ppm, not Da!)
    mass_tolerance_ppm: float = 3.0

    # RT co-elution tolerance (seconds)
    rt_tolerance_sec: float = 6.0

    # Minimum intensity for features
    min_intensity: float = 1e4

    @classmethod
    def for_instrument(cls, instrument: InstrumentType) -> 'ChargeConsolidationParams':
        """Create parameters optimized for specific instrument type.

        Args:
            instrument: Instrument type enum

        Returns:
            ChargeConsolidationParams with instrument-specific defaults
        """
        if instrument == InstrumentType.MR_TOF:
            return cls(
                mass_tolerance_ppm=1.2,  # Ultra-tight for TOF
                rt_tolerance_sec=5.0,
                min_intensity=1e4
            )
        elif instrument in (InstrumentType.ORBITRAP, InstrumentType.ASTRAL):
            return cls(
                mass_tolerance_ppm=3.0,  # Learned from AlphaDIA
                rt_tolerance_sec=6.0,    # Learned from AlphaDIA
                min_intensity=1e4
            )
        else:
            raise ValueError(f"Unknown instrument type: {instrument}")

    @classmethod
    def learn_from_ground_truth(
        cls,
        features_mz: np.ndarray,
        features_charge: np.ndarray,
        features_sequence: List[str],
        features_score: np.ndarray,
        features_rt: np.ndarray,
        score_threshold: float = 0.8,
        percentile: float = 95.0
    ) -> 'ChargeConsolidationParams':
        """Learn optimal consolidation parameters from high-confidence matches.

        This is the AlphaPeptFast equivalent of the learn_consolidation_parameters.py script.

        Args:
            features_mz: m/z values for all features
            features_charge: Charge states for all features
            features_sequence: Peptide sequences (empty string if unmatched)
            features_score: Match scores (0-1)
            features_rt: Retention times (seconds)
            score_threshold: Minimum score for high-confidence matches
            percentile: Percentile to use for robust parameter selection (95 or 99)

        Returns:
            ChargeConsolidationParams with learned values
        """
        # Filter to high-confidence matches
        confident_mask = features_score >= score_threshold

        if confident_mask.sum() < 100:
            print(f"Warning: Only {confident_mask.sum()} high-confidence matches, using defaults")
            return cls()

        # Group by peptide sequence
        sequence_to_indices = defaultdict(list)
        for idx in np.where(confident_mask)[0]:
            seq = features_sequence[idx]
            if seq:  # Skip empty sequences
                sequence_to_indices[seq].append(idx)

        # Find multi-charge peptides
        mass_errors_ppm = []
        rt_differences_sec = []

        for seq, indices in sequence_to_indices.items():
            if len(indices) < 2:
                continue

            charges = features_charge[indices]
            unique_charges = set(charges)

            if len(unique_charges) < 2:
                continue

            # Calculate neutral masses
            mzs = features_mz[indices]
            neutral_masses = np.array([
                calculate_neutral_mass_scalar(mz, charge)
                for mz, charge in zip(mzs, charges)
            ])

            # Mass consistency (ppm error between charge states)
            mean_mass = np.mean(neutral_masses)
            for mass in neutral_masses:
                ppm_error = abs((mass - mean_mass) / mean_mass * 1e6)
                mass_errors_ppm.append(ppm_error)

            # RT differences (all pairwise)
            rts = features_rt[indices]
            for i in range(len(rts)):
                for j in range(i + 1, len(rts)):
                    rt_diff = abs(rts[i] - rts[j])
                    rt_differences_sec.append(rt_diff)

        if len(mass_errors_ppm) == 0:
            print("Warning: No multi-charge peptides found, using defaults")
            return cls()

        # Calculate recommended parameters
        mass_errors_ppm = np.array(mass_errors_ppm)
        rt_differences_sec = np.array(rt_differences_sec)

        recommended_mass_ppm = np.percentile(mass_errors_ppm, percentile)
        recommended_rt_sec = np.percentile(rt_differences_sec, percentile)

        # Round up for safety
        recommended_mass_ppm = np.ceil(recommended_mass_ppm * 2) / 2  # Round to 0.5 ppm
        recommended_rt_sec = np.ceil(recommended_rt_sec)  # Round to 1 sec

        print(f"Learned consolidation parameters from {len(sequence_to_indices)} peptides:")
        print(f"  Mass tolerance: {recommended_mass_ppm:.1f} ppm ({percentile}th percentile)")
        print(f"  RT tolerance: {recommended_rt_sec:.0f} sec ({percentile}th percentile)")
        print(f"  Multi-charge rate: {len(mass_errors_ppm)/len(sequence_to_indices)*100:.1f}%")

        return cls(
            mass_tolerance_ppm=recommended_mass_ppm,
            rt_tolerance_sec=recommended_rt_sec,
            min_intensity=1e4
        )


def calculate_neutral_mass_scalar(mz: float, charge: int) -> float:
    """Calculate neutral mass from m/z and charge (scalar version).

    Args:
        mz: m/z value
        charge: Charge state

    Returns:
        Neutral mass in Da
    """
    return mz * charge - charge * PROTON_MASS


@njit
def calculate_neutral_mass(mz: float, charge: int) -> float:
    """Calculate neutral mass from m/z and charge (numba version).

    M = (m/z) × z - z × proton_mass

    Args:
        mz: Mass-to-charge ratio
        charge: Charge state (2 or 3)

    Returns:
        Neutral mass in Da
    """
    return mz * charge - charge * 1.00727647  # PROTON_MASS


@njit
def binary_search_mass_range_ppm(
    neutral_masses_sorted: np.ndarray,
    target_mass: float,
    tolerance_ppm: float
) -> Tuple[int, int]:
    """Find all features within mass tolerance using binary search.

    CORRECTED: Uses ppm tolerance instead of Da.

    Args:
        neutral_masses_sorted: Sorted array of neutral masses
        target_mass: Target neutral mass
        tolerance_ppm: Mass tolerance in ppm

    Returns:
        (left_idx, right_idx) indices defining the matching range
    """
    # Convert ppm to Da at target mass
    tolerance_da = (tolerance_ppm / 1e6) * target_mass

    # Find leftmost position
    left = np.searchsorted(neutral_masses_sorted, target_mass - tolerance_da, side='left')
    # Find rightmost position
    right = np.searchsorted(neutral_masses_sorted, target_mass + tolerance_da, side='right')

    return left, right


def find_charge_state_pairs(
    isotope_groups: List[IsotopeGroup],
    params: ChargeConsolidationParams
) -> List[Tuple[int, int, float, float]]:
    """Find charge state pairs (z=2, z=3) using neutral mass matching.

    CORRECTED: Uses ppm-based tolerance.

    Args:
        isotope_groups: List of IsotopeGroup objects with charge assignments
        params: Matching parameters

    Returns:
        List of (idx_2plus, idx_3plus, mass_error_ppm, rt_diff_sec) tuples
    """
    # Filter valid groups with good charge assignments and sufficient intensity
    valid_indices = []
    for i, group in enumerate(isotope_groups):
        if group.charge > 0 and group.m0_intensity >= params.min_intensity:
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return []

    # Separate by charge state
    z2_indices = []
    z3_indices = []

    for i in valid_indices:
        group = isotope_groups[i]
        if group.charge == 2:
            z2_indices.append(i)
        elif group.charge == 3:
            z3_indices.append(i)

    if len(z2_indices) == 0 or len(z3_indices) == 0:
        return []

    # Calculate neutral masses
    z2_neutral = np.array([
        calculate_neutral_mass_scalar(isotope_groups[i].m0_mz, 2)
        for i in z2_indices
    ])
    z3_neutral = np.array([
        calculate_neutral_mass_scalar(isotope_groups[i].m0_mz, 3)
        for i in z3_indices
    ])

    # Get RTs
    z2_rt = np.array([isotope_groups[i].m0_rt for i in z2_indices])
    z3_rt = np.array([isotope_groups[i].m0_rt for i in z3_indices])

    # Sort z=3 by neutral mass for binary search
    z3_sort_idx = np.argsort(z3_neutral)
    z3_neutral_sorted = z3_neutral[z3_sort_idx]
    z3_indices_sorted = [z3_indices[i] for i in z3_sort_idx]
    z3_rt_sorted = z3_rt[z3_sort_idx]

    # Find pairs
    pairs = []

    for i, z2_idx in enumerate(z2_indices):
        z2_mass = z2_neutral[i]
        z2_time = z2_rt[i]

        # Binary search for matching z=3 features (ppm-based)
        left, right = binary_search_mass_range_ppm(
            z3_neutral_sorted,
            z2_mass,
            params.mass_tolerance_ppm
        )

        # Check each matching z=3 feature for RT co-elution
        for j in range(left, right):
            z3_idx = z3_indices_sorted[j]
            z3_mass = z3_neutral_sorted[j]
            z3_time = z3_rt_sorted[j]

            # Check RT tolerance
            rt_diff = abs(z2_time - z3_time)
            if rt_diff <= params.rt_tolerance_sec:
                # Calculate ppm error
                mass_error_ppm = ((z2_mass - z3_mass) / z3_mass) * 1e6
                pairs.append((z2_idx, z3_idx, mass_error_ppm, rt_diff))

    return pairs


@dataclass
class ConsolidatedFeature:
    """A consolidated MS1 feature representing one peptide across multiple charge states."""

    # Charge-independent identifiers
    monoisotopic_mass: float  # Da
    apex_rt: float            # seconds
    charge_states: List[int]  # [2, 3]

    # Per-charge data (indexed by charge state)
    mz_by_charge: Dict[int, float]
    intensity_by_charge: Dict[int, float]
    isotope_groups_by_charge: Dict[int, IsotopeGroup]

    # Combined metrics
    total_intensity: float       # Sum across charges
    best_charge: int             # Highest intensity charge state
    mass_consistency_ppm: float  # RSD across charge states

    # Indices to original isotope groups
    group_indices: List[int]


def consolidate_features(
    isotope_groups: List[IsotopeGroup],
    params: ChargeConsolidationParams
) -> List[ConsolidatedFeature]:
    """Consolidate isotope groups across charge states.

    Args:
        isotope_groups: List of IsotopeGroup objects
        params: Consolidation parameters

    Returns:
        List of ConsolidatedFeature objects
    """
    # Find charge state pairs
    pairs = find_charge_state_pairs(isotope_groups, params)

    # Build graph of connected features
    connections = defaultdict(set)
    for z2_idx, z3_idx, _, _ in pairs:
        connections[z2_idx].add(z3_idx)
        connections[z3_idx].add(z2_idx)

    # Find connected components (features with shared charge states)
    visited = set()
    consolidated = []

    # First, group connected features
    for start_idx in range(len(isotope_groups)):
        if start_idx in visited:
            continue

        # BFS to find all connected features
        component = set()
        queue = [start_idx]

        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue

            visited.add(idx)
            component.add(idx)

            for neighbor in connections[idx]:
                if neighbor not in visited:
                    queue.append(neighbor)

        # Create consolidated feature from component
        if len(component) > 0:
            feature = create_consolidated_feature(list(component), isotope_groups)
            consolidated.append(feature)

    return consolidated


def create_consolidated_feature(
    group_indices: List[int],
    isotope_groups: List[IsotopeGroup]
) -> ConsolidatedFeature:
    """Create a ConsolidatedFeature from a list of isotope group indices.

    Args:
        group_indices: Indices of isotope groups to consolidate
        isotope_groups: Full list of isotope groups

    Returns:
        ConsolidatedFeature object
    """
    # Organize by charge state
    by_charge = defaultdict(list)
    for idx in group_indices:
        group = isotope_groups[idx]
        if group.charge > 0:
            by_charge[group.charge].append((idx, group))

    # Calculate neutral masses
    neutral_masses = []
    for charge, groups in by_charge.items():
        for idx, group in groups:
            neutral_mass = calculate_neutral_mass_scalar(group.m0_mz, charge)
            neutral_masses.append(neutral_mass)

    # Average neutral mass and RT
    mean_mass = np.mean(neutral_masses)
    mean_rt = np.mean([isotope_groups[idx].m0_rt for idx in group_indices])

    # Mass consistency (RSD in ppm)
    mass_rsd_ppm = (np.std(neutral_masses) / mean_mass) * 1e6 if len(neutral_masses) > 1 else 0.0

    # Aggregate by charge
    charge_states = sorted(by_charge.keys())
    mz_by_charge = {}
    intensity_by_charge = {}
    isotope_groups_by_charge = {}

    total_intensity = 0.0
    best_charge = charge_states[0]
    best_intensity = 0.0

    for charge in charge_states:
        groups = by_charge[charge]

        # Use highest intensity group for this charge
        best_group_idx, best_group = max(groups, key=lambda x: x[1].m0_intensity)

        mz_by_charge[charge] = best_group.m0_mz
        intensity_by_charge[charge] = best_group.m0_intensity
        isotope_groups_by_charge[charge] = best_group

        total_intensity += best_group.m0_intensity

        if best_group.m0_intensity > best_intensity:
            best_intensity = best_group.m0_intensity
            best_charge = charge

    return ConsolidatedFeature(
        monoisotopic_mass=mean_mass,
        apex_rt=mean_rt,
        charge_states=charge_states,
        mz_by_charge=mz_by_charge,
        intensity_by_charge=intensity_by_charge,
        isotope_groups_by_charge=isotope_groups_by_charge,
        total_intensity=total_intensity,
        best_charge=best_charge,
        mass_consistency_ppm=mass_rsd_ppm,
        group_indices=group_indices
    )

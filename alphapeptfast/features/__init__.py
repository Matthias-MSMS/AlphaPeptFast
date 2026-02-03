"""MS1 feature detection and consolidation.

This module provides:
- Feature finding using intensity-weighted peak grouping (argmax algorithm)
- Isotope pattern detection with automatic charge state determination
- Charge state consolidation (grouping z=2, z=3, z=4 of same peptide)
- Instrument-specific parameter presets (Orbitrap, MR-TOF, Astral)
- Parameter learning from ground truth data
"""

from .feature_finding import (
    FeatureFinderParams,
    FeatureFinder,
    find_features_numba,
    find_features_core_anneal,
    find_isotope_patterns as find_isotope_patterns_simple,
    find_charge_pairs,
    C13_MASS_DIFF,
)

from .isotope_grouping import (
    IsotopeGroup,
    IsotopeGroupingParams,
    InstrumentType,
    detect_isotope_patterns,
)

from .charge_consolidation import (
    ChargeConsolidationParams,
    ConsolidatedFeature,
    find_charge_state_pairs,
    consolidate_features,
)

from .quality_scoring import (
    calculate_base_quality_score,
    calculate_isotope_quality,
    calculate_isotope_group_quality,
    calculate_charge_consistency_bonus,
    calculate_consolidated_feature_quality,
    score_isotope_groups,
    score_consolidated_features,
    filter_by_quality,
)

__all__ = [
    # Feature finding (argmax algorithm)
    'FeatureFinderParams',
    'FeatureFinder',
    'find_features_numba',
    'find_features_core_anneal',
    'find_isotope_patterns_simple',
    'find_charge_pairs',
    'C13_MASS_DIFF',

    # Isotope grouping
    'IsotopeGroup',
    'IsotopeGroupingParams',
    'InstrumentType',
    'detect_isotope_patterns',

    # Charge consolidation
    'ChargeConsolidationParams',
    'ConsolidatedFeature',
    'find_charge_state_pairs',
    'consolidate_features',

    # Quality scoring
    'calculate_base_quality_score',
    'calculate_isotope_quality',
    'calculate_isotope_group_quality',
    'calculate_charge_consistency_bonus',
    'calculate_consolidated_feature_quality',
    'score_isotope_groups',
    'score_consolidated_features',
    'filter_by_quality',
]

"""Scoring and statistical validation modules for DIA/PRM data.

This module provides statistical tools for proteomics data analysis including:
- FDR calculation using target-decoy approach
- Mass recalibration with RT-segmented calibration
- Peak grouping using cosine similarity
- Co-elution detection
- Composite spectrum building

Key Features
------------
- Numba-accelerated for maximum performance
- Pure NumPy/Numba implementation (no pandas dependency)
- Target-decoy FDR with q-value calculation
- RT-segmented mass recalibration with adaptive binning
- Cosine similarity-based RT profile comparison
- Fragment co-elution detection

Examples
--------
>>> from alphapeptfast.scoring import calculate_fdr, MassRecalibrator
>>>
>>> # Calculate FDR
>>> scores = np.array([10.0, 9.0, 8.0, 7.0])
>>> is_decoy = np.array([False, False, True, False])
>>> fdr, qvalue = calculate_fdr(scores, is_decoy)
>>>
>>> # Mass recalibration
>>> calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)
>>> corrected_mz = calibrator.apply(new_mz, new_rt)
"""

from .fdr import (
    add_decoy_peptides,
    calculate_fdr,
    calculate_fdr_statistics,
)
from .intensity_scoring import (
    AlphaPeptDeepLoader,
    IntensityScorer,
)
from .isotope_scoring import (
    MS1IsotopeScorer,
    calculate_isotope_distribution,
    calculate_isotope_mz_values,
    find_isotope_envelope,
    score_isotope_envelope,
)
from .mass_recalibration import (
    MassRecalibrator,
    estimate_mass_error_from_charge_states,
)
from .peak_grouping import (
    build_composite_spectrum,
    cosine_similarity,
    extract_rt_profiles_around_peak,
    find_coeluting_peaks,
    group_coeluting_peaks,
)

__all__ = [
    # FDR calculation
    "calculate_fdr",
    "add_decoy_peptides",
    "calculate_fdr_statistics",
    # Intensity scoring
    "IntensityScorer",
    "AlphaPeptDeepLoader",
    # MS1 isotope scoring
    "MS1IsotopeScorer",
    "calculate_isotope_distribution",
    "calculate_isotope_mz_values",
    "find_isotope_envelope",
    "score_isotope_envelope",
    # Mass recalibration
    "MassRecalibrator",
    "estimate_mass_error_from_charge_states",
    # Peak grouping
    "cosine_similarity",
    "extract_rt_profiles_around_peak",
    "find_coeluting_peaks",
    "group_coeluting_peaks",
    "build_composite_spectrum",
]

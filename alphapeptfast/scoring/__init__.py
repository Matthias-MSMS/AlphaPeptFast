"""Scoring and statistical validation modules for DIA/PRM data.

This module provides statistical tools for proteomics data analysis including:
- FDR calculation using target-decoy approach
- Peak grouping using cosine similarity
- Co-elution detection
- Composite spectrum building

Key Features
------------
- Numba-accelerated for maximum performance
- Pure NumPy/Numba implementation (no pandas dependency)
- Target-decoy FDR with q-value calculation
- Cosine similarity-based RT profile comparison
- Fragment co-elution detection

Examples
--------
>>> from alphapeptfast.scoring import calculate_fdr, cosine_similarity
>>>
>>> # Calculate FDR
>>> scores = np.array([10.0, 9.0, 8.0, 7.0])
>>> is_decoy = np.array([False, False, True, False])
>>> fdr, qvalue = calculate_fdr(scores, is_decoy)
>>>
>>> # Calculate RT profile similarity
>>> similarity = cosine_similarity(profile1, profile2)
"""

from .fdr import (
    add_decoy_peptides,
    calculate_fdr,
    calculate_fdr_statistics,
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
    # Peak grouping
    "cosine_similarity",
    "extract_rt_profiles_around_peak",
    "find_coeluting_peaks",
    "group_coeluting_peaks",
    "build_composite_spectrum",
]

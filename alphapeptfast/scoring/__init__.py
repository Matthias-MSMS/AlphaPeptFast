"""Scoring and statistical validation modules for DIA/PRM data.

This module provides statistical tools for proteomics data analysis including:
- Peak grouping using cosine similarity
- Co-elution detection
- Composite spectrum building

Key Features
------------
- Numba-accelerated for maximum performance
- Pure NumPy/Numba implementation (no pandas dependency)
- Cosine similarity-based RT profile comparison
- Fragment co-elution detection

Examples
--------
>>> from alphapeptfast.scoring import cosine_similarity, find_coeluting_peaks
>>>
>>> # Calculate RT profile similarity
>>> similarity = cosine_similarity(profile1, profile2)
>>>
>>> # Find co-eluting fragments
>>> coeluting = find_coeluting_peaks(rt_profiles, min_similarity=0.7)
"""

from .peak_grouping import (
    cosine_similarity,
    extract_rt_profiles_around_peak,
    find_coeluting_peaks,
    group_coeluting_peaks,
    build_composite_spectrum,
)

__all__ = [
    "cosine_similarity",
    "extract_rt_profiles_around_peak",
    "find_coeluting_peaks",
    "group_coeluting_peaks",
    "build_composite_spectrum",
]

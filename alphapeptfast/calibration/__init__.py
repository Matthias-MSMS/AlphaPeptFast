"""Calibration functions for learning instrument and method parameters.

This module provides functions to learn kernel parameters (FWHM, offset)
from Q1 profiles and RT profiles. The same algorithms work for both:
- Q1 kernel: How intensity varies as Q1 slides over a precursor
- RT kernel: Chromatographic peak shape (intensity vs retention time)
"""

from .kernels import (
    LearnedKernel,
    learn_kernel_from_profiles,
    measure_profile_fwhm,
    measure_profile_center,
    detect_flat_top,
)

__all__ = [
    "LearnedKernel",
    "learn_kernel_from_profiles",
    "measure_profile_fwhm",
    "measure_profile_center",
    "detect_flat_top",
]

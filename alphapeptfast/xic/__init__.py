"""Ultra-fast XIC extraction for DIA/PRM data.

This module provides high-performance extracted ion chromatogram (XIC) extraction
for Data-Independent Acquisition (DIA) and Parallel Reaction Monitoring (PRM)
proteomics workflows.

Key Features
------------
- Binary search on m/z-sorted data for O(log n) lookup
- Parallel processing with Numba for maximum throughput
- Mass error tracking for quality assessment
- XIC correlation-based scoring
- Gaussian smoothing and FWHM calculation
- >28,000 spectra/second performance

Examples
--------
>>> from alphapeptfast.xic import UltraFastXICExtractor
>>>
>>> # Create extractor
>>> extractor = UltraFastXICExtractor(ppm_tolerance=20.0)
>>>
>>> # Extract XICs
>>> result = extractor.extract_xics(
...     mz_array, intensity_array, scan_array,
...     fragment_mzs, n_scans=1000
... )
>>>
>>> # Score peptides
>>> scores = extractor.score_peptides(
...     result['xic_matrix'], fragment_mzs,
...     result.get('mass_sum_matrix'),
...     result.get('mass_count_matrix')
... )
>>>
>>> # Smooth and calculate FWHM
>>> from alphapeptfast.xic import smooth_and_calculate_fwhm
>>> fwhm = smooth_and_calculate_fwhm(rt_values, xic_intensities)
"""

from .extraction import (
    binary_search_mz_range,
    build_xics_ultrafast,
    build_xics_with_mass_matrix,
    calculate_mass_error_features,
    score_xic_correlation,
    score_peptide_with_mass_errors,
    UltraFastXICExtractor,
)

from .smoothing import (
    smooth_gaussian_1d,
    auto_smooth_xic,
    calculate_fwhm,
    calculate_fwhm_with_apex,
    calculate_peak_quality,
    smooth_and_calculate_fwhm,
)

__all__ = [
    # Extraction
    "binary_search_mz_range",
    "build_xics_ultrafast",
    "build_xics_with_mass_matrix",
    "calculate_mass_error_features",
    "score_xic_correlation",
    "score_peptide_with_mass_errors",
    "UltraFastXICExtractor",
    # Smoothing and peak analysis
    "smooth_gaussian_1d",
    "auto_smooth_xic",
    "calculate_fwhm",
    "calculate_fwhm_with_apex",
    "calculate_peak_quality",
    "smooth_and_calculate_fwhm",
]

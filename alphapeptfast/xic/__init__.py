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

__all__ = [
    "binary_search_mz_range",
    "build_xics_ultrafast",
    "build_xics_with_mass_matrix",
    "calculate_mass_error_features",
    "score_xic_correlation",
    "score_peptide_with_mass_errors",
    "UltraFastXICExtractor",
]

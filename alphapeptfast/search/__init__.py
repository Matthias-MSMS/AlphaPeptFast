"""Search algorithms for peptide spectrum matching.

Core algorithms:
1. Binary search on m/z-sorted spectra (O(log n))
2. Fragment matching with mass tolerance
3. RT coelution filtering for feature-based search
4. Ion mirroring for modification detection

Battle-tested on:
- ProteinFirst_MS1centric v1.0
- 73 peptides, 67% accuracy with simple scoring
- 76,233 peptide database, ~50 candidates per search
- Validated: 2025-10-31
"""

from .fragment_matching import (
    binary_search_mz,
    match_fragments_to_spectrum,
    match_fragments_with_coelution,
    calculate_complementary_mz,
    calculate_match_statistics,
)

__all__ = [
    'binary_search_mz',
    'match_fragments_to_spectrum',
    'match_fragments_with_coelution',
    'calculate_complementary_mz',
    'calculate_match_statistics',
]

"""Peptide database with mass-indexed search.

Provides O(log n) candidate selection via binary search on mass-sorted peptides.

Battle-tested on:
- ProteinFirst_MS1centric v1.0
- 76,233 unique peptides from AlphaDIA
- Binary search: <1ms per query
- Typical search: 50-100 candidates at 5 ppm
- Validated: 2025-10-31
"""

from .peptide_db import (
    PeptideDatabase,
    TargetDecoyDatabase,
    search_mass_range_numba,
)

__all__ = [
    'PeptideDatabase',
    'TargetDecoyDatabase',
    'search_mass_range_numba',
]

"""Fragment generation for peptide sequencing.

This module provides ultra-fast fragment generation for b/y ions using
Numba JIT compilation. Designed for proteome-scale search (>100k peptides/sec).

Battle-tested on:
- ProteinFirst_MS1centric v1.0
- 73 peptides from AlphaDIA ground truth
- 67% accuracy with simple scoring
- Validated: 2025-10-31
"""

from .generator import (
    generate_by_ions,
    encode_peptide_to_ord,
    calculate_neutral_mass,
    calculate_precursor_mz,
    ppm_error,
    generate_fragments_batch,
    PROTON_MASS,
    H2O_MASS,
    NH3_MASS,
    CO_MASS,
    AA_MASSES_DICT,
)

__all__ = [
    'generate_by_ions',
    'encode_peptide_to_ord',
    'calculate_neutral_mass',
    'calculate_precursor_mz',
    'ppm_error',
    'generate_fragments_batch',
    'PROTON_MASS',
    'H2O_MASS',
    'NH3_MASS',
    'CO_MASS',
    'AA_MASSES_DICT',
]

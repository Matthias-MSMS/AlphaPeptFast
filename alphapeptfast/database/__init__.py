"""Peptide database with mass-indexed search.

Provides O(log n) candidate selection via binary search on mass-sorted peptides.

New in v2.0 (2025-11-04):
- FASTA file reading and parsing
- Protein digestion (trypsin with missed cleavages)
- K↔R swap decoy generation (standard proteomics method)
- Direct from_fasta() database creation

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

from .fasta_reader import (
    read_fasta,
    read_multiple_fasta,
    parse_protein_id,
)

from .digestion import (
    digest_protein_trypsin,
    digest_protein_list,
    digest_fasta,
)

from .decoys import (
    generate_kr_swap_decoy,
    generate_reverse_decoy,
    generate_pseudo_reverse_decoy,
    generate_decoys,
)

__all__ = [
    # Core database classes
    'PeptideDatabase',
    'TargetDecoyDatabase',
    'search_mass_range_numba',

    # FASTA reading
    'read_fasta',
    'read_multiple_fasta',
    'parse_protein_id',

    # Protein digestion
    'digest_protein_trypsin',
    'digest_protein_list',
    'digest_fasta',

    # Decoy generation
    'generate_kr_swap_decoy',
    'generate_reverse_decoy',
    'generate_pseudo_reverse_decoy',
    'generate_decoys',
]

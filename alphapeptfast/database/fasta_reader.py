"""FASTA file reading and parsing.

Lightweight FASTA parser for proteomics workflows. Supports:
- UniProt and generic FASTA formats
- Multi-FASTA files
- Protein ID extraction
- Metadata parsing

Design principles:
1. Pure Python (no dependencies except pathlib)
2. Memory efficient (streaming parser)
3. Flexible ID extraction
"""

import logging
from pathlib import Path
from typing import Union, List, Tuple, Optional

logger = logging.getLogger(__name__)


def parse_protein_id(header: str) -> Tuple[str, str]:
    """Extract protein ID and description from FASTA header.

    Supports multiple formats:
    - UniProt: >sp|P12345|NAME_HUMAN Description...
    - UniProt: >tr|A0A123|NAME_HUMAN Description...
    - Generic: >PROTEIN_ID Description...

    Parameters
    ----------
    header : str
        FASTA header line (without leading '>')

    Returns
    -------
    protein_id : str
        Extracted protein identifier
    description : str
        Full header line

    Examples
    --------
    >>> parse_protein_id("sp|P12345|NAME_HUMAN Some protein")
    ('P12345', 'sp|P12345|NAME_HUMAN Some protein')

    >>> parse_protein_id("PROT123 Description here")
    ('PROT123', 'PROT123 Description here')
    """
    description = header.strip()

    # Try UniProt format first: sp|P12345|NAME_HUMAN or tr|A0A123|NAME_HUMAN
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            # Second field is accession (P12345, A0A123, etc.)
            protein_id = parts[1]
        else:
            # Fallback to first field
            protein_id = header.split()[0]
    else:
        # Generic format: first whitespace-separated token
        protein_id = header.split()[0]

    return protein_id, description


def read_fasta(
    fasta_path: Union[str, Path],
    min_length: int = 0,
) -> List[Tuple[str, str, str]]:
    """Read FASTA file and return list of (protein_id, sequence, description).

    Parameters
    ----------
    fasta_path : str or Path
        Path to FASTA file
    min_length : int
        Minimum protein length (default: 0, no filter)

    Returns
    -------
    proteins : List[Tuple[str, str, str]]
        List of (protein_id, sequence, description) tuples

    Examples
    --------
    >>> proteins = read_fasta("human.fasta", min_length=7)
    >>> protein_id, sequence, description = proteins[0]
    >>> print(f"ID: {protein_id}, Length: {len(sequence)}")
    """
    fasta_path = Path(fasta_path)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    logger.info(f"Reading FASTA file: {fasta_path.name}")

    proteins = []
    current_id = None
    current_description = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # Process previous protein if exists
                if current_id and current_seq:
                    sequence = ''.join(current_seq)
                    if len(sequence) >= min_length:
                        proteins.append((current_id, sequence, current_description))

                # Parse new header
                header = line[1:].strip()
                current_id, current_description = parse_protein_id(header)
                current_seq = []
            else:
                # Accumulate sequence lines
                current_seq.append(line.strip())

        # Don't forget last protein
        if current_id and current_seq:
            sequence = ''.join(current_seq)
            if len(sequence) >= min_length:
                proteins.append((current_id, sequence, current_description))

    logger.info(f"✓ Read {len(proteins):,} proteins from {fasta_path.name}")

    return proteins


def read_multiple_fasta(
    fasta_paths: List[Union[str, Path]],
    min_length: int = 0,
) -> List[Tuple[str, str, str]]:
    """Read multiple FASTA files and combine results.

    Parameters
    ----------
    fasta_paths : List[str or Path]
        List of paths to FASTA files
    min_length : int
        Minimum protein length

    Returns
    -------
    proteins : List[Tuple[str, str, str]]
        Combined list of (protein_id, sequence, description) tuples
    """
    all_proteins = []

    for fasta_path in fasta_paths:
        proteins = read_fasta(fasta_path, min_length=min_length)
        all_proteins.extend(proteins)

    logger.info(f"✓ Combined {len(all_proteins):,} proteins from {len(fasta_paths)} files")

    return all_proteins

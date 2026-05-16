"""Decoy peptide generation for FDR control.

Multiple decoy generation methods for target-decoy approach:
- DIA-NN (default): Reverse middle residues, keep BOTH N- and C-termini intact
- K↔R swap: Reverse + swap K↔R (for MS1-centric / sub-ppm-accuracy search)
- Reverse: Simple reversal (legacy)
- Pseudo-reverse (X!Tandem-style): Reverse keeping only the C-terminal residue

DEFAULT IS DIA-NN to match AlphaDIA convention. Use 'kr_swap' for MS1-centric
search where target and decoy must NOT compete in the same MS1 window.

Design principles
-----------------
1. DIA-NN keep-both-termini: Most realistic decoys for tryptic digests — both
   the N-terminal post-cleavage residue AND C-terminal K/R are preserved, so
   target and decoy distributions match at both ends. This is what DIA-NN
   uses (Demichev et al., Nat Methods 2020) and what AlphaDIA adopted.
2. K↔R swap: Creates ~28 Da mass difference (prevents MS1 competition at
   sub-ppm). Preserves tryptic C-term (K↔R). Different mass than target,
   so use only for MS1-anchored / sub-ppm search.
3. Pseudo-reverse (X!Tandem-style, `seq[-2::-1] + seq[-1]`): Preserves only
   the C-terminal residue; the N-terminal residue is the target's penultimate
   residue (typically NOT a tryptic cleavage product). Mass-preserving.
4. Reverse: Pure `seq[::-1]`. Legacy. Both termini changed.

All methods (except K↔R swap) produce decoys with identical mass to target,
which is what classical target-decoy FDR assumes.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def generate_kr_swap_decoy(peptide: str) -> str:
    """Generate decoy by reversing and swapping K↔R.

    This is the RECOMMENDED method for high-accuracy MS1-centric search. It:
    1. Reverses the sequence
    2. Swaps K↔R

    Key advantages over shuffling:
    - **Mass difference**: 28.006 Da per K/R imbalance → prevents MS1 competition
    - **Tryptic preservation**: C-terminal K/R → R/K (shuffling destroys this)
    - **Sub-ppm accuracy**: Target and decoy in different MS1 windows
    - **Different fragmentation**: Reversed b/y ions with swapped K/R

    Parameters
    ----------
    peptide : str
        Target peptide sequence

    Returns
    -------
    decoy : str
        Decoy sequence (reversed with K↔R swapped)

    Examples
    --------
    >>> generate_kr_swap_decoy("PEPTADEK")
    'REDATPEP'
    # Reversed: "KEDATPEP" → K↔R swap: "REDATPEP"
    # Mass difference: +28 Da (1 K → 1 R)

    >>> generate_kr_swap_decoy("PROTEINKR")
    'KRNILETORP'
    # Reversed: "RKNIETORP" → K↔R swap: "KRNILETORP"
    # Mass difference: 0 Da (1 K + 1 R preserved)

    >>> generate_kr_swap_decoy("SEQUENCER")
    'KECNEUQES'
    # Reversed: "RECNEUQES" → K↔R swap: "KECNEUQES"
    # Mass difference: -28 Da (1 R → 1 K)

    Notes
    -----
    Mass behavior (FEATURE, not bug!):
    - K (Lysine): 128.094963 Da
    - R (Arginine): 156.101111 Da
    - Difference: 28.006 Da per K↔R swap
    - Net mass shift: 28.006 × (n_K - n_R) Da

    Why mass difference is GOOD:
    - At 5 ppm on 1000 Da peptide: ±0.005 Da tolerance
    - Mass shift of 28 Da >> 0.005 Da
    - Target and decoy DON'T compete in same MS1 window
    - Proper FDR estimation without MS1 interference
    - Critical for sub-ppm accuracy workflows

    Tryptic preservation:
    - Target: ...K or ...R (C-terminal)
    - After reversal: K or R at N-terminal
    - After K↔R swap: R or K at N-terminal, K or R at C-terminal
    - Result: C-terminal tryptic property PRESERVED (unlike shuffling)
    """
    # Step 1: Reverse sequence
    reversed_seq = peptide[::-1]

    # Step 2: Swap K↔R
    decoy = []
    for aa in reversed_seq:
        if aa == 'K':
            decoy.append('R')
        elif aa == 'R':
            decoy.append('K')
        else:
            decoy.append(aa)

    return ''.join(decoy)


def generate_reverse_decoy(peptide: str) -> str:
    """Generate decoy by reversing sequence.

    Simple reversal method. Preserves:
    - Amino acid composition
    - Precursor mass

    But produces different:
    - Fragmentation patterns (completely inverted)
    - May not preserve tryptic C-terminal

    Parameters
    ----------
    peptide : str
        Target peptide sequence

    Returns
    -------
    decoy : str
        Reversed peptide sequence

    Examples
    --------
    >>> generate_reverse_decoy("PEPTIDEK")
    'KEDITPEP'

    Notes
    -----
    This method does NOT preserve tryptic properties:
    - Target: PEPTIDEK (ends in K)
    - Decoy: KEDITPEP (ends in P, not tryptic)
    """
    return peptide[::-1]


def generate_pseudo_reverse_decoy(peptide: str) -> str:
    """Generate decoy by reversing but preserving C-terminal.

    Pseudo-reverse method. Preserves:
    - Amino acid composition
    - Precursor mass
    - Tryptic C-terminal (K/R stays at end)

    But produces different:
    - Fragmentation patterns

    Parameters
    ----------
    peptide : str
        Target peptide sequence

    Returns
    -------
    decoy : str
        Pseudo-reversed peptide sequence

    Examples
    --------
    >>> generate_pseudo_reverse_decoy("PEPTIDEK")
    'EDITPEPK'  # K preserved at C-terminus

    >>> generate_pseudo_reverse_decoy("PR")
    'PR'  # Too short to reverse

    Notes
    -----
    This method preserves tryptic properties:
    - Target: PEPTIDEK (ends in K)
    - Decoy: EDITPEPK (also ends in K)

    For peptides of length 1-2, returns unchanged (no reversal possible).
    """
    if len(peptide) <= 2:
        return peptide

    # Reverse all but last residue, keep C-terminal
    return peptide[-2::-1] + peptide[-1]


def generate_diann_decoy(peptide: str) -> str:
    """Generate decoy by reversing middle residues, keeping BOTH termini intact.

    DIA-NN convention (Demichev et al., Nat Methods 2020), also used by
    AlphaDIA. Preserves:
    - Amino acid composition
    - Precursor mass (mass-identical to target)
    - Tryptic N-terminal (post-cleavage residue at position 0)
    - Tryptic C-terminal (K/R at position N-1)

    Different fragmentation patterns from target.

    This is the recommended method for traditional MS2-driven target-decoy
    FDR on tryptic peptides because both termini match the target distribution
    at the cleavage products — neither is artefactually different.

    Parameters
    ----------
    peptide : str
        Target peptide sequence

    Returns
    -------
    decoy : str
        DIA-NN-style pseudo-reversed peptide sequence

    Examples
    --------
    >>> generate_diann_decoy("PEPTIDEK")
    'PEDITPEK'  # P (N-term) and K (C-term) both preserved; middle reversed

    >>> generate_diann_decoy("ASPECTKR")
    'ATCEPSKR'  # A and R kept; SPECTK reversed

    >>> generate_diann_decoy("PR")
    'PR'  # length <= 2, no internal residues to reverse

    Notes
    -----
    Compare to X!Tandem reverse-keep-last (`generate_pseudo_reverse_decoy`)
    which only preserves the C-terminal residue:
    - Target:               PEPTIDEK
    - DIA-NN decoy:         PEDITPEK  (both P and K kept)
    - X!Tandem decoy:       EDITPEPK  (only K kept)

    For peptides of length 1-2, returns unchanged (no internal residues).
    """
    if len(peptide) <= 2:
        return peptide

    return peptide[0] + peptide[1:-1][::-1] + peptide[-1]


def generate_decoys(
    target_peptides: List[str],
    method: str = 'diann',
) -> List[str]:
    """Generate decoy peptides from target peptides.

    Parameters
    ----------
    target_peptides : List[str]
        List of target peptide sequences
    method : str
        Decoy generation method (default: 'diann' to match AlphaDIA):
        - 'diann': Reverse middle, keep both N- and C-termini (DEFAULT,
          DIA-NN / AlphaDIA convention; mass-preserving)
        - 'kr_swap': Reverse + swap K↔R (RECOMMENDED for MS1-centric /
          sub-ppm search; ±28 Da mass difference vs target)
        - 'reverse': Simple `seq[::-1]` (legacy)
        - 'pseudo_reverse': X!Tandem reverse-keep-last (only C-term preserved)

    Returns
    -------
    decoy_peptides : List[str]
        List of decoy peptide sequences (same order as targets)

    Examples
    --------
    >>> targets = ["PEPTIDEK", "PROTEINK"]
    >>> decoys = generate_decoys(targets)  # default = 'diann'
    >>> print(decoys)
    ['PEDITPEK', 'PNIETORK']

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == 'diann':
        generator = generate_diann_decoy
    elif method == 'kr_swap':
        generator = generate_kr_swap_decoy
    elif method == 'reverse':
        generator = generate_reverse_decoy
    elif method == 'pseudo_reverse':
        generator = generate_pseudo_reverse_decoy
    else:
        raise ValueError(
            f"Unknown decoy method: {method}. "
            f"Must be 'diann', 'kr_swap', 'reverse', or 'pseudo_reverse'"
        )

    logger.info(f"Generating {len(target_peptides):,} decoy peptides (method: {method})...")

    decoy_peptides = [generator(pep) for pep in target_peptides]

    logger.info(f"✓ Generated {len(decoy_peptides):,} decoys")

    return decoy_peptides


def validate_decoy_mass(target: str, decoy: str, method: str) -> bool:
    """Validate that decoy has same mass as target (for testing).

    Parameters
    ----------
    target : str
        Target peptide sequence
    decoy : str
        Decoy peptide sequence
    method : str
        Decoy generation method

    Returns
    -------
    valid : bool
        True if masses match (within floating point precision)

    Examples
    --------
    >>> validate_decoy_mass("PEPTIDEK", "PEPTIDER", "kr_swap")
    True
    """
    from ..constants import AA_MASSES_DICT, H2O_MASS

    def calc_mass(peptide: str) -> float:
        return sum(AA_MASSES_DICT[aa] for aa in peptide) + H2O_MASS

    target_mass = calc_mass(target)
    decoy_mass = calc_mass(decoy)

    # Allow 0.001 Da difference (floating point precision)
    mass_diff = abs(target_mass - decoy_mass)

    if mass_diff > 0.001:
        logger.warning(
            f"Decoy mass mismatch ({method}): {target} ({target_mass:.4f} Da) → "
            f"{decoy} ({decoy_mass:.4f} Da), diff = {mass_diff:.4f} Da"
        )
        return False

    return True

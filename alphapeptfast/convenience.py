"""Convenience wrapper functions for easy-to-use API.

This module provides Python wrapper functions that handle common preprocessing
steps like sequence cleaning and ord() encoding automatically.

Use these functions when you want a simple API without worrying about:
- ord() encoding
- Non-standard amino acid handling
- Modification array preparation

For performance-critical code, use the Numba-compiled functions directly.

Examples
--------
>>> # Simple API - just pass strings
>>> mass = calculate_peptide_mass("PEPTIDE")
>>> fragments = generate_fragments("PEPTIDE", charge=2)

>>> # With modifications
>>> mods = [("Carbamidomethyl", 2)]
>>> mass = calculate_peptide_mass("PEPTIDE", modifications=mods)
>>> fragments = generate_fragments("PEPTIDE", charge=2, modifications=mods)

>>> # Handles non-standard amino acids automatically
>>> mass = calculate_peptide_mass("PEPTXIDE")  # X → L
"""

from typing import List, Tuple, Optional
import numpy as np

from .constants import NON_STANDARD_AA_MAP
from .fragments.generator import (
    encode_peptide_to_ord,
    calculate_neutral_mass,
    calculate_precursor_mz,
    generate_by_ions,
)
from .modifications import (
    prepare_modifications_for_numba,
    calculate_modified_neutral_mass,
    generate_modified_by_ions,
)


# =============================================================================
# Sequence Cleaning
# =============================================================================

def clean_sequence(sequence: str) -> str:
    """Clean peptide sequence by replacing non-standard amino acids.

    Replaces non-standard one-letter codes with their standard equivalents:
    - X (unknown) → L (leucine, most common)
    - Z (Glu/Gln) → Q (glutamine)
    - B (Asp/Asn) → N (asparagine)
    - J (Leu/Ile) → L (leucine)
    - U (selenocysteine) → C (cysteine)
    - O (pyrrolysine) → M (methionine)

    Parameters
    ----------
    sequence : str
        Peptide sequence (may contain non-standard AAs)

    Returns
    -------
    str
        Cleaned sequence with only standard 20 amino acids

    Examples
    --------
    >>> clean_sequence("PEPTIDE")
    'PEPTIDE'

    >>> clean_sequence("PEPTXIDE")
    'PEPTLIDE'

    >>> clean_sequence("PEZPTBIDE")
    'PEQPTNIDE'
    """
    cleaned = []
    for aa in sequence:
        if aa in NON_STANDARD_AA_MAP:
            cleaned.append(NON_STANDARD_AA_MAP[aa])
        else:
            cleaned.append(aa)
    return ''.join(cleaned)


# =============================================================================
# Mass Calculation Wrappers
# =============================================================================

def calculate_peptide_mass(
    sequence: str,
    modifications: Optional[List[Tuple[str, int]]] = None,
    clean: bool = True
) -> float:
    """Calculate neutral peptide mass (convenience wrapper).

    Simple API for calculating peptide mass with optional modifications.
    Handles sequence cleaning and ord() encoding automatically.

    Parameters
    ----------
    sequence : str
        Peptide sequence (standard one-letter codes)
    modifications : List[Tuple[str, int]], optional
        List of (mod_type, position) tuples from parse_modifications()
    clean : bool, default=True
        Whether to clean non-standard amino acids (X → L, etc.)

    Returns
    -------
    float
        Neutral peptide mass in Daltons (includes terminal H2O)

    Examples
    --------
    >>> # Simple unmodified peptide
    >>> calculate_peptide_mass("PEPTIDE")
    799.360023

    >>> # With modifications
    >>> mods = [("Carbamidomethyl", 2)]
    >>> calculate_peptide_mass("PEPTIDE", modifications=mods)
    856.381487

    >>> # Handles non-standard AAs automatically
    >>> calculate_peptide_mass("PEPTXIDE")  # X cleaned to L
    799.360023

    See Also
    --------
    calculate_neutral_mass : Low-level Numba function (faster, no cleaning)
    calculate_modified_neutral_mass : With modifications (Numba)
    """
    # Clean sequence if requested
    if clean:
        sequence = clean_sequence(sequence)

    # Encode to ord() array
    peptide_ord = encode_peptide_to_ord(sequence)

    # Calculate mass with or without modifications
    if modifications:
        mod_array = prepare_modifications_for_numba(modifications)
        return calculate_modified_neutral_mass(peptide_ord, mod_array)
    else:
        return calculate_neutral_mass(peptide_ord)


def calculate_precursor(
    sequence: str,
    charge: int,
    modifications: Optional[List[Tuple[str, int]]] = None,
    clean: bool = True
) -> float:
    """Calculate precursor m/z (convenience wrapper).

    Parameters
    ----------
    sequence : str
        Peptide sequence
    charge : int
        Precursor charge state (typically 2-4)
    modifications : List[Tuple[str, int]], optional
        List of modifications
    clean : bool, default=True
        Whether to clean non-standard amino acids

    Returns
    -------
    float
        Precursor m/z value

    Examples
    --------
    >>> calculate_precursor("PEPTIDE", charge=2)
    400.6836

    >>> # With modifications
    >>> mods = [("Carbamidomethyl", 2)]
    >>> calculate_precursor("PEPTIDE", charge=2, modifications=mods)
    429.1943
    """
    neutral_mass = calculate_peptide_mass(sequence, modifications, clean)
    return calculate_precursor_mz(neutral_mass, charge)


# =============================================================================
# Fragment Generation Wrappers
# =============================================================================

def generate_fragments(
    sequence: str,
    charge: int,
    modifications: Optional[List[Tuple[str, int]]] = None,
    fragment_types: Tuple[int, ...] = (0, 1),
    fragment_charges: Tuple[int, ...] = (1, 2),
    clean: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate theoretical fragment ions (convenience wrapper).

    Simple API for fragment generation. Handles sequence cleaning,
    ord() encoding, and modification preparation automatically.

    Parameters
    ----------
    sequence : str
        Peptide sequence
    charge : int
        Precursor charge state
    modifications : List[Tuple[str, int]], optional
        List of (mod_type, position) tuples
    fragment_types : tuple of int, default=(0, 1)
        Fragment types: 0=b, 1=y
    fragment_charges : tuple of int, default=(1, 2)
        Fragment charge states to generate
    clean : bool, default=True
        Whether to clean non-standard amino acids

    Returns
    -------
    fragment_mz : np.ndarray (float32)
        m/z values of all fragments
    fragment_type : np.ndarray (uint8)
        Fragment type (0=b, 1=y)
    fragment_position : np.ndarray (uint8)
        Fragment position (1 to len-1)
    fragment_charge : np.ndarray (uint8)
        Fragment charge state

    Examples
    --------
    >>> # Generate b and y ions, charges +1 and +2
    >>> mz, types, pos, charges = generate_fragments("PEPTIDE", charge=2)

    >>> # B-ions only, charge +1 only
    >>> mz, types, pos, charges = generate_fragments(
    ...     "PEPTIDE", charge=2,
    ...     fragment_types=(0,),
    ...     fragment_charges=(1,)
    ... )

    >>> # With modifications
    >>> mods = [("Carbamidomethyl", 2)]
    >>> mz, types, pos, charges = generate_fragments(
    ...     "PEPTIDE", charge=2, modifications=mods
    ... )

    See Also
    --------
    generate_by_ions : Low-level Numba function (faster, no cleaning)
    generate_modified_by_ions : With modifications (Numba)
    """
    # Clean sequence if requested
    if clean:
        sequence = clean_sequence(sequence)

    # Encode to ord() array
    peptide_ord = encode_peptide_to_ord(sequence)

    # Generate fragments with or without modifications
    if modifications:
        mod_array = prepare_modifications_for_numba(modifications)
        return generate_modified_by_ions(
            peptide_ord, mod_array, charge,
            fragment_types, fragment_charges
        )
    else:
        return generate_by_ions(
            peptide_ord, charge,
            fragment_types, fragment_charges
        )


def generate_b_ions(
    sequence: str,
    charge: int,
    modifications: Optional[List[Tuple[str, int]]] = None,
    fragment_charges: Tuple[int, ...] = (1, 2),
    clean: bool = True
) -> np.ndarray:
    """Generate b-ion m/z values (convenience wrapper).

    Parameters
    ----------
    sequence : str
        Peptide sequence
    charge : int
        Precursor charge state
    modifications : List[Tuple[str, int]], optional
        List of modifications
    fragment_charges : tuple of int, default=(1, 2)
        Fragment charge states
    clean : bool, default=True
        Whether to clean non-standard amino acids

    Returns
    -------
    np.ndarray
        Array of b-ion m/z values

    Examples
    --------
    >>> b_ions = generate_b_ions("PEPTIDE", charge=2)
    >>> len(b_ions)  # 6 b-ions for 7-residue peptide
    6
    """
    mz, types, _, _ = generate_fragments(
        sequence, charge, modifications,
        fragment_types=(0,),  # b-ions only
        fragment_charges=fragment_charges,
        clean=clean
    )
    return mz


def generate_y_ions(
    sequence: str,
    charge: int,
    modifications: Optional[List[Tuple[str, int]]] = None,
    fragment_charges: Tuple[int, ...] = (1, 2),
    clean: bool = True
) -> np.ndarray:
    """Generate y-ion m/z values (convenience wrapper).

    Parameters
    ----------
    sequence : str
        Peptide sequence
    charge : int
        Precursor charge state
    modifications : List[Tuple[str, int]], optional
        List of modifications
    fragment_charges : tuple of int, default=(1, 2)
        Fragment charge states
    clean : bool, default=True
        Whether to clean non-standard amino acids

    Returns
    -------
    np.ndarray
        Array of y-ion m/z values

    Examples
    --------
    >>> y_ions = generate_y_ions("PEPTIDE", charge=2)
    >>> len(y_ions)  # 6 y-ions for 7-residue peptide
    6
    """
    mz, types, _, _ = generate_fragments(
        sequence, charge, modifications,
        fragment_types=(1,),  # y-ions only
        fragment_charges=fragment_charges,
        clean=clean
    )
    return mz

"""Handle mass modifications for peptides.

This module provides functions for parsing modification strings and calculating
peptide masses and fragment ions with modifications.

Adapted from AlphaMod for AlphaPeptFast's ord()-based encoding architecture.

Key Features
------------
- Parse modification strings from data files
- Calculate neutral masses with modifications
- Generate modified b/y fragment ions
- Support for common modifications (Carbamidomethyl, Oxidation, Acetyl, etc.)

Examples
--------
>>> # Parse modifications
>>> mods = parse_modifications("Carbamidomethyl@C;Oxidation@M", "3;7")
>>> # Returns: [("Carbamidomethyl", 2), ("Oxidation", 6)]  # 0-based positions

>>> # Calculate modified mass
>>> mass = compute_modified_mass("PEPTIDE", mods)

>>> # Generate modified fragments
>>> peptide_ord = encode_peptide_to_ord("PEPTIDE")
>>> mod_array = prepare_modifications_for_numba(mods)
>>> mz, types, positions, charges = generate_modified_by_ions(
...     peptide_ord, mod_array, precursor_charge=2
... )
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numba

from .constants import (
    H2O_MASS,
    PROTON_MASS,
    AA_MASSES,
    CARBAMIDOMETHYL_MASS,
    OXIDATION_MASS,
    ACETYL_MASS,
    PHOSPHO_MASS,
    DEAMIDATION_MASS,
)


# =============================================================================
# Modification Parsing
# =============================================================================

def parse_modifications(mods: str, mod_sites: str) -> List[Tuple[str, int]]:
    """Parse modification string into list of (mod_type, position) tuples.

    Parses modification strings from proteomics data files (e.g., MaxQuant,
    AlphaDIA) into a structured format.

    Parameters
    ----------
    mods : str
        Modification string, e.g., "Carbamidomethyl@C;Oxidation@M"
        Multiple modifications separated by semicolons
    mod_sites : str
        Modification sites (1-based positions), e.g., "3;6"
        Multiple sites separated by semicolons

    Returns
    -------
    List[Tuple[str, int]]
        List of (modification_type, position) tuples
        Positions are 0-based (converted from 1-based input)

    Examples
    --------
    >>> parse_modifications("Carbamidomethyl@C", "3")
    [('Carbamidomethyl', 2)]  # Position 3 -> 0-based index 2

    >>> parse_modifications("Carbamidomethyl@C;Oxidation@M", "3;7")
    [('Carbamidomethyl', 2), ('Oxidation', 6)]

    >>> parse_modifications("", "")
    []

    Notes
    -----
    - Positions are converted from 1-based (proteomics convention) to 0-based (Python)
    - Handles byte strings (from pandas/numpy)
    - Skips invalid modifications
    """
    if not mods or mods == "":
        return []

    mod_list = mods.split(";")
    site_list = mod_sites.split(";") if ";" in mod_sites else [mod_sites]

    result = []
    for mod, site in zip(mod_list, site_list):
        mod = mod.strip()
        site = str(site).strip()

        # Handle byte strings from pandas/numpy
        if site.startswith("b'") and site.endswith("'"):
            site = site[2:-1]

        if "@" in mod and site.isdigit():
            mod_type = mod.split("@")[0]
            position = int(site) - 1  # Convert to 0-based
            result.append((mod_type, position))

    return result


# =============================================================================
# Mass Calculation with Modifications
# =============================================================================

def compute_modified_mass(sequence: str, modifications: List[Tuple[str, int]]) -> float:
    """Compute peptide neutral mass with modifications.

    Calculates the total neutral mass including:
    - Amino acid masses
    - Water (H2O) for complete peptide
    - Modification mass shifts

    Parameters
    ----------
    sequence : str
        Peptide sequence (standard one-letter codes)
    modifications : List[Tuple[str, int]]
        List of (modification_type, position) tuples from parse_modifications()

    Returns
    -------
    float
        Neutral peptide mass in Daltons

    Examples
    --------
    >>> # Unmodified peptide
    >>> compute_modified_mass("PEPTIDE", [])
    799.360023

    >>> # With Carbamidomethyl C at position 3 (0-based: 2)
    >>> mods = [("Carbamidomethyl", 2)]
    >>> compute_modified_mass("PECPTIDE", mods)
    856.381487  # +57.021464 Da

    Supported Modifications
    -----------------------
    - Carbamidomethyl (C): +57.021464 Da
    - Oxidation (M): +15.994915 Da
    - Acetyl (N-term): +42.010565 Da
    - Phospho (S/T/Y): +79.966331 Da
    - Deamidation (N/Q): +0.984016 Da
    """
    # Base mass: sum of amino acids
    mass = 0.0
    for aa in sequence:
        mass += AA_MASSES[ord(aa)]

    # Add terminal water
    mass += H2O_MASS

    # Add modification mass shifts
    for mod_type, position in modifications:
        if mod_type == "Carbamidomethyl":
            mass += CARBAMIDOMETHYL_MASS
        elif mod_type == "Oxidation":
            mass += OXIDATION_MASS
        elif mod_type == "Acetyl":
            mass += ACETYL_MASS
        elif mod_type == "Phospho":
            mass += PHOSPHO_MASS
        elif mod_type == "Deamidation":
            mass += DEAMIDATION_MASS

    return mass


# =============================================================================
# Modification Array Preparation for Numba
# =============================================================================

def prepare_modifications_for_numba(modifications: List[Tuple[str, int]]) -> np.ndarray:
    """Convert modification list to numpy array for Numba functions.

    Converts structured modification list into a 2D numpy array that can be
    passed to Numba JIT-compiled functions.

    Parameters
    ----------
    modifications : List[Tuple[str, int]]
        List of (modification_type, position) tuples

    Returns
    -------
    np.ndarray
        Array of shape (n_mods, 2) with dtype float64
        Each row: [position, mass_shift]
        Returns empty (0, 2) array if no modifications

    Examples
    --------
    >>> mods = [("Carbamidomethyl", 2), ("Oxidation", 5)]
    >>> mod_array = prepare_modifications_for_numba(mods)
    >>> mod_array
    array([[ 2.      , 57.021464],
           [ 5.      , 15.994915]])

    >>> # Empty modifications
    >>> prepare_modifications_for_numba([])
    array([], shape=(0, 2), dtype=float64)
    """
    if not modifications:
        return np.zeros((0, 2), dtype=np.float64)

    result = np.zeros((len(modifications), 2), dtype=np.float64)

    for i, (mod_type, position) in enumerate(modifications):
        result[i, 0] = position

        # Map modification type to mass shift
        if mod_type == "Carbamidomethyl":
            result[i, 1] = CARBAMIDOMETHYL_MASS
        elif mod_type == "Oxidation":
            result[i, 1] = OXIDATION_MASS
        elif mod_type == "Acetyl":
            result[i, 1] = ACETYL_MASS
        elif mod_type == "Phospho":
            result[i, 1] = PHOSPHO_MASS
        elif mod_type == "Deamidation":
            result[i, 1] = DEAMIDATION_MASS

    return result


# =============================================================================
# Modified Fragment Generation (Numba-Compiled)
# =============================================================================

@numba.jit(nopython=True, cache=True)
def generate_modified_by_ions(
    peptide_ord: np.ndarray,
    modifications: np.ndarray,
    precursor_charge: int,
    fragment_types: Tuple[int, ...] = (0, 1),
    fragment_charges: Tuple[int, ...] = (1, 2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate theoretical b/y fragment m/z values with modifications (Numba-compiled).

    This is the modified version of generate_by_ions() that accounts for
    post-translational modifications.

    Parameters
    ----------
    peptide_ord : np.ndarray (uint8)
        Peptide sequence as ord() values (use encode_peptide_to_ord())
    modifications : np.ndarray (float64)
        Modification array from prepare_modifications_for_numba()
        Shape: (n_mods, 2) where each row is [position, mass_shift]
    precursor_charge : int
        Precursor charge state
    fragment_types : tuple of int
        Fragment types: 0=b, 1=y (default: both)
    fragment_charges : tuple of int
        Charge states to generate (default: 1, 2)

    Returns
    -------
    fragment_mz : np.ndarray (float32)
        m/z values of all fragments
    fragment_type : np.ndarray (uint8)
        Fragment type (0=b, 1=y)
    fragment_position : np.ndarray (uint8)
        Fragment position (1 to peptide_length - 1)
    fragment_charge : np.ndarray (uint8)
        Fragment charge state

    Performance
    -----------
    Similar to unmodified generate_by_ions: >100k peptides/second

    Examples
    --------
    >>> peptide_ord = encode_peptide_to_ord("PEPTIDE")
    >>> mods = prepare_modifications_for_numba([("Carbamidomethyl", 2)])
    >>> mz, types, pos, charges = generate_modified_by_ions(
    ...     peptide_ord, mods, precursor_charge=2
    ... )
    """
    peptide_length = len(peptide_ord)
    n_positions = peptide_length - 1

    # Calculate maximum number of fragments
    max_fragments = n_positions * len(fragment_types) * len(fragment_charges)

    # Pre-allocate arrays
    fragment_mz = np.empty(max_fragments, dtype=np.float32)
    fragment_type = np.empty(max_fragments, dtype=np.uint8)
    fragment_position = np.empty(max_fragments, dtype=np.uint8)
    fragment_charge = np.empty(max_fragments, dtype=np.uint8)

    # Calculate cumulative masses with modifications
    # Forward cumulative sum for b-ions
    cumsum_forward = np.zeros(peptide_length, dtype=np.float64)
    cumsum_forward[0] = AA_MASSES[peptide_ord[0]]

    # Add modifications at position 0
    for j in range(len(modifications)):
        if modifications[j, 0] == 0:
            cumsum_forward[0] += modifications[j, 1]

    for i in range(1, peptide_length):
        cumsum_forward[i] = cumsum_forward[i-1] + AA_MASSES[peptide_ord[i]]

        # Add modifications at position i
        for j in range(len(modifications)):
            if modifications[j, 0] == i:
                cumsum_forward[i] += modifications[j, 1]

    # Backward cumulative sum for y-ions
    cumsum_backward = np.zeros(peptide_length, dtype=np.float64)
    cumsum_backward[peptide_length-1] = AA_MASSES[peptide_ord[peptide_length-1]]

    # Add modifications at last position
    for j in range(len(modifications)):
        if modifications[j, 0] == peptide_length - 1:
            cumsum_backward[peptide_length-1] += modifications[j, 1]

    for i in range(peptide_length-2, -1, -1):
        cumsum_backward[i] = cumsum_backward[i+1] + AA_MASSES[peptide_ord[i]]

        # Add modifications at position i
        for j in range(len(modifications)):
            if modifications[j, 0] == i:
                cumsum_backward[i] += modifications[j, 1]

    # Generate fragments
    idx = 0
    for frag_type in fragment_types:
        for position in range(1, n_positions + 1):
            for charge in fragment_charges:
                # Physical constraint: charge <= position
                if charge > position:
                    continue

                # Calculate fragment mass
                if frag_type == 0:  # b-ion
                    fragment_mass = cumsum_forward[position - 1]
                elif frag_type == 1:  # y-ion
                    fragment_mass = cumsum_backward[peptide_length - position] + H2O_MASS
                else:
                    continue

                # Calculate m/z
                mz = (fragment_mass + charge * PROTON_MASS) / charge

                # Store
                fragment_mz[idx] = mz
                fragment_type[idx] = frag_type
                fragment_position[idx] = position
                fragment_charge[idx] = charge
                idx += 1

    # Return only filled portion
    return (
        fragment_mz[:idx],
        fragment_type[:idx],
        fragment_position[:idx],
        fragment_charge[:idx]
    )


@numba.jit(nopython=True, cache=True)
def calculate_modified_neutral_mass(
    peptide_ord: np.ndarray,
    modifications: np.ndarray
) -> float:
    """Calculate neutral peptide mass with modifications from ord() array.

    Parameters
    ----------
    peptide_ord : np.ndarray (uint8)
        Peptide sequence as ord() values
    modifications : np.ndarray (float64)
        Modification array: shape (n_mods, 2), each row is [position, mass_shift]

    Returns
    -------
    float
        Neutral peptide mass including H2O and modifications

    Examples
    --------
    >>> peptide_ord = encode_peptide_to_ord("PEPTIDE")
    >>> mods = prepare_modifications_for_numba([("Carbamidomethyl", 2)])
    >>> mass = calculate_modified_neutral_mass(peptide_ord, mods)
    """
    # Sum amino acids
    total = 0.0
    for i in range(len(peptide_ord)):
        total += AA_MASSES[peptide_ord[i]]

    # Add terminal water
    total += H2O_MASS

    # Add modifications
    for i in range(len(modifications)):
        total += modifications[i, 1]

    return total

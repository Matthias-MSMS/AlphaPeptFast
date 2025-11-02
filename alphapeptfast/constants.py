"""Physical constants and amino acid masses for mass spectrometry calculations.

This module provides all physical constants, amino acid masses, and tolerance
settings used throughout AlphaPeptFast. All values are sourced from NIST or
established proteomics standards.

Constants are provided in both dictionary and ord()-indexed array formats
for compatibility with both standard Python and Numba JIT-compiled code.

Key Features
------------
- Correct PROTON_MASS (1.007276466622 Da, not hydrogen atom mass!)
- ord()-indexed AA_MASSES array for high-performance Numba code
- Support for non-standard amino acids (X, Z, B, J, U, O)
- Common modification masses (Carbamidomethyl, Oxidation, Acetyl)
- Default tolerance settings for MS1/MS2

Sources
-------
- NIST physical constants: https://physics.nist.gov/cgi-bin/cuu/Value
- IUPAC amino acid masses: https://www.unimod.org/masses.html
- Unimod modification masses: https://www.unimod.org/modifications_list.php
"""

import numpy as np
from numba import types
from numba.typed import Dict as NumbaDict

# =============================================================================
# Fundamental Physical Constants (NIST values)
# =============================================================================

# Proton mass (NOT hydrogen atom mass!)
# Source: NIST 2018 CODATA
# CRITICAL: Use 1.007276466622, not 1.007825 (which is H atom mass)
PROTON_MASS = 1.007276466622  # Da

# Electron mass
# Source: NIST 2018 CODATA
ELECTRON_MASS = 0.000548579909  # Da

# Water mass (H2O)
# Calculated: 2*1.007825 + 15.994915 = 18.010564684
H2O_MASS = 18.010564684  # Da

# Ammonia mass (NH3)
# Calculated: 14.003074 + 3*1.007825 = 17.026549101
NH3_MASS = 17.026549101  # Da

# Carbon monoxide mass (CO)
# Calculated: 12.000000 + 15.994915 = 27.994914620
CO_MASS = 27.994914620  # Da

# =============================================================================
# Ion Type Offsets
# =============================================================================

# b-ions: N-terminal fragments
# Formula: [M + H]+
# Offset = PROTON_MASS
B_ION_OFFSET = PROTON_MASS

# y-ions: C-terminal fragments
# Formula: [M + H2O + H]+
# Offset = H2O_MASS + PROTON_MASS
Y_ION_OFFSET = H2O_MASS + PROTON_MASS

# a-ions: b-ions minus CO
# Not commonly used in DIA, but included for completeness
A_ION_OFFSET = PROTON_MASS - CO_MASS

# c-ions: b-ions plus NH3
# Not commonly used in DIA, but included for completeness
C_ION_OFFSET = PROTON_MASS + NH3_MASS

# =============================================================================
# Isotope Masses
# =============================================================================

# Mass difference between C12 and C13
# Used for isotope envelope calculations
ISOTOPE_MASS_DIFFERENCE = 1.003355  # Da

# =============================================================================
# Amino Acid Monoisotopic Masses (Da)
# =============================================================================

# Standard 20 amino acids (unmodified)
# Source: IUPAC/Unimod mass tables
# Values are monoisotopic masses of residues (not including N/C terminals)
AA_MASSES_DICT = {
    'A': 71.037114,   # Alanine
    'R': 156.101111,  # Arginine
    'N': 114.042927,  # Asparagine
    'D': 115.026943,  # Aspartic acid
    'C': 103.009185,  # Cysteine (unmodified)
    'E': 129.042593,  # Glutamic acid
    'Q': 128.058578,  # Glutamine
    'G': 57.021464,   # Glycine
    'H': 137.058912,  # Histidine
    'I': 113.084064,  # Isoleucine
    'L': 113.084064,  # Leucine
    'K': 128.094963,  # Lysine
    'M': 131.040485,  # Methionine
    'F': 147.068414,  # Phenylalanine
    'P': 97.052764,   # Proline
    'S': 87.032028,   # Serine
    'T': 101.047679,  # Threonine
    'W': 186.079313,  # Tryptophan
    'Y': 163.063320,  # Tyrosine (Note: AlphaMod has 163.063329, minor diff)
    'V': 99.068414,   # Valine
}

# Common modifications built into amino acid masses
# These are often pre-applied in proteomics workflows
AA_MASSES_MODIFIED = {
    'C[Carbamidomethyl]': 160.030649,  # C + 57.021464
    'M[Oxidation]': 147.035400,         # M + 15.994915
}

# Non-standard amino acids mapped to standard equivalents
# These appear in some sequence databases or low-quality data
# Map to closest mass or most common interpretation
AA_MASSES_NONSTANDARD = {
    'X': 113.084064,  # Unknown → Leu/Ile (most common)
    'Z': 128.058578,  # Glu/Gln → Gln
    'B': 114.042927,  # Asp/Asn → Asn
    'J': 113.084064,  # Leu/Ile → Leu
    'U': 103.009185,  # Selenocysteine → Cys (similar mass)
    'O': 131.040485,  # Pyrrolysine → Met (closest mass)
}

# =============================================================================
# Mapping of Non-Standard Amino Acids
# =============================================================================

# Used for sequence cleaning before fragment generation
# Maps non-standard one-letter codes to standard amino acids
NON_STANDARD_AA_MAP = {
    'X': 'L',  # Unknown → Leucine (most common)
    'Z': 'Q',  # Glu/Gln → Glutamine
    'B': 'N',  # Asp/Asn → Asparagine
    'J': 'L',  # Leu/Ile → Leucine
    'U': 'C',  # Selenocysteine → Cysteine
    'O': 'M',  # Pyrrolysine → Methionine
}

# =============================================================================
# ord()-Indexed Arrays for Numba
# =============================================================================

# Create ord()-indexed lookup array for fast Numba access
# Array size 256 covers full ASCII range
# Access via: AA_MASSES[ord('A')] → 71.037114
AA_MASSES = np.zeros(256, dtype=np.float64)

# Populate with standard amino acids
for aa, mass in AA_MASSES_DICT.items():
    AA_MASSES[ord(aa)] = mass

# Populate with non-standard amino acids (for compatibility)
for aa, mass in AA_MASSES_NONSTANDARD.items():
    AA_MASSES[ord(aa)] = mass

# =============================================================================
# Numba-Typed Dictionary (for legacy compatibility)
# =============================================================================

# Some existing code uses Numba TypedDict instead of ord() arrays
# Provided for compatibility with AlphaMod-style code
aa_masses_numba = NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)
for aa, mass in AA_MASSES_DICT.items():
    aa_masses_numba[aa] = mass
for aa, mass in AA_MASSES_NONSTANDARD.items():
    aa_masses_numba[aa] = mass

# =============================================================================
# Common Modification Masses
# =============================================================================

# Carbamidomethylation of Cysteine (Unimod:4)
# Most common fixed modification in proteomics
# C2H3NO: 57.021464 Da
CARBAMIDOMETHYL_MASS = 57.021464

# Oxidation of Methionine (Unimod:35)
# Most common variable modification
# O: 15.994915 Da
OXIDATION_MASS = 15.994915

# Acetylation (Protein N-term, Unimod:1)
# Common N-terminal modification
# C2H2O: 42.010565 Da
ACETYL_MASS = 42.010565

# Phosphorylation (Unimod:21)
# Important for phosphoproteomics
# HPO3: 79.966331 Da
PHOSPHO_MASS = 79.966331

# Deamidation (Unimod:7)
# Common artifact on N and Q
# NH → O: 0.984016 Da
DEAMIDATION_MASS = 0.984016

# =============================================================================
# Default Tolerance Settings
# =============================================================================

# Default MS1 (precursor) mass tolerance in PPM
# Typical for high-resolution MS (Orbitrap, Q-TOF)
DEFAULT_MS1_TOLERANCE = 10.0  # ppm

# Default MS2 (fragment) mass tolerance in PPM
# Typically wider than MS1 due to faster scan rates
DEFAULT_MS2_TOLERANCE = 20.0  # ppm

# Default isotope detection tolerance in PPM
# Used for detecting C13 isotope peaks
DEFAULT_ISOTOPE_TOLERANCE = 5.0  # ppm

# =============================================================================
# Mass Accuracy Validation
# =============================================================================

def validate_constants():
    """Validate that constants are physically reasonable.

    Raises AssertionError if any constant is out of expected range.
    This is a sanity check to catch copy-paste errors or typos.
    """
    # Proton mass should be ~1.007276, NOT 1.007825 (hydrogen atom)
    assert 1.0072 < PROTON_MASS < 1.0073, f"PROTON_MASS is wrong: {PROTON_MASS}"

    # Electron mass should be ~0.000549
    assert 0.0005 < ELECTRON_MASS < 0.0006, f"ELECTRON_MASS is wrong: {ELECTRON_MASS}"

    # Water mass should be ~18.01
    assert 18.00 < H2O_MASS < 18.02, f"H2O_MASS is wrong: {H2O_MASS}"

    # Check that hydrogen atom = proton + electron (within floating point precision)
    H_ATOM_MASS = PROTON_MASS + ELECTRON_MASS
    assert abs(H_ATOM_MASS - 1.007825) < 0.000001, \
        f"H atom mass inconsistent: {H_ATOM_MASS}"

    # All standard amino acids should have mass > 0
    for aa, mass in AA_MASSES_DICT.items():
        assert mass > 50.0, f"AA {aa} mass is too low: {mass}"
        assert mass < 250.0, f"AA {aa} mass is too high: {mass}"

    print("✓ All constants validated successfully")

# Run validation on module import (catches errors early)
# Comment out if this causes issues during import
# validate_constants()

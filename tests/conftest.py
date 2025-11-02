"""Pytest configuration for AlphaPeptFast tests.

This module provides common fixtures and configuration for all tests.
Simplified compared to AlphaMod since AlphaPeptFast focuses on pure
computation without I/O dependencies.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_peptide():
    """Simple peptide for basic tests."""
    return "PEPTIDE"


@pytest.fixture
def simple_peptide_ord():
    """Simple peptide as ord() array."""
    from alphapeptfast.fragments.generator import encode_peptide_to_ord
    return encode_peptide_to_ord("PEPTIDE")


@pytest.fixture
def tryptic_peptides():
    """Collection of typical tryptic peptides."""
    return [
        "PEPTIDE",
        "ACDEK",
        "TESTPEPTIDER",
        "YGGFMTSEK",
        "LGEHNIDVLEGNEQFINAAK",
    ]


@pytest.fixture
def known_peptide_masses():
    """Known peptide masses for validation.

    Calculated using established tools (AlphaMod, Unimod calculator).
    Used to verify mass calculation accuracy.
    """
    return {
        "PEPTIDE": 799.360023,   # Neutral mass with H2O
        "ACDEK": 565.217789,     # With Carbamidomethyl C
        "TESTPEPTIDER": 1031.478569,
        "YGGFMTSEK": 1002.439819,
    }


@pytest.fixture
def aa_masses():
    """Amino acid masses for testing."""
    from alphapeptfast.constants import AA_MASSES
    return AA_MASSES


@pytest.fixture
def aa_masses_dict():
    """Amino acid masses dictionary."""
    from alphapeptfast.constants import AA_MASSES_DICT
    return AA_MASSES_DICT


@pytest.fixture
def proton_mass():
    """Proton mass constant."""
    from alphapeptfast.constants import PROTON_MASS
    return PROTON_MASS


@pytest.fixture
def h2o_mass():
    """Water mass constant."""
    from alphapeptfast.constants import H2O_MASS
    return H2O_MASS


# Random seed for reproducibility
@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)

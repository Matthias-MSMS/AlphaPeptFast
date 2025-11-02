"""Unit tests for the modifications module.

Adapted from AlphaMod for AlphaPeptFast's ord()-based encoding architecture.
"""

import numpy as np
import pytest

from alphapeptfast.constants import (
    H2O_MASS,
    PROTON_MASS,
    AA_MASSES_DICT,
    CARBAMIDOMETHYL_MASS,
    OXIDATION_MASS,
    ACETYL_MASS,
)
from alphapeptfast.modifications import (
    parse_modifications,
    compute_modified_mass,
    prepare_modifications_for_numba,
    generate_modified_by_ions,
    calculate_modified_neutral_mass,
)
from alphapeptfast.fragments.generator import encode_peptide_to_ord


class TestParseModifications:
    """Test modification parsing functionality."""

    def test_empty_modifications(self):
        """Test parsing empty modification strings."""
        assert parse_modifications("", "") == []
        assert parse_modifications(None, "") == []
        assert parse_modifications("", "1") == []

    def test_single_modification(self):
        """Test parsing single modification."""
        mods, sites = "Carbamidomethyl@C", "3"
        result = parse_modifications(mods, sites)
        assert result == [("Carbamidomethyl", 2)]  # 0-based position

    def test_multiple_modifications(self):
        """Test parsing multiple modifications."""
        mods = "Carbamidomethyl@C;Oxidation@M"
        sites = "3;6"
        result = parse_modifications(mods, sites)
        assert result == [("Carbamidomethyl", 2), ("Oxidation", 5)]

    def test_byte_string_sites(self):
        """Test parsing sites as byte strings."""
        mods = "Oxidation@M"
        sites = "b'5'"
        result = parse_modifications(mods, sites)
        assert result == [("Oxidation", 4)]

    def test_invalid_format(self):
        """Test handling of invalid modification format."""
        # Missing @ symbol
        mods = "Carbamidomethyl"
        sites = "3"
        result = parse_modifications(mods, sites)
        assert result == []

        # Non-numeric site
        mods = "Carbamidomethyl@C"
        sites = "abc"
        result = parse_modifications(mods, sites)
        assert result == []

    def test_whitespace_handling(self):
        """Test handling of whitespace in modifications."""
        mods = " Carbamidomethyl@C ; Oxidation@M "
        sites = " 3 ; 6 "
        result = parse_modifications(mods, sites)
        assert result == [("Carbamidomethyl", 2), ("Oxidation", 5)]


class TestComputeModifiedMass:
    """Test modified mass calculation."""

    def test_unmodified_peptide(self):
        """Test mass calculation for unmodified peptide."""
        sequence = "PEPTIDE"
        modifications = []
        mass = compute_modified_mass(sequence, modifications)

        # Calculate expected mass
        expected = sum(AA_MASSES_DICT[aa] for aa in sequence) + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_single_carbamidomethyl(self):
        """Test mass with single carbamidomethyl modification."""
        sequence = "PEPTCIDE"  # C at position 4 (0-based: 3)
        modifications = [("Carbamidomethyl", 3)]
        mass = compute_modified_mass(sequence, modifications)

        expected = sum(AA_MASSES_DICT[aa] for aa in sequence) + H2O_MASS + CARBAMIDOMETHYL_MASS
        assert abs(mass - expected) < 0.001

    def test_single_oxidation(self):
        """Test mass with single oxidation modification."""
        sequence = "PEPTMIDE"  # M at position 4 (0-based: 3)
        modifications = [("Oxidation", 3)]
        mass = compute_modified_mass(sequence, modifications)

        expected = sum(AA_MASSES_DICT[aa] for aa in sequence) + H2O_MASS + OXIDATION_MASS
        assert abs(mass - expected) < 0.001

    def test_multiple_modifications(self):
        """Test mass with multiple modifications."""
        sequence = "CMPTCMDE"  # C at 0,4 and M at 1,5
        modifications = [
            ("Carbamidomethyl", 0),
            ("Oxidation", 1),
            ("Carbamidomethyl", 4),
            ("Oxidation", 5)
        ]
        mass = compute_modified_mass(sequence, modifications)

        expected = (
            sum(AA_MASSES_DICT[aa] for aa in sequence) +
            H2O_MASS +
            2 * CARBAMIDOMETHYL_MASS +
            2 * OXIDATION_MASS
        )
        assert abs(mass - expected) < 0.001

    def test_acetyl_modification(self):
        """Test mass with N-terminal acetylation."""
        sequence = "PEPTIDE"
        modifications = [("Acetyl", 0)]  # N-terminal
        mass = compute_modified_mass(sequence, modifications)

        expected = sum(AA_MASSES_DICT[aa] for aa in sequence) + H2O_MASS + ACETYL_MASS
        assert abs(mass - expected) < 0.001


class TestPrepareModificationsForNumba:
    """Test modification preparation for Numba."""

    def test_empty_modifications(self):
        """Test preparing empty modification list."""
        result = prepare_modifications_for_numba([])
        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    def test_single_modification(self):
        """Test preparing single modification."""
        modifications = [("Carbamidomethyl", 3)]
        result = prepare_modifications_for_numba(modifications)

        assert result.shape == (1, 2)
        assert result[0, 0] == 3
        assert result[0, 1] == CARBAMIDOMETHYL_MASS

    def test_multiple_modifications(self):
        """Test preparing multiple modifications."""
        modifications = [("Carbamidomethyl", 0), ("Oxidation", 5), ("Acetyl", 0)]
        result = prepare_modifications_for_numba(modifications)

        assert result.shape == (3, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == CARBAMIDOMETHYL_MASS
        assert result[1, 0] == 5
        assert result[1, 1] == OXIDATION_MASS
        assert result[2, 0] == 0
        assert result[2, 1] == ACETYL_MASS

    def test_unknown_modification_type(self):
        """Test handling of unknown modification type."""
        modifications = [("UnknownMod", 3)]
        result = prepare_modifications_for_numba(modifications)

        assert result.shape == (1, 2)
        assert result[0, 0] == 3
        assert result[0, 1] == 0.0  # Unknown modification gets 0 mass


class TestModifiedNeutralMass:
    """Test neutral mass calculation with modifications (ord-based)."""

    def test_unmodified_mass_ord(self):
        """Test neutral mass calculation without modifications."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = np.zeros((0, 2), dtype=np.float64)

        mass = calculate_modified_neutral_mass(peptide_ord, modifications)

        expected = sum(AA_MASSES_DICT[aa] for aa in peptide) + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_modified_mass_ord(self):
        """Test neutral mass calculation with modifications."""
        peptide = "PEPTCIDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([("Carbamidomethyl", 4)])

        mass = calculate_modified_neutral_mass(peptide_ord, modifications)

        expected = (
            sum(AA_MASSES_DICT[aa] for aa in peptide) +
            H2O_MASS +
            CARBAMIDOMETHYL_MASS
        )
        assert abs(mass - expected) < 0.001

    def test_multiple_modifications_ord(self):
        """Test neutral mass with multiple modifications."""
        peptide = "CMPTCMDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([
            ("Carbamidomethyl", 0),
            ("Oxidation", 1),
            ("Carbamidomethyl", 4),
            ("Oxidation", 5)
        ])

        mass = calculate_modified_neutral_mass(peptide_ord, modifications)

        expected = (
            sum(AA_MASSES_DICT[aa] for aa in peptide) +
            H2O_MASS +
            2 * CARBAMIDOMETHYL_MASS +
            2 * OXIDATION_MASS
        )
        assert abs(mass - expected) < 0.001


class TestModifiedFragmentGeneration:
    """Test modified fragment generation with ord-based encoding."""

    def test_unmodified_fragments(self):
        """Test fragment generation without modifications."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = np.zeros((0, 2), dtype=np.float64)

        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=2,
            fragment_types=(0, 1), fragment_charges=(1,)
        )

        # Should generate b and y ions
        assert len(mz) > 0
        assert np.any(types == 0)  # b-ions present
        assert np.any(types == 1)  # y-ions present

        # Verify b1 mass (P + H+)
        b1_idx = np.where((types == 0) & (positions == 1) & (charges == 1))[0]
        assert len(b1_idx) == 1
        expected_b1 = AA_MASSES_DICT["P"] + PROTON_MASS
        assert abs(mz[b1_idx[0]] - expected_b1) < 0.001

    def test_modified_b_ions(self):
        """Test b-ion generation with modifications."""
        peptide = "PEPTCIDE"  # C at position 4 (0-based)
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([("Carbamidomethyl", 4)])

        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)
        )

        # b4 should NOT include modification (PEPT)
        b4_idx = np.where((types == 0) & (positions == 4) & (charges == 1))[0]
        assert len(b4_idx) == 1
        expected_b4 = sum(AA_MASSES_DICT[aa] for aa in "PEPT") + PROTON_MASS
        assert abs(mz[b4_idx[0]] - expected_b4) < 0.001

        # b5 should include modification (PEPTC + mod)
        b5_idx = np.where((types == 0) & (positions == 5) & (charges == 1))[0]
        assert len(b5_idx) == 1
        expected_b5 = (
            sum(AA_MASSES_DICT[aa] for aa in "PEPTC") +
            CARBAMIDOMETHYL_MASS +
            PROTON_MASS
        )
        assert abs(mz[b5_idx[0]] - expected_b5) < 0.001

    def test_modified_y_ions(self):
        """Test y-ion generation with modifications."""
        peptide = "PEPTCIDE"  # C at position 4 (0-based)
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([("Carbamidomethyl", 4)])

        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=2,
            fragment_types=(1,),  # y-ions only
            fragment_charges=(1,)
        )

        # y3 should NOT include modification (IDE)
        y3_idx = np.where((types == 1) & (positions == 3) & (charges == 1))[0]
        assert len(y3_idx) == 1
        expected_y3 = sum(AA_MASSES_DICT[aa] for aa in "IDE") + H2O_MASS + PROTON_MASS
        assert abs(mz[y3_idx[0]] - expected_y3) < 0.001

        # y4 should include modification (CIDE + mod)
        y4_idx = np.where((types == 1) & (positions == 4) & (charges == 1))[0]
        assert len(y4_idx) == 1
        expected_y4 = (
            sum(AA_MASSES_DICT[aa] for aa in "CIDE") +
            CARBAMIDOMETHYL_MASS +
            H2O_MASS +
            PROTON_MASS
        )
        assert abs(mz[y4_idx[0]] - expected_y4) < 0.001

    def test_multiple_modifications_fragments(self):
        """Test fragment generation with multiple modifications."""
        peptide = "CMPTCMDE"  # C at 0,4 and M at 1,5
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([
            ("Carbamidomethyl", 0),
            ("Oxidation", 1),
            ("Carbamidomethyl", 4),
            ("Oxidation", 5)
        ])

        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)
        )

        # b1 should include C modification
        b1_idx = np.where((types == 0) & (positions == 1) & (charges == 1))[0]
        assert len(b1_idx) == 1
        expected_b1 = AA_MASSES_DICT["C"] + CARBAMIDOMETHYL_MASS + PROTON_MASS
        assert abs(mz[b1_idx[0]] - expected_b1) < 0.001

        # b2 should include both C and M modifications
        b2_idx = np.where((types == 0) & (positions == 2) & (charges == 1))[0]
        assert len(b2_idx) == 1
        expected_b2 = (
            AA_MASSES_DICT["C"] + AA_MASSES_DICT["M"] +
            CARBAMIDOMETHYL_MASS + OXIDATION_MASS +
            PROTON_MASS
        )
        assert abs(mz[b2_idx[0]] - expected_b2) < 0.001

    def test_modified_fragments_with_multiple_charges(self):
        """Test that modified fragments work with multiple charge states."""
        peptide = "PEPTCIDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        modifications = prepare_modifications_for_numba([("Carbamidomethyl", 4)])

        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=3,
            fragment_types=(0, 1),
            fragment_charges=(1, 2)
        )

        # Should have both charge 1 and charge 2
        assert np.any(charges == 1)
        assert np.any(charges == 2)

        # All fragments should obey charge <= position constraint
        for pos, charge in zip(positions, charges):
            assert charge <= pos


class TestIntegrationModifications:
    """Integration tests for modifications."""

    def test_full_pipeline(self):
        """Test complete modification pipeline."""
        # Parse from string
        mods = parse_modifications("Carbamidomethyl@C;Oxidation@M", "3;5")

        # Prepare for Numba
        mod_array = prepare_modifications_for_numba(mods)

        # Calculate mass
        sequence = "PEPCTMIDE"
        mass = compute_modified_mass(sequence, mods)

        # Generate fragments
        peptide_ord = encode_peptide_to_ord(sequence)
        mz, types, positions, charges = generate_modified_by_ions(
            peptide_ord, mod_array, precursor_charge=2
        )

        # Verify we got fragments
        assert len(mz) > 0

        # Verify mass is correct
        expected = (
            sum(AA_MASSES_DICT[aa] for aa in sequence) +
            H2O_MASS +
            CARBAMIDOMETHYL_MASS +
            OXIDATION_MASS
        )
        assert abs(mass - expected) < 0.001

    def test_consistency_with_unmodified(self):
        """Test that unmodified generates same results as no modifications."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        # Generate with modifications module (empty modifications)
        modifications = np.zeros((0, 2), dtype=np.float64)
        mz_mod, types_mod, pos_mod, charges_mod = generate_modified_by_ions(
            peptide_ord, modifications, precursor_charge=2,
            fragment_types=(0, 1), fragment_charges=(1,)
        )

        # Generate with regular function
        from alphapeptfast.fragments.generator import generate_by_ions
        mz_reg, types_reg, pos_reg, charges_reg = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1), fragment_charges=(1,)
        )

        # Should be identical
        assert len(mz_mod) == len(mz_reg)
        np.testing.assert_allclose(mz_mod, mz_reg, rtol=1e-6)
        np.testing.assert_array_equal(types_mod, types_reg)
        np.testing.assert_array_equal(pos_mod, pos_reg)
        np.testing.assert_array_equal(charges_mod, charges_reg)

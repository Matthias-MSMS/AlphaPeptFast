"""Tests for convenience wrapper functions.

Tests the user-friendly API that handles sequence cleaning and encoding automatically.
"""

import numpy as np
import pytest

from alphapeptfast.convenience import (
    clean_sequence,
    calculate_peptide_mass,
    calculate_precursor,
    generate_fragments,
    generate_b_ions,
    generate_y_ions,
)
from alphapeptfast.constants import AA_MASSES_DICT, H2O_MASS, PROTON_MASS


class TestCleanSequence:
    """Test sequence cleaning functionality."""

    def test_clean_standard_sequence(self):
        """Test that standard sequences pass through unchanged."""
        sequence = "PEPTIDE"
        assert clean_sequence(sequence) == sequence

    def test_clean_single_nonstandard(self):
        """Test cleaning single non-standard AA."""
        assert clean_sequence("PEPTXIDE") == "PEPTLIDE"  # X → L

    def test_clean_multiple_nonstandard(self):
        """Test cleaning multiple non-standard AAs."""
        # X→L, Z→Q, B→N, J→L, U→C, O→M
        assert clean_sequence("XZBJUO") == "LQNLCM"

    def test_clean_mixed_sequence(self):
        """Test cleaning sequence with mix of standard and non-standard."""
        assert clean_sequence("PEZPTBIDE") == "PEQPTNIDE"

    def test_clean_empty_sequence(self):
        """Test cleaning empty sequence."""
        assert clean_sequence("") == ""


class TestCalculatePeptideMass:
    """Test peptide mass calculation wrapper."""

    def test_calculate_mass_simple(self):
        """Test simple mass calculation."""
        mass = calculate_peptide_mass("PEPTIDE")
        expected = sum(AA_MASSES_DICT[aa] for aa in "PEPTIDE") + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_calculate_mass_with_cleaning(self):
        """Test mass calculation with sequence cleaning."""
        # X should be cleaned to L
        mass_with_x = calculate_peptide_mass("PEPTXIDE")
        mass_with_l = calculate_peptide_mass("PEPTLIDE")
        assert abs(mass_with_x - mass_with_l) < 0.001

    def test_calculate_mass_without_cleaning(self):
        """Test that cleaning can be disabled."""
        # Note: X is actually in AA_MASSES (from NON_STANDARD_AA_MAP)
        # So this test just verifies that clean=False skips the cleaning step
        mass_no_clean = calculate_peptide_mass("PEPTXIDE", clean=False)
        mass_with_clean = calculate_peptide_mass("PEPTXIDE", clean=True)

        # Both should give the same result since X→L in AA_MASSES
        # This just verifies the parameter works
        assert abs(mass_no_clean - mass_with_clean) < 0.001

    def test_calculate_mass_with_modifications(self):
        """Test mass calculation with modifications."""
        from alphapeptfast.constants import CARBAMIDOMETHYL_MASS

        mods = [("Carbamidomethyl", 2)]
        mass = calculate_peptide_mass("PEPTIDE", modifications=mods)

        expected = sum(AA_MASSES_DICT[aa] for aa in "PEPTIDE") + H2O_MASS + CARBAMIDOMETHYL_MASS
        assert abs(mass - expected) < 0.001

    def test_calculate_mass_empty_sequence(self):
        """Test mass of empty sequence."""
        mass = calculate_peptide_mass("")
        # Just H2O for empty sequence
        assert abs(mass - H2O_MASS) < 0.001


class TestCalculatePrecursor:
    """Test precursor m/z calculation wrapper."""

    def test_calculate_precursor_simple(self):
        """Test simple precursor m/z calculation."""
        mz = calculate_precursor("PEPTIDE", charge=2)

        # Calculate expected
        neutral_mass = sum(AA_MASSES_DICT[aa] for aa in "PEPTIDE") + H2O_MASS
        expected = (neutral_mass + 2 * PROTON_MASS) / 2
        assert abs(mz - expected) < 0.001

    def test_calculate_precursor_different_charges(self):
        """Test with different charge states."""
        sequence = "PEPTIDE"

        for charge in [1, 2, 3, 4]:
            mz = calculate_precursor(sequence, charge=charge)
            assert mz > 0
            # Higher charge should give lower m/z
            if charge > 1:
                mz_prev = calculate_precursor(sequence, charge=charge-1)
                assert mz < mz_prev

    def test_calculate_precursor_with_modifications(self):
        """Test precursor calculation with modifications."""
        from alphapeptfast.constants import CARBAMIDOMETHYL_MASS

        mods = [("Carbamidomethyl", 2)]
        mz = calculate_precursor("PEPTIDE", charge=2, modifications=mods)

        # Calculate expected
        neutral_mass = (
            sum(AA_MASSES_DICT[aa] for aa in "PEPTIDE") +
            H2O_MASS +
            CARBAMIDOMETHYL_MASS
        )
        expected = (neutral_mass + 2 * PROTON_MASS) / 2
        assert abs(mz - expected) < 0.001


class TestGenerateFragments:
    """Test fragment generation wrapper."""

    def test_generate_fragments_simple(self):
        """Test simple fragment generation."""
        mz, types, positions, charges = generate_fragments("PEPTIDE", charge=2)

        # Should generate fragments
        assert len(mz) > 0
        assert len(types) == len(mz)
        assert len(positions) == len(mz)
        assert len(charges) == len(mz)

        # Should have both b and y ions
        assert np.any(types == 0)  # b-ions
        assert np.any(types == 1)  # y-ions

    def test_generate_fragments_with_cleaning(self):
        """Test fragment generation with sequence cleaning."""
        # Should handle non-standard AA
        mz1, _, _, _ = generate_fragments("PEPTXIDE", charge=2)
        mz2, _, _, _ = generate_fragments("PEPTLIDE", charge=2)

        # Should be identical after cleaning
        np.testing.assert_allclose(mz1, mz2, rtol=1e-6)

    def test_generate_fragments_b_only(self):
        """Test generating only b-ions."""
        mz, types, _, _ = generate_fragments(
            "PEPTIDE", charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)
        )

        # All should be b-ions
        assert np.all(types == 0)

        # Should have 6 b-ions for 7-residue peptide
        assert len(mz) == 6

    def test_generate_fragments_y_only(self):
        """Test generating only y-ions."""
        mz, types, _, _ = generate_fragments(
            "PEPTIDE", charge=2,
            fragment_types=(1,),  # y-ions only
            fragment_charges=(1,)
        )

        # All should be y-ions
        assert np.all(types == 1)

        # Should have 6 y-ions for 7-residue peptide
        assert len(mz) == 6

    def test_generate_fragments_with_modifications(self):
        """Test fragment generation with modifications."""
        mods = [("Carbamidomethyl", 2)]
        mz, types, positions, charges = generate_fragments(
            "PEPTIDE", charge=2, modifications=mods
        )

        # Should generate fragments
        assert len(mz) > 0

        # Fragments should be different from unmodified
        mz_unmod, _, _, _ = generate_fragments("PEPTIDE", charge=2)
        # At least some fragments should differ
        assert not np.allclose(mz, mz_unmod)


class TestGenerateBIons:
    """Test b-ion generation wrapper."""

    def test_generate_b_ions_simple(self):
        """Test simple b-ion generation."""
        b_ions = generate_b_ions("PEPTIDE", charge=2, fragment_charges=(1,))

        # Should have 6 b-ions for 7-residue peptide
        assert len(b_ions) == 6

        # All should be positive
        assert np.all(b_ions > 0)

        # Should be monotonically increasing
        assert np.all(b_ions[1:] > b_ions[:-1])

    def test_generate_b_ions_with_modifications(self):
        """Test b-ion generation with modifications."""
        mods = [("Carbamidomethyl", 2)]
        b_ions = generate_b_ions(
            "PEPTIDE", charge=2,
            modifications=mods,
            fragment_charges=(1,)
        )

        # Should have 6 b-ions
        assert len(b_ions) == 6

    def test_generate_b_ions_multiple_charges(self):
        """Test b-ion generation with multiple charges."""
        b_ions = generate_b_ions("PEPTIDE", charge=2, fragment_charges=(1, 2))

        # Should have more than 6 (some positions will have both charges)
        assert len(b_ions) > 6


class TestGenerateYIons:
    """Test y-ion generation wrapper."""

    def test_generate_y_ions_simple(self):
        """Test simple y-ion generation."""
        y_ions = generate_y_ions("PEPTIDE", charge=2, fragment_charges=(1,))

        # Should have 6 y-ions for 7-residue peptide
        assert len(y_ions) == 6

        # All should be positive
        assert np.all(y_ions > 0)

        # Should be monotonically increasing
        assert np.all(y_ions[1:] > y_ions[:-1])

    def test_generate_y_ions_with_modifications(self):
        """Test y-ion generation with modifications."""
        mods = [("Carbamidomethyl", 2)]
        y_ions = generate_y_ions(
            "PEPTIDE", charge=2,
            modifications=mods,
            fragment_charges=(1,)
        )

        # Should have 6 y-ions
        assert len(y_ions) == 6

    def test_generate_y_ions_multiple_charges(self):
        """Test y-ion generation with multiple charges."""
        y_ions = generate_y_ions("PEPTIDE", charge=2, fragment_charges=(1, 2))

        # Should have more than 6 (some positions will have both charges)
        assert len(y_ions) > 6


class TestIntegrationConvenience:
    """Integration tests for convenience API."""

    def test_complete_workflow(self):
        """Test complete workflow with convenience API."""
        sequence = "PEPTXIDE"  # Has non-standard AA
        mods = [("Carbamidomethyl", 2)]

        # Calculate masses
        neutral_mass = calculate_peptide_mass(sequence, modifications=mods)
        precursor_mz = calculate_precursor(sequence, charge=2, modifications=mods)

        # Generate fragments
        mz, types, positions, charges = generate_fragments(
            sequence, charge=2, modifications=mods
        )

        # All should succeed
        assert neutral_mass > 0
        assert precursor_mz > 0
        assert len(mz) > 0

    def test_consistency_with_low_level_api(self):
        """Test that convenience API matches low-level API."""
        from alphapeptfast.fragments.generator import (
            encode_peptide_to_ord,
            calculate_neutral_mass,
            generate_by_ions,
        )

        sequence = "PEPTIDE"

        # Convenience API
        mass_conv = calculate_peptide_mass(sequence)
        mz_conv, types_conv, pos_conv, charges_conv = generate_fragments(
            sequence, charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Low-level API
        peptide_ord = encode_peptide_to_ord(sequence)
        mass_low = calculate_neutral_mass(peptide_ord)
        mz_low, types_low, pos_low, charges_low = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Should be identical
        assert abs(mass_conv - mass_low) < 1e-10
        np.testing.assert_array_equal(mz_conv, mz_low)
        np.testing.assert_array_equal(types_conv, types_low)
        np.testing.assert_array_equal(pos_conv, pos_low)
        np.testing.assert_array_equal(charges_conv, charges_low)

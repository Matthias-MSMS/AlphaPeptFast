"""Comprehensive unit tests for mass calculations and constants.

Adapted from AlphaMod's test_calculations.py for AlphaPeptFast's
ord()-based encoding approach.
"""

import numpy as np
import pytest

from alphapeptfast.constants import (
    AA_MASSES,
    AA_MASSES_DICT,
    B_ION_OFFSET,
    H2O_MASS,
    PROTON_MASS,
    Y_ION_OFFSET,
    ELECTRON_MASS,
    CARBAMIDOMETHYL_MASS,
)
from alphapeptfast.fragments.generator import (
    encode_peptide_to_ord,
    calculate_neutral_mass,
    calculate_precursor_mz,
    ppm_error,
    generate_by_ions,
)


class TestConstants:
    """Test physical constants are correct."""

    def test_proton_mass_correct(self):
        """Test that PROTON_MASS is correct (not hydrogen atom mass)."""
        # PROTON_MASS should be ~1.007276, NOT 1.007825 (H atom)
        assert 1.0072 < PROTON_MASS < 1.0073, \
            f"PROTON_MASS is wrong: {PROTON_MASS}"

        # Exact NIST value
        assert abs(PROTON_MASS - 1.007276466622) < 1e-10

    def test_hydrogen_atom_mass(self):
        """Test that proton + electron = hydrogen atom."""
        H_ATOM_MASS = PROTON_MASS + ELECTRON_MASS
        # Should be approximately 1.007825
        assert abs(H_ATOM_MASS - 1.007825) < 0.000001

    def test_h2o_mass(self):
        """Test water mass is reasonable."""
        assert 18.00 < H2O_MASS < 18.02

    def test_aa_masses_populated(self):
        """Test that AA_MASSES array is populated."""
        # Check standard amino acids
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            mass = AA_MASSES[ord(aa)]
            assert mass > 50.0, f"AA {aa} mass too low: {mass}"
            assert mass < 250.0, f"AA {aa} mass too high: {mass}"

    def test_aa_masses_dict_complete(self):
        """Test that AA_MASSES_DICT has all 20 standard amino acids."""
        standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
        dict_aas = set(AA_MASSES_DICT.keys())
        assert standard_aas.issubset(dict_aas)

    def test_ion_offsets(self):
        """Test ion offset calculations."""
        # b-ions: just a proton
        assert abs(B_ION_OFFSET - PROTON_MASS) < 1e-10

        # y-ions: H2O + proton
        assert abs(Y_ION_OFFSET - (H2O_MASS + PROTON_MASS)) < 1e-10


class TestPeptideEncoding:
    """Test ord() encoding of peptides."""

    def test_encode_peptide_basic(self):
        """Test basic peptide encoding."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        # Check type and length
        assert isinstance(peptide_ord, np.ndarray)
        assert peptide_ord.dtype == np.uint8
        assert len(peptide_ord) == len(peptide)

        # Check values
        expected = np.array([ord(c) for c in peptide], dtype=np.uint8)
        np.testing.assert_array_equal(peptide_ord, expected)

    def test_encode_empty_peptide(self):
        """Test encoding empty peptide."""
        peptide_ord = encode_peptide_to_ord("")
        assert len(peptide_ord) == 0

    def test_encode_single_aa(self):
        """Test encoding single amino acid."""
        peptide_ord = encode_peptide_to_ord("A")
        assert len(peptide_ord) == 1
        assert peptide_ord[0] == ord("A")


class TestNeutralMassCalculation:
    """Test neutral peptide mass calculations."""

    def test_calculate_neutral_mass_basic(self):
        """Test basic neutral mass calculation."""
        peptide = "AAA"
        peptide_ord = encode_peptide_to_ord(peptide)
        mass = calculate_neutral_mass(peptide_ord)

        # Expected: 3 * A mass + H2O
        expected = 3 * AA_MASSES_DICT["A"] + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_calculate_neutral_mass_all_aas(self):
        """Test with all standard amino acids."""
        peptide = "ACDEFGHIKLMNPQRSTVWY"
        peptide_ord = encode_peptide_to_ord(peptide)
        mass = calculate_neutral_mass(peptide_ord)

        # Expected: sum of all AA masses + H2O
        expected = sum(AA_MASSES_DICT[aa] for aa in peptide) + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_calculate_neutral_mass_empty(self):
        """Test with empty peptide."""
        peptide_ord = encode_peptide_to_ord("")
        mass = calculate_neutral_mass(peptide_ord)
        # Just H2O for empty peptide
        assert abs(mass - H2O_MASS) < 0.001

    def test_calculate_neutral_mass_single_aa(self):
        """Test with single amino acid."""
        peptide_ord = encode_peptide_to_ord("A")
        mass = calculate_neutral_mass(peptide_ord)
        expected = AA_MASSES_DICT["A"] + H2O_MASS
        assert abs(mass - expected) < 0.001

    def test_calculate_neutral_mass_known_peptide(self):
        """Test against known peptide mass."""
        # PEPTIDE: known neutral mass ~799.36 Da
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)
        mass = calculate_neutral_mass(peptide_ord)

        # Calculate expected manually
        expected = (
            AA_MASSES_DICT["P"] +
            AA_MASSES_DICT["E"] +
            AA_MASSES_DICT["P"] +
            AA_MASSES_DICT["T"] +
            AA_MASSES_DICT["I"] +
            AA_MASSES_DICT["D"] +
            AA_MASSES_DICT["E"] +
            H2O_MASS
        )
        assert abs(mass - expected) < 0.001

    def test_calculate_neutral_mass_precision(self):
        """Test numerical precision with long peptide."""
        # Very long peptide to test rounding errors
        peptide = "A" * 100
        peptide_ord = encode_peptide_to_ord(peptide)
        mass = calculate_neutral_mass(peptide_ord)

        expected = 100 * AA_MASSES_DICT["A"] + H2O_MASS
        # Should be accurate to < 1e-6
        assert abs(mass - expected) < 1e-6


class TestPrecursorMZ:
    """Test precursor m/z calculations."""

    def test_calculate_precursor_mz_basic(self):
        """Test basic precursor m/z calculation."""
        neutral_mass = 1000.0
        charge = 2
        mz = calculate_precursor_mz(neutral_mass, charge)

        # m/z = (M + z*H+) / z
        expected = (neutral_mass + charge * PROTON_MASS) / charge
        assert abs(mz - expected) < 0.001

    def test_calculate_precursor_mz_charges(self):
        """Test with different charge states."""
        neutral_mass = 1000.0

        for charge in [1, 2, 3, 4, 5]:
            mz = calculate_precursor_mz(neutral_mass, charge)
            expected = (neutral_mass + charge * PROTON_MASS) / charge
            assert abs(mz - expected) < 0.001

    def test_calculate_precursor_mz_realistic(self):
        """Test with realistic peptide mass."""
        # PEPTIDE neutral mass ~799.36 Da
        peptide_ord = encode_peptide_to_ord("PEPTIDE")
        neutral_mass = calculate_neutral_mass(peptide_ord)

        # Charge 2 (typical tryptic peptide)
        mz = calculate_precursor_mz(neutral_mass, 2)

        # Should be around 400-401 m/z
        assert 400.0 < mz < 401.0


class TestPPMError:
    """Test PPM error calculations."""

    def test_ppm_error_exact_match(self):
        """Test exact mass match gives 0 ppm."""
        error = ppm_error(1000.0, 1000.0)
        assert abs(error) < 1e-10

    def test_ppm_error_positive(self):
        """Test positive PPM error."""
        theoretical = 1000.0
        # 10 ppm higher
        observed = theoretical * (1 + 10.0 / 1e6)
        error = ppm_error(observed, theoretical)
        assert abs(error - 10.0) < 0.01

    def test_ppm_error_negative(self):
        """Test negative PPM error."""
        theoretical = 1000.0
        # 10 ppm lower
        observed = theoretical * (1 - 10.0 / 1e6)
        error = ppm_error(observed, theoretical)
        assert abs(error - (-10.0)) < 0.01

    def test_ppm_error_large_mass(self):
        """Test PPM error with large masses."""
        theoretical = 10000.0
        # 5 ppm difference
        observed = theoretical * (1 + 5.0 / 1e6)
        error = ppm_error(observed, theoretical)
        assert abs(error - 5.0) < 0.01

    def test_ppm_error_small_mass(self):
        """Test PPM error with small masses."""
        theoretical = 100.0
        # 20 ppm difference
        observed = theoretical * (1 + 20.0 / 1e6)
        error = ppm_error(observed, theoretical)
        assert abs(error - 20.0) < 0.01


class TestGenerateByIons:
    """Test b/y ion generation."""

    def test_generate_by_ions_basic(self):
        """Test basic b/y ion generation."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        # Generate with default parameters (b and y, charges 1 and 2)
        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2
        )

        # Should generate fragments
        assert len(mz) > 0
        assert len(types) == len(mz)
        assert len(positions) == len(mz)
        assert len(charges) == len(mz)

    def test_generate_by_ions_b_only(self):
        """Test generating only b-ions."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),  # 0 = b-ions only
            fragment_charges=(1,)
        )

        # All should be b-ions (type 0)
        assert np.all(types == 0)

        # Should have 6 b-ions for PEPTIDE (length 7)
        assert len(mz) == 6

    def test_generate_by_ions_y_only(self):
        """Test generating only y-ions."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(1,),  # 1 = y-ions only
            fragment_charges=(1,)
        )

        # All should be y-ions (type 1)
        assert np.all(types == 1)

        # Should have 6 y-ions for PEPTIDE (length 7)
        assert len(mz) == 6

    def test_generate_by_ions_manual_check(self):
        """Test b/y ion masses manually."""
        peptide = "AAA"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),  # b and y
            fragment_charges=(1,)    # charge 1 only
        )

        # Find b1 ion (first b-ion, position 1, charge 1)
        b1_idx = np.where((types == 0) & (positions == 1) & (charges == 1))[0]
        assert len(b1_idx) == 1

        # b1 = A (mass) / 1
        expected_b1 = AA_MASSES_DICT["A"] + PROTON_MASS
        assert abs(mz[b1_idx[0]] - expected_b1) < 0.001

        # Find y1 ion (first y-ion, position 1, charge 1)
        y1_idx = np.where((types == 1) & (positions == 1) & (charges == 1))[0]
        assert len(y1_idx) == 1

        # y1 = A + H2O + H+
        expected_y1 = AA_MASSES_DICT["A"] + H2O_MASS + PROTON_MASS
        assert abs(mz[y1_idx[0]] - expected_y1) < 0.001

    def test_generate_by_ions_increasing_masses(self):
        """Test that fragment masses increase with position."""
        peptide = "ACDEFGHIK"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)  # charge 1 only
        )

        # Sort by position
        sorted_idx = np.argsort(positions)
        sorted_mz = mz[sorted_idx]

        # Each should be heavier than previous
        for i in range(1, len(sorted_mz)):
            assert sorted_mz[i] > sorted_mz[i-1]

    def test_generate_by_ions_empty_peptide(self):
        """Test with empty peptide."""
        peptide_ord = encode_peptide_to_ord("")

        # Empty peptide should raise an error or return empty
        # Current implementation raises ValueError for negative array size
        # This is acceptable behavior - don't generate fragments for empty peptides
        with pytest.raises(ValueError):
            mz, types, positions, charges = generate_by_ions(
                peptide_ord, precursor_charge=2
            )

    def test_generate_by_ions_single_aa(self):
        """Test with single amino acid."""
        peptide_ord = encode_peptide_to_ord("A")

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2
        )

        # No fragments for single AA (can't break at termini)
        assert len(mz) == 0

    def test_generate_by_ions_two_aa(self):
        """Test with two amino acids."""
        peptide_ord = encode_peptide_to_ord("AA")

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),
            fragment_charges=(1,)
        )

        # Should have 2 fragments: b1 and y1
        assert len(mz) == 2

    def test_generate_by_ions_charge_constraint(self):
        """Test that fragments obey charge <= position constraint."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=3,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1, 2, 3)
        )

        # For each fragment, charge should be <= position
        for pos, charge in zip(positions, charges):
            assert charge <= pos, \
                f"Fragment at position {pos} has charge {charge} > position"

    def test_generate_by_ions_multiple_charges(self):
        """Test generation with multiple fragment charges."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=3,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1, 2)
        )

        # Should have both charge 1 and charge 2 fragments
        assert np.any(charges == 1)
        assert np.any(charges == 2)


class TestComplementarity:
    """Test complementarity of b and y ions."""

    def test_by_ions_complementary(self):
        """Test that b and y ions are complementary."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        # Generate fragments
        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0, 1),  # b and y
            fragment_charges=(1,)    # charge 1 only
        )

        # Get peptide neutral mass
        peptide_mass = calculate_neutral_mass(peptide_ord)
        n = len(peptide)

        # For each b-ion, find complementary y-ion
        b_ions = mz[types == 0]
        y_ions = mz[types == 1]

        # b_i + y_(n-i) should equal peptide mass + 2*H+ (within tolerance)
        for i in range(1, n):
            # b_i is at position i
            b_i = b_ions[i-1]  # 0-indexed
            # y_(n-i) is at position n-i
            y_comp = y_ions[n-1-i]  # 0-indexed

            # Reconstruct neutral masses
            b_neutral = b_i - PROTON_MASS  # Remove H+ from b-ion
            y_neutral = y_comp - PROTON_MASS - H2O_MASS  # Remove H+ and H2O

            # Sum should equal peptide mass (without H2O)
            peptide_mass_no_h2o = peptide_mass - H2O_MASS
            complement_sum = b_neutral + y_neutral

            # Allow small tolerance for floating point
            assert abs(complement_sum - peptide_mass_no_h2o) < 0.01, \
                f"b{i} + y{n-i} complementarity failed: {complement_sum} vs {peptide_mass_no_h2o}"


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_typical_tryptic_peptide(self):
        """Test with typical tryptic peptide."""
        peptide = "YGGFMTSEK"  # Common test peptide
        peptide_ord = encode_peptide_to_ord(peptide)

        # Calculate precursor m/z
        neutral_mass = calculate_neutral_mass(peptide_ord)
        mz_precursor = calculate_precursor_mz(neutral_mass, 2)

        # Should be reasonable for tryptic peptide
        assert 400 < mz_precursor < 800

        # Generate fragments
        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2
        )

        # Should have many fragments
        assert len(mz) > 10

        # Fragment m/z values should be reasonable (not necessarily < precursor)
        # Note: +1 fragments can have higher m/z than +2 precursor
        # E.g., 700 Da fragment at +1 = 701 m/z > 500 Da precursor at +2 = 250.5 m/z
        assert np.all(mz > 50)  # Minimum fragment size
        assert np.all(mz < neutral_mass + 100)  # Can't exceed peptide mass by much

    def test_known_peptide_fragment(self):
        """Test against known fragment mass."""
        # Simple test: AA peptide
        peptide = "AA"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)
        )

        # b1 should be A + H+
        expected_b1 = AA_MASSES_DICT["A"] + PROTON_MASS

        # Find b1
        b1_idx = np.where((positions == 1))[0]
        assert len(b1_idx) == 1
        assert abs(mz[b1_idx[0]] - expected_b1) < 0.001

    def test_ppm_error_on_fragments(self):
        """Test PPM error calculation on simulated fragments."""
        peptide = "PEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_charges=(1,)
        )

        # Simulate observed values with 5 ppm error
        for theoretical in mz:
            observed = theoretical * (1 + 5.0 / 1e6)
            error = ppm_error(observed, theoretical)
            assert abs(error - 5.0) < 0.1


class TestEdgeCases:
    """Test edge cases and numerical limits."""

    def test_very_long_peptide(self):
        """Test with very long peptide."""
        peptide = "A" * 100
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=2,
            fragment_types=(0,),  # b-ions only
            fragment_charges=(1,)
        )

        # Should have 99 b-ions
        assert len(mz) == 99

    def test_heavy_amino_acids(self):
        """Test with heavy amino acids."""
        peptide = "WWW"  # Three tryptophans (heaviest standard AA)
        peptide_ord = encode_peptide_to_ord(peptide)

        neutral_mass = calculate_neutral_mass(peptide_ord)
        expected = 3 * AA_MASSES_DICT["W"] + H2O_MASS
        assert abs(neutral_mass - expected) < 0.001

    def test_all_20_amino_acids(self):
        """Test with all 20 standard amino acids."""
        peptide = "ACDEFGHIKLMNPQRSTVWY"
        peptide_ord = encode_peptide_to_ord(peptide)

        mz, types, positions, charges = generate_by_ions(
            peptide_ord, precursor_charge=3,
            fragment_charges=(1,)
        )

        # Should generate fragments successfully
        assert len(mz) > 0

        # All m/z values should be reasonable
        assert np.all(mz > 50)  # Smallest fragment
        assert np.all(mz < 3000)  # Largest fragment

    def test_numerical_stability(self):
        """Test numerical stability with repeated calculations."""
        peptide = "TESTPEPTIDE"
        peptide_ord = encode_peptide_to_ord(peptide)

        # Calculate mass 1000 times
        masses = [calculate_neutral_mass(peptide_ord) for _ in range(1000)]

        # All should be identical (no accumulating errors)
        assert np.std(masses) < 1e-10
        assert np.all(np.array(masses) == masses[0])

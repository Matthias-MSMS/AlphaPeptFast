"""Tests for charge state consolidation module.

Tests:
- PPM-based neutral mass matching (CORRECTED from Da)
- Charge state pairing (z=2, z=3)
- Parameter learning from ground truth
- Instrument-specific presets
- ConsolidatedFeature creation
- Edge cases and validation
"""

import unittest
import numpy as np

from alphapeptfast.features.isotope_grouping import IsotopeGroup, InstrumentType
from alphapeptfast.features.charge_consolidation import (
    ChargeConsolidationParams,
    ConsolidatedFeature,
    find_charge_state_pairs,
    consolidate_features,
    calculate_neutral_mass_scalar,
    binary_search_mass_range_ppm,
)
from alphapeptfast.constants import PROTON_MASS


class TestChargeConsolidationParams(unittest.TestCase):
    """Test parameter presets and learning."""

    def test_orbitrap_preset(self):
        """Test Orbitrap parameter preset uses PPM (not Da)."""
        params = ChargeConsolidationParams.for_instrument(InstrumentType.ORBITRAP)

        # CORRECTED: Should be ppm-based, not Da
        self.assertEqual(params.mass_tolerance_ppm, 3.0)
        self.assertEqual(params.rt_tolerance_sec, 6.0)

    def test_mrtof_preset(self):
        """Test MR-TOF parameter preset."""
        params = ChargeConsolidationParams.for_instrument(InstrumentType.MR_TOF)

        # Ultra-tight for TOF
        self.assertEqual(params.mass_tolerance_ppm, 1.2)
        self.assertEqual(params.rt_tolerance_sec, 5.0)

    def test_parameter_learning_basic(self):
        """Test parameter learning from synthetic ground truth."""
        # Create synthetic data: 100 peptides, each at z=2 and z=3
        n_peptides = 100
        np.random.seed(42)

        neutral_masses = np.random.uniform(1000, 3000, n_peptides)

        # Create z=2 and z=3 features with small errors
        features_mz = []
        features_charge = []
        features_sequence = []
        features_score = []
        features_rt = []

        for i, neutral_mass in enumerate(neutral_masses):
            seq = f"PEPTIDE{i}"
            rt = np.random.uniform(100, 1000)

            # z=2 with small mass error (~1 ppm)
            mass_z2 = neutral_mass * (1 + np.random.uniform(-1e-6, 1e-6))
            mz_z2 = (mass_z2 + 2 * PROTON_MASS) / 2
            features_mz.append(mz_z2)
            features_charge.append(2)
            features_sequence.append(seq)
            features_score.append(0.95)
            features_rt.append(rt + np.random.uniform(-2, 2))  # Small RT diff

            # z=3 with small mass error
            mass_z3 = neutral_mass * (1 + np.random.uniform(-1e-6, 1e-6))
            mz_z3 = (mass_z3 + 3 * PROTON_MASS) / 3
            features_mz.append(mz_z3)
            features_charge.append(3)
            features_sequence.append(seq)
            features_score.append(0.95)
            features_rt.append(rt + np.random.uniform(-2, 2))

        # Learn parameters
        params = ChargeConsolidationParams.learn_from_ground_truth(
            features_mz=np.array(features_mz),
            features_charge=np.array(features_charge),
            features_sequence=features_sequence,
            features_score=np.array(features_score),
            features_rt=np.array(features_rt),
            score_threshold=0.8,
            percentile=95.0
        )

        # Should learn tight tolerance (data is clean)
        self.assertLess(params.mass_tolerance_ppm, 5.0)
        self.assertLess(params.rt_tolerance_sec, 10.0)

    def test_parameter_learning_insufficient_data(self):
        """Test fallback to defaults with insufficient data."""
        # Only 10 features (too few)
        params = ChargeConsolidationParams.learn_from_ground_truth(
            features_mz=np.random.uniform(400, 1000, 10),
            features_charge=np.full(10, 2),
            features_sequence=["PEP"] * 10,
            features_score=np.full(10, 0.5),  # Low scores
            features_rt=np.random.uniform(100, 1000, 10),
            score_threshold=0.8
        )

        # Should use defaults
        self.assertGreater(params.mass_tolerance_ppm, 0)


class TestNeutralMassCalculation(unittest.TestCase):
    """Test neutral mass calculation."""

    def test_neutral_mass_z2(self):
        """Test neutral mass calculation for z=2."""
        mz = 650.0
        charge = 2

        neutral_mass = calculate_neutral_mass_scalar(mz, charge)

        # M = m/z × z - z × proton_mass
        expected = 650.0 * 2 - 2 * PROTON_MASS
        self.assertAlmostEqual(neutral_mass, expected, places=4)

    def test_neutral_mass_z3(self):
        """Test neutral mass calculation for z=3."""
        mz = 433.7
        charge = 3

        neutral_mass = calculate_neutral_mass_scalar(mz, charge)

        expected = 433.7 * 3 - 3 * PROTON_MASS
        self.assertAlmostEqual(neutral_mass, expected, places=4)


class TestBinarySearchPPM(unittest.TestCase):
    """Test PPM-based binary search (CORRECTED from Da)."""

    def test_binary_search_finds_matches(self):
        """Test that binary search finds features within PPM tolerance."""
        # Sorted neutral masses
        neutral_masses = np.array([1000.0, 1010.0, 1020.0, 1030.0, 1040.0])

        target_mass = 1020.0
        tolerance_ppm = 5000.0  # 5000 ppm = 0.5% (wide tolerance for testing)

        left, right = binary_search_mass_range_ppm(neutral_masses, target_mass, tolerance_ppm)

        # At 1020 Da, 5000 ppm = 5.1 Da window
        # Should find 1020 ± 5.1 Da → indices 1, 2, 3 (1010, 1020, 1030)
        matches = neutral_masses[left:right]

        self.assertIn(1010.0, matches)
        self.assertIn(1020.0, matches)
        self.assertIn(1030.0, matches)
        self.assertNotIn(1000.0, matches)
        self.assertNotIn(1040.0, matches)

    def test_binary_search_tight_tolerance(self):
        """Test tight tolerance correctly limits matches."""
        neutral_masses = np.array([1000.0, 1000.001, 1000.002, 1000.01, 1000.1])

        target_mass = 1000.0
        tolerance_ppm = 1.0  # 1 ppm at 1000 Da = 0.001 Da

        left, right = binary_search_mass_range_ppm(neutral_masses, target_mass, tolerance_ppm)

        matches = neutral_masses[left:right]

        # Should only find 1000.0 and 1000.001 (within 1 ppm)
        self.assertEqual(len(matches), 2)
        self.assertIn(1000.0, matches)
        self.assertIn(1000.001, matches)


class TestChargeStatePairing(unittest.TestCase):
    """Test charge state pair finding."""

    def create_isotope_group(self, mz, charge, rt, intensity=1e6):
        """Helper to create IsotopeGroup."""
        return IsotopeGroup(
            m0_idx=0,
            m0_mz=mz,
            m0_rt=rt,
            m0_intensity=intensity,
            charge=charge,
            has_m1=True,
            has_m2=False
        )

    def test_find_charge_pairs_basic(self):
        """Test finding z=2/z=3 pairs of same peptide."""
        # Same peptide at z=2 and z=3
        neutral_mass = 1300.0  # Da

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        groups = [
            self.create_isotope_group(mz_z2, charge=2, rt=100.0),
            self.create_isotope_group(mz_z3, charge=3, rt=102.0),  # 2 sec RT diff
        ]

        params = ChargeConsolidationParams(
            mass_tolerance_ppm=5.0,
            rt_tolerance_sec=5.0,
            min_intensity=1e4
        )

        pairs = find_charge_state_pairs(groups, params)

        # Should find one pair
        self.assertEqual(len(pairs), 1)
        z2_idx, z3_idx, mass_error_ppm, rt_diff = pairs[0]

        self.assertEqual(z2_idx, 0)
        self.assertEqual(z3_idx, 1)
        self.assertLess(abs(mass_error_ppm), 1.0)  # Small error
        self.assertAlmostEqual(rt_diff, 2.0, places=1)

    def test_no_pairs_different_peptides(self):
        """Test that different peptides don't pair."""
        # Two different peptides
        mz_peptide1_z2 = (1000.0 + 2 * PROTON_MASS) / 2
        mz_peptide2_z3 = (2000.0 + 3 * PROTON_MASS) / 3

        groups = [
            self.create_isotope_group(mz_peptide1_z2, charge=2, rt=100.0),
            self.create_isotope_group(mz_peptide2_z3, charge=3, rt=100.0),
        ]

        params = ChargeConsolidationParams(mass_tolerance_ppm=5.0)
        pairs = find_charge_state_pairs(groups, params)

        # Should find no pairs (masses don't match)
        self.assertEqual(len(pairs), 0)

    def test_rt_filtering(self):
        """Test that RT tolerance filters non-coeluting features."""
        neutral_mass = 1300.0

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        groups = [
            self.create_isotope_group(mz_z2, charge=2, rt=100.0),
            self.create_isotope_group(mz_z3, charge=3, rt=200.0),  # 100 sec away
        ]

        params = ChargeConsolidationParams(
            mass_tolerance_ppm=5.0,
            rt_tolerance_sec=10.0  # Tight RT tolerance
        )

        pairs = find_charge_state_pairs(groups, params)

        # Should find no pairs (RT too different)
        self.assertEqual(len(pairs), 0)

    def test_intensity_filtering(self):
        """Test that low-intensity features are filtered."""
        neutral_mass = 1300.0

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        groups = [
            self.create_isotope_group(mz_z2, charge=2, rt=100.0, intensity=1e3),  # Too low
            self.create_isotope_group(mz_z3, charge=3, rt=100.0, intensity=1e6),
        ]

        params = ChargeConsolidationParams(
            mass_tolerance_ppm=5.0,
            min_intensity=1e4  # Higher than z=2 intensity
        )

        pairs = find_charge_state_pairs(groups, params)

        # Should find no pairs (z=2 filtered by intensity)
        self.assertEqual(len(pairs), 0)


class TestConsolidation(unittest.TestCase):
    """Test feature consolidation."""

    def create_isotope_group(self, mz, charge, rt, intensity=1e6):
        """Helper to create IsotopeGroup."""
        return IsotopeGroup(
            m0_idx=0,
            m0_mz=mz,
            m0_rt=rt,
            m0_intensity=intensity,
            charge=charge,
            has_m1=True
        )

    def test_consolidate_simple_pair(self):
        """Test consolidating a z=2/z=3 pair."""
        neutral_mass = 1300.0

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        groups = [
            self.create_isotope_group(mz_z2, charge=2, rt=100.0, intensity=1e7),
            self.create_isotope_group(mz_z3, charge=3, rt=100.0, intensity=5e6),
        ]

        params = ChargeConsolidationParams(mass_tolerance_ppm=5.0)
        consolidated = consolidate_features(groups, params)

        # Should create one consolidated feature
        self.assertEqual(len(consolidated), 1)

        feature = consolidated[0]
        self.assertEqual(len(feature.charge_states), 2)
        self.assertIn(2, feature.charge_states)
        self.assertIn(3, feature.charge_states)

        # Total intensity should be sum
        expected_total = 1e7 + 5e6
        self.assertAlmostEqual(feature.total_intensity, expected_total, places=-4)

        # Best charge should be z=2 (higher intensity)
        self.assertEqual(feature.best_charge, 2)

    def test_consolidate_isolated_features(self):
        """Test that isolated features remain separate."""
        # Two peptides with no shared charge states
        mz_peptide1_z2 = (1000.0 + 2 * PROTON_MASS) / 2
        mz_peptide2_z2 = (2000.0 + 2 * PROTON_MASS) / 2

        groups = [
            self.create_isotope_group(mz_peptide1_z2, charge=2, rt=100.0),
            self.create_isotope_group(mz_peptide2_z2, charge=2, rt=200.0),
        ]

        params = ChargeConsolidationParams(mass_tolerance_ppm=5.0)
        consolidated = consolidate_features(groups, params)

        # Should create two separate features
        self.assertEqual(len(consolidated), 2)

        # Each should have only one charge state
        for feature in consolidated:
            self.assertEqual(len(feature.charge_states), 1)


class TestCorrectionFromDaToPPM(unittest.TestCase):
    """Critical test: Verify Da tolerances were corrected to PPM."""

    def test_tolerance_is_ppm_not_da(self):
        """CRITICAL: Verify tolerances are PPM-based, not Da.

        The old MSC_MS1_high_res code had mass_tolerance_da=0.05,
        which is 50 ppm at 1000 Da - way too loose!

        This test verifies the corrected alphapeptfast code uses ppm.
        """
        params = ChargeConsolidationParams()

        # Should be ppm, not Da
        self.assertIsInstance(params.mass_tolerance_ppm, float)
        self.assertGreater(params.mass_tolerance_ppm, 0)

        # Should be reasonable ppm values (not large Da values)
        self.assertLess(params.mass_tolerance_ppm, 10.0)  # Typical: 1-5 ppm

        # Verify binary search uses ppm correctly
        neutral_masses = np.array([1000.0, 1001.0, 1002.0])
        target = 1000.0

        # At 1000 Da, 3 ppm = 0.003 Da
        left, right = binary_search_mass_range_ppm(neutral_masses, target, 3.0)

        matches = neutral_masses[left:right]

        # Should only find 1000.0 (3 ppm = 0.003 Da, so 1001.0 is 1000 ppm away)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0], 1000.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_groups(self):
        """Test with no isotope groups."""
        params = ChargeConsolidationParams()
        pairs = find_charge_state_pairs([], params)

        self.assertEqual(len(pairs), 0)

    def test_single_charge_state(self):
        """Test with features all at same charge state."""
        def create_group(mz):
            return IsotopeGroup(
                m0_idx=0, m0_mz=mz, m0_rt=100.0,
                m0_intensity=1e6, charge=2
            )

        groups = [create_group(650.0), create_group(700.0), create_group(750.0)]

        params = ChargeConsolidationParams()
        pairs = find_charge_state_pairs(groups, params)

        # Should find no pairs (all z=2)
        self.assertEqual(len(pairs), 0)


if __name__ == '__main__':
    unittest.main()

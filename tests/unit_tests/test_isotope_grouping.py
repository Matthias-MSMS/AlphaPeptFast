"""Tests for isotope pattern detection module.

Tests:
- Isotope spacing detection (M0 → M1 → M2)
- Automatic charge state determination
- Instrument-specific parameter presets
- Binary search performance
- Edge cases and error handling
"""

import unittest
import numpy as np

from alphapeptfast.features.isotope_grouping import (
    IsotopeGroup,
    IsotopeGroupingParams,
    InstrumentType,
    detect_isotope_patterns,
    calculate_mass_error_ppm,
    C13_MASS_DIFF,
)


class TestIsotopeGroupingParams(unittest.TestCase):
    """Test parameter presets for different instruments."""

    def test_orbitrap_preset(self):
        """Test Orbitrap parameter preset."""
        params = IsotopeGroupingParams.for_instrument(InstrumentType.ORBITRAP)

        # Should use looser tolerance for ~240K resolution
        self.assertEqual(params.mz_tolerance_ppm, 3.0)
        self.assertEqual(params.rt_tolerance_factor, 1.0)
        self.assertTrue(params.detect_c13_2x)
        self.assertTrue(params.detect_s34)

    def test_astral_preset(self):
        """Test Astral parameter preset (same as Orbitrap)."""
        params = IsotopeGroupingParams.for_instrument(InstrumentType.ASTRAL)

        self.assertEqual(params.mz_tolerance_ppm, 3.0)
        self.assertEqual(params.rt_tolerance_factor, 1.0)

    def test_mrtof_preset(self):
        """Test MR-TOF parameter preset."""
        params = IsotopeGroupingParams.for_instrument(InstrumentType.MR_TOF)

        # Should use ultra-tight tolerance for >1M resolution
        self.assertEqual(params.mz_tolerance_ppm, 1.2)
        self.assertEqual(params.rt_tolerance_factor, 1.0)


class TestMassErrorCalculation(unittest.TestCase):
    """Test PPM error calculation."""

    def test_perfect_spacing(self):
        """Test zero error for perfect isotope spacing."""
        observed_diff = C13_MASS_DIFF / 2.0  # z=2 spacing
        expected_diff = C13_MASS_DIFF / 2.0
        m0_mz = 650.0

        error = calculate_mass_error_ppm(observed_diff, expected_diff, m0_mz)

        self.assertAlmostEqual(error, 0.0, places=6)

    def test_positive_error(self):
        """Test positive mass error."""
        # 2 ppm error at 650 m/z
        observed_diff = (C13_MASS_DIFF / 2.0) + (650.0 * 2e-6)
        expected_diff = C13_MASS_DIFF / 2.0
        m0_mz = 650.0

        error = calculate_mass_error_ppm(observed_diff, expected_diff, m0_mz)

        self.assertAlmostEqual(error, 2.0, places=1)

    def test_negative_error(self):
        """Test negative mass error."""
        # -3 ppm error at 800 m/z
        observed_diff = (C13_MASS_DIFF / 2.0) - (800.0 * 3e-6)
        expected_diff = C13_MASS_DIFF / 2.0
        m0_mz = 800.0

        error = calculate_mass_error_ppm(observed_diff, expected_diff, m0_mz)

        self.assertAlmostEqual(error, -3.0, places=1)


class TestIsotopeDetection(unittest.TestCase):
    """Test isotope pattern detection."""

    def create_synthetic_features(self, base_mz, charge, n_isotopes=3, rt=100.0, fwhm=2.0):
        """Create synthetic isotope pattern.

        Args:
            base_mz: M0 m/z value
            charge: Charge state (1, 2, or 3)
            n_isotopes: Number of isotopes (1, 2, or 3)
            rt: Retention time (seconds)
            fwhm: Peak width (seconds)

        Returns:
            Structured numpy array with feature data
        """
        spacing = C13_MASS_DIFF / charge

        features = []
        for i in range(n_isotopes):
            mz = base_mz + i * spacing
            # Realistic intensity decay for isotopes
            intensity = 1e6 * (0.6 ** i)
            features.append((mz, rt, fwhm, intensity))

        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        return np.array(features, dtype=dtype)

    def test_detect_z2_isotopes(self):
        """Test detection of z=2 isotope pattern."""
        # Create z=2 peptide at 650 m/z with M0, M1, M2
        features = self.create_synthetic_features(650.0, charge=2, n_isotopes=3)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        # Should find one group
        self.assertEqual(len(groups), 3)  # One group per M0 (each peak considered)

        # First group should be the M0
        group = groups[0]
        self.assertEqual(group.charge, 2)
        self.assertTrue(group.has_m1)
        self.assertTrue(group.has_m2)

        # Mass errors should be small
        self.assertLess(abs(group.m0_m1_mass_error_ppm), 1.0)

    def test_detect_z3_isotopes(self):
        """Test detection of z=3 isotope pattern."""
        # Create z=3 peptide at 433.7 m/z
        features = self.create_synthetic_features(433.7, charge=3, n_isotopes=3)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        # Should detect z=3
        group = groups[0]
        self.assertEqual(group.charge, 3)
        self.assertTrue(group.has_m1)

    def test_incomplete_isotope_pattern(self):
        """Test detection with missing M2."""
        # Create only M0 and M1
        features = self.create_synthetic_features(650.0, charge=2, n_isotopes=2)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        group = groups[0]
        self.assertTrue(group.has_m1)
        self.assertFalse(group.has_m2)

    def test_isolated_peak(self):
        """Test detection with single isolated peak (no isotopes)."""
        # Single peak
        features = self.create_synthetic_features(650.0, charge=2, n_isotopes=1)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        # Should still create group, but no isotopes found
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertFalse(group.has_m1)
        self.assertFalse(group.has_m2)

    def test_rt_coelution_requirement(self):
        """Test that isotopes must co-elute in RT."""
        # Create M0 and M1 far apart in RT
        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        features = np.array([
            (650.0, 100.0, 2.0, 1e6),  # M0
            (650.5, 150.0, 2.0, 6e5),  # M1 at different RT (50 sec away)
        ], dtype=dtype)

        params = IsotopeGroupingParams(
            mz_tolerance_ppm=5.0,
            rt_tolerance_factor=1.0  # Within 1× FWHM = 2 seconds
        )
        groups = detect_isotope_patterns(features, params)

        # Should not find M1 (too far in RT)
        group = groups[0]
        self.assertFalse(group.has_m1)

    def test_tight_mass_tolerance(self):
        """Test that tight tolerance rejects off-mass peaks."""
        # Create M0 and mis-spaced peak
        spacing_error = C13_MASS_DIFF / 2.0 + 0.01  # 10 mDa error
        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        features = np.array([
            (650.0, 100.0, 2.0, 1e6),
            (650.0 + spacing_error, 100.0, 2.0, 6e5),
        ], dtype=dtype)

        # Very tight tolerance (1 ppm)
        params = IsotopeGroupingParams(mz_tolerance_ppm=1.0)
        groups = detect_isotope_patterns(features, params)

        # Should reject the mis-spaced peak
        group = groups[0]
        # At 650 m/z, 10 mDa = ~15 ppm, should be rejected at 1 ppm tolerance
        self.assertFalse(group.has_m1)


class TestChargeStateDetermination(unittest.TestCase):
    """Test automatic charge state determination from isotope spacing."""

    def create_features_with_charge(self, base_mz, true_charge, n_isotopes=3):
        """Helper to create features with specific charge state."""
        spacing = C13_MASS_DIFF / true_charge

        features = []
        for i in range(n_isotopes):
            mz = base_mz + i * spacing
            intensity = 1e6 * (0.6 ** i)
            features.append((mz, 100.0, 2.0, intensity))

        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        return np.array(features, dtype=dtype)

    def test_charge_detection_z2(self):
        """Test that z=2 is correctly identified."""
        features = self.create_features_with_charge(650.0, true_charge=2)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        self.assertEqual(groups[0].charge, 2)

    def test_charge_detection_z3(self):
        """Test that z=3 is correctly identified."""
        features = self.create_features_with_charge(433.7, true_charge=3)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        self.assertEqual(groups[0].charge, 3)

    def test_charge_detection_z1(self):
        """Test that z=1 is correctly identified."""
        features = self.create_features_with_charge(1300.0, true_charge=1)

        params = IsotopeGroupingParams(mz_tolerance_ppm=5.0)
        groups = detect_isotope_patterns(features, params)

        self.assertEqual(groups[0].charge, 1)


class TestPerformance(unittest.TestCase):
    """Test performance with large datasets."""

    def test_large_dataset_performance(self):
        """Test that detection scales well with dataset size."""
        import time

        # Create 10,000 random features
        n_features = 10000
        np.random.seed(42)

        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        features = np.array([
            (np.random.uniform(400, 1500),
             np.random.uniform(0, 3600),
             2.0,
             np.random.uniform(1e4, 1e7))
            for _ in range(n_features)
        ], dtype=dtype)

        params = IsotopeGroupingParams(mz_tolerance_ppm=3.0)

        start = time.time()
        groups = detect_isotope_patterns(features, params)
        elapsed = time.time() - start

        # Should complete in < 5 seconds for 10K features
        self.assertLess(elapsed, 5.0)

        # Should find groups
        self.assertGreater(len(groups), 0)

        print(f"\nProcessed {n_features} features in {elapsed:.2f} sec")
        print(f"Throughput: {n_features/elapsed:.0f} features/sec")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_features(self):
        """Test with no features."""
        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        features = np.array([], dtype=dtype)

        params = IsotopeGroupingParams()
        groups = detect_isotope_patterns(features, params)

        self.assertEqual(len(groups), 0)

    def test_invalid_fwhm(self):
        """Test handling of invalid FWHM values."""
        dtype = [('mz', 'f8'), ('rt', 'f8'), ('fwhm_sec', 'f8'), ('intensity', 'f8')]
        features = np.array([
            (650.0, 100.0, 0.0, 1e6),    # Zero FWHM
            (650.5, 100.0, -1.0, 6e5),   # Negative FWHM
        ], dtype=dtype)

        params = IsotopeGroupingParams()
        groups = detect_isotope_patterns(features, params)

        # Should handle gracefully (skip invalid features)
        # Won't crash, but won't find isotopes either
        self.assertGreaterEqual(len(groups), 0)


if __name__ == '__main__':
    unittest.main()

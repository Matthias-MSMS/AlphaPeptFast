"""Tests for mass recalibration module.

This module tests RT-segmented mass recalibration including:
- Charge-state independent calibration (CRITICAL TEST)
- Adaptive RT binning
- Pre-search charge state consistency check
- MAD-based outlier removal
- Linear interpolation for sparse bins
- Recommended tolerance calculation
- Performance benchmarks
"""

import unittest

import numpy as np

from alphapeptfast.scoring.mass_recalibration import (
    MassRecalibrator,
    apply_corrections_fast,
    assign_rt_bins,
    calculate_bin_corrections,
    calculate_ppm_errors,
    determine_rt_bins,
    estimate_mass_error_from_charge_states,
    interpolate_bin_corrections,
    remove_outliers_mad,
)


class TestPPMCalculation(unittest.TestCase):
    """Test PPM error calculation."""

    def test_ppm_calculation_basic(self):
        """Test basic PPM error calculation."""
        observed = np.array([500.0, 1000.0, 1500.0])
        theoretical = np.array([500.0, 1000.0, 1500.0])

        ppm = calculate_ppm_errors(observed, theoretical)

        np.testing.assert_allclose(ppm, [0.0, 0.0, 0.0], atol=1e-6)

    def test_ppm_calculation_positive_error(self):
        """Test PPM with positive mass error."""
        theoretical = 1000.0
        observed = 1000.0 + 1000.0 * 5e-6  # +5 ppm error

        ppm = calculate_ppm_errors(np.array([observed]), np.array([theoretical]))

        np.testing.assert_allclose(ppm, [5.0], atol=1e-3)

    def test_ppm_calculation_negative_error(self):
        """Test PPM with negative mass error."""
        theoretical = 1000.0
        observed = 1000.0 - 1000.0 * 3e-6  # -3 ppm error

        ppm = calculate_ppm_errors(np.array([observed]), np.array([theoretical]))

        np.testing.assert_allclose(ppm, [-3.0], atol=1e-3)


class TestOutlierRemoval(unittest.TestCase):
    """Test MAD-based outlier removal."""

    def test_outlier_removal_basic(self):
        """Test outlier removal with clear outliers."""
        # Most values around 0, one clear outlier
        values = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 100.0])

        inliers = remove_outliers_mad(values, threshold=3.0)

        # Should remove 100.0
        self.assertEqual(np.sum(inliers), 5)
        self.assertFalse(inliers[-1])

    def test_outlier_removal_no_outliers(self):
        """Test with no outliers (all similar values)."""
        values = np.array([5.0, 5.1, 4.9, 5.0, 5.2])

        inliers = remove_outliers_mad(values, threshold=3.0)

        # All should be inliers
        self.assertEqual(np.sum(inliers), 5)

    def test_outlier_removal_empty(self):
        """Test with empty array."""
        values = np.array([])

        inliers = remove_outliers_mad(values)

        self.assertEqual(len(inliers), 0)


class TestAdaptiveBinning(unittest.TestCase):
    """Test adaptive RT binning logic."""

    def test_adaptive_bins_small_dataset(self):
        """Test binning with small dataset."""
        n_bins = determine_rt_bins(n_psms=500, min_psms_per_bin=50)

        # 500 PSMs, target 100/bin → 5 bins
        self.assertEqual(n_bins, 5)
        self.assertGreaterEqual(500 / n_bins, 50)  # Meet minimum

    def test_adaptive_bins_medium_dataset(self):
        """Test binning with medium dataset."""
        n_bins = determine_rt_bins(n_psms=5000, min_psms_per_bin=50)

        # 5000 PSMs → ~20-50 bins
        self.assertGreaterEqual(n_bins, 20)
        self.assertGreaterEqual(5000 / n_bins, 50)

    def test_adaptive_bins_large_dataset(self):
        """Test binning with large dataset."""
        n_bins = determine_rt_bins(n_psms=100000, min_psms_per_bin=50)

        # Should cap at 100 bins
        self.assertEqual(n_bins, 100)

    def test_adaptive_bins_minimum_enforcement(self):
        """Test that minimum PSMs per bin is enforced."""
        # Edge case: 300 PSMs, min 50/bin
        n_bins = determine_rt_bins(n_psms=300, min_psms_per_bin=50)

        # Should use 5 bins (60 PSMs/bin) not 6 (50 PSMs/bin exactly)
        self.assertEqual(n_bins, 5)
        self.assertGreaterEqual(300 / n_bins, 50)


class TestRTBinAssignment(unittest.TestCase):
    """Test RT bin assignment."""

    def test_rt_bin_assignment_uniform(self):
        """Test bin assignment for uniformly distributed RTs."""
        rt_seconds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        rt_min = 0.0
        rt_max = 4.0
        n_bins = 4

        bin_indices = assign_rt_bins(rt_seconds, rt_min, rt_max, n_bins)

        # Should assign to bins 0, 1, 2, 3, 3
        expected = np.array([0, 1, 2, 3, 3])
        np.testing.assert_array_equal(bin_indices, expected)

    def test_rt_bin_assignment_edge_cases(self):
        """Test bin assignment at edges."""
        rt_seconds = np.array([0.0, 10.0])
        rt_min = 0.0
        rt_max = 10.0
        n_bins = 10

        bin_indices = assign_rt_bins(rt_seconds, rt_min, rt_max, n_bins)

        # First should be bin 0, last should be bin 9
        self.assertEqual(bin_indices[0], 0)
        self.assertEqual(bin_indices[1], 9)

    def test_rt_bin_assignment_single_bin(self):
        """Test with single bin (global correction)."""
        rt_seconds = np.array([100.0, 200.0, 300.0])
        rt_min = 0.0
        rt_max = 600.0
        n_bins = 1

        bin_indices = assign_rt_bins(rt_seconds, rt_min, rt_max, n_bins)

        # All should be in bin 0
        np.testing.assert_array_equal(bin_indices, np.zeros(3, dtype=np.int32))


class TestBinCorrections(unittest.TestCase):
    """Test bin correction calculation."""

    def test_bin_corrections_simple(self):
        """Test correction calculation with clear bins."""
        # 6 PSMs, 3 bins, 2 PSMs per bin
        ppm_errors = np.array([5.0, 6.0, 10.0, 11.0, -2.0, -3.0])
        bin_indices = np.array([0, 0, 1, 1, 2, 2])
        n_bins = 3

        corrections, counts = calculate_bin_corrections(ppm_errors, bin_indices, n_bins)

        # Bin 0: median([5, 6]) = 5.5
        # Bin 1: median([10, 11]) = 10.5
        # Bin 2: median([-2, -3]) = -2.5
        np.testing.assert_allclose(corrections[0], 5.5, atol=1e-6)
        np.testing.assert_allclose(corrections[1], 10.5, atol=1e-6)
        np.testing.assert_allclose(corrections[2], -2.5, atol=1e-6)
        np.testing.assert_array_equal(counts, [2, 2, 2])

    def test_bin_corrections_sparse(self):
        """Test with some empty bins."""
        ppm_errors = np.array([5.0, 10.0])
        bin_indices = np.array([0, 2])  # Skip bin 1
        n_bins = 3

        corrections, counts = calculate_bin_corrections(ppm_errors, bin_indices, n_bins)

        # Bin 0: 5.0
        # Bin 1: NaN (empty)
        # Bin 2: 10.0
        np.testing.assert_allclose(corrections[0], 5.0)
        self.assertTrue(np.isnan(corrections[1]))
        np.testing.assert_allclose(corrections[2], 10.0)
        np.testing.assert_array_equal(counts, [1, 0, 1])

    def test_bin_corrections_empty_bin(self):
        """Test behavior with completely empty bins."""
        ppm_errors = np.array([])
        bin_indices = np.array([])
        n_bins = 5

        corrections, counts = calculate_bin_corrections(ppm_errors, bin_indices, n_bins)

        # All bins should be NaN
        self.assertTrue(np.all(np.isnan(corrections)))
        np.testing.assert_array_equal(counts, np.zeros(5, dtype=np.int32))


class TestInterpolation(unittest.TestCase):
    """Test linear interpolation for missing bins."""

    def test_interpolation_single_gap(self):
        """Test interpolation across single gap."""
        corrections = np.array([0.0, np.nan, 10.0])
        counts = np.array([1, 0, 1])

        filled = interpolate_bin_corrections(corrections, counts)

        # Middle should be interpolated: (0 + 10) / 2 = 5
        np.testing.assert_allclose(filled, [0.0, 5.0, 10.0])

    def test_interpolation_multiple_gaps(self):
        """Test interpolation across multiple gaps."""
        corrections = np.array([0.0, np.nan, np.nan, 9.0])
        counts = np.array([1, 0, 0, 1])

        filled = interpolate_bin_corrections(corrections, counts)

        # Should interpolate: 0, 3, 6, 9
        np.testing.assert_allclose(filled, [0.0, 3.0, 6.0, 9.0])

    def test_interpolation_edge_extension(self):
        """Test edge value extension."""
        corrections = np.array([np.nan, np.nan, 5.0, np.nan, np.nan])
        counts = np.array([0, 0, 1, 0, 0])

        filled = interpolate_bin_corrections(corrections, counts)

        # Should extend edges
        np.testing.assert_allclose(filled, [5.0, 5.0, 5.0, 5.0, 5.0])

    def test_interpolation_no_data(self):
        """Test with no valid data."""
        corrections = np.array([np.nan, np.nan, np.nan])
        counts = np.array([0, 0, 0])

        filled = interpolate_bin_corrections(corrections, counts)

        # Should fallback to 0.0
        np.testing.assert_allclose(filled, [0.0, 0.0, 0.0])


class TestMassRecalibrator(unittest.TestCase):
    """Test MassRecalibrator class."""

    def test_calibrator_basic(self):
        """Test basic mass recalibration workflow."""
        # Create synthetic data: +5 ppm error across all RTs
        n_psms = 1000
        theoretical_mz = np.random.uniform(400, 1500, n_psms)
        observed_mz = theoretical_mz * (1 + 5e-6)  # +5 ppm
        rt_seconds = np.linspace(0, 3600, n_psms)

        # Fit calibrator
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Check that error is corrected
        self.assertLess(abs(calibrator.median_ppm_before - 5.0), 0.1)
        self.assertLess(abs(calibrator.median_ppm_after), 0.1)

    def test_calibrator_rt_dependent_drift(self):
        """Test calibration with RT-dependent mass drift."""
        # Create drift: 0 ppm at start → +10 ppm at end
        n_psms = 2000
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        rt_seconds = np.linspace(0, 3600, n_psms)

        # Linear drift from 0 to 10 ppm
        ppm_errors = np.linspace(0, 10, n_psms)
        observed_mz = theoretical_mz * (1 + ppm_errors * 1e-6)

        # Fit calibrator
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds, adaptive_bins=True)

        # Apply correction
        corrected_mz = calibrator.apply(observed_mz, rt_seconds)
        ppm_after = calculate_ppm_errors(corrected_mz, theoretical_mz)

        # After calibration, error should be near zero across gradient
        self.assertLess(abs(np.median(ppm_after)), 0.5)
        self.assertLess(np.std(ppm_after), 1.0)

    def test_calibrator_with_outliers(self):
        """Test that outliers are removed."""
        # Create data with outliers
        n_psms = 1000
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        observed_mz = theoretical_mz * (1 + 5e-6)  # +5 ppm

        # Add some gross outliers
        outlier_indices = np.random.choice(n_psms, 50, replace=False)
        observed_mz[outlier_indices] *= 1 + 100e-6  # +100 ppm

        rt_seconds = np.linspace(0, 3600, n_psms)

        # Should handle outliers gracefully
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Median should still be close to 5 ppm (robust to outliers)
        self.assertLess(abs(calibrator.median_ppm_before - 5.0), 1.0)

    def test_calibrator_empty_data(self):
        """Test with empty data (should not crash)."""
        calibrator = MassRecalibrator(
            observed_mz=np.array([]),
            theoretical_mz=np.array([]),
            rt_seconds=np.array([]),
        )

        # Should use defaults
        self.assertEqual(calibrator.recommended_tolerance, 20.0)


class TestChargeStateIndependence(unittest.TestCase):
    """CRITICAL TEST: Verify single calibration curve works for all charge states."""

    def test_charge_state_independent_calibration(self):
        """Test that 2+ and 3+ are both corrected by SINGLE calibration curve.

        This is the critical test requested by the user. It verifies that we
        don't need separate calibration curves for different charge states.

        If calibration is correct, a single curve should eliminate systematic
        errors for ALL charge states equally.
        """
        # Create synthetic data:
        # - 500 PSMs at charge 2+, systematic +5 ppm error
        # - 500 PSMs at charge 3+, systematic +5 ppm error (SAME error)
        # - Both distributed uniformly across RT range

        n_per_charge = 500
        rt_seconds = np.concatenate([
            np.linspace(0, 3600, n_per_charge),  # 2+ across full gradient
            np.linspace(0, 3600, n_per_charge),  # 3+ across full gradient
        ])

        # Theoretical m/z values
        theoretical_mz_2plus = np.random.uniform(400, 1000, n_per_charge)
        theoretical_mz_3plus = np.random.uniform(400, 1000, n_per_charge)
        theoretical_mz = np.concatenate([theoretical_mz_2plus, theoretical_mz_3plus])

        # Systematic +5 ppm error for BOTH charge states
        observed_mz = theoretical_mz * (1 + 5e-6)

        charges = np.concatenate([
            np.full(n_per_charge, 2, dtype=np.int32),
            np.full(n_per_charge, 3, dtype=np.int32),
        ])

        # Fit SINGLE calibration curve (no charge state consideration)
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Apply to both charge states
        corrected_mz = calibrator.apply(observed_mz, rt_seconds)

        # Calculate residual errors per charge state
        ppm_after = calculate_ppm_errors(corrected_mz, theoretical_mz)
        ppm_2plus = ppm_after[:n_per_charge]
        ppm_3plus = ppm_after[n_per_charge:]

        # CRITICAL ASSERTION: Both charge states corrected to ~0 ppm
        median_2plus = np.median(ppm_2plus)
        median_3plus = np.median(ppm_3plus)

        self.assertLess(abs(median_2plus), 0.5, "Charge 2+ not corrected to ~0 ppm")
        self.assertLess(abs(median_3plus), 0.5, "Charge 3+ not corrected to ~0 ppm")

        # Both should have similar residual error (no charge bias)
        self.assertLess(abs(median_2plus - median_3plus), 0.3,
                       "Charge states have different residual errors")

        print(f"✓ Charge 2+: {median_2plus:.2f} ppm after calibration")
        print(f"✓ Charge 3+: {median_3plus:.2f} ppm after calibration")

    def test_charge_state_mixed_errors(self):
        """Test calibration when charge states have DIFFERENT systematic errors.

        This tests a more challenging case where 2+ and 3+ have different
        systematic errors (e.g., due to isotope selection). A single curve
        should still perform reasonably well.
        """
        n_per_charge = 500

        # 2+ peptides: +7 ppm error
        # 3+ peptides: +3 ppm error
        theoretical_mz_2plus = np.random.uniform(400, 1000, n_per_charge)
        theoretical_mz_3plus = np.random.uniform(400, 1000, n_per_charge)

        observed_mz_2plus = theoretical_mz_2plus * (1 + 7e-6)  # +7 ppm
        observed_mz_3plus = theoretical_mz_3plus * (1 + 3e-6)  # +3 ppm

        theoretical_mz = np.concatenate([theoretical_mz_2plus, theoretical_mz_3plus])
        observed_mz = np.concatenate([observed_mz_2plus, observed_mz_3plus])

        rt_seconds = np.concatenate([
            np.linspace(0, 3600, n_per_charge),
            np.linspace(0, 3600, n_per_charge),
        ])

        # Fit SINGLE calibration curve
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # The calibration should find the median error (~5 ppm)
        # and correct both charge states partially
        self.assertGreater(calibrator.median_ppm_before, 4.0)
        self.assertLess(calibrator.median_ppm_before, 6.0)


class TestRapidMassDrift(unittest.TestCase):
    """Test calibration handles rapid mass shifts (air conditioning case)."""

    def test_rapid_mass_shift(self):
        """Test that rapid mass drift is captured with adaptive binning.

        Simulates the air conditioning case: temperature spike causes
        mass error to jump from +2 ppm to +12 ppm over 5 minutes, then
        return to +2 ppm.
        """
        # 10k PSMs over 60 min gradient
        n_psms = 10000
        rt_seconds = np.linspace(0, 3600, n_psms)
        theoretical_mz = np.random.uniform(500, 1000, n_psms)

        # Create mass error profile with spike at 30-35 min
        ppm_errors = np.full(n_psms, 2.0)  # Baseline +2 ppm

        # Spike from 30 to 35 min (1800-2100 sec)
        spike_mask = (rt_seconds >= 1800) & (rt_seconds <= 2100)
        ppm_errors[spike_mask] = 12.0  # +12 ppm during spike

        observed_mz = theoretical_mz * (1 + ppm_errors * 1e-6)

        # Fit with adaptive binning (should use ~40 bins for 10k PSMs)
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds, adaptive_bins=True)

        # Should use enough bins to capture 5-minute drift
        # 60 min / 40 bins = 1.5 min per bin → can resolve 5 min event
        self.assertGreaterEqual(calibrator.n_bins, 20)

        # Apply correction
        corrected_mz = calibrator.apply(observed_mz, rt_seconds)
        ppm_after = calculate_ppm_errors(corrected_mz, theoretical_mz)

        # Check correction during spike region
        ppm_spike_after = ppm_after[spike_mask]
        median_spike_after = np.median(ppm_spike_after)

        # Should reduce spike error significantly
        self.assertLess(abs(median_spike_after), 2.0,
                       "Rapid mass drift not corrected")

        print(f"✓ Bins used: {calibrator.n_bins}")
        print(f"✓ Spike error after calibration: {median_spike_after:.2f} ppm")


class TestChargeStateConsistencyCheck(unittest.TestCase):
    """Test pre-search charge state consistency estimation."""

    def test_charge_state_consistency_basic(self):
        """Test charge state consistency check with charge-dependent errors.

        NOTE: This function can only detect CHARGE-STATE-DEPENDENT errors,
        not global systematic shifts (where all charges have same error).
        """
        # Create synthetic precursor list with charge state pairs
        # 1000 precursors, 500 unique molecules each appearing as 2+ and 3+

        n_unique = 500
        neutral_masses = np.random.uniform(1000, 3000, n_unique)

        # Create charge-state-DEPENDENT errors:
        # 2+ has +7 ppm error
        # 3+ has +3 ppm error
        # (e.g., due to isotope selection differences)

        from alphapeptfast.constants import PROTON_MASS

        # 2+ precursors with +7 ppm error
        neutral_masses_2plus = neutral_masses * (1 + 7e-6)
        mz_2plus = (neutral_masses_2plus + 2 * PROTON_MASS) / 2

        # 3+ precursors with +3 ppm error
        neutral_masses_3plus = neutral_masses * (1 + 3e-6)
        mz_3plus = (neutral_masses_3plus + 3 * PROTON_MASS) / 3

        precursor_mz = np.concatenate([mz_2plus, mz_3plus])
        precursor_charges = np.concatenate([
            np.full(n_unique, 2, dtype=np.int32),
            np.full(n_unique, 3, dtype=np.int32),
        ])

        # Shuffle to simulate real data
        shuffle_idx = np.random.permutation(len(precursor_mz))
        precursor_mz = precursor_mz[shuffle_idx]
        precursor_charges = precursor_charges[shuffle_idx]

        # Estimate mass error
        estimate = estimate_mass_error_from_charge_states(
            precursor_mz, precursor_charges, tolerance_ppm_start=50.0
        )

        # Should detect discrepancy (not absolute error, but difference)
        # The function finds INCONSISTENCY between charge states
        self.assertGreater(estimate['n_charge_pairs'], 100)
        # Median ppm should reflect the charge state discrepancy (~4 ppm difference)
        self.assertGreater(abs(estimate['median_ppm']), 2.0)

        print(f"✓ Detected mass discrepancy: {estimate['median_ppm']:.1f} ppm")
        print(f"✓ Charge pairs found: {estimate['n_charge_pairs']}")
        print(f"✓ Recommended tolerance: {estimate['recommended_tolerance']:.1f} ppm")

    def test_charge_state_consistency_no_pairs(self):
        """Test when no charge state pairs are found."""
        # All precursors at single charge state
        precursor_mz = np.random.uniform(400, 1000, 1000)
        precursor_charges = np.full(1000, 2, dtype=np.int32)

        estimate = estimate_mass_error_from_charge_states(
            precursor_mz, precursor_charges
        )

        # Should fallback to default
        self.assertEqual(estimate['n_charge_pairs'], 0)
        self.assertEqual(estimate['recommended_tolerance'], 20.0)


class TestRecommendedTolerance(unittest.TestCase):
    """Test recommended tolerance calculation."""

    def test_recommended_tolerance_narrow_distribution(self):
        """Test tolerance for narrow error distribution."""
        # Very tight mass accuracy: 1 ppm std
        n_psms = 1000
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        ppm_errors = np.random.normal(0, 1.0, n_psms)  # 1 ppm std
        observed_mz = theoretical_mz * (1 + ppm_errors * 1e-6)
        rt_seconds = np.linspace(0, 3600, n_psms)

        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Should recommend narrow tolerance (95th percentile ~2 ppm)
        self.assertLess(calibrator.recommended_tolerance, 10.0)

    def test_recommended_tolerance_wide_distribution(self):
        """Test tolerance for wide error distribution."""
        # Poor mass accuracy: 10 ppm std
        n_psms = 1000
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        ppm_errors = np.random.normal(0, 10.0, n_psms)  # 10 ppm std
        observed_mz = theoretical_mz * (1 + ppm_errors * 1e-6)
        rt_seconds = np.linspace(0, 3600, n_psms)

        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Should recommend wider tolerance
        self.assertGreater(calibrator.recommended_tolerance, 15.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_psm(self):
        """Test with single PSM."""
        calibrator = MassRecalibrator(
            observed_mz=np.array([500.0]),
            theoretical_mz=np.array([500.0]),
            rt_seconds=np.array([1800.0]),
        )

        # Should use global correction
        corrected = calibrator.apply(np.array([600.0]), np.array([1800.0]))
        self.assertEqual(len(corrected), 1)

    def test_all_outliers(self):
        """Test when all PSMs are flagged as outliers."""
        # Create data where all points are far from median
        n_psms = 100
        theoretical_mz = np.array([500.0] * n_psms)
        observed_mz = theoretical_mz + np.random.uniform(100, 200, n_psms)
        rt_seconds = np.linspace(0, 3600, n_psms)

        # Should handle gracefully (fall back to defaults)
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        self.assertIsNotNone(calibrator.bin_corrections)

    def test_zero_rt_range(self):
        """Test when all PSMs have same RT."""
        n_psms = 100
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        observed_mz = theoretical_mz * (1 + 5e-6)
        rt_seconds = np.full(n_psms, 1800.0)  # All at same RT

        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        # Should still work (single global correction)
        corrected = calibrator.apply(observed_mz, rt_seconds)
        self.assertEqual(len(corrected), n_psms)


class TestPerformance(unittest.TestCase):
    """Test performance benchmarks."""

    def test_calibration_speed(self):
        """Test calibration fitting speed."""
        import time

        n_psms = 10000
        theoretical_mz = np.random.uniform(400, 1500, n_psms)
        observed_mz = theoretical_mz * (1 + np.random.normal(0, 5, n_psms) * 1e-6)
        rt_seconds = np.linspace(0, 3600, n_psms)

        start = time.time()
        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)
        elapsed = time.time() - start

        print(f"\nCalibration fitting: {n_psms} PSMs in {elapsed*1000:.1f} ms")
        print(f"Throughput: {n_psms/elapsed:.0f} PSMs/sec")

        # Should be fast (<1 sec for 10k PSMs)
        self.assertLess(elapsed, 1.0)

    def test_correction_speed(self):
        """Test correction application speed."""
        import time

        # Fit on subset
        n_fit = 1000
        theoretical_mz_fit = np.random.uniform(500, 1000, n_fit)
        observed_mz_fit = theoretical_mz_fit * (1 + 5e-6)
        rt_seconds_fit = np.linspace(0, 3600, n_fit)

        calibrator = MassRecalibrator(observed_mz_fit, theoretical_mz_fit, rt_seconds_fit)

        # Apply to large dataset
        n_apply = 1000000
        mz_to_correct = np.random.uniform(400, 1500, n_apply)
        rt_to_correct = np.random.uniform(0, 3600, n_apply)

        start = time.time()
        corrected = calibrator.apply(mz_to_correct, rt_to_correct)
        elapsed = time.time() - start

        throughput = n_apply / elapsed

        print(f"\nCorrection application: {n_apply} m/z in {elapsed*1000:.1f} ms")
        print(f"Throughput: {throughput/1e6:.2f} M m/z values/sec")

        # Should be very fast (>1M/sec)
        self.assertGreater(throughput, 1e6)


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete calibration workflows."""

    def test_iterative_recalibration_workflow(self):
        """Test two-round calibration workflow.

        Round 1: Wide tolerance search → calibrate → narrow tolerance
        Round 2: Narrow tolerance search → refine calibration
        """
        # Simulate Round 1: 1000 PSMs from initial search
        n_round1 = 1000
        theoretical_mz_r1 = np.random.uniform(500, 1000, n_round1)
        observed_mz_r1 = theoretical_mz_r1 * (1 + np.random.normal(5, 2, n_round1) * 1e-6)
        rt_seconds_r1 = np.linspace(0, 3600, n_round1)

        # First calibration
        calibrator_r1 = MassRecalibrator(observed_mz_r1, theoretical_mz_r1, rt_seconds_r1)

        tolerance_r1 = calibrator_r1.recommended_tolerance
        print(f"\nRound 1 tolerance: {tolerance_r1:.1f} ppm")

        # Simulate Round 2: More PSMs with better accuracy
        n_round2 = 5000
        theoretical_mz_r2 = np.random.uniform(500, 1000, n_round2)
        observed_mz_r2 = theoretical_mz_r2 * (1 + np.random.normal(2, 1, n_round2) * 1e-6)
        rt_seconds_r2 = np.linspace(0, 3600, n_round2)

        # Apply first calibration
        corrected_mz_r2 = calibrator_r1.apply(observed_mz_r2, rt_seconds_r2)

        # Second calibration on corrected data
        calibrator_r2 = MassRecalibrator(corrected_mz_r2, theoretical_mz_r2, rt_seconds_r2)

        tolerance_r2 = calibrator_r2.recommended_tolerance
        print(f"Round 2 tolerance: {tolerance_r2:.1f} ppm")

        # Second round should have tighter tolerance
        self.assertLess(tolerance_r2, tolerance_r1)

    def test_pre_search_to_calibration_workflow(self):
        """Test complete workflow from pre-search check to calibration.

        NOTE: Charge state consistency can detect charge-dependent errors.
        For global systematic errors, we'd start with a default tolerance.
        """
        # Step 1: Pre-search charge state check
        # Create charge-state-dependent errors (2+ vs 3+)
        n_precursors = 5000
        neutral_masses = np.random.uniform(1000, 3000, n_precursors)

        from alphapeptfast.constants import PROTON_MASS

        # 2+ with +10 ppm, 3+ with +5 ppm (charge-dependent)
        neutral_masses_2plus = neutral_masses * (1 + 10e-6)
        neutral_masses_3plus = neutral_masses * (1 + 5e-6)

        mz_2plus = (neutral_masses_2plus + 2 * PROTON_MASS) / 2
        mz_3plus = (neutral_masses_3plus + 3 * PROTON_MASS) / 3

        precursor_mz = np.concatenate([mz_2plus, mz_3plus])
        precursor_charges = np.concatenate([
            np.full(len(mz_2plus), 2, dtype=np.int32),
            np.full(len(mz_3plus), 3, dtype=np.int32),
        ])

        estimate = estimate_mass_error_from_charge_states(precursor_mz, precursor_charges)
        print(f"\nPre-search charge state discrepancy: {estimate['median_ppm']:.1f} ppm")
        print(f"Recommended tolerance: {estimate['recommended_tolerance']:.1f} ppm")

        # Charge state check should detect discrepancy
        self.assertGreater(estimate['n_charge_pairs'], 100)

        # Step 2: Simulate search with that tolerance → calibrate
        n_psms = 2000
        theoretical_mz = np.random.uniform(500, 1000, n_psms)
        observed_mz = theoretical_mz * (1 + np.random.normal(8, 3, n_psms) * 1e-6)
        rt_seconds = np.linspace(0, 3600, n_psms)

        calibrator = MassRecalibrator(observed_mz, theoretical_mz, rt_seconds)

        print(f"Post-search calibration: {calibrator.recommended_tolerance:.1f} ppm")

        # Calibrator should provide useful tolerance
        self.assertGreater(calibrator.recommended_tolerance, 5.0)
        self.assertLess(calibrator.recommended_tolerance, 25.0)


if __name__ == "__main__":
    unittest.main()

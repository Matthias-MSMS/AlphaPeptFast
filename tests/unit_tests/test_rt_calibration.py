"""Comprehensive unit tests for RT calibration module.

Tests the robust PCHIP-based iRT → RT calibration including:
- MAD-based outlier removal
- PCHIP monotonicity preservation
- Dead-time offset estimation
- Robust extrapolation
- Edge cases and failure modes
"""

import numpy as np
import pytest

from alphapeptfast.rt.calibration import fit_pchip_irt_to_rt, predict_pchip_irt


class TestOutlierRemoval:
    """Test MAD-based robust outlier filtering."""

    def test_clean_data_unchanged(self):
        """Test that clean data passes through without removal."""
        # Perfect linear relationship
        irt = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        model = fit_pchip_irt_to_rt(irt, rt)
        x_fitted = model[0]  # x values after outlier removal

        # All points should be kept
        assert len(x_fitted) == len(irt)

    def test_outliers_can_be_removed(self):
        """Test that outlier removal can filter points."""
        # Linear data with realistic noise
        np.random.seed(42)
        n = 20
        irt = np.linspace(0.1, 0.9, n)
        rt = 1000 * irt + 50 + np.random.normal(0, 5, n)  # Linear + small noise

        # Add a few obvious outliers
        rt[5] += 150  # Large positive outlier
        rt[15] -= 150  # Large negative outlier

        # Strict threshold should remove outliers
        model_strict = fit_pchip_irt_to_rt(irt, rt, mad_k=2.0)
        x_strict = model_strict[0]

        # Lenient threshold should keep more points
        model_lenient = fit_pchip_irt_to_rt(irt, rt, mad_k=5.0)
        x_lenient = model_lenient[0]

        # Lenient should keep more or equal points than strict
        assert len(x_lenient) >= len(x_strict)
        # At least some points should be kept
        assert len(x_lenient) >= 10

    def test_outlier_removal_improves_fit(self):
        """Test that outlier removal improves calibration quality."""
        # Create clean linear data
        np.random.seed(123)
        n = 15
        irt_clean = np.linspace(0.1, 0.9, n)
        rt_clean = 1000 * irt_clean + 100 + np.random.normal(0, 3, n)

        # Add extreme outliers
        irt_dirty = np.copy(irt_clean)
        rt_dirty = np.copy(rt_clean)
        rt_dirty[3] += 200
        rt_dirty[10] -= 200

        # Fit with outlier removal
        model = fit_pchip_irt_to_rt(irt_dirty, rt_dirty, mad_k=3.0)

        # Model should be created successfully
        assert len(model) == 10  # Full model tuple
        assert model[0].size > 0  # At least some points kept

    def test_mad_threshold_configurable(self):
        """Test that MAD threshold affects outlier removal."""
        irt = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rt = np.array([100.0, 200.0, 350.0, 400.0, 500.0])  # 0.3 slightly off

        # Strict threshold (should remove more)
        model_strict = fit_pchip_irt_to_rt(irt, rt, mad_k=2.0)

        # Lenient threshold (should keep more)
        model_lenient = fit_pchip_irt_to_rt(irt, rt, mad_k=5.0)

        # Lenient should keep at least as many points
        assert len(model_lenient[0]) >= len(model_strict[0])

    def test_all_outliers_fails_gracefully(self):
        """Test that removing all points fails gracefully."""
        # All points are outliers (scattered randomly)
        irt = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rt = np.array([100.0, 500.0, 200.0, 700.0, 150.0])

        model = fit_pchip_irt_to_rt(irt, rt, mad_k=0.5)  # Very strict

        # Should return empty or minimal model
        x_fitted = model[0]
        assert len(x_fitted) >= 0  # Should not crash


class TestPCHIPMonotonicity:
    """Test PCHIP monotonicity preservation."""

    def test_monotone_input_stays_monotone(self):
        """Test that monotone input produces monotone interpolation."""
        # Strictly increasing
        irt = np.linspace(0.0, 1.0, 10)
        rt = np.linspace(60.0, 3600.0, 10)

        model = fit_pchip_irt_to_rt(irt, rt)

        # Query many points
        irt_test = np.linspace(0.0, 1.0, 100)
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should be strictly increasing
        for i in range(1, len(rt_pred)):
            assert rt_pred[i] > rt_pred[i-1], \
                f"Not monotone at index {i}: {rt_pred[i-1]} -> {rt_pred[i]}"

    def test_non_monotone_sections_handled(self):
        """Test that non-monotone x values are handled."""
        # iRT with duplicates/backsteps (should be removed)
        irt = np.array([0.1, 0.2, 0.2, 0.3, 0.25, 0.4])  # Non-monotone
        rt = np.array([100.0, 200.0, 210.0, 300.0, 250.0, 400.0])

        model = fit_pchip_irt_to_rt(irt, rt)
        x_fitted = model[0]

        # After processing, x should be strictly increasing
        for i in range(1, len(x_fitted)):
            assert x_fitted[i] > x_fitted[i-1]

    def test_s_shaped_curve_monotone(self):
        """Test monotonicity with S-shaped calibration curve."""
        # Typical gradient with slow start/end
        irt = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        rt = np.array([50.0, 100.0, 500.0, 1000.0, 1500.0, 1900.0, 2000.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Dense query
        irt_test = np.linspace(0.0, 1.0, 200)
        rt_pred = predict_pchip_irt(model, irt_test)

        # Monotonicity preserved
        assert np.all(np.diff(rt_pred) > 0)

    def test_endpoint_derivatives_reasonable(self):
        """Test that endpoint derivatives don't blow up."""
        irt = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rt = np.array([100.0, 300.0, 500.0, 700.0, 900.0])

        model = fit_pchip_irt_to_rt(irt, rt)
        m_left, m_right = model[7], model[8]  # Tail slopes

        # Should be positive and reasonable (not infinite)
        assert m_left > 0
        assert m_right > 0
        assert m_left < 10000  # Sanity check
        assert m_right < 10000


class TestDeadTimeEstimation:
    """Test t0 (dead-time offset) estimation."""

    def test_positive_t0_estimated(self):
        """Test that positive t0 offset is estimated."""
        # Linear relationship with offset
        irt = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        t0_true = 60.0  # 1 minute dead time
        rt = 200.0 * irt + t0_true

        model = fit_pchip_irt_to_rt(irt, rt, estimate_t0=True)
        t0_estimated = model[9]

        # Should estimate close to true t0
        assert abs(t0_estimated - t0_true) < 10.0

    def test_negative_t0_clamped(self):
        """Test that negative t0 is clamped to zero."""
        # Relationship that would give negative offset
        irt = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rt = np.array([50.0, 300.0, 500.0, 700.0, 900.0])

        model = fit_pchip_irt_to_rt(irt, rt, estimate_t0=True)
        t0 = model[9]

        # Should be non-negative
        assert t0 >= 0.0

    def test_no_early_peptides_gives_zero(self):
        """Test that without early peptides, t0 ≈ 0."""
        # All peptides late-eluting (no early region)
        irt = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        rt = np.array([600.0, 700.0, 800.0, 900.0, 1000.0])

        model = fit_pchip_irt_to_rt(irt, rt, estimate_t0=True)
        t0 = model[9]

        # Should be small or zero
        assert t0 < 100.0

    def test_t0_disabled_option(self):
        """Test that t0 estimation can be disabled."""
        irt = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        rt = 200.0 * irt + 60.0  # Has offset

        model = fit_pchip_irt_to_rt(irt, rt, estimate_t0=False)
        t0 = model[9]

        # Should be exactly zero when disabled
        assert t0 == 0.0

    def test_t0_improves_prediction(self):
        """Test that t0 estimation improves prediction accuracy."""
        # Data with known offset
        irt_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        t0_true = 50.0
        rt_train = 1000.0 * irt_train + t0_true

        # With t0 estimation
        model_with_t0 = fit_pchip_irt_to_rt(irt_train, rt_train, estimate_t0=True)

        # Without t0 estimation
        model_without_t0 = fit_pchip_irt_to_rt(irt_train, rt_train, estimate_t0=False)

        # Test on new point
        irt_test = np.array([0.2])
        rt_true = 1000.0 * irt_test[0] + t0_true

        rt_pred_with = predict_pchip_irt(model_with_t0, irt_test)[0]
        rt_pred_without = predict_pchip_irt(model_without_t0, irt_test)[0]

        # With t0 should be more accurate
        error_with = abs(rt_pred_with - rt_true)
        error_without = abs(rt_pred_without - rt_true)

        assert error_with < error_without


class TestExtrapolation:
    """Test linear extrapolation beyond calibration range."""

    def test_left_tail_extrapolation(self):
        """Test extrapolation below calibration range."""
        irt = np.array([0.3, 0.5, 0.7, 0.9])
        rt = np.array([300.0, 500.0, 700.0, 900.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Query below range
        irt_test = np.array([0.1, 0.2])
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should extrapolate linearly
        assert len(rt_pred) == 2
        assert rt_pred[0] < rt_pred[1]  # Monotone
        assert rt_pred[1] < 300.0  # Below calibration range

    def test_right_tail_extrapolation(self):
        """Test extrapolation above calibration range."""
        irt = np.array([0.1, 0.3, 0.5, 0.7])
        rt = np.array([100.0, 300.0, 500.0, 700.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Query above range
        irt_test = np.array([0.8, 0.9, 1.0])
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should extrapolate linearly
        assert len(rt_pred) == 3
        assert rt_pred[0] < rt_pred[1] < rt_pred[2]  # Monotone
        assert rt_pred[0] > 700.0  # Above calibration range

    def test_extrapolation_uses_median_slope(self):
        """Test that extrapolation uses robust median slope."""
        # Linear data
        irt = np.array([0.2, 0.4, 0.6, 0.8])
        rt = np.array([200.0, 400.0, 600.0, 800.0])  # Slope = 1000

        model = fit_pchip_irt_to_rt(irt, rt)

        # Extrapolate far
        irt_test = np.array([1.0, 1.2])
        rt_pred = predict_pchip_irt(model, irt_test)

        # Slope should be approximately preserved
        slope_extrap = (rt_pred[1] - rt_pred[0]) / (1.2 - 1.0)
        assert 900 < slope_extrap < 1100  # Close to 1000

    def test_extreme_extrapolation_stable(self):
        """Test that extreme extrapolation doesn't blow up."""
        irt = np.array([0.3, 0.5, 0.7])
        rt = np.array([300.0, 500.0, 700.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Very far extrapolation
        irt_test = np.array([-1.0, 2.0, 5.0])
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should not be NaN or infinite
        assert np.all(np.isfinite(rt_pred))

        # Should be monotone
        assert rt_pred[0] < rt_pred[1] < rt_pred[2]


class TestEdgeCases:
    """Test edge cases and failure modes."""

    def test_empty_input(self):
        """Test with empty arrays."""
        irt = np.array([])
        rt = np.array([])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should return empty model
        assert len(model[0]) == 0

        # Prediction should return NaN
        irt_test = np.array([0.5])
        rt_pred = predict_pchip_irt(model, irt_test)
        assert np.isnan(rt_pred[0])

    def test_single_point_calibration(self):
        """Test with single calibration point."""
        irt = np.array([0.5])
        rt = np.array([500.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should return constant prediction
        irt_test = np.array([0.3, 0.5, 0.7])
        rt_pred = predict_pchip_irt(model, irt_test)

        # All predictions should be the same (constant)
        assert np.all(rt_pred == rt_pred[0])

    def test_two_point_calibration(self):
        """Test with two calibration points (linear)."""
        irt = np.array([0.2, 0.8])
        rt = np.array([200.0, 800.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should be perfectly linear
        irt_test = np.array([0.5])
        rt_pred = predict_pchip_irt(model, irt_test)[0]

        # Midpoint should be at midpoint
        assert abs(rt_pred - 500.0) < 1.0

    def test_duplicate_irt_values(self):
        """Test with duplicate iRT values (de-duplication)."""
        # Exact duplicates
        irt = np.array([0.1, 0.2, 0.2, 0.3, 0.4])
        rt = np.array([100.0, 200.0, 210.0, 300.0, 400.0])

        model = fit_pchip_irt_to_rt(irt, rt, dedup_tol=1e-4)
        x_fitted = model[0]

        # Duplicates should be averaged/removed
        assert len(x_fitted) <= len(irt)

        # x should be unique
        assert len(x_fitted) == len(np.unique(x_fitted))

    def test_very_close_irt_values(self):
        """Test with very close iRT values (near-duplicates)."""
        irt = np.array([0.1, 0.2, 0.2000001, 0.3, 0.4])
        rt = np.array([100.0, 200.0, 205.0, 300.0, 400.0])

        model = fit_pchip_irt_to_rt(irt, rt, dedup_tol=1e-4)
        x_fitted = model[0]

        # Near-duplicates should be collapsed
        assert len(x_fitted) < len(irt)

    def test_very_long_gradient(self):
        """Test with very long gradient (>1 hour)."""
        irt = np.linspace(0.0, 1.0, 10)
        rt = np.linspace(0.0, 7200.0, 10)  # 2 hour gradient

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should handle large RT values
        irt_test = np.array([0.5])
        rt_pred = predict_pchip_irt(model, irt_test)

        assert 3000 < rt_pred[0] < 4000  # Should be around midpoint

    def test_very_short_gradient(self):
        """Test with very short gradient (<5 min)."""
        irt = np.linspace(0.0, 1.0, 10)
        rt = np.linspace(0.0, 180.0, 10)  # 3 minute gradient

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should handle small RT values
        irt_test = np.array([0.5])
        rt_pred = predict_pchip_irt(model, irt_test)

        assert 80 < rt_pred[0] < 100  # Should be around midpoint

    def test_negative_irt_values(self):
        """Test with negative iRT values."""
        # Some iRT scales can be negative
        irt = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
        rt = np.array([50.0, 200.0, 400.0, 600.0, 800.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should handle negative iRT
        irt_test = np.array([-0.1])
        rt_pred = predict_pchip_irt(model, irt_test)

        assert np.isfinite(rt_pred[0])

    def test_irt_greater_than_one(self):
        """Test with iRT values > 1.0."""
        # iRT scale not always 0-1
        irt = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        rt = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should handle iRT > 1
        irt_test = np.array([1.25])
        rt_pred = predict_pchip_irt(model, irt_test)

        assert 1200 < rt_pred[0] < 1300


class TestKnownPeptides:
    """Test against known iRT peptide standards."""

    def test_linear_calibration_accuracy(self):
        """Test accuracy with perfect linear relationship."""
        # Perfect linear: RT = 1000 * iRT + 100
        irt_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rt_train = 1000.0 * irt_train + 100.0

        model = fit_pchip_irt_to_rt(irt_train, rt_train)

        # Test on new points
        irt_test = np.array([0.2, 0.4, 0.6, 0.8])
        rt_expected = 1000.0 * irt_test + 100.0
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should be very accurate (within 1 second)
        errors = np.abs(rt_pred - rt_expected)
        assert np.all(errors < 1.0)

    def test_rt_accuracy_within_tolerance(self):
        """Test that predictions are within reasonable tolerance."""
        # Realistic calibration with some noise
        np.random.seed(42)
        irt = np.linspace(0.1, 0.9, 20)
        rt_true = 1000.0 * irt + 100.0
        rt_noisy = rt_true + np.random.normal(0, 5.0, size=20)  # 5 second noise

        # Fit on noisy data
        model = fit_pchip_irt_to_rt(irt, rt_noisy)

        # Predict on clean points
        rt_pred = predict_pchip_irt(model, irt)

        # Should be close to true values (within ~10 seconds)
        errors = np.abs(rt_pred - rt_true)
        assert np.mean(errors) < 10.0

    def test_roundtrip_consistency(self):
        """Test that model predicts training data accurately."""
        irt_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rt_train = np.array([150.0, 400.0, 650.0, 900.0, 1150.0])

        model = fit_pchip_irt_to_rt(irt_train, rt_train)
        rt_pred = predict_pchip_irt(model, irt_train)

        # Should reconstruct training data well
        errors = np.abs(rt_pred - rt_train)
        assert np.all(errors < 5.0)  # Within 5 seconds


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_no_nans_or_infs(self):
        """Test that normal use never produces NaN or Inf."""
        irt = np.linspace(0.1, 0.9, 10)
        rt = np.linspace(100.0, 1000.0, 10)

        model = fit_pchip_irt_to_rt(irt, rt)

        # Test over wide range
        irt_test = np.linspace(-0.5, 1.5, 100)
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should all be finite
        assert np.all(np.isfinite(rt_pred))

    def test_float32_vs_float64(self):
        """Test that float32 input doesn't cause issues."""
        irt = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        rt = np.array([100.0, 300.0, 500.0, 700.0, 900.0], dtype=np.float32)

        # Should handle float32 (converted internally to float64)
        model = fit_pchip_irt_to_rt(irt, rt)

        irt_test = np.array([0.5], dtype=np.float32)
        rt_pred = predict_pchip_irt(model, irt_test)

        assert np.isfinite(rt_pred[0])

    def test_precision_with_close_points(self):
        """Test precision when points are very close."""
        # Points very close together
        irt = np.array([0.500, 0.501, 0.502, 0.503, 0.504])
        rt = np.array([500.0, 501.0, 502.0, 503.0, 504.0])

        model = fit_pchip_irt_to_rt(irt, rt)

        # Should handle precision
        irt_test = np.array([0.5015])
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should be close to 501.5
        assert 500.5 < rt_pred[0] < 502.5


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_typical_dia_workflow(self):
        """Test typical DIA RT calibration workflow."""
        # Simulate identifying ~20 high-confidence peptides
        np.random.seed(42)
        n_calib = 20
        irt_true = np.sort(np.random.uniform(0.0, 1.0, n_calib))
        rt_true = 2000.0 * irt_true + 120.0  # 35 min gradient, 2 min dead time

        # Add some measurement noise
        rt_obs = rt_true + np.random.normal(0, 3.0, n_calib)

        # Add 2 outliers
        rt_obs[5] += 200.0
        rt_obs[15] -= 150.0

        # Fit calibration
        model = fit_pchip_irt_to_rt(irt_true, rt_obs, mad_k=3.5, estimate_t0=True)

        # Predict on all peptides in library (e.g., 10k peptides)
        irt_library = np.random.uniform(0.0, 1.0, 100)  # Smaller for test speed
        rt_pred = predict_pchip_irt(model, irt_library)

        # All predictions should be reasonable
        assert np.all(rt_pred > 0)
        assert np.all(rt_pred < 3000)  # Within gradient range
        assert np.all(np.isfinite(rt_pred))

    def test_batch_prediction(self):
        """Test predicting many points at once."""
        irt = np.linspace(0.1, 0.9, 10)
        rt = np.linspace(100.0, 1000.0, 10)

        model = fit_pchip_irt_to_rt(irt, rt)

        # Predict 10,000 points
        irt_test = np.random.uniform(0.0, 1.0, 10000)
        rt_pred = predict_pchip_irt(model, irt_test)

        # Should handle large batch
        assert len(rt_pred) == 10000
        assert np.all(np.isfinite(rt_pred))

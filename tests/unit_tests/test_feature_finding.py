"""
Tests for feature finding module.

Validates the argmax feature finding algorithm including:
- Peak grouping with ppm/RT tolerances
- Intensity-weighted centroiding
- Isotope pattern detection
- Charge pair analysis for mass accuracy validation

Author: Claude Code
Date: February 2026
"""

import numpy as np
import pytest

from alphapeptfast.features.feature_finding import (
    find_features_numba,
    find_isotope_patterns,
    find_charge_pairs,
    FeatureFinderParams,
    C13_MASS_DIFF,
)
from alphapeptfast.constants import PROTON_MASS


# =============================================================================
# Test Parameters - Instrument-agnostic defaults
# =============================================================================

DEFAULT_PPM_TOL = 10.0
DEFAULT_RT_TOL_SEC = 5.0
DEFAULT_INTENSITY_THRESHOLD = 100.0
DEFAULT_MIN_PEAKS = 3


# =============================================================================
# Tests for Core Feature Finding
# =============================================================================

class TestFindFeaturesNumba:
    """Tests for the core argmax feature finding algorithm."""

    def test_single_feature_basic(self):
        """Simple case: peaks at same m/z and RT -> 1 feature."""
        base_mz = 500.0
        mz = np.array([499.998, 499.999, 500.0, 500.001, 500.002], dtype=np.float64)
        intensity = np.array([100, 500, 1000, 500, 100], dtype=np.float64)
        scan = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        rt = np.array([60.0, 60.1, 60.2, 60.3, 60.4], dtype=np.float64)

        # Sort by m/z (required by algorithm)
        order = np.argsort(mz)
        mz, intensity, scan, rt = mz[order], intensity[order], scan[order], rt[order]

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=1.0,
            intensity_threshold=50.0,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        feat_mz, feat_rt, feat_intensity, feat_mz_std, _, _, feat_n_peaks, _, n_features = result

        assert n_features == 1, f"Expected 1 feature, got {n_features}"
        assert feat_n_peaks[0] == 5, f"Expected 5 peaks, got {feat_n_peaks[0]}"
        assert abs(feat_mz[0] - 500.0) < 0.002, f"Feature m/z {feat_mz[0]} not near 500.0"

    def test_two_features_different_mz(self):
        """Two peaks at different m/z -> 2 features."""
        mz1 = np.array([499.999, 500.0, 500.001], dtype=np.float64)
        int1 = np.array([500, 1000, 500], dtype=np.float64)
        rt1 = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        mz2 = np.array([599.999, 600.0, 600.001], dtype=np.float64)
        int2 = np.array([500, 1000, 500], dtype=np.float64)
        rt2 = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        mz = np.concatenate([mz1, mz2])
        intensity = np.concatenate([int1, int2])
        rt = np.concatenate([rt1, rt2])
        scan = np.arange(len(mz), dtype=np.int32)

        order = np.argsort(mz)
        mz, intensity, scan, rt = mz[order], intensity[order], scan[order], rt[order]

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, _, _, n_features = result
        assert n_features == 2, f"Expected 2 features, got {n_features}"

    def test_two_features_different_rt(self):
        """Same m/z, different RT -> 2 features."""
        mz1 = np.array([499.999, 500.0, 500.001], dtype=np.float64)
        int1 = np.array([500, 1000, 500], dtype=np.float64)
        rt1 = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        mz2 = np.array([499.999, 500.0, 500.001], dtype=np.float64)
        int2 = np.array([500, 1000, 500], dtype=np.float64)
        rt2 = np.array([120.0, 120.0, 120.0], dtype=np.float64)

        mz = np.concatenate([mz1, mz2])
        intensity = np.concatenate([int1, int2])
        rt = np.concatenate([rt1, rt2])
        scan = np.arange(len(mz), dtype=np.int32)

        order = np.argsort(mz)
        mz, intensity, scan, rt = mz[order], intensity[order], scan[order], rt[order]

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, _, _, n_features = result
        assert n_features == 2, f"Expected 2 features (different RT), got {n_features}"

    def test_intensity_weighted_centroid(self):
        """Feature m/z should be intensity-weighted mean, not simple mean."""
        mz = np.array([500.000, 500.001, 500.002], dtype=np.float64)
        intensity = np.array([100.0, 100.0, 1000.0], dtype=np.float64)
        scan = np.array([0, 1, 2], dtype=np.int32)
        rt = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=50.0,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        feat_mz, _, _, _, _, _, _, _, n_features = result

        assert n_features == 1

        weighted_mean = (500.0*100 + 500.001*100 + 500.002*1000) / 1200
        simple_mean = np.mean(mz)

        err_weighted = abs(feat_mz[0] - weighted_mean)
        err_simple = abs(feat_mz[0] - simple_mean)

        assert err_weighted < err_simple, \
            f"Not intensity-weighted! feat={feat_mz[0]:.6f}, weighted={weighted_mean:.6f}, simple={simple_mean:.6f}"
        assert err_weighted < 1e-6, \
            f"Feature m/z {feat_mz[0]} != expected {weighted_mean}"

    def test_min_peaks_filter(self):
        """Features with < min_peaks should be rejected."""
        mz = np.array([500.0, 500.001], dtype=np.float64)
        intensity = np.array([1000, 1000], dtype=np.float64)
        scan = np.array([0, 1], dtype=np.int32)
        rt = np.array([60.0, 60.0], dtype=np.float64)

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, _, _, n_features = result
        assert n_features == 0, f"Should reject feature with only 2 peaks"

    def test_intensity_threshold(self):
        """Peaks below intensity threshold should be ignored."""
        mz = np.array([500.0, 500.001, 500.002], dtype=np.float64)
        intensity = np.array([50, 50, 50], dtype=np.float64)
        scan = np.array([0, 1, 2], dtype=np.int32)
        rt = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, _, _, n_features = result
        assert n_features == 0, f"Should reject feature below intensity threshold"

    @pytest.mark.parametrize("ppm_tol", [5.0, 10.0, 20.0])
    def test_ppm_boundary_inside(self, ppm_tol):
        """Peak just INSIDE ppm tolerance should be included."""
        base_mz = 500.0
        inside_ppm = ppm_tol * 0.9
        offset_mz = base_mz + base_mz * inside_ppm / 1e6

        mz = np.array([base_mz, base_mz, offset_mz], dtype=np.float64)
        intensity = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
        scan = np.array([0, 1, 2], dtype=np.int32)
        rt = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        order = np.argsort(mz)
        mz, intensity, scan, rt = mz[order], intensity[order], scan[order], rt[order]

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=ppm_tol,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, feat_n_peaks, _, n_features = result
        assert n_features == 1, f"Should find 1 feature at {ppm_tol} ppm tolerance"
        assert feat_n_peaks[0] == 3, f"All 3 peaks should be in feature"

    @pytest.mark.parametrize("ppm_tol", [5.0, 10.0, 20.0])
    def test_ppm_boundary_outside(self, ppm_tol):
        """Peak just OUTSIDE ppm tolerance should NOT be included."""
        base_mz = 500.0
        outside_ppm = ppm_tol * 1.2
        offset_mz = base_mz + base_mz * outside_ppm / 1e6

        mz = np.array([base_mz, base_mz, offset_mz], dtype=np.float64)
        intensity = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
        scan = np.array([0, 1, 2], dtype=np.int32)
        rt = np.array([60.0, 60.0, 60.0], dtype=np.float64)

        order = np.argsort(mz)
        mz, intensity, scan, rt = mz[order], intensity[order], scan[order], rt[order]

        result = find_features_numba(
            mz, intensity, scan, rt,
            ppm_tol=ppm_tol,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
            min_peaks=DEFAULT_MIN_PEAKS,
        )

        _, _, _, _, _, _, feat_n_peaks, _, n_features = result
        if n_features == 1:
            assert feat_n_peaks[0] == 2, f"Feature should only have 2 peaks"
        else:
            assert n_features == 0, "No feature if outside tolerance leaves < min_peaks"


# =============================================================================
# Tests for Isotope Pattern Detection
# =============================================================================

class TestIsotopePatterns:
    """Tests for isotope pattern detection and charge state inference."""

    def test_z2_isotope_pattern(self):
        """z=2 peptide: M+1 at 0.5 Da spacing, M+2 at 1.0 Da."""
        base_mz = 500.0
        spacing_z2 = C13_MASS_DIFF / 2

        feature_mz = np.array([
            base_mz,
            base_mz + spacing_z2,
            base_mz + 2 * spacing_z2,
        ], dtype=np.float64)
        feature_rt = np.array([120.0, 120.0, 120.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 600.0, 200.0], dtype=np.float64)

        charge, m1_idx, m1_rt_diff = find_isotope_patterns(
            feature_mz, feature_rt, feature_intensity,
            n_features=3,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
        )

        assert charge[0] == 2, f"Expected z=2, got {charge[0]}"
        assert m1_idx[0] == 1, f"M+1 should be at index 1, got {m1_idx[0]}"

    def test_z3_isotope_pattern(self):
        """z=3 peptide: M+1 at 0.33 Da spacing."""
        base_mz = 400.0
        spacing_z3 = C13_MASS_DIFF / 3

        feature_mz = np.array([
            base_mz,
            base_mz + spacing_z3,
            base_mz + 2 * spacing_z3,
        ], dtype=np.float64)
        feature_rt = np.array([120.0, 120.0, 120.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 700.0, 300.0], dtype=np.float64)

        charge, m1_idx, _ = find_isotope_patterns(
            feature_mz, feature_rt, feature_intensity,
            n_features=3,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
        )

        assert charge[0] == 3, f"Expected z=3, got {charge[0]}"
        assert m1_idx[0] == 1, "M+1 should be at index 1"

    def test_no_isotope_pattern(self):
        """Isolated peak with no M+1 should have unknown charge (0)."""
        feature_mz = np.array([500.0], dtype=np.float64)
        feature_rt = np.array([120.0], dtype=np.float64)
        feature_intensity = np.array([1000.0], dtype=np.float64)

        charge, m1_idx, _ = find_isotope_patterns(
            feature_mz, feature_rt, feature_intensity,
            n_features=1,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
        )

        assert charge[0] == 0, f"No isotope -> charge should be 0, got {charge[0]}"
        assert m1_idx[0] == -1, "No M+1 found"

    def test_isotope_rt_coelution_required(self):
        """M+1 at correct m/z but different RT should NOT match."""
        base_mz = 500.0
        spacing_z2 = C13_MASS_DIFF / 2

        feature_mz = np.array([base_mz, base_mz + spacing_z2], dtype=np.float64)
        feature_rt = np.array([60.0, 180.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 600.0], dtype=np.float64)

        charge, m1_idx, _ = find_isotope_patterns(
            feature_mz, feature_rt, feature_intensity,
            n_features=2,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=5.0,
        )

        assert charge[0] == 0, "Should not match M+1 at different RT"
        assert m1_idx[0] == -1

    def test_multiple_isotope_patterns(self):
        """Multiple independent isotope patterns should each be identified."""
        spacing_z2 = C13_MASS_DIFF / 2
        spacing_z3 = C13_MASS_DIFF / 3

        feature_mz = np.array([
            500.0, 500.0 + spacing_z2, 500.0 + 2 * spacing_z2,
            400.0, 400.0 + spacing_z3, 400.0 + 2 * spacing_z3,
        ], dtype=np.float64)
        feature_rt = np.array([60.0, 60.0, 60.0, 120.0, 120.0, 120.0], dtype=np.float64)
        feature_intensity = np.array([
            1000.0, 600.0, 200.0,
            800.0, 560.0, 240.0,
        ], dtype=np.float64)

        charge, _, _ = find_isotope_patterns(
            feature_mz, feature_rt, feature_intensity,
            n_features=6,
            ppm_tol=DEFAULT_PPM_TOL,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
        )

        assert charge[0] == 2, f"First peptide should be z=2, got {charge[0]}"
        assert charge[3] == 3, f"Second peptide should be z=3, got {charge[3]}"

    def test_isotope_spacing_constant(self):
        """Verify C13_MASS_DIFF is the expected value."""
        expected = 1.003355
        assert abs(C13_MASS_DIFF - expected) < 0.001, \
            f"C13_MASS_DIFF should be ~{expected}, got {C13_MASS_DIFF}"


# =============================================================================
# Tests for Charge Pair Detection
# =============================================================================

class TestChargePairs:
    """Tests for charge pair detection (mass accuracy validation)."""

    def test_charge_pair_neutral_mass_calculation(self):
        """Verify neutral mass calculation from m/z and charge."""
        neutral_mass = 1234.5678

        for charge in [2, 3, 4]:
            mz = (neutral_mass + charge * PROTON_MASS) / charge
            calculated_neutral = mz * charge - charge * PROTON_MASS

            assert abs(calculated_neutral - neutral_mass) < 1e-6, \
                f"z={charge}: calculated {calculated_neutral} != {neutral_mass}"

    def test_perfect_charge_pair(self):
        """Perfect z=2/z=3 pair should be found with zero mass error."""
        neutral_mass = 1000.0
        rt = 120.0

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        feature_mz = np.array([mz_z2, mz_z3], dtype=np.float64)
        feature_rt = np.array([rt, rt], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0], dtype=np.float64)
        charge = np.array([2, 3], dtype=np.int32)

        has_partner, partner_idx, mass_error_da, mass_error_ppm = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=2,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.1,
        )

        assert has_partner[0] == True, "z=2 should have z=3 partner"
        assert partner_idx[0] == 1, "Partner index should be 1"
        assert abs(mass_error_da[0]) < 1e-9, f"Mass error should be ~0, got {mass_error_da[0]}"
        assert abs(mass_error_ppm[0]) < 1e-6, f"PPM error should be ~0, got {mass_error_ppm[0]}"

    def test_charge_pair_with_mass_error(self):
        """Charge pair with realistic mass error should still be found."""
        neutral_mass = 1000.0
        rt = 120.0

        ppm_error = 2.0
        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3_with_error = ((neutral_mass + 3 * PROTON_MASS) / 3) * (1 + ppm_error / 1e6)

        feature_mz = np.array([mz_z2, mz_z3_with_error], dtype=np.float64)
        feature_rt = np.array([rt, rt], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0], dtype=np.float64)
        charge = np.array([2, 3], dtype=np.int32)

        has_partner, partner_idx, mass_error_da, mass_error_ppm = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=2,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.01,
        )

        assert has_partner[0] == True, "Should find partner within tolerance"
        assert mass_error_ppm[0] < 10.0, f"PPM error {mass_error_ppm[0]} should be < 10"

    def test_no_partner_different_rt(self):
        """z=2 and z=3 at different RT should NOT match."""
        neutral_mass = 1000.0

        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        feature_mz = np.array([mz_z2, mz_z3], dtype=np.float64)
        feature_rt = np.array([60.0, 180.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0], dtype=np.float64)
        charge = np.array([2, 3], dtype=np.int32)

        has_partner, partner_idx, _, _ = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=2,
            rt_tol_sec=5.0,
            mass_tol_da=0.1,
        )

        assert has_partner[0] == False, "Should NOT find partner (RT too different)"
        assert partner_idx[0] == -1

    def test_no_partner_different_mass(self):
        """z=2 and z=3 from different peptides should NOT match."""
        neutral_mass_1 = 1000.0
        neutral_mass_2 = 1500.0

        mz_z2 = (neutral_mass_1 + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass_2 + 3 * PROTON_MASS) / 3

        feature_mz = np.array([mz_z2, mz_z3], dtype=np.float64)
        feature_rt = np.array([120.0, 120.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0], dtype=np.float64)
        charge = np.array([2, 3], dtype=np.int32)

        has_partner, partner_idx, _, _ = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=2,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.1,
        )

        assert has_partner[0] == False, "Should NOT find partner (different neutral mass)"
        assert partner_idx[0] == -1

    def test_multiple_pairs(self):
        """Multiple charge pairs should all be found."""
        neutral_mass_1 = 1000.0
        neutral_mass_2 = 1500.0

        mz_z2_1 = (neutral_mass_1 + 2 * PROTON_MASS) / 2
        mz_z3_1 = (neutral_mass_1 + 3 * PROTON_MASS) / 3
        mz_z2_2 = (neutral_mass_2 + 2 * PROTON_MASS) / 2
        mz_z3_2 = (neutral_mass_2 + 3 * PROTON_MASS) / 3

        feature_mz = np.array([mz_z2_1, mz_z3_1, mz_z2_2, mz_z3_2], dtype=np.float64)
        feature_rt = np.array([120.0, 120.0, 180.0, 180.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0, 500.0, 400.0], dtype=np.float64)
        charge = np.array([2, 3, 2, 3], dtype=np.int32)

        has_partner, partner_idx, _, _ = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=4,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.1,
        )

        assert has_partner[0] == True, "First z=2 should have partner"
        assert partner_idx[0] == 1, "First z=2 partner should be index 1"
        assert has_partner[2] == True, "Second z=2 should have partner"
        assert partner_idx[2] == 3, "Second z=2 partner should be index 3"

    def test_no_z3_features(self):
        """No z=3 features -> no pairs possible."""
        feature_mz = np.array([500.0, 600.0], dtype=np.float64)
        feature_rt = np.array([120.0, 120.0], dtype=np.float64)
        feature_intensity = np.array([1000.0, 800.0], dtype=np.float64)
        charge = np.array([2, 2], dtype=np.int32)

        has_partner, partner_idx, _, _ = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=2,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.1,
        )

        assert not has_partner.any(), "No pairs possible without z=3 features"

    def test_empty_input(self):
        """Empty input should return empty results."""
        feature_mz = np.array([], dtype=np.float64)
        feature_rt = np.array([], dtype=np.float64)
        feature_intensity = np.array([], dtype=np.float64)
        charge = np.array([], dtype=np.int32)

        has_partner, partner_idx, mass_error_da, mass_error_ppm = find_charge_pairs(
            feature_mz, feature_rt, feature_intensity, charge,
            n_features=0,
            rt_tol_sec=DEFAULT_RT_TOL_SEC,
            mass_tol_da=0.1,
        )

        assert len(has_partner) == 0
        assert len(partner_idx) == 0


# =============================================================================
# Tests for PPM Calculations
# =============================================================================

class TestPpmCalculations:
    """Tests for PPM error calculations."""

    def test_ppm_to_daltons(self):
        """PPM tolerance conversion to Daltons."""
        mz = 500.0
        ppm = 10.0

        delta_da = mz * ppm / 1e6

        assert abs(delta_da - 0.005) < 1e-9, f"10 ppm at 500 m/z = 0.005 Da"

    def test_ppm_error_calculation(self):
        """PPM error between observed and theoretical."""
        observed = 500.005
        theoretical = 500.0

        ppm_error = (observed - theoretical) / theoretical * 1e6

        assert abs(ppm_error - 10.0) < 1e-6, f"Expected 10 ppm, got {ppm_error}"

    def test_ppm_scales_with_mass(self):
        """Same absolute error = different PPM at different masses."""
        delta_da = 0.01

        ppm_at_500 = delta_da / 500.0 * 1e6
        ppm_at_1000 = delta_da / 1000.0 * 1e6

        assert abs(ppm_at_500 - 20.0) < 1e-6
        assert abs(ppm_at_1000 - 10.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

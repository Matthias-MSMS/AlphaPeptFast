"""Tests for feature quality scoring module.

Tests:
- Base quality score calculation (mass accuracy, FWHM, peak count, etc.)
- Isotope pattern quality bonus
- Charge state consistency bonus
- IsotopeGroup quality scoring
- ConsolidatedFeature quality scoring
- Batch scoring functions
- Quality filtering
"""

import unittest
import numpy as np

from alphapeptfast.features.isotope_grouping import IsotopeGroup, InstrumentType
from alphapeptfast.features.charge_consolidation import (
    ConsolidatedFeature,
    calculate_neutral_mass_scalar,
)
from alphapeptfast.features.quality_scoring import (
    calculate_base_quality_score,
    calculate_isotope_quality,
    calculate_isotope_group_quality,
    calculate_charge_consistency_bonus,
    calculate_consolidated_feature_quality,
    score_isotope_groups,
    score_consolidated_features,
    filter_by_quality,
)
from alphapeptfast.constants import PROTON_MASS


class TestBaseQualityScore(unittest.TestCase):
    """Test base quality score calculation."""

    def test_perfect_feature(self):
        """Test scoring for perfect feature (high quality)."""
        score = calculate_base_quality_score(
            mz_std_ppm=0.5,      # Excellent mass accuracy (30 pts)
            fwhm=5.0,            # Perfect FWHM (30 pts)
            n_peaks=10,          # Many peaks (20 pts)
            n_scans=5,           # Many scans (10 pts)
            intensity=1000000,   # 1000× threshold (10 pts)
            intensity_threshold=1000.0
        )

        # Should get maximum score
        self.assertEqual(score, 100.0)

    def test_mass_accuracy_component(self):
        """Test mass accuracy scoring."""
        # Excellent mass accuracy (< 1 ppm)
        score1 = calculate_base_quality_score(
            mz_std_ppm=0.8, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Good mass accuracy (< 2 ppm)
        score2 = calculate_base_quality_score(
            mz_std_ppm=1.5, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Score1 should be higher due to better mass accuracy
        self.assertGreater(score1, score2)
        self.assertGreater(score1 - score2, 3)  # At least 5 point difference

    def test_fwhm_component(self):
        """Test FWHM (elution shape) scoring."""
        # Ideal FWHM (1-10 seconds)
        score1 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Acceptable FWHM (0.5-20 seconds)
        score2 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=15.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Poor FWHM (> 20 seconds)
        score3 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=25.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Ideal should be best
        self.assertGreater(score1, score2)
        self.assertGreater(score2, score3)

    def test_peak_count_component(self):
        """Test peak count scoring."""
        # Many peaks (≥ 10)
        score1 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=10, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Moderate peaks (≥ 5)
        score2 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # Few peaks (≥ 3)
        score3 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=3, n_scans=3,
            intensity=10000, intensity_threshold=1000.0
        )

        # More peaks = higher score
        self.assertGreater(score1, score2)
        self.assertGreater(score2, score3)

    def test_intensity_component(self):
        """Test intensity relative to threshold scoring."""
        # Very high intensity (>100× threshold) - 10 pts
        score1 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=200000, intensity_threshold=1000.0
        )

        # High intensity (>10× threshold) - 7 pts
        score2 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=15000, intensity_threshold=1000.0
        )

        # Low intensity (>1× threshold) - 3 pts
        score3 = calculate_base_quality_score(
            mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3,
            intensity=2000, intensity_threshold=1000.0
        )

        # Higher intensity = higher score
        self.assertGreater(score1, score2)
        self.assertGreater(score2, score3)


class TestIsotopeQuality(unittest.TestCase):
    """Test isotope pattern quality bonus."""

    def test_no_isotopes(self):
        """Test scoring for feature with no isotopes detected."""
        group = IsotopeGroup(
            m0_idx=0,
            m0_mz=650.0,
            m0_rt=100.0,
            m0_intensity=1e6,
            charge=2,
            has_m1=False,
            has_m2=False
        )

        bonus = calculate_isotope_quality(group)

        # Should get 0 bonus (no isotopes)
        self.assertEqual(bonus, 0.0)

    def test_m1_only(self):
        """Test scoring for feature with M1 isotope."""
        group = IsotopeGroup(
            m0_idx=0,
            m0_mz=650.0,
            m0_rt=100.0,
            m0_intensity=1e6,
            charge=2,
            has_m1=True,
            m0_m1_mass_error_ppm=0.5,  # Excellent mass error
            has_m2=False
        )

        bonus = calculate_isotope_quality(group)

        # Should get M1 bonus (10) + mass error bonus (5) = 15
        self.assertEqual(bonus, 15.0)

    def test_m1_and_m2(self):
        """Test scoring for feature with M1 and M2 isotopes."""
        group = IsotopeGroup(
            m0_idx=0,
            m0_mz=650.0,
            m0_rt=100.0,
            m0_intensity=1e6,
            charge=2,
            has_m1=True,
            m0_m1_mass_error_ppm=0.8,
            has_m2=True
        )

        bonus = calculate_isotope_quality(group)

        # Should get M1 (10) + M2 (5) + mass error (5) = 20 (maximum)
        self.assertEqual(bonus, 20.0)

    def test_mass_error_penalty(self):
        """Test that high mass error reduces bonus."""
        # Good mass error
        group1 = IsotopeGroup(
            m0_idx=0, m0_mz=650.0, m0_rt=100.0, m0_intensity=1e6,
            charge=2, has_m1=True, m0_m1_mass_error_ppm=0.5
        )

        # Poor mass error
        group2 = IsotopeGroup(
            m0_idx=0, m0_mz=650.0, m0_rt=100.0, m0_intensity=1e6,
            charge=2, has_m1=True, m0_m1_mass_error_ppm=10.0
        )

        bonus1 = calculate_isotope_quality(group1)
        bonus2 = calculate_isotope_quality(group2)

        # Good mass error should get higher bonus
        self.assertGreater(bonus1, bonus2)


class TestIsotopeGroupQualityScore(unittest.TestCase):
    """Test complete quality scoring for IsotopeGroup."""

    def test_high_quality_feature(self):
        """Test high-quality feature with isotopes."""
        group = IsotopeGroup(
            m0_idx=0,
            m0_mz=650.0,
            m0_rt=100.0,
            m0_intensity=1e6,
            charge=2,
            has_m1=True,
            m0_m1_mass_error_ppm=0.5,
            has_m2=True
        )

        score = calculate_isotope_group_quality(
            group,
            mz_std_ppm=0.8,
            fwhm=5.0,
            n_peaks=10,
            n_scans=5,
            intensity_threshold=1000.0
        )

        # Should get high score (base + isotope bonus, rescaled)
        self.assertGreater(score, 90.0)

    def test_low_quality_feature(self):
        """Test low-quality feature."""
        group = IsotopeGroup(
            m0_idx=0,
            m0_mz=650.0,
            m0_rt=100.0,
            m0_intensity=1500,  # Low intensity
            charge=2,
            has_m1=False,  # No isotopes
            has_m2=False
        )

        score = calculate_isotope_group_quality(
            group,
            mz_std_ppm=15.0,  # Poor mass accuracy
            fwhm=25.0,        # Poor FWHM
            n_peaks=2,        # Few peaks
            n_scans=1,        # Few scans
            intensity_threshold=1000.0
        )

        # Should get low score
        self.assertLess(score, 30.0)


class TestChargeConsistencyBonus(unittest.TestCase):
    """Test charge state consistency bonus."""

    def test_single_charge_state(self):
        """Test that single charge state gets no bonus."""
        bonus = calculate_charge_consistency_bonus(
            mass_consistency_ppm=1.0,
            n_charge_states=1
        )

        # Should get 0 bonus (single charge)
        self.assertEqual(bonus, 0.0)

    def test_multi_charge_excellent_consistency(self):
        """Test multi-charge with excellent mass consistency."""
        bonus = calculate_charge_consistency_bonus(
            mass_consistency_ppm=1.0,
            n_charge_states=2
        )

        # Should get maximum bonus (10 points)
        self.assertEqual(bonus, 10.0)

    def test_multi_charge_poor_consistency(self):
        """Test multi-charge with poor mass consistency."""
        bonus = calculate_charge_consistency_bonus(
            mass_consistency_ppm=15.0,
            n_charge_states=2
        )

        # Should get no bonus (poor consistency)
        self.assertEqual(bonus, 0.0)


class TestConsolidatedFeatureQualityScore(unittest.TestCase):
    """Test quality scoring for ConsolidatedFeature."""

    def create_test_feature(self):
        """Create a test consolidated feature."""
        neutral_mass = 1300.0
        mz_z2 = (neutral_mass + 2 * PROTON_MASS) / 2
        mz_z3 = (neutral_mass + 3 * PROTON_MASS) / 3

        group_z2 = IsotopeGroup(
            m0_idx=0, m0_mz=mz_z2, m0_rt=100.0, m0_intensity=1e7,
            charge=2, has_m1=True, m0_m1_mass_error_ppm=0.5, has_m2=True
        )

        group_z3 = IsotopeGroup(
            m0_idx=1, m0_mz=mz_z3, m0_rt=100.0, m0_intensity=5e6,
            charge=3, has_m1=True, m0_m1_mass_error_ppm=0.8, has_m2=False
        )

        return ConsolidatedFeature(
            monoisotopic_mass=neutral_mass,
            apex_rt=100.0,
            charge_states=[2, 3],
            mz_by_charge={2: mz_z2, 3: mz_z3},
            intensity_by_charge={2: 1e7, 3: 5e6},
            isotope_groups_by_charge={2: group_z2, 3: group_z3},
            total_intensity=1.5e7,
            best_charge=2,
            mass_consistency_ppm=1.5,
            group_indices=[0, 1]
        )

    def test_high_quality_consolidated_feature(self):
        """Test high-quality consolidated feature."""
        feature = self.create_test_feature()

        score = calculate_consolidated_feature_quality(
            feature,
            mz_std_ppm=0.8,
            fwhm=5.0,
            n_peaks=10,
            n_scans=5,
            intensity_threshold=1000.0
        )

        # Should get very high score (base + isotope + charge bonus)
        self.assertGreater(score, 90.0)

    def test_charge_consistency_improves_score(self):
        """Test that good charge consistency improves score."""
        # Good consistency
        feature1 = self.create_test_feature()
        feature1.mass_consistency_ppm = 1.0

        # Poor consistency
        feature2 = self.create_test_feature()
        feature2.mass_consistency_ppm = 15.0

        score1 = calculate_consolidated_feature_quality(
            feature1, mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3
        )

        score2 = calculate_consolidated_feature_quality(
            feature2, mz_std_ppm=1.0, fwhm=5.0, n_peaks=5, n_scans=3
        )

        # Better consistency = higher score
        self.assertGreater(score1, score2)


class TestBatchScoring(unittest.TestCase):
    """Test batch scoring functions."""

    def test_score_isotope_groups_batch(self):
        """Test batch scoring for isotope groups."""
        # Create 3 test groups
        groups = [
            IsotopeGroup(
                m0_idx=i, m0_mz=650.0 + i, m0_rt=100.0, m0_intensity=1e6,
                charge=2, has_m1=True, m0_m1_mass_error_ppm=0.5, has_m2=(i > 0)
            )
            for i in range(3)
        ]

        mz_std_ppm = np.array([0.8, 1.5, 2.0])
        fwhm = np.array([5.0, 5.0, 5.0])
        n_peaks = np.array([10, 5, 3])
        n_scans = np.array([5, 3, 2])

        scores = score_isotope_groups(
            groups, mz_std_ppm, fwhm, n_peaks, n_scans,
            intensity_threshold=1000.0
        )

        # Should return array of 3 scores
        self.assertEqual(len(scores), 3)

        # All scores should be 0-100
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 100))

        # First group should have highest score (best parameters)
        self.assertEqual(np.argmax(scores), 0)


class TestQualityFiltering(unittest.TestCase):
    """Test quality-based filtering."""

    def test_filter_by_quality(self):
        """Test filtering features by quality threshold."""
        groups = [
            IsotopeGroup(
                m0_idx=i, m0_mz=650.0 + i, m0_rt=100.0, m0_intensity=1e6,
                charge=2, has_m1=True, m0_m1_mass_error_ppm=0.5
            )
            for i in range(5)
        ]

        # Scores: 90, 80, 70, 60, 50
        quality_scores = np.array([90.0, 80.0, 70.0, 60.0, 50.0])

        # Filter with threshold 70
        filtered = filter_by_quality(groups, quality_scores, min_quality=70.0)

        # Should keep only first 3 features
        self.assertEqual(len(filtered), 3)


class TestScoreRanges(unittest.TestCase):
    """Test that all scores are within valid ranges."""

    def test_base_score_range(self):
        """Test that base score stays in 0-100 range."""
        # Test extreme values
        test_cases = [
            # (mz_std, fwhm, n_peaks, n_scans, intensity)
            (0.1, 5.0, 20, 10, 1e9),  # Perfect
            (50.0, 100.0, 1, 1, 10),  # Terrible
            (5.0, -1.0, 5, 3, 1e5),   # Missing FWHM
        ]

        for mz_std, fwhm, n_peaks, n_scans, intensity in test_cases:
            score = calculate_base_quality_score(
                mz_std, fwhm, n_peaks, n_scans, intensity, 1000.0
            )

            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_isotope_quality_range(self):
        """Test that isotope quality stays in 0-20 range."""
        # Maximum bonus
        group = IsotopeGroup(
            m0_idx=0, m0_mz=650.0, m0_rt=100.0, m0_intensity=1e6,
            charge=2, has_m1=True, m0_m1_mass_error_ppm=0.5, has_m2=True
        )

        bonus = calculate_isotope_quality(group)

        self.assertGreaterEqual(bonus, 0.0)
        self.assertLessEqual(bonus, 20.0)


if __name__ == '__main__':
    unittest.main()

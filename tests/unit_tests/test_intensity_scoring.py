"""Tests for fragment intensity scoring module.

CRITICAL: These tests verify that tuple-based alignment correctly matches
fragments between our generator and AlphaPeptDeep predictions, fixing the
ordering bug that prevented intensity scoring from working.
"""

import unittest
from pathlib import Path

import numpy as np

from alphapeptfast.fragments.generator import encode_peptide_to_ord, generate_by_ions
from alphapeptfast.scoring.intensity_scoring import (
    AlphaPeptDeepLoader,
    IntensityScorer,
    normalize_intensities,
    pearson_correlation,
)


class TestNormalization(unittest.TestCase):
    """Test intensity normalization."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        intensities = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_intensities(intensities)

        np.testing.assert_allclose(normalized, [0.2, 0.4, 0.6, 0.8, 1.0])

    def test_normalize_empty(self):
        """Test empty array."""
        intensities = np.array([])
        normalized = normalize_intensities(intensities)

        self.assertEqual(len(normalized), 0)

    def test_normalize_zeros(self):
        """Test all zeros."""
        intensities = np.array([0.0, 0.0, 0.0])
        normalized = normalize_intensities(intensities)

        np.testing.assert_array_equal(normalized, [0.0, 0.0, 0.0])

    def test_normalize_preserves_ratios(self):
        """Test that normalization preserves intensity ratios."""
        intensities = np.array([100.0, 50.0, 25.0])
        normalized = normalize_intensities(intensities)

        # Ratios should be preserved
        ratio_original = intensities[1] / intensities[0]
        ratio_normalized = normalized[1] / normalized[0]

        self.assertAlmostEqual(ratio_original, ratio_normalized, places=6)


class TestPearsonCorrelation(unittest.TestCase):
    """Test Pearson correlation calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2*x

        corr = pearson_correlation(x, y)

        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_negative_correlation(self):
        """Test perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # y = -x + 6

        corr = pearson_correlation(x, y)

        self.assertAlmostEqual(corr, -1.0, places=6)

    def test_no_correlation(self):
        """Test no correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant

        corr = pearson_correlation(x, y)

        self.assertAlmostEqual(corr, 0.0, places=6)

    def test_insufficient_data(self):
        """Test with too few points."""
        x = np.array([1.0])
        y = np.array([2.0])

        corr = pearson_correlation(x, y)

        self.assertEqual(corr, 0.0)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        corr = pearson_correlation(x, y)

        self.assertEqual(corr, 0.0)


class TestTupleBasedAlignment(unittest.TestCase):
    """CRITICAL: Test tuple-based alignment fixes ordering bug."""

    def test_alignment_concept(self):
        """Test that tuple-based alignment works correctly.

        This is the fix for the ordering bug. Our generator organizes
        fragments by TYPE (all b-ions, then y-ions), but AlphaPeptDeep
        organizes by POSITION (all ion types per position).

        Tuple-based matching ensures we compare the right fragments!
        """
        # Simulate predictions organized by position (AlphaPeptDeep style)
        predictions = {
            ('b', 1, 1): (100.0, 0.01),  # b1+ (low intensity)
            ('b', 2, 1): (200.0, 0.50),  # b2+ (good)
            ('b', 2, 2): (100.5, 0.01),  # b2++ (physically unlikely)
            ('b', 3, 1): (300.0, 0.30),  # b3+ (good)
            ('y', 1, 1): (175.0, 1.00),  # y1+ (strong)
            ('y', 2, 1): (275.0, 0.80),  # y2+ (strong)
            ('y', 3, 1): (375.0, 0.60),  # y3+ (good)
        }

        # Simulate our generator (organized by type)
        # Note: b2++ might be skipped by charge > position constraint
        our_fragments = [
            (0, 1, 1, 100.0),   # (type=b, pos=1, charge=1, mz)
            (0, 2, 1, 200.0),   # b2+
            (0, 3, 1, 300.0),   # b3+ (skipped b2++ and b3++)
            (1, 1, 1, 175.0),   # y1+
            (1, 2, 1, 275.0),   # y2+
            (1, 3, 1, 375.0),   # y3+
        ]

        # Tuple-based alignment
        matched_count = 0
        for frag_type, frag_pos, frag_charge, frag_mz in our_fragments:
            ion_type = 'b' if frag_type == 0 else 'y'
            key = (ion_type, frag_pos, frag_charge)

            if key in predictions:
                pred_mz, pred_intensity = predictions[key]
                # Check that m/z matches (within tolerance)
                self.assertAlmostEqual(pred_mz, frag_mz, places=1)
                matched_count += 1

        # Should match 6 fragments (all in our list)
        self.assertEqual(matched_count, 6)

    def test_alignment_with_skipped_charges(self):
        """Test alignment when some 2+ charges are skipped.

        This specifically tests the scenario where:
        - b2++ is IMPOSSIBLE (charge > position for low fragments)
        - Our generator skips it
        - But index-based alignment would fail
        - Tuple-based alignment succeeds
        """
        # Predictions (AlphaPeptDeep provides all, even if intensity=0)
        predictions = {
            ('b', 1, 1): (100.0, 0.01),
            ('b', 1, 2): (50.5, 0.00),   # Impossible, intensity=0
            ('b', 2, 1): (200.0, 0.50),
            ('b', 2, 2): (100.5, 0.00),  # Unlikely, intensity=0
            ('b', 3, 1): (300.0, 0.30),
            ('b', 3, 2): (150.5, 0.01),  # Still unlikely at position 3
            ('b', 7, 1): (700.0, 0.20),
            ('b', 7, 2): (350.5, 0.40),  # Now realistic!
        }

        # Our generator (skips charges where charge > position)
        # For a peptide, we might generate: b1+, b2+, b3+, ..., b7+, b7++
        # Note: b7++ is the first 2+ b-ion (realistic for larger fragments)
        our_fragments = [
            (0, 1, 1, 100.0),  # b1+
            (0, 2, 1, 200.0),  # b2+
            (0, 3, 1, 300.0),  # b3+
            (0, 7, 1, 700.0),  # b7+
            (0, 7, 2, 350.5),  # b7++ (first 2+ ion)
        ]

        # With INDEX-based alignment (the bug):
        # Index 0: pred b1+ → our b1+    ✓
        # Index 1: pred b1++ → our b2+   ✗ WRONG!
        # Index 2: pred b2+ → our b3+    ✗ WRONG!
        # Everything misaligned!

        # With TUPLE-based alignment (the fix):
        matches = {}
        for frag_type, frag_pos, frag_charge, frag_mz in our_fragments:
            ion_type = 'b' if frag_type == 0 else 'y'
            key = (ion_type, frag_pos, frag_charge)

            if key in predictions:
                pred_mz, pred_intensity = predictions[key]
                matches[key] = (pred_mz, pred_intensity, frag_mz)

        # All 5 fragments should match correctly
        self.assertEqual(len(matches), 5)

        # Verify correct alignments
        self.assertIn(('b', 1, 1), matches)
        self.assertIn(('b', 2, 1), matches)
        self.assertIn(('b', 3, 1), matches)
        self.assertIn(('b', 7, 1), matches)
        self.assertIn(('b', 7, 2), matches)

        # Verify m/z values match
        for key, (pred_mz, pred_int, our_mz) in matches.items():
            self.assertAlmostEqual(pred_mz, our_mz, places=1)


class TestAlphaPeptDeepLoader(unittest.TestCase):
    """Test AlphaPeptDeep prediction loader (if HDF5 available)."""

    def setUp(self):
        """Set up test - skip if no test HDF5 available."""
        self.test_hdf = Path("/Users/matthiasmann/Documents/projects/AlphaNovo_RT_projection/alphadia_7min_retest_defaults/library/speclib.hdf")

        if not self.test_hdf.exists():
            self.skipTest(f"Test HDF5 not available: {self.test_hdf}")

    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = AlphaPeptDeepLoader(self.test_hdf)
        self.assertIsNotNone(loader)

    def test_load_peptide_returns_dict(self):
        """Test loading a peptide returns predictions dict."""
        loader = AlphaPeptDeepLoader(self.test_hdf)

        # AASLSER should be in library (we saw it earlier)
        predictions = loader.load_peptide_predictions("AASLSER", charge=2)

        # Should return a dict
        self.assertIsInstance(predictions, dict)

        # Keys should be tuples
        if predictions:
            first_key = next(iter(predictions.keys()))
            self.assertIsInstance(first_key, tuple)
            self.assertEqual(len(first_key), 3)  # (type, position, charge)

    def test_tuple_keys_correct_format(self):
        """Test that tuple keys have correct format."""
        loader = AlphaPeptDeepLoader(self.test_hdf)
        predictions = loader.load_peptide_predictions("AASLSER", charge=2)

        for key, (mz, intensity) in predictions.items():
            ion_type, position, charge = key

            # Ion type should be 'b' or 'y'
            self.assertIn(ion_type, ['b', 'y'])

            # Position should be positive integer
            self.assertGreater(position, 0)
            self.assertIsInstance(position, int)

            # Charge should be 1 or 2
            self.assertIn(charge, [1, 2])

            # m/z should be positive
            self.assertGreater(mz, 0.0)

            # Intensity should be >= 0
            self.assertGreaterEqual(intensity, 0.0)

    def test_only_significant_intensities(self):
        """Test that only fragments with intensity > threshold are stored."""
        loader = AlphaPeptDeepLoader(self.test_hdf)
        predictions = loader.load_peptide_predictions("AASLSER", charge=2)

        # All stored intensities should be > 0.01
        for key, (mz, intensity) in predictions.items():
            self.assertGreater(intensity, 0.01)

    def test_caching_works(self):
        """Test that predictions are cached."""
        loader = AlphaPeptDeepLoader(self.test_hdf)

        # Load twice
        pred1 = loader.load_peptide_predictions("AASLSER", charge=2)
        pred2 = loader.load_peptide_predictions("AASLSER", charge=2)

        # Should be same object (cached)
        self.assertIs(pred1, pred2)


class TestIntensityScorer(unittest.TestCase):
    """Test IntensityScorer with synthetic data."""

    def test_scorer_initialization(self):
        """Test scorer can be initialized."""
        # Use dummy path (won't actually load)
        scorer = IntensityScorer("dummy.hdf")
        self.assertIsNotNone(scorer)

    def test_score_match_synthetic(self):
        """Test scoring with synthetic data."""
        # Create mock scorer with manual predictions
        scorer = IntensityScorer("dummy.hdf")

        # Manually set predictions
        scorer.loader.cache["PEPTIDE_2"] = {
            ('b', 2, 1): (200.0, 0.50),
            ('b', 3, 1): (300.0, 0.30),
            ('y', 2, 1): (275.0, 0.80),
            ('y', 3, 1): (375.0, 0.60),
        }

        # Create synthetic observed spectrum (MUST BE SORTED for binary search!)
        observed_mz = np.array([200.0, 275.0, 300.0, 375.0])
        observed_intensity = np.array([0.50, 0.80, 0.30, 0.60])

        # Create theoretical fragments (from generator)
        fragment_mz = np.array([200.0, 300.0, 275.0, 375.0])
        fragment_type = np.array([0, 0, 1, 1])  # b, b, y, y
        fragment_position = np.array([2, 3, 2, 3])
        fragment_charge = np.array([1, 1, 1, 1])

        # Score
        result = scorer.score_match(
            peptide="PEPTIDE",
            charge=2,
            observed_mz=observed_mz,
            observed_intensity=observed_intensity,
            fragment_mz=fragment_mz,
            fragment_type=fragment_type,
            fragment_position=fragment_position,
            fragment_charge=fragment_charge,
            mz_tolerance_ppm=20.0,
        )

        # Perfect match should give correlation ~1.0
        self.assertGreater(result['correlation'], 0.95)
        self.assertEqual(result['n_matched'], 4)
        self.assertEqual(result['coverage'], 1.0)

    def test_score_no_predictions(self):
        """Test scoring when no predictions available."""
        scorer = IntensityScorer("dummy.hdf")

        # Manually set empty cache (avoid file access)
        scorer.loader.cache["UNKNOWN_2"] = {}

        observed_mz = np.array([200.0, 300.0])
        observed_intensity = np.array([0.5, 0.3])
        fragment_mz = np.array([200.0, 300.0])
        fragment_type = np.array([0, 0])
        fragment_position = np.array([2, 3])
        fragment_charge = np.array([1, 1])

        result = scorer.score_match(
            peptide="UNKNOWN",
            charge=2,
            observed_mz=observed_mz,
            observed_intensity=observed_intensity,
            fragment_mz=fragment_mz,
            fragment_type=fragment_type,
            fragment_position=fragment_position,
            fragment_charge=fragment_charge,
        )

        # Should return zeros
        self.assertEqual(result['correlation'], 0.0)
        self.assertEqual(result['n_matched'], 0)

    def test_score_insufficient_matches(self):
        """Test scoring with too few matches for correlation."""
        scorer = IntensityScorer("dummy.hdf")

        # Only 2 predictions (need 3+ for correlation)
        scorer.loader.cache["PEPTIDE_2"] = {
            ('b', 2, 1): (200.0, 0.50),
            ('b', 3, 1): (300.0, 0.30),
        }

        # MUST BE SORTED for binary search
        observed_mz = np.array([200.0, 300.0])
        observed_intensity = np.array([0.5, 0.3])
        fragment_mz = np.array([200.0, 300.0])
        fragment_type = np.array([0, 0])
        fragment_position = np.array([2, 3])
        fragment_charge = np.array([1, 1])

        result = scorer.score_match(
            peptide="PEPTIDE",
            charge=2,
            observed_mz=observed_mz,
            observed_intensity=observed_intensity,
            fragment_mz=fragment_mz,
            fragment_type=fragment_type,
            fragment_position=fragment_position,
            fragment_charge=fragment_charge,
        )

        # Should have matches but no correlation (need 3+)
        self.assertEqual(result['n_matched'], 2)
        self.assertEqual(result['correlation'], 0.0)


class TestIntegrationWithGenerator(unittest.TestCase):
    """Test integration with fragment generator."""

    def test_generator_fragments_align_with_predictions(self):
        """Test that fragments from generator can be aligned with predictions.

        This is the ultimate test: generate fragments with our generator,
        create mock predictions, and verify alignment works correctly.
        """
        # Generate fragments for a peptide
        peptide = "PEPTIDER"
        peptide_ord = encode_peptide_to_ord(peptide)
        fragment_mz, fragment_type, fragment_position, fragment_charge = generate_by_ions(
            peptide_ord, precursor_charge=2
        )

        # Create mock predictions matching the generated fragments
        predictions = {}
        for i in range(len(fragment_mz)):
            ion_type = 'b' if fragment_type[i] == 0 else 'y'
            key = (ion_type, int(fragment_position[i]), int(fragment_charge[i]))
            # Mock intensity (use position as proxy for intensity)
            mock_intensity = float(fragment_position[i]) / 10.0
            predictions[key] = (float(fragment_mz[i]), mock_intensity)

        # Verify we can look up each generated fragment
        for i in range(len(fragment_mz)):
            ion_type = 'b' if fragment_type[i] == 0 else 'y'
            key = (ion_type, int(fragment_position[i]), int(fragment_charge[i]))

            self.assertIn(key, predictions)
            pred_mz, pred_intensity = predictions[key]
            self.assertAlmostEqual(pred_mz, fragment_mz[i], places=3)


if __name__ == "__main__":
    unittest.main()

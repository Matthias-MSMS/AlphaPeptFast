"""Unit tests for kernel learning functions.

Tests cover:
1. measure_profile_fwhm - FWHM measurement via interpolation
2. measure_profile_center - Intensity-weighted center
3. detect_flat_top - Trapezoid detection
4. learn_kernel_from_profiles - End-to-end kernel learning
"""

import numpy as np
import pytest

from alphapeptfast.calibration import (
    LearnedKernel,
    learn_kernel_from_profiles,
    measure_profile_fwhm,
    measure_profile_center,
    detect_flat_top,
)


class TestMeasureProfileFWHM:
    """Tests for measure_profile_fwhm function."""

    def test_gaussian_profile(self):
        """Test FWHM measurement on a Gaussian profile."""
        # Create Gaussian with known FWHM = 2.355 * sigma
        sigma = 2.0
        expected_fwhm = 2.355 * sigma  # ~4.71

        positions = np.linspace(-10, 10, 101)
        intensities = np.exp(-0.5 * (positions / sigma) ** 2) * 1000

        fwhm = measure_profile_fwhm(positions, intensities)

        # Should be close to expected
        assert abs(fwhm - expected_fwhm) < 0.2

    def test_narrow_gaussian(self):
        """Test FWHM on narrow Gaussian (similar to Q1 kernel)."""
        # FWHM = 2.8 Da, typical for quadrupole
        sigma = 2.8 / 2.355  # ~1.19
        expected_fwhm = 2.8

        positions = np.linspace(-5, 5, 51)
        intensities = np.exp(-0.5 * (positions / sigma) ** 2) * 10000

        fwhm = measure_profile_fwhm(positions, intensities)

        assert abs(fwhm - expected_fwhm) < 0.15

    def test_wide_profile(self):
        """Test FWHM on wide profile (like 20 Th Q1 window)."""
        # FWHM = 16 Da
        sigma = 16 / 2.355
        expected_fwhm = 16

        positions = np.linspace(-20, 20, 101)
        intensities = np.exp(-0.5 * (positions / sigma) ** 2) * 5000

        fwhm = measure_profile_fwhm(positions, intensities)

        assert abs(fwhm - expected_fwhm) < 1.0

    def test_sparse_sampling(self):
        """Test FWHM with sparse sampling (5-6 points like Q1 data)."""
        # FWHM = 16 Da, but only 5 points at 4 Da spacing
        sigma = 16 / 2.355
        expected_fwhm = 16

        positions = np.array([-8, -4, 0, 4, 8], dtype=np.float32)
        intensities = np.exp(-0.5 * (positions / sigma) ** 2) * 10000

        fwhm = measure_profile_fwhm(positions, intensities)

        # With sparse sampling, expect larger error
        assert abs(fwhm - expected_fwhm) < 3.0

    def test_too_few_points(self):
        """Test FWHM returns NaN with too few points."""
        positions = np.array([0.0, 1.0])
        intensities = np.array([100.0, 50.0])

        fwhm = measure_profile_fwhm(positions, intensities)

        assert np.isnan(fwhm)

    def test_no_signal(self):
        """Test FWHM returns NaN with no signal."""
        positions = np.linspace(0, 10, 11)
        intensities = np.zeros(11)

        fwhm = measure_profile_fwhm(positions, intensities)

        assert np.isnan(fwhm)

    def test_flat_profile(self):
        """Test FWHM on flat profile (all above half-max)."""
        positions = np.linspace(0, 10, 11)
        intensities = np.ones(11) * 1000

        fwhm = measure_profile_fwhm(positions, intensities)

        # Should span full range
        assert abs(fwhm - 10.0) < 0.5


class TestMeasureProfileCenter:
    """Tests for measure_profile_center function."""

    def test_symmetric_profile(self):
        """Test center of symmetric profile at origin."""
        positions = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        intensities = np.array([100, 500, 1000, 500, 100], dtype=np.float32)

        center = measure_profile_center(positions, intensities)

        assert abs(center - 0.0) < 0.01

    def test_shifted_profile(self):
        """Test center of profile shifted from expected."""
        # Profile centered at +0.5
        positions = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        intensities = np.array([100, 500, 1000, 500, 100], dtype=np.float32)

        center = measure_profile_center(positions, intensities)

        # Intensity-weighted center should be at 2.0
        assert abs(center - 2.0) < 0.01

    def test_asymmetric_profile(self):
        """Test center of asymmetric profile."""
        positions = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        intensities = np.array([100, 200, 1000, 300, 100], dtype=np.float32)

        center = measure_profile_center(positions, intensities)

        # Should be pulled toward the high intensity at position 2
        assert 1.5 < center < 2.5

    def test_no_signal(self):
        """Test center returns NaN with no signal."""
        positions = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        intensities = np.zeros(5, dtype=np.float32)

        center = measure_profile_center(positions, intensities)

        assert np.isnan(center)


class TestDetectFlatTop:
    """Tests for detect_flat_top function."""

    def test_trapezoid_profile(self):
        """Test flat top detection on trapezoidal profile."""
        # Trapezoid: flat from -5 to +5, flanks outside
        positions = np.array([-10, -7, -5, -2, 0, 2, 5, 7, 10], dtype=np.float32)
        intensities = np.array([100, 500, 1000, 1000, 1000, 1000, 1000, 500, 100], dtype=np.float32)

        flat_top = detect_flat_top(positions, intensities, threshold=0.9)

        # Flat top should span from -5 to +5 = 10 units
        assert flat_top >= 7.0  # Allow some tolerance

    def test_gaussian_no_flat_top(self):
        """Test no flat top detected on Gaussian."""
        sigma = 2.0
        positions = np.linspace(-6, 6, 25)
        intensities = np.exp(-0.5 * (positions / sigma) ** 2) * 1000

        flat_top = detect_flat_top(positions, intensities, threshold=0.9)

        # Gaussian should have minimal flat top (at most 1-2 grid spacings)
        fwhm = 2.355 * sigma
        assert flat_top < fwhm * 0.3  # Flat top should be small relative to FWHM

    def test_narrow_peak(self):
        """Test no flat top on narrow peak."""
        positions = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        intensities = np.array([100, 500, 1000, 500, 100], dtype=np.float32)

        flat_top = detect_flat_top(positions, intensities, threshold=0.9)

        # Single peak above threshold
        assert flat_top == 0.0

    def test_empty_profile(self):
        """Test flat top returns 0 on empty profile."""
        positions = np.array([0, 1], dtype=np.float32)
        intensities = np.array([0, 0], dtype=np.float32)

        flat_top = detect_flat_top(positions, intensities)

        assert flat_top == 0.0


class TestLearnKernelFromProfiles:
    """Tests for learn_kernel_from_profiles function."""

    def test_learn_gaussian_kernel(self):
        """Test learning Gaussian kernel from synthetic profiles."""
        # Create 20 Gaussian profiles with known parameters
        true_fwhm = 2.8  # Da
        true_offset = 0.1  # Da
        sigma = true_fwhm / 2.355

        np.random.seed(42)
        n_profiles = 20

        positions_list = []
        intensities_list = []
        expected_centers = []

        for i in range(n_profiles):
            # Expected center with small offset
            expected = 500.0 + i * 10  # Different precursor m/z
            expected_centers.append(expected)

            # Actual center is offset
            actual_center = expected + true_offset

            # Create profile
            pos = np.linspace(expected - 5, expected + 5, 21)
            inten = np.exp(-0.5 * ((pos - actual_center) / sigma) ** 2) * 10000
            # Add noise
            inten += np.random.randn(len(inten)) * 100

            positions_list.append(pos)
            intensities_list.append(inten)

        kernel = learn_kernel_from_profiles(
            positions_list,
            intensities_list,
            np.array(expected_centers),
        )

        # Check learned parameters
        assert abs(kernel.fwhm - true_fwhm) < 0.3
        assert abs(kernel.offset - true_offset) < 0.05
        assert kernel.kernel_type == "gaussian"
        assert kernel.n_profiles >= 18  # Most should be valid

    def test_learn_wide_kernel(self):
        """Test learning wide kernel (like 20 Th Q1)."""
        # Create profiles with FWHM = 16 Da
        true_fwhm = 16.0
        true_offset = 0.5
        sigma = true_fwhm / 2.355

        np.random.seed(123)
        n_profiles = 30

        positions_list = []
        intensities_list = []
        expected_centers = []

        for i in range(n_profiles):
            expected = 600.0 + i * 5
            expected_centers.append(expected)
            actual_center = expected + true_offset

            pos = np.linspace(expected - 20, expected + 20, 41)
            inten = np.exp(-0.5 * ((pos - actual_center) / sigma) ** 2) * 5000
            inten += np.random.randn(len(inten)) * 50

            positions_list.append(pos)
            intensities_list.append(inten)

        kernel = learn_kernel_from_profiles(
            positions_list,
            intensities_list,
            np.array(expected_centers),
        )

        assert abs(kernel.fwhm - true_fwhm) < 1.5
        assert abs(kernel.offset - true_offset) < 0.2

    def test_sparse_sampling(self):
        """Test kernel learning with sparse sampling (5-6 points)."""
        # Simulate real Q1 data: 5 points at 4 Da spacing
        true_fwhm = 16.0
        true_offset = 0.0
        sigma = true_fwhm / 2.355

        np.random.seed(456)
        n_profiles = 50

        positions_list = []
        intensities_list = []
        expected_centers = []

        for i in range(n_profiles):
            expected = 500.0 + i * 2
            expected_centers.append(expected)

            # Only 5-6 points at 4 Da spacing
            pos = np.array([expected - 8, expected - 4, expected, expected + 4, expected + 8])
            inten = np.exp(-0.5 * ((pos - expected) / sigma) ** 2) * 10000
            inten += np.random.randn(len(inten)) * 200

            positions_list.append(pos)
            intensities_list.append(np.maximum(inten, 0))

        kernel = learn_kernel_from_profiles(
            positions_list,
            intensities_list,
            np.array(expected_centers),
        )

        # With sparse sampling, expect larger uncertainty
        assert abs(kernel.fwhm - true_fwhm) < 3.0
        assert kernel.fwhm_std > 0  # Should have uncertainty
        assert kernel.n_profiles >= 40

    def test_insufficient_profiles(self):
        """Test error with insufficient valid profiles."""
        positions_list = [np.array([0, 1]), np.array([0, 1])]
        intensities_list = [np.array([0, 0]), np.array([0, 0])]
        expected_centers = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="Insufficient"):
            learn_kernel_from_profiles(
                positions_list,
                intensities_list,
                expected_centers,
            )

    def test_mismatched_lengths(self):
        """Test error with mismatched input lengths."""
        positions_list = [np.array([0, 1, 2])]
        intensities_list = [np.array([100, 200, 100]), np.array([100, 200, 100])]
        expected_centers = np.array([1.0])

        with pytest.raises(ValueError, match="same length"):
            learn_kernel_from_profiles(
                positions_list,
                intensities_list,
                expected_centers,
            )


class TestLearnedKernelDataclass:
    """Tests for LearnedKernel dataclass."""

    def test_sigma_property(self):
        """Test sigma property derived from FWHM."""
        kernel = LearnedKernel(
            fwhm=2.355,
            offset=0.0,
            fwhm_std=0.1,
            offset_std=0.01,
            n_profiles=10,
        )

        assert abs(kernel.sigma - 1.0) < 0.001

    def test_repr(self):
        """Test string representation."""
        kernel = LearnedKernel(
            fwhm=2.8,
            offset=0.1,
            fwhm_std=0.2,
            offset_std=0.05,
            n_profiles=100,
            kernel_type="gaussian",
        )

        repr_str = repr(kernel)
        assert "2.800" in repr_str
        assert "gaussian" in repr_str
        assert "n=100" in repr_str


class TestIntegration:
    """Integration tests combining kernel learning functions."""

    def test_q1_kernel_workflow(self):
        """Test typical Q1 kernel learning workflow."""
        # Simulate extracting Q1 profiles and learning kernel
        np.random.seed(789)

        # Parameters similar to ZenoTOF 20 Th
        true_fwhm = 16.0
        true_offset = 0.1
        sigma = true_fwhm / 2.355

        # Create 100 profiles
        positions_list = []
        intensities_list = []
        expected_centers = []

        for i in range(100):
            precursor_mz = 400 + i * 6  # Range 400-1000 m/z
            expected_centers.append(precursor_mz)

            # Q1 values at 4 Da spacing around precursor
            q1_vals = np.arange(precursor_mz - 12, precursor_mz + 13, 4)
            actual_center = precursor_mz + true_offset

            # Gaussian profile with noise
            inten = np.exp(-0.5 * ((q1_vals - actual_center) / sigma) ** 2)
            inten = inten * (5000 + np.random.randn() * 500)
            inten = np.maximum(inten + np.random.randn(len(inten)) * 100, 0)

            positions_list.append(q1_vals.astype(np.float32))
            intensities_list.append(inten.astype(np.float32))

        # Learn kernel
        kernel = learn_kernel_from_profiles(
            positions_list,
            intensities_list,
            np.array(expected_centers),
        )

        # Verify learned parameters
        assert 12.0 < kernel.fwhm < 20.0  # Reasonable range
        assert abs(kernel.offset - true_offset) < 0.3
        assert kernel.n_profiles >= 90
        assert kernel.fwhm_std > 0
        assert kernel.offset_std > 0

    def test_rt_kernel_workflow(self):
        """Test typical RT kernel learning workflow."""
        # Simulate RT profiles (chromatographic peaks)
        np.random.seed(321)

        true_fwhm = 20.0  # seconds
        true_offset = 0.5  # seconds
        sigma = true_fwhm / 2.355

        positions_list = []
        intensities_list = []
        expected_centers = []

        for i in range(50):
            apex_rt = 300 + i * 20  # RT from 300 to 1300 seconds
            expected_centers.append(apex_rt)

            # RT values around apex
            rt_vals = np.linspace(apex_rt - 40, apex_rt + 40, 41)
            actual_center = apex_rt + true_offset

            # Gaussian peak
            inten = np.exp(-0.5 * ((rt_vals - actual_center) / sigma) ** 2)
            inten = inten * (10000 + np.random.randn() * 1000)
            inten = np.maximum(inten + np.random.randn(len(inten)) * 200, 0)

            positions_list.append(rt_vals.astype(np.float32))
            intensities_list.append(inten.astype(np.float32))

        kernel = learn_kernel_from_profiles(
            positions_list,
            intensities_list,
            np.array(expected_centers),
        )

        assert 15.0 < kernel.fwhm < 25.0
        assert abs(kernel.offset - true_offset) < 1.0
        assert kernel.n_profiles >= 45

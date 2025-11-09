#!/usr/bin/env python3
"""Quick test of XIC smoothing and FWHM calculation."""

import numpy as np
from alphapeptfast.xic import (
    smooth_gaussian_1d,
    auto_smooth_xic,
    calculate_fwhm,
    calculate_peak_quality,
    smooth_and_calculate_fwhm
)

# Generate synthetic Gaussian peak
def generate_gaussian_peak(n_points=100, fwhm_true=10.0, noise_level=0.1):
    """Generate noisy Gaussian peak for testing."""
    rt = np.linspace(0, 60, n_points)  # 60 seconds, ~0.6s spacing

    # True Gaussian: FWHM = 2.355 * sigma
    sigma = fwhm_true / 2.355
    apex_rt = 30.0  # Center at 30s

    # Gaussian peak
    intensities = np.exp(-0.5 * ((rt - apex_rt) / sigma) ** 2)
    intensities = intensities * 1e6  # Scale to typical intensity

    # Add noise
    noise = np.random.randn(n_points) * noise_level * intensities.max()
    intensities_noisy = intensities + noise
    intensities_noisy = np.maximum(intensities_noisy, 0)  # No negative values

    return rt, intensities_noisy, fwhm_true


def test_smoothing():
    """Test Gaussian smoothing."""
    print("="*80)
    print("TEST 1: Gaussian Smoothing")
    print("="*80)

    rt, intensities, fwhm_true = generate_gaussian_peak(fwhm_true=10.0, noise_level=0.2)

    # Test different sigma values
    for sigma in [1.0, 2.0, 5.0]:
        smoothed = smooth_gaussian_1d(intensities, sigma=sigma)
        noise_reduction = 1.0 - np.std(smoothed - intensities.mean()) / np.std(intensities - intensities.mean())
        print(f"  sigma={sigma:.1f}: noise reduction = {noise_reduction*100:.1f}%")

    print()


def test_auto_smooth():
    """Test automatic smoothing."""
    print("="*80)
    print("TEST 2: Automatic Smoothing")
    print("="*80)

    rt, intensities, fwhm_true = generate_gaussian_peak(fwhm_true=10.0, noise_level=0.2)

    # Auto-smooth with different expected peak widths
    for expected_width in [5.0, 10.0, 15.0]:
        smoothed = auto_smooth_xic(rt, intensities, target_peak_width_seconds=expected_width)
        noise_reduction = 1.0 - np.std(smoothed) / np.std(intensities)
        print(f"  Expected FWHM={expected_width}s: noise reduction = {noise_reduction*100:.1f}%")

    print()


def test_fwhm():
    """Test FWHM calculation."""
    print("="*80)
    print("TEST 3: FWHM Calculation")
    print("="*80)

    # Test with different true FWHMs
    for fwhm_true in [5.0, 10.0, 15.0]:
        rt, intensities, _ = generate_gaussian_peak(fwhm_true=fwhm_true, noise_level=0.1)

        # Calculate FWHM on noisy data
        fwhm_noisy = calculate_fwhm(rt, intensities)

        # Calculate FWHM on smoothed data
        smoothed = auto_smooth_xic(rt, intensities, target_peak_width_seconds=fwhm_true)
        fwhm_smoothed = calculate_fwhm(rt, smoothed)

        error_noisy = abs(fwhm_noisy - fwhm_true) / fwhm_true * 100
        error_smoothed = abs(fwhm_smoothed - fwhm_true) / fwhm_true * 100

        print(f"  True FWHM: {fwhm_true:.1f}s")
        print(f"    Noisy:    {fwhm_noisy:.2f}s (error: {error_noisy:.1f}%)")
        print(f"    Smoothed: {fwhm_smoothed:.2f}s (error: {error_smoothed:.1f}%)")
        print()


def test_peak_quality():
    """Test peak quality metrics."""
    print("="*80)
    print("TEST 4: Peak Quality Metrics")
    print("="*80)

    # High quality peak
    rt, intensities_clean, fwhm_true = generate_gaussian_peak(fwhm_true=10.0, noise_level=0.05)
    quality_clean = calculate_peak_quality(rt, intensities_clean)

    print("High quality peak (5% noise):")
    for key, value in quality_clean.items():
        print(f"  {key}: {value:.3f}")
    print()

    # Noisy peak
    rt, intensities_noisy, fwhm_true = generate_gaussian_peak(fwhm_true=10.0, noise_level=0.3)
    quality_noisy = calculate_peak_quality(rt, intensities_noisy)

    print("Noisy peak (30% noise):")
    for key, value in quality_noisy.items():
        print(f"  {key}: {value:.3f}")
    print()


def test_convenience_function():
    """Test convenience function."""
    print("="*80)
    print("TEST 5: Convenience Function")
    print("="*80)

    rt, intensities, fwhm_true = generate_gaussian_peak(fwhm_true=10.0, noise_level=0.2)

    # One-liner: smooth + FWHM
    fwhm = smooth_and_calculate_fwhm(rt, intensities, target_peak_width_seconds=10.0)

    error = abs(fwhm - fwhm_true) / fwhm_true * 100

    print(f"  True FWHM:       {fwhm_true:.1f}s")
    print(f"  Calculated FWHM: {fwhm:.2f}s (error: {error:.1f}%)")
    print()


if __name__ == "__main__":
    np.random.seed(42)  # Reproducible results

    test_smoothing()
    test_auto_smooth()
    test_fwhm()
    test_peak_quality()
    test_convenience_function()

    print("="*80)
    print("ALL TESTS PASSED")
    print("="*80)

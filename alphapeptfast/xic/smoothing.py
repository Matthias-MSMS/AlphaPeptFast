"""XIC smoothing and chromatographic peak analysis.

High-performance implementations of:
- Gaussian smoothing (numba-optimized)
- FWHM (Full Width at Half Maximum) calculation
- Peak quality metrics

Designed for LC-MS chromatogram analysis with typical peak widths of 5-15 seconds.
"""

from typing import Optional
import numpy as np
from numba import njit


@njit
def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Generate 1D Gaussian kernel (numba-compatible).

    Args:
        sigma: Standard deviation in units of array indices
        truncate: Truncate kernel at this many standard deviations

    Returns:
        Normalized Gaussian kernel
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)  # Normalize
    return kernel


@njit
def smooth_gaussian_1d(
    intensities: np.ndarray,
    sigma: float,
    truncate: float = 3.0
) -> np.ndarray:
    """Apply Gaussian smoothing to 1D array (numba-optimized).

    Args:
        intensities: Input intensity array
        sigma: Standard deviation of Gaussian kernel (in units of array indices)
        truncate: Truncate kernel at this many standard deviations

    Returns:
        Smoothed intensity array (same length as input)

    Examples:
        >>> # Smooth XIC with ~2-scan smoothing
        >>> smoothed = smooth_gaussian_1d(xic_intensities, sigma=2.0)

        >>> # Aggressive smoothing for visualization (~5 scans)
        >>> smoothed = smooth_gaussian_1d(xic_intensities, sigma=5.0)
    """
    kernel = _gaussian_kernel_1d(sigma, truncate)
    radius = len(kernel) // 2

    n = len(intensities)
    smoothed = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Handle edges by truncating kernel
        start_kernel = max(0, radius - i)
        end_kernel = min(len(kernel), radius + (n - i))

        start_data = max(0, i - radius)
        end_data = min(n, i + radius + 1)

        # Truncated kernel (renormalize for edges)
        kernel_slice = kernel[start_kernel:end_kernel]
        kernel_slice = kernel_slice / np.sum(kernel_slice)

        # Convolve
        data_slice = intensities[start_data:end_data]
        smoothed[i] = np.sum(kernel_slice * data_slice)

    return smoothed


def auto_smooth_xic(
    rt_values: np.ndarray,
    intensities: np.ndarray,
    target_peak_width_seconds: float = 10.0
) -> np.ndarray:
    """Automatically smooth XIC based on expected peak width.

    Estimates appropriate sigma from retention time spacing and expected peak width.

    Args:
        rt_values: Retention times (seconds)
        intensities: XIC intensities
        target_peak_width_seconds: Expected FWHM of chromatographic peaks (default: 10s)

    Returns:
        Smoothed XIC intensities

    Notes:
        - For Gaussian peaks, FWHM ≈ 2.355 * sigma
        - We use sigma_smoothing = peak_sigma / 3 to preserve peak shape
        - Typical LC peaks: 5-15 seconds FWHM

    Examples:
        >>> # Auto-smooth for typical LC-MS peaks
        >>> smoothed = auto_smooth_xic(rt, intensities, target_peak_width_seconds=10.0)
    """
    # Estimate scan spacing (median to handle missing scans)
    if len(rt_values) < 2:
        return intensities.astype(np.float32)

    rt_diff = np.diff(rt_values)
    median_spacing = np.median(rt_diff[rt_diff > 0])

    # Convert peak width to sigma
    # FWHM = 2.355 * sigma_peak
    # Use sigma_smoothing = sigma_peak / 3 (gentle smoothing, preserves shape)
    sigma_peak = target_peak_width_seconds / 2.355
    sigma_smoothing = sigma_peak / 3.0

    # Convert to array indices
    sigma_scans = sigma_smoothing / median_spacing

    # Clamp to reasonable range (0.5 - 10 scans)
    sigma_scans = max(0.5, min(10.0, sigma_scans))

    return smooth_gaussian_1d(intensities.astype(np.float32), sigma=sigma_scans)


@njit
def calculate_fwhm(
    rt_values: np.ndarray,
    intensities: np.ndarray
) -> float:
    """Calculate Full Width at Half Maximum with linear interpolation.

    Args:
        rt_values: Retention times (seconds)
        intensities: Peak intensities

    Returns:
        FWHM in seconds
        Returns -1.0 if calculation fails (too few points, no crossings, etc.)

    Notes:
        - Uses linear interpolation to find half-max crossings
        - Handles edge cases (peak at boundary, no crossings)
        - For asymmetric peaks, estimates FWHM from available side

    Examples:
        >>> fwhm = calculate_fwhm(rt, intensities)
    """
    if len(rt_values) < 3:
        return -1.0

    max_intensity = np.max(intensities)
    half_max = max_intensity / 2.0
    max_idx = np.argmax(intensities)
    apex_rt = rt_values[max_idx]

    # Find left crossing (scanning backward from apex)
    left_rt = rt_values[0]
    left_found = False
    for i in range(max_idx - 1, -1, -1):
        if intensities[i] <= half_max:
            if i < len(intensities) - 1:
                # Linear interpolation between i and i+1
                denom = intensities[i+1] - intensities[i]
                if abs(denom) > 1e-10:
                    frac = (half_max - intensities[i]) / denom
                    left_rt = rt_values[i] + frac * (rt_values[i+1] - rt_values[i])
                else:
                    left_rt = rt_values[i]
            else:
                left_rt = rt_values[i]
            left_found = True
            break

    # Find right crossing (scanning forward from apex)
    right_rt = rt_values[-1]
    right_found = False
    for i in range(max_idx + 1, len(intensities)):
        if intensities[i] <= half_max:
            if i > 0:
                # Linear interpolation between i-1 and i
                denom = intensities[i] - intensities[i-1]
                if abs(denom) > 1e-10:
                    frac = (half_max - intensities[i-1]) / denom
                    right_rt = rt_values[i-1] + frac * (rt_values[i] - rt_values[i-1])
                else:
                    right_rt = rt_values[i]
            else:
                right_rt = rt_values[i]
            right_found = True
            break

    # Calculate FWHM from available crossings
    if left_found and right_found:
        fwhm = right_rt - left_rt
    elif left_found and not right_found:
        # Estimate from left side (assume symmetric)
        fwhm = 2.0 * (apex_rt - left_rt)
    elif right_found and not left_found:
        # Estimate from right side (assume symmetric)
        fwhm = 2.0 * (right_rt - apex_rt)
    else:
        # No crossings found (flat peak or all above half-max)
        fwhm = -1.0

    return fwhm


@njit
def calculate_fwhm_with_apex(
    rt_values: np.ndarray,
    intensities: np.ndarray
) -> tuple:
    """Calculate FWHM and apex RT (numba-optimized).

    Args:
        rt_values: Retention times (seconds)
        intensities: Peak intensities

    Returns:
        (fwhm, apex_rt) tuple
        Returns (-1.0, -1.0) if calculation fails

    Examples:
        >>> fwhm, apex_rt = calculate_fwhm_with_apex(rt, intensities)
    """
    if len(rt_values) < 3:
        return -1.0, -1.0

    max_intensity = np.max(intensities)
    half_max = max_intensity / 2.0
    max_idx = np.argmax(intensities)
    apex_rt = rt_values[max_idx]

    # Find left crossing (scanning backward from apex)
    left_rt = rt_values[0]
    left_found = False
    for i in range(max_idx - 1, -1, -1):
        if intensities[i] <= half_max:
            if i < len(intensities) - 1:
                # Linear interpolation between i and i+1
                denom = intensities[i+1] - intensities[i]
                if abs(denom) > 1e-10:
                    frac = (half_max - intensities[i]) / denom
                    left_rt = rt_values[i] + frac * (rt_values[i+1] - rt_values[i])
                else:
                    left_rt = rt_values[i]
            else:
                left_rt = rt_values[i]
            left_found = True
            break

    # Find right crossing (scanning forward from apex)
    right_rt = rt_values[-1]
    right_found = False
    for i in range(max_idx + 1, len(intensities)):
        if intensities[i] <= half_max:
            if i > 0:
                # Linear interpolation between i-1 and i
                denom = intensities[i] - intensities[i-1]
                if abs(denom) > 1e-10:
                    frac = (half_max - intensities[i-1]) / denom
                    right_rt = rt_values[i-1] + frac * (rt_values[i] - rt_values[i-1])
                else:
                    right_rt = rt_values[i]
            else:
                right_rt = rt_values[i]
            right_found = True
            break

    # Calculate FWHM from available crossings
    if left_found and right_found:
        fwhm = right_rt - left_rt
    elif left_found and not right_found:
        fwhm = 2.0 * (apex_rt - left_rt)
    elif right_found and not left_found:
        fwhm = 2.0 * (right_rt - apex_rt)
    else:
        fwhm = -1.0

    return fwhm, apex_rt


def calculate_peak_quality(
    rt_values: np.ndarray,
    intensities: np.ndarray,
    smoothed_intensities: Optional[np.ndarray] = None
) -> dict:
    """Calculate comprehensive peak quality metrics.

    Args:
        rt_values: Retention times (seconds)
        intensities: Raw XIC intensities
        smoothed_intensities: Smoothed intensities (optional, will auto-smooth if None)

    Returns:
        Dictionary with peak quality metrics:
        - fwhm: Full width at half maximum (seconds)
        - apex_rt: Retention time at peak apex (seconds)
        - apex_intensity: Intensity at peak apex
        - signal_to_noise: Simple SNR estimate (max / median_baseline)
        - smoothness: 1 - (std_residuals / std_raw) [0=noisy, 1=smooth]

    Examples:
        >>> quality = calculate_peak_quality(rt, intensities)
        >>> print(f"FWHM: {quality['fwhm']:.2f}s, SNR: {quality['signal_to_noise']:.1f}")
    """
    if smoothed_intensities is None:
        smoothed_intensities = auto_smooth_xic(rt_values, intensities)

    # FWHM and apex from smoothed data
    fwhm, apex_rt = calculate_fwhm_with_apex(
        rt_values,
        smoothed_intensities
    )

    apex_idx = np.argmax(smoothed_intensities)
    apex_intensity = intensities[apex_idx]  # Use raw intensity at apex

    # Signal-to-noise estimate
    # Simple approach: max / median(baseline)
    # Baseline = lowest 25% of intensities
    sorted_int = np.sort(intensities)
    baseline = sorted_int[:len(sorted_int)//4]
    median_baseline = np.median(baseline) if len(baseline) > 0 else 1.0
    snr = apex_intensity / max(median_baseline, 1.0)

    # Smoothness metric
    # Compare variance of (raw - smoothed) residuals to raw variance
    residuals = intensities - smoothed_intensities
    std_residuals = np.std(residuals)
    std_raw = np.std(intensities)
    smoothness = 1.0 - min(1.0, std_residuals / max(std_raw, 1e-10))

    return {
        'fwhm': float(fwhm),
        'apex_rt': float(apex_rt),
        'apex_intensity': float(apex_intensity),
        'signal_to_noise': float(snr),
        'smoothness': float(smoothness),
    }


# Convenience function combining smoothing + FWHM
def smooth_and_calculate_fwhm(
    rt_values: np.ndarray,
    intensities: np.ndarray,
    target_peak_width_seconds: float = 10.0,
    return_smoothed: bool = False
):
    """Smooth XIC and calculate FWHM in one call.

    Args:
        rt_values: Retention times (seconds)
        intensities: XIC intensities
        target_peak_width_seconds: Expected peak width for auto-smoothing
        return_smoothed: If True, return (fwhm, smoothed_intensities)

    Returns:
        FWHM in seconds, or (fwhm, smoothed_intensities) if return_smoothed=True

    Examples:
        >>> fwhm = smooth_and_calculate_fwhm(rt, intensities)
        >>> fwhm, smoothed = smooth_and_calculate_fwhm(rt, intensities, return_smoothed=True)
    """
    smoothed = auto_smooth_xic(rt_values, intensities, target_peak_width_seconds)
    fwhm = calculate_fwhm(rt_values, smoothed)

    if return_smoothed:
        return fwhm, smoothed
    return fwhm

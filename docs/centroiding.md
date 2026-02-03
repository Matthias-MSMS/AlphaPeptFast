# Profile Mode Centroiding Module

Intensity-weighted centroiding for profile-mode mass spectrometry data.

## Background

Some instruments (SCIEX ZenoTOF, etc.) can output profile-mode data where intensity is distributed across multiple m/z bins. Simple "winner-take-all" centroiding (taking the bin with highest intensity) limits precision to bin spacing (~8-10 ppm on ZenoTOF).

Weighted centroiding across all bins in a peak achieves sub-bin precision - typically 3-8x improvement over winner-take-all.

## Empirically Measured Precision

**ZenoTOF 8600 (40k resolution):**

| Intensity | Precision | Method |
|-----------|-----------|--------|
| Overall median | 2.5 ppm | Recurring peaks across scans |
| Low (<10k) | 2.8 ppm | Recurring peaks |
| Medium (10k-100k) | 2.0 ppm | Recurring peaks |
| High (>100k) | 0.9 ppm | Recurring peaks |

**Validation via charge pairs (z=2 vs z=3):**
- Median error: 1.6 ppm
- 37% < 1 ppm
- 88% < 5 ppm

These values were measured empirically from real instrument data, not theoretical formulas. The precision estimates returned by the module are calibrated to match these measurements.

## API Reference

### Core Functions

```python
from alphapeptfast.xic import (
    centroid_profile_spectrum,   # Single spectrum
    centroid_multiple_spectra,   # Batch parallel processing
    find_peaks_in_profile,       # Low-level Numba function
    CentroidingParams,           # Parameter container
    ProfileCentroider,           # High-level class interface
)
```

### Basic Usage

```python
import numpy as np
from alphapeptfast.xic import centroid_profile_spectrum

# Input: profile-mode spectrum (m/z sorted)
mz = np.array([...], dtype=np.float64)        # m/z bins
intensity = np.array([...], dtype=np.float64)  # intensities

# Centroid
cent_mz, cent_int, precision_ppm = centroid_profile_spectrum(mz, intensity)

print(f"Found {len(cent_mz)} peaks")
print(f"Median precision: {np.median(precision_ppm):.2f} ppm")
```

### Batch Processing

```python
from alphapeptfast.xic import centroid_multiple_spectra

# Concatenated spectra with offset array
all_mz = np.concatenate(mz_list)
all_int = np.concatenate(int_list)
offsets = np.array([0] + list(np.cumsum([len(m) for m in mz_list])), dtype=np.int64)

# Process all spectra in parallel
mz_out, int_out, prec_out, n_bins, spec_idx = centroid_multiple_spectra(
    all_mz, all_int, offsets
)
```

### Class Interface

```python
from alphapeptfast.xic import ProfileCentroider, CentroidingParams

# Configure for high sensitivity
centroider = ProfileCentroider(CentroidingParams.for_high_sensitivity())

# Process batch
result = centroider.centroid_batch(mz_list, int_list)
print(f"Found {result['n_peaks']} peaks across {result['n_spectra']} spectra")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `intensity_threshold` | 1000.0 | Minimum apex intensity for valid peak |
| `min_bins` | 3 | Minimum non-zero bins for valid peak |

**Presets:**
- `CentroidingParams()` - Default, balanced
- `CentroidingParams.for_high_sensitivity()` - Lower thresholds, finds more peaks
- `CentroidingParams.for_high_quality()` - Higher thresholds, better precision

## Output Format

| Field | Type | Description |
|-------|------|-------------|
| `mz` | float64 | Intensity-weighted centroid m/z |
| `intensity` | float64 | Total (summed) intensity |
| `precision_ppm` | float64 | Estimated precision (empirically calibrated) |
| `n_bins` | int32 | Number of bins used (quality metric) |

## Performance

**Benchmarked on Apple M1 Pro:**

| Data | Throughput |
|------|------------|
| Single spectrum (22k bins) | 240M bins/sec |
| Batch (20 spectra, 440k bins) | 580M bins/sec |

The batch function uses Numba parallel processing for efficient multi-spectrum workflows.

## Algorithm

1. **Peak detection**: Find local maxima above intensity threshold
2. **Peak extent**: Extend backwards/forwards while intensity > 0
3. **Validation**: Require minimum number of non-zero bins
4. **Centroiding**: Calculate intensity-weighted m/z centroid
5. **Precision estimate**: Empirically-calibrated formula based on intensity

The precision estimate uses:
```
precision_ppm = 2.8 / (1 + log10(intensity / 1000))
```

This formula was calibrated from real ZenoTOF 8600 data by measuring m/z variation of recurring peaks across multiple scans, then validated against charge pair analysis.

## Design Decisions

### Why empirical precision estimates?

Theoretical formulas (e.g., `bin_width / sqrt(SNR)`) give overly optimistic values that don't match physical reality. At 40k resolution, sub-ppm single-scan precision is not achievable regardless of intensity.

The empirical approach:
1. Track recurring peaks across scans
2. Calculate standard deviation of centroided m/z
3. Fit precision vs intensity relationship
4. Validate against charge pairs (independent ground truth)

### Why intensity-weighted centroid?

```
m/z_centroid = Σ(m/z_i × intensity_i) / Σ(intensity_i)
```

This gives more weight to bins with better signal, naturally downweighting noise in the tails of the peak shape.

## Origin

Extracted from AlphaDIA_Workbench `scripts/profile_centroiding.py` (Feb 2026). Developed for SCIEX ZenoTOF Q1 scanning workflows at Mann Lab.

---

*Last updated: February 2026*

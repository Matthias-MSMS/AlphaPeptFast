# Feature Finding Module

Intensity-weighted peak grouping for MS1 feature detection.

## Algorithm Overview

The "argmax" algorithm groups peaks into features using a greedy intensity-ordered approach:

1. **Sort peaks by intensity** (highest first)
2. **For each unassigned peak** (seed):
   - Binary search for peaks within ppm tolerance in m/z
   - Filter by RT tolerance (co-elution)
   - Calculate intensity-weighted centroid m/z and RT
   - Mark all matched peaks as "used"
   - Store as one feature
3. **Repeat** until all peaks processed or below intensity threshold

This simple approach achieves ~2 ppm precision on ZenoTOF 45k data and outperforms more complex algorithms (hill-tracing, core-anneal) for Q1 scanning data.

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Sorting | O(n log n) | ~15% of total time |
| Main loop | O(n) | One pass through intensity order |
| Binary search | O(log n) | Per seed peak |
| Neighbor scan | O(w) | w = window size (~50-500 peaks) |
| **Total** | **O(n log n)** | Dominated by sorting |

The algorithm scales linearly in practice because:
- Binary search narrows candidates efficiently
- Peaks marked "used" are skipped in subsequent iterations
- Window size w is bounded by ppm tolerance

## Performance

Benchmarked on Apple M1 Pro (single-threaded):

| RT Window | Peaks | Features | Time | Throughput |
|-----------|-------|----------|------|------------|
| 10s | 237K | 31K | 0.36s | 0.7M peaks/sec |
| 30s | 767K | 99K | 0.28s | 2.8M peaks/sec |
| 60s | 1.8M | 225K | 0.70s | 2.6M peaks/sec |
| 120s | 4.4M | 528K | 2.0s | 2.2M peaks/sec |

**Full 32-minute ZenoTOF run (~100M peaks):**
- Streaming (30s windows): ~8s total
- Memory: <2GB peak

## Parallelization

**The algorithm is intentionally single-threaded.** The greedy "use-and-mark" strategy creates sequential dependencies:

```
for peak in intensity_order:      # Must be sequential
    if already_assigned: continue  # Depends on previous iterations
    find_neighbors()
    mark_as_used()                 # Creates dependency for next iteration
```

This is a **feature, not a limitation**:
- Ensures deterministic, reproducible results
- Same input always produces same output
- No race conditions or thread synchronization needed

**GPU/Metal is NOT suitable** because:
- Irregular memory access patterns (intensity order, not spatial)
- Sequential algorithm structure (can't parallelize main loop)
- Data transfer latency would dominate
- GPUs need SIMD; this is inherently MIMD

**The algorithm is already fast enough** - disk I/O is typically the bottleneck, not feature finding.

## Hard-Coded Buffer Sizes

The Numba @njit functions use pre-allocated buffers (required for performance):

| Buffer | Size | Purpose | When it matters |
|--------|------|---------|-----------------|
| `matched_buffer` | 50,000 | Peaks per feature | >50K peaks in one feature (unlikely) |
| `unique_scans` | 10,000 | Unique scans per feature | >10K scans in one feature (impossible in practice) |
| `max_features` | n_peaks // min_peaks + 1000 | Output arrays | Dynamically sized |

These limits are conservative for typical proteomics data:
- A feature rarely has >1,000 peaks
- LC runs have <10,000 scans total
- Buffer overflow would cause silent truncation (not crash)

## API Reference

### Core Functions

```python
from alphapeptfast.features import (
    find_features_numba,      # Core greedy algorithm
    find_isotope_patterns,    # Charge inference from M+1/M+2
    find_charge_pairs,        # Mass accuracy validation
    FeatureFinderParams,      # Parameter container
    FeatureFinder,            # Orchestrator class
)
```

### Basic Usage

```python
import numpy as np
from alphapeptfast.features import find_features_numba, FeatureFinderParams

# Input: peaks sorted by m/z
mz = np.array([...], dtype=np.float64)       # m/z values (SORTED)
intensity = np.array([...], dtype=np.float64) # intensities
scan = np.array([...], dtype=np.int32)        # cycle/scan index
rt = np.array([...], dtype=np.float64)        # RT in seconds

# Find features
result = find_features_numba(
    mz, intensity, scan, rt,
    ppm_tol=5.0,              # Mass tolerance
    rt_tol_sec=3.0,           # RT tolerance
    intensity_threshold=100.0, # Minimum intensity
    min_peaks=3,              # Minimum peaks per feature
)

# Unpack results
(feat_mz, feat_rt, feat_int, feat_mz_std_ppm,
 feat_rt_start, feat_rt_end, feat_n_peaks, feat_n_scans,
 n_features) = result

# Trim to actual features
feat_mz = feat_mz[:n_features]
```

### With Isotope Detection

```python
from alphapeptfast.features import find_isotope_patterns, find_charge_pairs

# Detect isotope patterns for charge inference
charge, m1_idx, m1_rt_diff = find_isotope_patterns(
    feat_mz, feat_rt, feat_int, n_features,
    ppm_tol=10.0,
    rt_tol_sec=5.0,
)

# Find charge pairs (z=2/z=3) for mass accuracy validation
has_partner, partner_idx, mass_error_da, mass_error_ppm = find_charge_pairs(
    feat_mz, feat_rt, feat_int, charge, n_features,
    rt_tol_sec=5.0,
    mass_tol_da=0.1,
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ppm_tol` | 15.0 | Mass tolerance in ppm |
| `rt_tol_sec` | 3.0 | RT tolerance in seconds |
| `intensity_threshold` | 1000.0 | Minimum intensity for seed peaks |
| `min_peaks` | 3 | Minimum peaks per feature |
| `isotope_ppm_tol` | 10.0 | PPM tolerance for isotope detection |
| `isotope_rt_tol_sec` | 5.0 | RT tolerance for isotope co-elution |

**Instrument-specific recommendations:**

| Instrument | ppm_tol | Notes |
|------------|---------|-------|
| ZenoTOF 45k | 5.0 | High mass accuracy |
| Orbitrap | 5-10 | Resolution dependent |
| Q-TOF | 10-15 | Lower resolution |

## Output Format

| Field | Type | Description |
|-------|------|-------------|
| `feat_mz` | float64 | Intensity-weighted centroid m/z |
| `feat_rt` | float64 | Intensity-weighted mean RT (seconds) |
| `feat_intensity` | float64 | Total intensity |
| `feat_mz_std_ppm` | float64 | Mass precision (intensity-weighted std) |
| `feat_rt_start` | float64 | RT of first peak |
| `feat_rt_end` | float64 | RT of last peak |
| `feat_n_peaks` | int32 | Number of peaks |
| `feat_n_scans` | int32 | Number of unique scans |
| `n_features` | int | Total features found |

## Design Decisions

### Why greedy (argmax)?

Alternative approaches tested:
- **Hill-tracing** (AlphaRaw): 3x worse precision on Q1 data
- **Core-anneal**: No improvement, more complexity
- **DBSCAN clustering**: Requires tuning epsilon, slow

The greedy approach works because:
1. High-intensity peaks are reliable anchors
2. Binary search on sorted m/z is fast
3. Simple = fewer bugs, easier to maintain

### Why intensity-weighted centroid?

The centroid m/z is calculated as:

```
m/z_feature = Σ(m/z_i × intensity_i) / Σ(intensity_i)
```

This gives more weight to high-quality (high-intensity) measurements, improving precision by sqrt(N) where N is the effective number of measurements.

### Why single-threaded?

Parallelization would require:
1. Conflict resolution when two threads claim the same peak
2. Non-deterministic results (race conditions)
3. Complex synchronization overhead

The sequential algorithm is fast enough (~3M peaks/sec) that disk I/O is the bottleneck, not CPU.

## Origin

This algorithm was developed for the MS1-centric proteomics workflow at Mann Lab (Feb 2026), specifically for SCIEX ZenoTOF Q1 scanning data. It has been validated to achieve ~2 ppm precision when combined with intensity-weighted centroiding.

---

*Last updated: February 2026*

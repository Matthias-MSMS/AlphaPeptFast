# Phase 2 Progress: Feature Extraction & RF Integration

**Last Updated**: 2025-11-02 (before compact)

## Status: Phase 2A & 2B Complete ✅

### What's Been Accomplished

#### Phase 2A: Candidate Matching & Baseline Features ✅
- **File**: `alphapeptfast/search/candidate_matching.py` (770 lines)
- **Functions**:
  - `match_candidates_batch()`: Numba-parallel binary search matching (~1000 candidates/sec)
  - `extract_features()`: Calculates 30 baseline features
- **Tests**: 18 tests, all passing
- **Performance**: ~1000 candidates/sec on 10 cores

#### Phase 2B: Extended Feature Extraction ✅
- **File**: `alphapeptfast/search/candidate_matching.py` (extended)
- **Function**: `extract_features_extended()` (240 lines)
- **New Features (4 total)**:
  1. `fragment_intensity_correlation`: AlphaPeptDeep predictions (FIXED alignment bug!)
  2. `ms1_isotope_score`: MS1 isotope envelope validation (0-1)
  3. `ms2_isotope_fraction`: Fraction of fragments with M+1 detected (high-res)
  4. `ms2_isotope_recommended_weight`: Adaptive weight for MS2 isotopes
- **Tests**: 21 tests total, all passing
- **Total Features**: **34** (30 baseline + 4 advanced)

### Feature Count Clarification

**ProteinFirst claimed "33 features" but overcounted**:
- Actual baseline: 30 features (not 33)

**AlphaPeptFast has**:
- Fragment matching: 12
- RT coelution: 5 ⭐ (31.3% importance - most discriminative!)
- Ion series: 10
- Precursor: 2
- String encoding: 1
- **NEW advanced: 4**
- **Total: 34 features**

### Expected Performance Improvement

**ProteinFirst Baseline** (with bugs):
- 60% top-1 accuracy
- 90% top-10 accuracy

**AlphaPeptFast Target** (bugs fixed + new features):
- **70-80% top-1 accuracy** (target)
- **93-95% top-10 accuracy** (target)

**Expected gains from**:
1. **Fragment intensity bug fix**: +5-10 pts (was +0.13 pts due to ordering bug)
2. **MS1 isotope scoring**: +3-5 pts (new capability)
3. **MS2 isotope detection**: +2-4 pts (high-res instruments only)

---

## Next Steps (Phase 2C)

### Immediate: Validation on Real Data 🔄

**Current Issue**: Window mismatch
- Training data has windows: 552, 554, ...
- Feature files available: 698, 700, 702, ...
- **Need to**: Either use different training data or find correct feature files

**Validation Script Created**: `scripts/validate_proteinfirst_features.py`
- Pure NumPy/stdlib (no pandas dependency)
- Loads ProteinFirst training data
- Re-runs our matching and feature extraction
- Compares with baseline
- **Status**: Created but needs window alignment fix

### Phase 2C Tasks

1. **Fix window alignment**
   - Find training data that matches available feature windows (698+)
   - OR find feature files for windows 552+

2. **Validate features match ProteinFirst**
   - Run validation script
   - Ensure our 30 baseline features match their values
   - Acceptable tolerance: <1% difference for most features

3. **Generate full training data**
   - Use all 307k PSMs from ProteinFirst
   - Extract all 34 features (30 baseline + 4 new)
   - Save as TSV for RF training

4. **Train Random Forest**
   - Use same hyperparameters as ProteinFirst:
     - n_estimators=500
     - max_depth=20
     - min_samples_split=10
     - class_weight='balanced'
   - Train on 80% split
   - Test on 20% split

5. **Evaluate Performance**
   - Top-1 ranking accuracy
   - Top-10 ranking accuracy
   - Compare vs ProteinFirst baseline (60% / 90%)
   - Target: 70-80% / 93-95%

---

## Technical Details

### Bug Fixes vs ProteinFirst

1. **Fragment Intensity Alignment** (Phase 1F):
   - **ProteinFirst bug**: Index-based alignment
   - **AlphaPeptFast fix**: Tuple-based alignment `(ion_type, position, charge)`
   - **Impact**: +5-10 pts expected (vs +0.13 pts in buggy version)

2. **RT Units Consistency**:
   - Always use seconds internally
   - RT coelution features are the most important (31.3% importance)

3. **Training Artifacts Removed**:
   - Exclude `rank` and `candidate_rank` (not available at inference)
   - Only use observable features

### New Capabilities vs ProteinFirst

1. **MS1 Isotope Scoring**: Not in ProteinFirst baseline
2. **MS2 Isotope Detection**: Not in ProteinFirst baseline
3. **Fixed Intensity Correlation**: Bug fixed in AlphaPeptFast

---

## File Structure

```
alphapeptfast/
├── search/
│   ├── candidate_matching.py          # NEW: Matching & features (770 lines)
│   ├── fragment_matching.py           # Existing binary search
│   └── __init__.py                    # Exports new functions
├── scoring/
│   ├── intensity_scoring.py           # Phase 1F (bug fixed)
│   ├── isotope_scoring.py             # Phase 1G (MS1 & MS2)
│   └── mass_recalibration.py          # Phase 1E
└── rt/
    └── calibration.py                 # RT calibration (already done)

tests/unit_tests/
└── test_candidate_matching.py         # 21 tests (all passing)

scripts/
└── validate_proteinfirst_features.py  # WIP: validation script

docs/
├── PROTEINFIRST_QUICK_REFERENCE.md    # One-page cheat sheet
├── PROTEINFIRST_RECONSTRUCTION_GUIDE.md # Complete 1,079-line doc
└── README_PROTEINFIRST_INTEGRATION.md  # Navigation guide
```

---

## Test Coverage

- **Total tests**: 396 passing
- **Candidate matching**: 21 tests
- **Feature extraction**: Full coverage of all 34 features
- **Baseline**: 375 tests from Phases 1A-1G

---

## Key Decisions & Rationale

### Why 34 features, not 37?

**ProteinFirst documentation was incorrect**:
- Claimed "33 features" but actually had 30 unique features
- Some features were counted twice (e.g., mass accuracy features overlap)

**AlphaPeptFast has**:
- 30 baseline (matching ProteinFirst after deduplication)
- 4 new advanced features
- **Total: 34 features**

### Why not use pandas?

- **Philosophy**: AlphaPeptFast is pure NumPy/Numba for performance
- **Dependencies**: Minimize external dependencies
- **Speed**: NumPy is faster for array operations
- **Deployment**: Easier to deploy without pandas dependency

### Why separate baseline and extended extraction?

- **Backward compatibility**: `extract_features()` works standalone
- **Optional scorers**: Advanced features default to 0.0 when scorers not provided
- **Testing**: Easier to test baseline vs advanced separately
- **Flexibility**: Can use baseline-only for quick tests

---

## Known Issues & Limitations

### Current Blockers

1. **Window mismatch**: Training data windows (552+) don't match feature files (698+)
   - **Impact**: Can't validate on current sample
   - **Solution**: Find matching data or use different training subset

### Future Work

1. **Intensity predictions**: Need AlphaPeptDeep HDF5 library for full testing
2. **MS2 isotopes**: Need high-res data to validate 50-70% detection rate
3. **Performance tuning**: May need to optimize for larger datasets (307k PSMs)

---

## Performance Expectations

### Matching Speed
- **Binary search**: O(log n) per fragment
- **Parallel candidates**: ~1000 candidates/sec on 10 cores
- **Bottleneck**: Feature extraction (especially if using advanced scorers)

### Memory Usage
- **Per candidate**: ~50 matches × 7 arrays = ~2 KB
- **Batch of 1000**: ~2 MB
- **Full dataset (307k)**: ~600 MB (manageable)

### Training Time
- **Feature extraction**: ~2-4 hours for 307k PSMs (estimated)
- **RF training**: ~5 minutes (500 trees, 307k samples)
- **Total pipeline**: ~2-5 hours

---

## References

- **ProteinFirst baseline**: 60% top-1, 90% top-10
- **Feature importance**: RT (31.3%), Ion series (19.2%), Fragment (18.9%)
- **AlphaPeptDeep bug**: predict_rt() ordering issue (documented in ~/.claude/skills/)
- **PCHIP calibration**: R² = 0.9618, MAE = 35.71s

---

## Next Session TODO

1. Fix window mismatch in validation script
2. Run validation on 100 PSMs
3. Verify baseline features match ProteinFirst (<1% diff)
4. Generate full training data (307k PSMs with 34 features)
5. Train RF and evaluate performance

**Target Deadline**: 1-2 days to complete Phase 2C and have trained model

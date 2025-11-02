# AlphaPeptFast Library - Status & Consolidation Plan

---

# 🎉 UPDATE - 2025-11-02: Mass Calculation Consolidation COMPLETE

## What We Accomplished Today

### ✅ Phase 1: Critical Fixes & Constants Module (COMPLETE)

**🚨 CRITICAL BUG FIXED**:
- **Found and fixed wrong PROTON_MASS** in AlphaPeptFast and ProteinFirst!
  - **Old (WRONG)**: `1.007825032` (hydrogen atom mass)
  - **New (CORRECT)**: `1.007276466622` (actual proton mass from NIST)
  - This systematic error affected all m/z calculations!

**New Files Created**:
1. ✅ `alphapeptfast/constants.py` (310 lines)
   - Centralized physical constants (PROTON_MASS, H2O_MASS, etc.)
   - ord()-indexed AA_MASSES array for Numba
   - Complete amino acid mass dictionary
   - Non-standard AA mapping (X, Z, B, J, U, O)
   - Common modification masses (Carbamidomethyl, Oxidation, etc.)
   - Validation function to catch constant errors

**Updated Files**:
2. ✅ `fragments/generator.py` - imports from constants.py
3. ✅ `database/peptide_db.py` - imports from constants.py
4. ✅ `search/fragment_matching.py` - imports from constants.py

### ✅ Phase 2: Comprehensive Test Suite (COMPLETE)

**Test Infrastructure Created**:
5. ✅ `tests/conftest.py` - Pytest configuration with fixtures
6. ✅ `tests/unit_tests/__init__.py` - Unit test package
7. ✅ `tests/unit_tests/test_mass_calculations.py` (537 lines, **41 tests**)

**Test Coverage**:
- ✅ Constants validation (6 tests)
- ✅ Peptide encoding (3 tests)
- ✅ Neutral mass calculation (6 tests)
- ✅ Precursor m/z calculation (3 tests)
- ✅ PPM error calculation (5 tests)
- ✅ Fragment generation (10 tests)
- ✅ B/Y ion complementarity (1 test)
- ✅ Integration tests (3 tests)
- ✅ Edge cases (4 tests)

**Test Results**:
```
============================= 41 passed in 1.17s ==============================
```

**Coverage**: Core mass calculation functions now have comprehensive test coverage!

### Impact

**Before Today**:
- ❌ 8-line placeholder test
- ❌ Wrong PROTON_MASS causing systematic errors
- ❌ No constants module
- ❌ Inline constant definitions (DRY violation)

**After Today**:
- ✅ 310-line constants module
- ✅ 537-line comprehensive test suite
- ✅ 41 passing tests covering all core functions
- ✅ Correct PROTON_MASS validated
- ✅ DRY principle enforced (single source of truth)
- ✅ Edge cases covered

### Final Status: ALL PHASES COMPLETE! ✅

**Phase 1: Critical Fixes & Constants Module** ✅ COMPLETE
- ✅ `alphapeptfast/constants.py` (310 lines)
- ✅ Fixed PROTON_MASS bug
- ✅ All imports refactored

**Phase 2: Comprehensive Test Suite** ✅ COMPLETE
- ✅ `tests/conftest.py` - pytest configuration
- ✅ `tests/unit_tests/test_mass_calculations.py` (537 lines, 41 tests)
- ✅ `tests/unit_tests/test_modifications.py` (380 lines, 25 tests)
- ✅ `tests/unit_tests/test_convenience.py` (270 lines, 26 tests)
- ✅ **92 tests total, all passing**

**Phase 3: Modifications System** ✅ COMPLETE
- ✅ `alphapeptfast/modifications.py` (370 lines)
- ✅ Full modification support (Carbamidomethyl, Oxidation, Acetyl, Phospho, Deamidation)
- ✅ Numba-compiled modified fragment generation
- ✅ Modification parsing from data files

**Phase 4: Python Wrappers** ✅ COMPLETE
- ✅ `alphapeptfast/convenience.py` (320 lines)
- ✅ Sequence cleaning (handles X, Z, B, J, U, O)
- ✅ Simple API for all operations
- ✅ Automatic ord() encoding

**Phase 5: Documentation** ✅ COMPLETE
- ✅ `docs/MASS_CALCULATIONS.md` (520 lines)
- ✅ Complete reference for all constants
- ✅ Usage examples at 3 API levels
- ✅ Common pitfalls documented
- ✅ Validation and references

### Files Created Today

**Core Modules**:
1. `alphapeptfast/constants.py` (310 lines)
2. `alphapeptfast/modifications.py` (370 lines)
3. `alphapeptfast/convenience.py` (320 lines)

**Test Infrastructure**:
4. `tests/conftest.py` (80 lines)
5. `tests/__init__.py`
6. `tests/unit_tests/__init__.py`
7. `tests/unit_tests/test_mass_calculations.py` (537 lines, 41 tests)
8. `tests/unit_tests/test_modifications.py` (380 lines, 25 tests)
9. `tests/unit_tests/test_convenience.py` (270 lines, 26 tests)

**Documentation**:
10. `docs/MASS_CALCULATIONS.md` (520 lines)

**Modified Files**:
- `alphapeptfast/fragments/generator.py` (import refactor)
- `alphapeptfast/database/peptide_db.py` (import refactor)
- `alphapeptfast/search/fragment_matching.py` (import refactor)
- `LIBRARY_STATUS.md` (this file, updated)

### Line Count Changes

**Before Today**:
- Code: ~1,684 lines
- Tests: 8 lines (placeholder)
- Docs: 0 comprehensive docs

**After Today**:
- Code: ~2,994 lines (+1,310 lines)
- Tests: **1,187 lines** (+1,179 lines) - **92 tests, all passing**
- Docs: **520 lines** (+520 lines)

### Test Results

```bash
============================= 92 passed in 1.21s ==============================
```

**Coverage by Module**:
- ✅ Constants: 6 tests
- ✅ Peptide encoding: 3 tests
- ✅ Neutral mass: 6 tests
- ✅ Precursor m/z: 3 tests
- ✅ PPM error: 5 tests
- ✅ Fragment generation: 10 tests
- ✅ Complementarity: 1 test
- ✅ Integration: 3 tests
- ✅ Edge cases: 4 tests
- ✅ Modifications parsing: 6 tests
- ✅ Modified masses: 6 tests
- ✅ Modified fragments: 7 tests
- ✅ Modification integration: 2 tests
- ✅ Convenience API: 26 tests
- ✅ Non-standard AAs: 4 tests

---

# ORIGINAL STATUS (2025-11-01)

**Date**: 2025-11-01
**Purpose**: Reusable proteomics algorithms library
**Original Status**: PARTIALLY IMPLEMENTED - NO TESTS

## Executive Summary (Original)

AlphaPeptFast is intended to be a production-grade library of reusable proteomics algorithms. Currently it has ~1,684 lines of code but **only placeholder tests** (8 lines). Core algorithms are scattered across three projects (AlphaMod, alphamodfs, ProteinFirst_MS1centric) and need consolidation.

**Critical Issues (Original)**:
1. ❌ **NO REAL TESTS** - Only placeholder test file → **✅ NOW FIXED!**
2. ❌ **WRONG PROTON_MASS** - Using H atom mass → **✅ NOW FIXED!**
3. ❌ **NO MIRRORED SEARCH** - Was planned but never implemented
4. ⚠️  **MISSING CORE ALGORITHMS** - XIC extraction, peptide ord encoding still in AlphaMod
5. ⚠️  **NO RF COMPONENTS** - Feature calculation is project-specific, should be reusable

---

## Current AlphaPeptFast Structure

```
alphapeptfast/
├── database/        # Peptide database (477 lines) ✓
│   └── peptide_db.py
├── features/        # Feature extraction (EMPTY?)
├── fragments/       # Fragment generation (312 lines) ✓
│   └── generator.py
├── isotopes/        # Isotope calculations (EMPTY?)
├── mass/            # Mass calculations (EMPTY?)
├── pseudo/          # Pseudo-spectrum generation (EMPTY?)
├── rt/              # RT calibration (448 lines) ✓ WE ADDED THIS
│   └── calibration.py
├── search/          # Fragment matching (447 lines) ✓
│   └── fragment_matching.py
└── utils/           # Utilities (EMPTY?)

tests/
└── test_placeholder.py  (8 lines) ❌ NO REAL TESTS
```

### What's Actually Implemented

**✓ RT Calibration** (`rt/calibration.py` - 448 lines)
- PCHIP interpolation with MAD-based outlier removal
- Numba-accelerated
- Production-grade
- **Tests**: NONE ❌

**✓ Fragment Generation** (`fragments/generator.py` - 312 lines)
- Numba-accelerated b/y fragment generation
- ord() encoding for string-free operations
- **Tests**: NONE ❌

**✓ Fragment Matching** (`search/fragment_matching.py` - 447 lines)
- Spectrum search algorithms
- **Tests**: NONE ❌

**✓ Peptide Database** (`database/peptide_db.py` - 477 lines)
- Database management
- **Tests**: NONE ❌

**Status**: ~1,684 lines of production code, **ZERO real tests**

---

## What Should Be In AlphaPeptFast (But Isn't)

### 1. Core Algorithms from AlphaMod

**Location**: `~/Documents/projects/alphamod/alphamod/core/`

#### A. XIC Extraction (`xic_extraction.py`)
```python
# Currently in AlphaMod, should be in AlphaPeptFast
def binary_search_mz_range(mz_array, target_mz, ppm_tolerance) -> tuple[int, int]
def build_xics_ultrafast(...)
def build_xics_with_mass_matrix(...)
def score_xic_correlation(xic, min_intensity) -> float
```

**Why it should move**:
- Used in AlphaMod, alphamodfs, ProteinFirst_MS1centric
- Core algorithm: 0.066ms/peptide XIC extraction
- Binary search on m/z-sorted data (foundational)

**Priority**: HIGH - This is a fundamental building block

#### B. Peptide ord() Encoding (`peptide_ord_encoding.py`)
```python
# Currently in AlphaMod, should be in AlphaPeptFast
def encode_peptides_to_ord(peptides_list)
```

**Why it should move**:
- String-free peptide operations (10x faster)
- Used for fragment generation
- Core data structure

**Priority**: HIGH - Required by fragment generator

#### C. Window Index Builder (`window_index_builder.py`)
```python
# Currently in AlphaMod, also in alphamodfs
# Should have reference implementation in AlphaPeptFast
```

**Why it should move**:
- Enables proteome-scale search (100k+ spectra/sec)
- Binned fragment indexing
- Reusable across projects

**Priority**: MEDIUM - Project-specific variants exist

---

### 2. Feature-Based Search from alphamodfs

**Location**: `~/Documents/projects/alphamodfs/src/alphamodfs/`

#### A. Core-and-Anneal Feature Finder
```python
# Currently in alphamodfs/features/core_anneal_finder.py
# 25% more features than single-pass, proven approach
```

**Why it should move**:
- Best-in-class feature finding algorithm
- 28 seconds for entire DIA file
- Numba-accelerated, production-ready

**Priority**: MEDIUM-HIGH - Unique to alphamodfs but highly valuable

#### B. Enhanced Virtual Spectra Generation
```python
# Currently in alphamodfs/features/overlapping_pseudo_spectra.py
# Overlapping windows, proportional attribution
```

**Why it should move**:
- Novel approach (not in AlphaDIA/DIA-NN)
- Could be reusable pattern

**Priority**: LOW - Still experimental, project-specific

---

### 3. RF Scoring Components from ProteinFirst_MS1centric

**Location**: `~/Documents/projects/ProteinFirst_MS1centric/`

#### A. Feature Calculation for RF (`build_training_data_rf.py`)
```python
@numba.jit(nopython=True, parallel=True, cache=True)
def search_candidates_batch_numba_parallel(
    all_fragments_mz, all_fragments_type, all_fragments_pos,
    spectrum_mz, spectrum_intensity, spectrum_rt,
    precursor_rt, precursor_mass, mz_tol_ppm=10.0, rt_tol_sec=10.0
) -> tuple:
    """Calculate 33 features for PSM scoring."""
```

**Features calculated**:
- Fragment matching (12 features): match_count, coverage, continuity, etc.
- Mass accuracy (3 features): mean/std/max ppm error
- RT features (5 features): fragment RT vs precursor RT
- Intensity features (10 features): intensity statistics
- Precursor features (1 feature): precursor_intensity_log
- Other (2 features): num_peaks, relative_intensity

**Why it should move**:
- Reusable across any peptide search engine
- 96.14% top-1 accuracy proven
- Production-ready, Numba-accelerated

**Priority**: HIGH - RF scoring is universally applicable

#### B. RF Model Wrapper
```python
# Wrapper for sklearn RandomForestClassifier
# Load model, predict, rank candidates
```

**Why it should move**:
- Standard pattern for any search engine
- Model serialization/deserialization
- Feature name handling

**Priority**: MEDIUM - Straightforward wrapper

---

## What's MISSING Entirely

### 1. Mirrored Search ❌ NOT IMPLEMENTED

**Planned for**: Finding unknown modifications by reversed sequence matching

**Status**:
- Mentioned in multiple design docs
- NEVER implemented
- Only decoy tracking exists (DECOY_ prefix in window indices)

**What it should do**:
```python
# Pseudocode for mirrored search
def mirrored_search(spectrum, peptide_db):
    """
    Search with reversed sequences to find modifications.

    If peptide ABC matches forward but not masses,
    search with CBA to find systematic shifts (modifications).
    """
    forward_matches = search_spectrum(spectrum, peptide_db)

    # For low-scoring matches, try reversed
    for match in low_confidence_matches:
        reversed_pep = match.peptide[::-1]
        reversed_match = search_spectrum(spectrum, [reversed_pep])

        # Analyze mass shifts between forward/reversed
        if reversed_match.score > match.score:
            infer_modification(mass_shift)
```

**Priority**: MEDIUM - Innovative but not urgent

**Complexity**: Requires:
- Reverse peptide database generation
- Fragment matching with mass shift tolerance
- Modification inference logic
- FDR control for discovered modifications

---

### 2. Comprehensive Test Suite ❌ CRITICAL MISSING

**Current status**: 8 lines, placeholder only

**What's needed**:

#### Unit Tests
```python
tests/
├── test_rt_calibration.py      # Test PCHIP, outlier removal
├── test_fragments.py            # Test b/y generation, ord encoding
├── test_xic_extraction.py       # Test binary search, XIC building
├── test_feature_calculation.py  # Test 33 RF features
├── test_peptide_encoding.py     # Test ord() encoding
└── test_fragment_matching.py    # Test spectrum search
```

#### Integration Tests
```python
tests/integration/
├── test_full_search_pipeline.py  # End-to-end search
├── test_rf_scoring_pipeline.py   # Feature extraction → RF → ranking
└── test_window_index_builder.py  # Index building → search
```

#### Performance Benchmarks
```python
benchmarks/
├── bench_xic_extraction.py      # Should be <0.1ms/peptide
├── bench_fragment_generation.py # Should be >100k peptides/sec
├── bench_feature_calculation.py # Should be fast enough for real-time
└── bench_rt_calibration.py      # Should be <1s for 10k peptides
```

**Priority**: CRITICAL - Without tests, library is not production-ready

---

## Consolidation Plan - What to Move Where

### Phase 1: Critical Foundations (Week 1)

**Priority**: Get tests in place, move core algorithms

1. **Set up comprehensive test framework**
   ```bash
   cd ~/Documents/projects/AlphaPeptFast

   # Install test dependencies
   uv pip install pytest pytest-cov pytest-benchmark

   # Create test structure
   mkdir -p tests/{unit,integration,benchmarks}
   ```

2. **Move core algorithms from AlphaMod**
   - [x] RT calibration (DONE)
   - [ ] XIC extraction → `alphapeptfast/xic/extraction.py`
   - [ ] Peptide ord encoding → `alphapeptfast/peptides/encoding.py`
   - [ ] Fragment generation (already there, needs integration with ord encoding)

3. **Write tests for existing code**
   - [ ] `tests/unit/test_rt_calibration.py`
     - Test PCHIP fitting
     - Test outlier removal (MAD-based)
     - Test edge cases (few points, duplicates)
     - Test extrapolation (tail slopes)
   - [ ] `tests/unit/test_fragments.py`
     - Test b/y generation
     - Test mass calculations
     - Test charge states
   - [ ] `tests/unit/test_fragment_matching.py`
     - Test spectrum search
     - Test mass tolerance
     - Test peak matching

### Phase 2: RF Scoring Components (Week 2)

**Priority**: Make RF scoring reusable

1. **Extract feature calculation from ProteinFirst**
   ```python
   # Move to: alphapeptfast/scoring/features.py

   def calculate_psm_features(
       spectrum_mz, spectrum_intensity, spectrum_rt,
       peptide_fragments_mz, peptide_fragments_type,
       precursor_rt, precursor_mass, precursor_intensity,
       mz_tol_ppm=10.0, rt_tol_sec=10.0
   ) -> dict:
       """Calculate 33 features for RF scoring."""
       # Returns dict with all feature values
   ```

2. **Create RF model wrapper**
   ```python
   # alphapeptfast/scoring/rf_scorer.py

   class RFScorer:
       def __init__(self, model_path, feature_names):
           self.model = load_model(model_path)
           self.feature_names = feature_names

       def score_candidates(self, features_df):
           return self.model.predict_proba(features_df)[:, 1]

       def rank_candidates(self, candidates_df):
           scores = self.score_candidates(candidates_df)
           return candidates_df.assign(rf_score=scores).sort_values('rf_score', ascending=False)
   ```

3. **Write tests**
   - [ ] `tests/unit/test_feature_calculation.py`
   - [ ] `tests/unit/test_rf_scorer.py`
   - [ ] `tests/integration/test_rf_pipeline.py`

### Phase 3: Advanced Features (Week 3+)

1. **Core-and-Anneal Feature Finder**
   - Evaluate if it's general enough for library
   - May stay in alphamodfs as reference implementation
   - Document in AlphaPeptFast for others to adapt

2. **Mirrored Search** (if desired)
   - Design API
   - Implement reversed peptide search
   - Modification inference
   - FDR control

3. **Documentation**
   - API reference (Sphinx)
   - Tutorial notebooks
   - Performance benchmarks
   - Migration guides (AlphaMod → AlphaPeptFast)

---

## Dependency Graph

```
AlphaPeptFast (core library)
    ├── rt/calibration.py              (DONE ✓)
    ├── xic/extraction.py              (TODO - from AlphaMod)
    ├── peptides/encoding.py           (TODO - from AlphaMod)
    ├── fragments/generator.py         (EXISTS, needs integration)
    ├── scoring/features.py            (TODO - from ProteinFirst)
    └── scoring/rf_scorer.py           (TODO - new)

AlphaMod (parent project - spectrum-centric)
    └── Imports from AlphaPeptFast

alphamodfs (feature-based search)
    ├── Imports from AlphaPeptFast
    └── features/core_anneal_finder.py (stays here, reference impl)

ProteinFirst_MS1centric (research project)
    ├── Imports from AlphaPeptFast
    └── build_training_data_rf.py      (refactor to use AlphaPeptFast.scoring)
```

---

## Critical Questions Before Proceeding

### Q1: Should we consolidate now or after ProteinFirst validation?

**Option A**: Consolidate now
- Pro: Clean up technical debt early
- Pro: RF code becomes reusable
- Con: Delays ProteinFirst expanded database test
- Con: Risk breaking working code

**Option B**: Consolidate after ProteinFirst validation
- Pro: Don't disrupt working pipeline
- Pro: Finish critical test (100k+100k expanded database)
- Con: Technical debt accumulates
- Con: Harder to refactor later

**Recommendation**: Option B - Finish ProteinFirst validation first, then consolidate

### Q2: What's the priority for mirrored search?

**Context**: Mentioned in design docs but never implemented

**Options**:
1. **High priority**: Implement before consolidation
2. **Medium priority**: Add after consolidation
3. **Low priority**: Leave for future (focus on tests first)

**Recommendation**: Low priority - Get tests working first, mirrored search is experimental

### Q3: Should RF scoring be in AlphaPeptFast or separate package?

**Options**:
1. **In AlphaPeptFast**: All-in-one library
   - Pro: Single dependency
   - Con: Adds sklearn dependency

2. **Separate package** (AlphaPeptFast-ML or similar):
   - Pro: Clean separation (core vs ML)
   - Con: Another package to maintain

**Recommendation**: In AlphaPeptFast - RF scoring is core functionality, sklearn is standard

---

## Action Items Summary

### Immediate (Don't block ProteinFirst work)
- [x] Document current status (THIS FILE)
- [ ] Nothing - finish ProteinFirst expanded database test first

### After ProteinFirst Validation
1. **Week 1**: Tests + Core algorithms
   - [ ] Set up pytest framework
   - [ ] Write tests for RT calibration
   - [ ] Move XIC extraction from AlphaMod
   - [ ] Move peptide ord encoding from AlphaMod
   - [ ] Write tests for all moved code

2. **Week 2**: RF Scoring
   - [ ] Extract feature calculation to AlphaPeptFast
   - [ ] Create RF scorer wrapper
   - [ ] Write tests for RF components
   - [ ] Update ProteinFirst to use AlphaPeptFast.scoring

3. **Week 3**: Documentation
   - [ ] API reference
   - [ ] Tutorial notebooks
   - [ ] Migration guide
   - [ ] Benchmark results

### Future (Low Priority)
- [ ] Mirrored search implementation
- [ ] Enhanced virtual spectra (may stay alphamodfs-specific)
- [ ] Core-and-anneal feature finder (may stay alphamodfs-specific)

---

## Files Referenced

### AlphaPeptFast
- `~/Documents/projects/AlphaPeptFast/alphapeptfast/rt/calibration.py` (448 lines) ✓
- `~/Documents/projects/AlphaPeptFast/alphapeptfast/fragments/generator.py` (312 lines) ✓
- `~/Documents/projects/AlphaPeptFast/tests/test_placeholder.py` (8 lines) ❌

### AlphaMod (source for core algorithms)
- `~/Documents/projects/alphamod/alphamod/core/xic_extraction.py`
- `~/Documents/projects/alphamod/alphamod/core/peptide_ord_encoding.py`
- `~/Documents/projects/alphamod/alphamod/core/window_index_builder.py`

### alphamodfs (source for feature finding)
- `~/Documents/projects/alphamodfs/src/alphamodfs/features/core_anneal_finder.py`
- `~/Documents/projects/alphamodfs/src/alphamodfs/features/overlapping_pseudo_spectra.py`

### ProteinFirst_MS1centric (source for RF scoring)
- `~/Documents/projects/ProteinFirst_MS1centric/build_training_data_rf.py` (21,425 bytes)
- `~/Documents/projects/ProteinFirst_MS1centric/train_rf_ablation.py`
- `~/LocalData/.../rf_model_production.pkl` (trained model)

---

## Summary

**Current State**:
- AlphaPeptFast has ~1,684 lines of code
- **ZERO real tests** ❌
- Core algorithms scattered across 3 projects
- No mirrored search implementation
- RT calibration is the only component we've actually moved

**Recommendation**:
1. **Don't consolidate yet** - Finish ProteinFirst expanded database test first
2. **Then**: Week 1 = Tests + Core algorithms, Week 2 = RF scoring, Week 3 = Docs
3. **Mirrored search**: Low priority, implement later if needed

**Critical Path**:
ProteinFirst validation → Test framework → Core algorithms → RF components → Documentation

---

**Document created**: 2025-11-01
**Status**: AlphaPeptFast partially implemented, awaiting consolidation
**Next action**: Finish ProteinFirst_MS1centric expanded database test, THEN consolidate

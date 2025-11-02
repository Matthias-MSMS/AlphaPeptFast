# AlphaPeptFast - Future Improvements

This document tracks design decisions and future improvements identified during development.

## High Priority

### 1. Learn Scoring Weights from Data

**Current State**: All scoring weights are hardcoded based on intuition.

**Affected Modules**:
- `alphapeptfast/scoring/isotope_scoring.py`: MS1 isotope scoring
  - Hardcoded: 30% peak coverage, 30% mass accuracy, 40% intensity correlation
- Future: Fragment intensity scoring, XIC correlation, combined PSM scoring

**Why This Matters**:
Data-driven weight optimization could significantly improve PSM discrimination, especially at low FDRs (< 1%).

**Proposed Approach**:
1. **Collect labeled data** from high-confidence PSMs at strict FDR (e.g., 0.1%)
   - Positives: PSMs passing strict FDR
   - Negatives: Decoy hits, or PSMs failing at relaxed FDR

2. **Feature engineering**:
   - Extract all sub-scores: peak coverage, mass accuracy, intensity correlation
   - Add context features: precursor m/z, charge, instrument type, sample type
   - Compute feature interactions (e.g., coverage × mass_accuracy)

3. **Model selection**:
   - **Option A**: Logistic regression (interpretable, fast)
   - **Option B**: Gradient boosting (XGBoost/LightGBM - best performance)
   - **Option C**: Grid search optimizing separation at 1% FDR (simplest)

4. **Validation**:
   - Cross-validation on multiple datasets
   - Test on different instruments (Orbitrap, Q-TOF, timsTOF)
   - Test on different samples (plasma, tissue, cell lysate)

**Expected Impact**:
- 10-30% more PSMs at 1% FDR
- Better calibrated probabilities
- Instrument/sample-specific optimization

**Implementation Timeline**:
- Phase 2A: Collect features from real data
- Phase 2B: Train initial model on single dataset
- Phase 2C: Validate on multiple datasets
- Phase 2D: Deploy learned weights as defaults

---

## Medium Priority

### 2. MS1 Isotope Distribution from Chemical Formula

**Current State**: Uses average peptide isotope approximation.

**Improvement**: Calculate exact isotope distribution from peptide sequence.

**Benefits**:
- More accurate for extreme compositions (e.g., high cysteine, proline-rich)
- Better scoring for modified peptides
- Handles unusual amino acids correctly

**Complexity**: Medium (need isotope pattern calculation library or implement)

**Reference**: `pyteomics.mass.isotopologues()` or similar

---

### 3. Adaptive Isotope Tolerance by Mass

**Current State**: Fixed 5-10 ppm tolerance for all m/z.

**Improvement**: Adaptive tolerance based on precursor m/z and instrument calibration.

**Rationale**:
- Low m/z (< 400): Tighter tolerance possible
- High m/z (> 1200): May need wider tolerance
- Post-calibration: Can use narrower tolerance

---

### 4. Charge State-Specific Isotope Scoring

**Current State**: Same scoring formula for all charges.

**Observation**:
- Charge 1: Isotope patterns less informative (less peaks doubly charged)
- Charge 3+: More complex patterns, different optimal weights

**Improvement**: Learn charge-specific scoring weights.

---

## Low Priority

### 5. Multi-Isotope Peak Detection

**Current State**: Takes most intense peak if multiple match.

**Improvement**: Use weighted average or sum of overlapping peaks.

**Use Case**: Overlapping precursors in DIA (common in plasma).

---

### 6. Missing Isotope Peak Imputation

**Current State**: Missing peaks scored as 0 intensity.

**Improvement**: Impute missing peaks from neighboring isotopes.

**Rationale**: Sometimes M+2 is missing due to noise threshold, but M+0, M+1, M+3 are present.

---

## Research Ideas

### 7. Deep Learning Isotope Scoring

**Idea**: Train a neural network to score isotope envelopes directly from raw spectrum.

**Input**: Cropped MS1 spectrum around precursor ± 5 m/z
**Output**: Probability that precursor is correct

**Advantages**:
- Learns complex patterns (satellite peaks, noise characteristics)
- No need for manual feature engineering
- Could detect co-eluting precursors

**Challenges**:
- Requires large labeled dataset
- Less interpretable than current approach
- Slower inference (but still fast with GPU)

---

### 8. Joint MS1/MS2 Scoring Model

**Idea**: Learn a joint model combining all evidence:
- MS1 isotope envelope
- Fragment intensity correlation
- Fragment mass accuracy
- XIC correlation
- RT prediction

**Approach**: Multi-task learning or stacking ensemble

**Expected Impact**: State-of-the-art PSM discrimination

---

## Design Decisions Log

### Phase 1G: MS1 Isotope Scoring (2025-11-02)

**Decision**: Use hardcoded scoring weights initially.

**Rationale**:
- Get working implementation deployed quickly
- Validate that features are informative
- Collect real data for learning weights later

**Future**: Replace with learned weights (see improvement #1)

---

### Phase 1F: Fragment Intensity Scoring (2025-11-02)

**Decision**: Use Pearson correlation instead of spectral angle or cosine similarity.

**Rationale**:
- Pearson correlation is scale-invariant (handles intensity differences)
- Well-understood statistical properties
- Fast to compute in Numba

**Alternative considered**: Spectral angle (less robust to intensity scale differences)

---

### Phase 1E: Mass Recalibration (2025-11-01)

**Decision**: Charge-state independent calibration.

**Rationale**:
- Systematic mass errors affect all charges equally (in modern MS)
- Simpler implementation
- Tests confirm single curve works for all charges

**Alternative considered**: Separate curves per charge (unnecessary complexity)

---

## Notes

This document is living and should be updated as:
- New design decisions are made
- Performance bottlenecks are identified
- User feedback is received
- New research papers suggest improvements

Last updated: 2025-11-02

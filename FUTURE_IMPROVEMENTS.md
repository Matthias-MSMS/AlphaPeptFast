# AlphaPeptFast - Future Improvements

This document tracks design decisions and future improvements identified during development.

## High Priority

### 1. MS2 Fragment Isotope Detection (High-Resolution Instruments)

**Motivation**: High-resolution TOF instruments (>1M resolution) can resolve M+1 isotopes
in MS2 fragments in ~70% of cases. This provides powerful orthogonal evidence for correct
fragment assignments.

**Current State**: Not implemented. We only use MS1 isotopes for precursor validation.

**Why This Matters**:
- Fragment isotopes validate charge state assignment
- Confirms fragment identity (not noise or interference)
- Instrument-specific advantage for high-res MS2
- Underutilized in current proteomics tools

**Implementation Design**:

```python
# In fragment_matching.py or new ms2_isotope_scoring.py

def detect_fragment_isotopes(
    spectrum_mz: np.ndarray,
    spectrum_intensity: np.ndarray,
    matched_fragment_mz: np.ndarray,
    matched_fragment_charge: np.ndarray,
    matched_fragment_mass: np.ndarray,
    tolerance_ppm: float = 10.0,
) -> dict:
    """Detect M+1 isotopes for matched fragments.

    Returns
    -------
    dict
        - 'n_with_isotope': Number of fragments with detectable M+1
        - 'isotope_fraction': Fraction of fragments showing isotopes
        - 'isotope_ratios': Observed M+1/M+0 intensity ratios
        - 'isotope_score': Combined score (0-1)
    """
    isotope_confirmations = 0
    isotope_ratios = []

    for i in range(len(matched_fragment_mz)):
        fragment_mz = matched_fragment_mz[i]
        fragment_charge = matched_fragment_charge[i]
        fragment_mass = matched_fragment_mass[i]

        # Calculate expected M+1 m/z
        m1_mz = fragment_mz + (ISOTOPE_MASS_DIFFERENCE / fragment_charge)

        # Binary search for M+1 peak
        m1_start, m1_end = binary_search_mz_range(spectrum_mz, m1_mz, tolerance_ppm)

        if m1_start == m1_end:
            continue  # No M+1 found

        # Find most intense peak in range (if multiple matches)
        m1_idx = m1_start + np.argmax(spectrum_intensity[m1_start:m1_end])

        # Check intensity ratio makes sense
        m0_intensity = spectrum_intensity[matched_indices[i]]
        m1_intensity = spectrum_intensity[m1_idx]

        # Expected M+1 ratio from fragment mass (~mass/1000 * 0.5 for peptides)
        expected_ratio = calculate_expected_m1_ratio(fragment_mass)
        observed_ratio = m1_intensity / m0_intensity

        # Accept if within 2x of expected (generous tolerance)
        if 0.5 * expected_ratio < observed_ratio < 2.0 * expected_ratio:
            isotope_confirmations += 1
            isotope_ratios.append(observed_ratio)

    isotope_fraction = isotope_confirmations / len(matched_fragment_mz)

    # Score only if enough evidence (adaptive threshold)
    if isotope_fraction > 0.3:  # At least 30% of fragments
        isotope_score = isotope_fraction  # Could be more sophisticated
    else:
        isotope_score = 0.0  # Not enough evidence (low-res instrument?)

    return {
        'n_with_isotope': isotope_confirmations,
        'isotope_fraction': isotope_fraction,
        'isotope_ratios': np.array(isotope_ratios),
        'isotope_score': isotope_score,
    }
```

**Adaptive Behavior**:
- **High-res instruments**: 50-70% of fragments show M+1 → strong evidence
- **Medium-res instruments**: 10-30% of fragments → weak evidence, low weight
- **Low-res instruments**: <10% of fragments → ignore (likely false positives)

**Integration into PSM Scoring**:
```python
# In combined PSM scoring
ms2_isotope_result = detect_fragment_isotopes(...)

if ms2_isotope_result['isotope_fraction'] > 0.3:
    # Add MS2 isotope evidence to combined score
    # Weight should be learned from data (see improvement #2)
    combined_score = (
        0.35 * fragment_intensity_score +
        0.25 * ms1_isotope_score +
        0.15 * ms2_isotope_score +  # NEW!
        0.15 * mass_accuracy_score +
        0.10 * rt_score
    )
```

**Expected Impact**:
- **High-res TOF**: 15-25% more PSMs at 1% FDR
- **Orbitrap**: 5-10% more PSMs (only large fragments)
- **Low-res**: No impact (feature not applicable)

**Implementation Complexity**: Medium
- Reuse binary search from MS1 isotope scoring
- Need expected M+1 ratio calculation (similar to MS1)
- Adaptive threshold logic
- Integration into PSM scoring

**Testing Requirements**:
- Test with high-res TOF data (timsTOF, Bruker maXis)
- Test with Orbitrap (should see partial isotopes)
- Test with low-res (should gracefully skip)
- Validate that ratios make physical sense

**References**:
- Proteome Discoverer uses fragment isotopes on high-res instruments
- MSFragger has experimental support for this
- Underutilized in most DIA tools

**Priority**: HIGH - This is a significant differentiator for high-res instruments
and relatively straightforward to implement given existing infrastructure.

**Timeline**:
- Phase 2B: Implement detection logic
- Phase 2C: Test on real high-res data
- Phase 2D: Integrate into combined scoring
- Phase 3A: Learn optimal weights

---

### 2. Learn Scoring Weights from Data

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

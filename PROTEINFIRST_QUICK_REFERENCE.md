# ProteinFirst Random Forest: Quick Reference

## One-Page Summary

**What**: Random Forest model for peptide-spectrum matching in DIA mass spectrometry
**Performance**: 96.14% top-1 ranking accuracy (median 4 candidates/spectrum)
**Key Features**: RT coelution dominates (+3.21 pts), intensity features negligible (+0.13 pts)
**Code Location**: `/Users/matthiasmann/Documents/projects/ProteinFirst_MS1centric`
**Model Location**: `/Users/matthiasmann/LocalData/mass_spec_data/ProteinFirst_MS1centric/data/results/`

---

## Data Pipeline (7 Stages)

```
Raw MS Data (HDF5)
       ↓
[1] Feature Detection (core_anneal_finder.py) → 271k MS2 features
       ↓
[2] MS1 Extraction (pre-done) → 271k precursors
       ↓
[3] Training Data Generation (build_training_data_rf.py) → 307k candidate pairs
       ↓
[4] RT Predictions (AlphaPeptDeep + PCHIP) → Calibrated iRT
       ↓
[5] Fragment Intensities (AlphaPeptDeep) → [SKIP - minimal benefit]
       ↓
[6] Feature Engineering (add_rt_features.py) → 33 production features
       ↓
[7] RF Training (train_random_forest.py) → Final model
```

**Critical files to run in order**:
1. `build_training_data_rf.py` (generates 307k training samples, 2-4 hours)
2. `generate_all_irt_predictions.py` (RT calibration, 15 seconds)
3. `create_clean_training_data.py` (cleanup, 1 minute)
4. `train_random_forest.py` (train model, 5 minutes)

---

## 33 Production Features (Categorized)

**RT Features (5)** ⭐ MOST IMPORTANT - 31.3% importance
- mean_rt_diff, median_rt_diff, std_rt_diff, min_rt_diff, max_rt_diff

**Ion Series (10)** - 19.2% importance
- n_b_ions, n_y_ions, y_to_b_ratio, b_series_continuity, y_series_continuity, max_continuity, n_high_mass_ions, n_low_mass_ions, n_mid_mass_ions, mean_fragment_spacing

**Fragment Match (12)** - 18.9% importance
- match_count, coverage, total_intensity, mean_intensity, max_intensity, median_intensity, intensity_std, mean_abs_ppm_error, ppm_error_std, max_abs_ppm_error, intensity_snr, match_efficiency

**Mass Accuracy (3)** - 10.8% importance
- mean_abs_ppm_error, ppm_error_std, max_abs_ppm_error

**Precursor (1)** - 8.2% importance
- precursor_intensity

**Other (2)** - 11.6% importance
- precursor_charge, matched_fragments_string

---

## Model Configuration

```python
RandomForestClassifier(
    n_estimators=500,       # 500 trees
    max_depth=20,           # Prevent overfitting
    min_samples_split=10,   # Minimum samples per split
    class_weight='balanced', # Handle class imbalance (15% positive)
    random_state=42,        # Reproducible
)
```

**Training Data**: 307,615 samples (47,845 positive, 259,770 negative)
**Train/Test Split**: 80/20 stratified
**Training Time**: ~5 minutes

---

## Performance Metrics

**Classification Accuracy**: 92.37%
- What fraction of (spectrum, peptide) pairs we classify correctly

**Top-1 Ranking Accuracy**: 96.14% ← PRODUCTION METRIC
- When correct peptide is in candidate list, how often is it ranked #1?
- Test set: 11,515 correct peptides, 11,070 ranked #1
- Median failure margin: 0.062 points (close calls)

**Per-Category Ablation**:
```
Fragment matching only:                92.80%
  + Mass accuracy:          +1.97 pts → 94.77%
  + Spectrum intensity:     +0.13 pts → 94.90%
  + RT coelution:           +3.21 pts → 98.11% ⭐
  + Precursor info:         +0.34 pts → 98.45%
  - Padding features:       -0.11 pts → 98.34%
Final (all 33):                        96.14%
```

---

## Key Insights

1. **RT Dominates**: RT coelution between fragments and precursor provides 3.21 point improvement. Most discriminative feature.

2. **Intensity Predictions Don't Help**: AlphaPeptDeep intensity predictions add 108 MB data and 10× computation but only +0.13 pts accuracy. Skip for AlphaPeptFast.

3. **AlphaPeptDeep Bug**: `predict_rt()` returns arbitrary order, not input order. Must merge on 'sequence'.

4. **RT Units**: Always use SECONDS internally. R = 0.98 with seconds, R = 0.01 with wrong units.

5. **Remove Training Artifacts**: `rank` and `candidate_rank` don't exist at inference time. Removing them has no accuracy loss.

6. **Simple Features Win**: Fragment matching alone gives 92.80%. Adding RT takes it to 96.14%.

---

## File Sizes & Times

```
Feature Extraction:
  build_training_data_rf.py     2-4 hours    Numba-parallel, 10 workers

RT Calibration:
  generate_all_irt_predictions  15 seconds   4,355 pep/sec (AlphaPeptDeep)
  PCHIP fitting                 <1 second

Training:
  train_random_forest.py        5 minutes    500 estimators, 307k samples
  Feature importance analysis   1 minute

Output Files:
  training_data_rf_clean.tsv    199 MB       307,615 rows × 33 features
  rf_model_production.pkl       429 MB       Sklearn pickle
  rf_feature_names.pkl          <1 KB        Column names list
```

---

## Critical Bug Examples

### AlphaPeptDeep Order Bug
```python
# WRONG - Gets r = 0.01
peptides = df['sequence'].tolist()
result = mm.predict_rt(df)
irt = result['rt_pred'].values  # Wrong order!

# CORRECT - Gets r = 0.98
result = mm.predict_rt(df)
df_merged = df.merge(result[['sequence', 'rt_pred']], on='sequence')
irt = df_merged['rt_pred'].values
```

### RT Units
```python
# WRONG - thinking RT is in minutes
if rt > 60:  # This is false for 20 minute run (1200 seconds)

# CORRECT
if rt_seconds > 600:  # More than 10 minutes
```

### Training Artifacts
```python
# WRONG - rank doesn't exist at inference time!
features['rank'] = final_rank

# CORRECT - only observable features
features['match_count'] = len(matched_fragments)
```

---

## To Port to AlphaPeptFast (11 hours total)

**HIGH PRIORITY** (2-3 hours):
- Binary search matching function (80 lines)
- Feature extraction function (150 lines)
- Window database building (25 lines)

**MEDIUM PRIORITY** (2-3 hours after Phase 2b):
- Scorer interface (100 lines)
- RF scorer implementation (100 lines)

**TOTAL EFFORT**: ~11 hours (1.5 days)

---

## Validation Checklist

- [ ] Can load training data (199 MB)
- [ ] Can load RT predictions (1.5 MB)
- [ ] Can load trained model (429 MB)
- [ ] Feature extraction produces 33 columns
- [ ] Model.predict_proba() works
- [ ] Top-1 accuracy >= 96%
- [ ] Unit tests pass
- [ ] Benchmarks comparable to original

---

## Reproduction Command

```bash
cd /Users/matthiasmann/Documents/projects/ProteinFirst_MS1centric
source .venv/bin/activate

python build_training_data_rf.py
python generate_all_irt_predictions.py
python create_clean_training_data.py
python train_random_forest.py
python calculate_top1_accuracy.py
```

Expected output: "Top-1 ranking accuracy: 96.14%"

---

## Contact Files

- **Full Documentation**: PROTEINFIRST_RECONSTRUCTION_GUIDE.md
- **RF Status**: RF_VALIDATION_STATUS.md
- **Integration Plan**: ALPHAPEPTFAST_INTEGRATION.md
- **AlphaPeptDeep Bug**: ~/.claude/skills/alphadia_deep.md


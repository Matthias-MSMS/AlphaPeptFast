#!/usr/bin/env python
"""Train Random Forest on 30 baseline features to reproduce ProteinFirst results.

This script verifies that AlphaPeptFast can independently reproduce ProteinFirst's
68.8% top-1 and 89.4% top-10 ranking accuracy using the same 30 baseline features.

Goal: Achieve independence from ProteinFirst codebase.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pickle
import csv
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 80)
print("AlphaPeptFast: Baseline RF Training (30 Features)")
print("=" * 80)
print()
print("Goal: Reproduce ProteinFirst's 68.8% top-1, 89.4% top-10 performance")
print("=" * 80)
print()

# Load ProteinFirst training data (has 30 baseline features already)
data_path = Path.home() / "LocalData/mass_spec_data/ProteinFirst_MS1centric/data/results/training_data_rf_clean.tsv"

print("[1] Loading training data...")
print(f"    Path: {data_path}")

rows = []
with open(data_path, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = list(reader.fieldnames)
    for row in reader:
        rows.append(row)

print(f"    ✓ Loaded {len(rows):,} samples")
print()

# Define 30 baseline features (production features, no training artifacts)
baseline_features = [
    # Fragment matching (12)
    'match_count', 'coverage',
    'n_b_ions', 'n_y_ions', 'y_to_b_ratio',
    'b_series_continuity', 'y_series_continuity', 'max_continuity',
    'n_high_mass_ions', 'n_low_mass_ions', 'n_mid_mass_ions',
    'mean_fragment_spacing',
    # Mass accuracy (3)
    'mean_abs_ppm_error', 'ppm_error_std', 'max_abs_ppm_error',
    # Intensity stats (10)
    'total_intensity', 'mean_intensity', 'max_intensity',
    'median_intensity', 'intensity_std', 'intensity_snr',
    # RT coelution (5) - most important!
    'mean_rt_diff', 'rt_diff_std', 'max_rt_diff', 'min_rt_diff', 'median_rt_diff',
    # Precursor (2)
    'precursor_charge', 'precursor_intensity',
    # Other (1)
    'match_efficiency',
]

print(f"[2] Preparing feature matrix...")
print(f"    Features: {len(baseline_features)}")
print(f"      Fragment matching: 12")
print(f"      Mass accuracy: 3")
print(f"      Intensity stats: 10")
print(f"      RT coelution: 5")
print(f"      Precursor: 2")
print(f"      Other: 1")
print()

# Build feature matrix
X = np.zeros((len(rows), len(baseline_features)), dtype=np.float32)
y = np.zeros(len(rows), dtype=np.int32)

for i, row in enumerate(rows):
    for j, feat in enumerate(baseline_features):
        val = row.get(feat, '0')
        X[i, j] = float(val) if val else 0.0
    y[i] = int(float(row.get('label', 0)))

# Handle NaNs
X = np.nan_to_num(X, nan=0.0)

print(f"    ✓ Feature matrix: {X.shape}")
print(f"    ✓ Labels: {y.shape}")
print(f"      Positives: {(y == 1).sum():,} ({100*(y==1).mean():.1f}%)")
print(f"      Negatives: {(y == 0).sum():,} ({100*(y==0).mean():.1f}%)")
print()

# Train/test split
print("[3] Creating train/test split...")
train_idx, test_idx = train_test_split(
    np.arange(len(rows)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"    ✓ Training: {len(train_idx):,} samples")
print(f"    ✓ Test: {len(test_idx):,} samples")
print()

# Train RF with ProteinFirst's hyperparameters
print("[4] Training Random Forest...")
print("    Hyperparameters (from ProteinFirst):")
print("      n_estimators: 500")
print("      max_depth: 20")
print("      min_samples_split: 10")
print("      class_weight: balanced")
print("      random_state: 42")
print()

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(X_train, y_train)
print()
print("    ✓ Model trained")
print()

# Classification accuracy
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
accuracy = (y_pred == y_test).mean()

print(f"    Classification accuracy: {100*accuracy:.2f}%")
print()

# Calculate TOP-1 and TOP-10 ranking accuracy
print("[5] Calculating ranking accuracy...")
print()

# Build dataframe-like structure for test set
test_data = []
for i, idx in enumerate(test_idx):
    row = rows[idx].copy()
    row['rf_score'] = y_proba[i]
    row['label'] = y[idx]
    test_data.append(row)

# Group by precursor (window, mz, rt, charge)
precursor_groups = {}
for row in test_data:
    key = (row['window'], row['precursor_mz'], row['precursor_rt'], row['precursor_charge'])
    if key not in precursor_groups:
        precursor_groups[key] = []
    precursor_groups[key].append(row)

print(f"    Total test samples: {len(test_data):,}")
print(f"    Unique precursors: {len(precursor_groups):,}")
print()

# Calculate top-1 and top-10 accuracy
top1_correct = 0
top10_correct = 0
total_precursors = 0

for key, candidates in precursor_groups.items():
    # Sort by RF score (descending)
    sorted_candidates = sorted(candidates, key=lambda x: x['rf_score'], reverse=True)

    # Check if any of the candidates is the true target (label=1)
    has_true_target = any(int(float(c['label'])) == 1 for c in sorted_candidates)

    if not has_true_target:
        continue  # Only count precursors where we have the true target

    total_precursors += 1

    # Check top-1
    if int(float(sorted_candidates[0]['label'])) == 1:
        top1_correct += 1
        top10_correct += 1  # Top-1 is also in top-10
    else:
        # Check top-10
        for i in range(min(10, len(sorted_candidates))):
            if int(float(sorted_candidates[i]['label'])) == 1:
                top10_correct += 1
                break

top1_accuracy = top1_correct / total_precursors if total_precursors > 0 else 0
top10_accuracy = top10_correct / total_precursors if total_precursors > 0 else 0

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Classification accuracy: {100*accuracy:.2f}%")
print(f"  (How often we correctly classify individual candidate pairs)")
print()
print(f"Top-1 ranking accuracy: {100*top1_accuracy:.2f}%")
print(f"  (How often correct peptide ranks #1 for its precursor)")
print()
print(f"Top-10 ranking accuracy: {100*top10_accuracy:.2f}%")
print(f"  (How often correct peptide is in top 10)")
print()
print(f"Test set breakdown:")
print(f"  Total precursors with true target: {total_precursors:,}")
print(f"  Correct peptide ranked #1: {top1_correct:,}")
print(f"  Correct peptide in top 10: {top10_correct:,}")
print()
print("=" * 80)
print("COMPARISON TO PROTEINFIRST BASELINE")
print("=" * 80)
print()
print(f"ProteinFirst (from analyze_top10_ranking.py):")
print(f"  Top-1:  68.8%")
print(f"  Top-10: 89.4%")
print()
print(f"AlphaPeptFast (this run):")
print(f"  Top-1:  {100*top1_accuracy:.1f}%")
print(f"  Top-10: {100*top10_accuracy:.1f}%")
print()

if abs(top1_accuracy - 0.688) < 0.05 and abs(top10_accuracy - 0.894) < 0.05:
    print("✅ SUCCESS! We've reproduced ProteinFirst's performance!")
    print("   AlphaPeptFast is now independent!")
else:
    print("⚠ Results differ from ProteinFirst baseline")
    print("  This may be due to:")
    print("  - Different train/test split (random seed)")
    print("  - Different subset of data")
    print("  - Need to investigate further")

print()
print("=" * 80)

# Save model
output_path = Path.cwd() / "rf_model_baseline_alphapeptfast.pkl"
print(f"[6] Saving model to {output_path}...")

model_data = {
    'model': rf,
    'feature_cols': baseline_features,
    'train_accuracy': accuracy,
    'top1_accuracy': top1_accuracy,
    'top10_accuracy': top10_accuracy,
}

with open(output_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"    ✓ Model saved")
print()
print("=" * 80)
print("DONE!")
print("=" * 80)

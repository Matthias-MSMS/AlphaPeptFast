#!/usr/bin/env python
"""Validate AlphaPeptFast feature extraction against ProteinFirst baseline.

This script:
1. Loads a sample of ProteinFirst training data
2. Loads the corresponding raw MS2 features
3. Re-runs our matching and feature extraction
4. Compares our features with ProteinFirst baseline
5. Reports any discrepancies

Usage:
    python scripts/validate_proteinfirst_features.py --n-samples 100
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
import argparse
from dataclasses import dataclass

from alphapeptfast.fragments.generator import generate_by_ions, encode_peptide_to_ord
from alphapeptfast.search.candidate_matching import match_candidates_batch, extract_features


# Stub classes for unpickling ProteinFirst data (from features.finder module)
@dataclass
class FeatureSet:
    """Container for feature finding results (stub for unpickling)."""
    mz: np.ndarray
    rt: np.ndarray
    mz_std: np.ndarray
    fwhm: np.ndarray
    intensity: np.ndarray
    quality: np.ndarray
    n_peaks: np.ndarray
    n_scans: np.ndarray
    rt_start: np.ndarray = None
    rt_stop: np.ndarray = None
    charge: np.ndarray = None
    charge_source: np.ndarray = None


# Register stub class for unpickling
import sys
sys.modules['features'] = sys.modules[__name__]
sys.modules['features.finder'] = sys.modules[__name__]


def load_training_sample(n_samples: int = 100) -> List[Dict]:
    """Load a small sample of ProteinFirst training data (pure Python/NumPy)."""
    data_path = Path.home() / "LocalData/mass_spec_data/ProteinFirst_MS1centric/data/results/training_data_rf_clean.tsv"

    print(f"Loading {n_samples} samples from {data_path}...")

    rows = []
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            rows.append(row)

    print(f"Loaded {len(rows)} rows")
    if rows:
        print(f"Columns: {list(rows[0].keys())}")
    return rows


def load_window_features(window: int):
    """Load MS2 features for a specific window."""
    # Use ms2_features_production directory which has windows 381-696+
    features_path = Path.home() / f"LocalData/mass_spec_data/ProteinFirst_MS1centric/data/ms2_features_production/features_{window}_core_anneal.pkl"

    if not features_path.exists():
        print(f"Warning: Window {window} features not found at {features_path}")
        return None

    with open(features_path, 'rb') as f:
        window_data = pickle.load(f)

    # Extract the core_features FeatureSet from the dict
    if isinstance(window_data, dict) and 'core_features' in window_data:
        return window_data['core_features']

    return window_data


def compare_features(our_features: Dict[str, float], pf_row: Dict) -> Dict[str, tuple]:
    """Compare our features with ProteinFirst baseline.

    Returns dict of {feature_name: (our_value, pf_value, diff)}
    """
    comparisons = {}

    # Map our feature names to ProteinFirst column names
    feature_mapping = {
        'match_count': 'match_count',
        'coverage': 'coverage',
        'n_b_ions': 'n_b_ions',
        'n_y_ions': 'n_y_ions',
        'y_to_b_ratio': 'y_to_b_ratio',
        'b_series_continuity': 'b_series_continuity',
        'y_series_continuity': 'y_series_continuity',
        'max_continuity': 'max_continuity',
        'n_high_mass_ions': 'n_high_mass_ions',
        'n_low_mass_ions': 'n_low_mass_ions',
        'n_mid_mass_ions': 'n_mid_mass_ions',
        'mean_fragment_spacing': 'mean_fragment_spacing',
        'mean_abs_ppm_error': 'mean_abs_ppm_error',
        'ppm_error_std': 'ppm_error_std',
        'max_abs_ppm_error': 'max_abs_ppm_error',
        'total_intensity': 'total_intensity',
        'mean_intensity': 'mean_intensity',
        'max_intensity': 'max_intensity',
        'median_intensity': 'median_intensity',
        'intensity_std': 'intensity_std',
        'intensity_snr': 'intensity_snr',
        'mean_rt_diff': 'mean_rt_diff',
        'std_rt_diff': 'rt_diff_std',  # Note: Different name!
        'max_rt_diff': 'max_rt_diff',
        'min_rt_diff': 'min_rt_diff',
        'median_rt_diff': 'median_rt_diff',
        'precursor_intensity': 'precursor_intensity',
        'precursor_charge': 'precursor_charge',
        'match_efficiency': 'match_efficiency',
    }

    for our_name, pf_name in feature_mapping.items():
        if our_name in our_features and pf_name in pf_row:
            our_val = our_features[our_name]
            pf_val_str = pf_row[pf_name]

            # Handle empty strings and convert to float
            try:
                pf_val = float(pf_val_str) if pf_val_str else 0.0
            except (ValueError, TypeError):
                pf_val = 0.0

            diff = abs(our_val - pf_val)
            comparisons[our_name] = (our_val, pf_val, diff)

    return comparisons


def validate_single_psm(row: Dict, window_features) -> Dict:
    """Validate a single PSM by re-extracting features."""

    if window_features is None:
        return None

    peptide = row['peptide']
    charge = int(float(row['precursor_charge']))  # Convert '2.0' -> 2.0 -> 2
    precursor_rt = float(row['precursor_rt'])
    precursor_intensity = float(row['precursor_intensity']) if 'precursor_intensity' in row else 1e6

    # Generate theoretical fragments
    try:
        peptide_ord = encode_peptide_to_ord(peptide)
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(peptide_ord, charge)
    except Exception as e:
        print(f"Error generating fragments for {peptide}: {e}")
        return None

    # Get spectrum from window features
    # NOTE: This is a simplified version - in real ProteinFirst, we'd need to
    # find the specific MS2 feature that matches this precursor
    # For now, we'll use the window's aggregated features

    spectrum_mz = window_features.mz
    spectrum_intensity = window_features.intensity
    spectrum_rt = window_features.rt

    if len(spectrum_mz) == 0:
        print(f"Empty spectrum for window {row['window']}")
        return None

    # Prepare for batch matching (single candidate)
    max_frags = len(theo_mz)
    candidate_mz = np.array([theo_mz], dtype=np.float64)
    candidate_type = np.array([theo_type], dtype=np.uint8)
    candidate_pos = np.array([theo_pos], dtype=np.uint8)
    candidate_charge = np.array([theo_charge], dtype=np.uint8)
    frags_per_cand = np.array([len(theo_mz)], dtype=np.int32)

    # Match fragments
    results = match_candidates_batch(
        candidate_mz, candidate_type, candidate_pos, candidate_charge, frags_per_cand,
        spectrum_mz, spectrum_intensity, spectrum_rt, precursor_rt,
        mz_tolerance_ppm=10.0, rt_tolerance_sec=10.0
    )

    match_count = results[0][0]
    match_intensities = results[1][0]
    match_mz_errors = results[2][0]
    match_rt_diffs = results[3][0]
    match_types = results[4][0]
    match_positions = results[5][0]
    match_charges = results[6][0]

    # Extract features
    our_features = extract_features(
        peptide=peptide,
        charge=charge,
        precursor_intensity=precursor_intensity,
        match_count=match_count,
        match_intensities=match_intensities,
        match_mz_errors=match_mz_errors,
        match_rt_diffs=match_rt_diffs,
        match_types=match_types,
        match_positions=match_positions,
        match_charges=match_charges,
        n_theoretical_fragments=len(theo_mz),
    )

    # Compare with ProteinFirst
    comparisons = compare_features(our_features, row)

    return {
        'peptide': peptide,
        'charge': charge,
        'our_features': our_features,
        'comparisons': comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate feature extraction on ProteinFirst data')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of PSMs to validate')
    parser.add_argument('--verbose', action='store_true', help='Print detailed comparisons')
    args = parser.parse_args()

    print("=" * 80)
    print("AlphaPeptFast Feature Extraction Validation")
    print("=" * 80)

    # Load training data
    rows = load_training_sample(args.n_samples)

    # Group by window to minimize file reads
    windows = set(row['window'] for row in rows)
    print(f"\nProcessing {len(windows)} unique windows...")

    all_comparisons = []
    successful_validations = 0
    failed_validations = 0

    for window in list(windows)[:5]:  # Limit to first 5 windows for now
        print(f"\n--- Window {window} ---")

        # Load window features
        window_features = load_window_features(int(window))
        if window_features is None:
            continue

        # Process PSMs in this window
        window_psms = [row for row in rows if row['window'] == window]
        print(f"Processing {len(window_psms)} PSMs...")

        for row in window_psms:
            result = validate_single_psm(row, window_features)

            if result is None:
                failed_validations += 1
                continue

            successful_validations += 1
            all_comparisons.append(result)

            if args.verbose:
                print(f"\nPeptide: {result['peptide']} (charge {result['charge']})")
                for feat_name, (our_val, pf_val, diff) in result['comparisons'].items():
                    if diff > 0.01:  # Only show significant differences
                        print(f"  {feat_name:25s}: ours={our_val:8.3f}, PF={pf_val:8.3f}, diff={diff:8.3f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Successful validations: {successful_validations}")
    print(f"Failed validations: {failed_validations}")

    if all_comparisons:
        print(f"\nAnalyzing {len(all_comparisons)} successful PSMs...")

        # Aggregate differences across all PSMs
        feature_diffs = {}
        for result in all_comparisons:
            for feat_name, (our_val, pf_val, diff) in result['comparisons'].items():
                if feat_name not in feature_diffs:
                    feature_diffs[feat_name] = []
                feature_diffs[feat_name].append(diff)

        print(f"\nMean Absolute Differences (across {len(all_comparisons)} PSMs):")
        print("-" * 60)
        for feat_name in sorted(feature_diffs.keys()):
            diffs = feature_diffs[feat_name]
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            print(f"{feat_name:25s}: mean={mean_diff:8.4f}, max={max_diff:8.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

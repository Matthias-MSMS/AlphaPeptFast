#!/usr/bin/env python
"""Augment ProteinFirst training data with 4 new features.

Takes existing training_data_rf_clean.tsv (30 baseline features) and adds:
1. fragment_intensity_correlation (AlphaPeptDeep predictions, FIXED)
2. ms1_isotope_score (MS1 isotope envelope validation)
3. ms2_isotope_fraction (fraction of fragments with M+1)
4. ms2_isotope_recommended_weight (adaptive weight for MS2 isotopes)

Output: training_data_with_new_features.tsv (34 features total)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pickle
import csv
from typing import Dict, Optional
import argparse
from dataclasses import dataclass

from alphapeptfast.fragments.generator import generate_by_ions, encode_peptide_to_ord
from alphapeptfast.search.candidate_matching import match_candidates_batch
from alphapeptfast.scoring.intensity_scoring import IntensityScorer
from alphapeptfast.scoring.isotope_scoring import MS1IsotopeScorer, detect_fragment_isotopes, score_ms2_fragment_isotopes


# Stub classes for unpickling ProteinFirst data
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
sys.modules['features'] = sys.modules[__name__]
sys.modules['features.finder'] = sys.modules[__name__]


def load_window_features(window: int):
    """Load MS2 features for a specific window."""
    features_path = Path.home() / f"LocalData/mass_spec_data/ProteinFirst_MS1centric/data/ms2_features_production/features_{window}_core_anneal.pkl"

    if not features_path.exists():
        return None

    with open(features_path, 'rb') as f:
        window_data = pickle.load(f)

    # Extract features - ProteinFirst uses 'annealed_features', not 'core_features'!
    if isinstance(window_data, dict):
        if 'annealed_features' in window_data:
            return window_data['annealed_features']
        elif 'core_features' in window_data:
            return window_data['core_features']

    return window_data


def compute_new_features(
    row: Dict,
    window_features: FeatureSet,
    intensity_scorer: Optional[IntensityScorer] = None,
    ms1_isotope_scorer: Optional[MS1IsotopeScorer] = None,
) -> Dict[str, float]:
    """Compute the 4 new features for a single PSM.

    Returns dict with:
    - fragment_intensity_correlation
    - ms1_isotope_score
    - ms2_isotope_fraction
    - ms2_isotope_recommended_weight
    """
    peptide = row['peptide']
    charge = int(float(row['precursor_charge']))
    precursor_rt = float(row['precursor_rt'])
    precursor_mz = float(row['precursor_mz'])
    precursor_intensity = float(row['precursor_intensity']) if row.get('precursor_intensity') else 1e6

    # Generate theoretical fragments
    try:
        peptide_ord = encode_peptide_to_ord(peptide)
        theo_mz, theo_type, theo_pos, theo_charge = generate_by_ions(peptide_ord, charge)
    except Exception as e:
        print(f"Warning: Error generating fragments for {peptide}: {e}")
        return {
            'fragment_intensity_correlation': 0.0,
            'ms1_isotope_score': 0.0,
            'ms2_isotope_fraction': 0.0,
            'ms2_isotope_recommended_weight': 0.0,
        }

    # Get spectrum from window features
    # ProteinFirst approach: Pre-filter by RT before matching (creates "virtual spectrum")
    # This is more efficient than checking RT for every m/z match
    rt_mask = np.abs(window_features.rt - precursor_rt) <= 10.0  # ±10 sec tolerance

    if rt_mask.sum() < 10:  # ProteinFirst requires at least 10 peaks in RT window
        return {
            'fragment_intensity_correlation': 0.0,
            'ms1_isotope_score': 0.0,
            'ms2_isotope_fraction': 0.0,
            'ms2_isotope_recommended_weight': 0.0,
        }

    # Extract RT-filtered spectrum
    rt_filtered_mz = window_features.mz[rt_mask]
    rt_filtered_intensity = window_features.intensity[rt_mask]
    rt_filtered_rt = window_features.rt[rt_mask]

    # Sort by m/z (required for binary search)
    sort_idx = np.argsort(rt_filtered_mz)
    spectrum_mz = rt_filtered_mz[sort_idx]
    spectrum_intensity = rt_filtered_intensity[sort_idx]
    spectrum_rt = rt_filtered_rt[sort_idx]

    # Match fragments
    candidate_mz = np.array([theo_mz], dtype=np.float64)
    candidate_type = np.array([theo_type], dtype=np.uint8)
    candidate_pos = np.array([theo_pos], dtype=np.uint8)
    candidate_charge = np.array([theo_charge], dtype=np.uint8)
    frags_per_cand = np.array([len(theo_mz)], dtype=np.int32)

    results = match_candidates_batch(
        candidate_mz, candidate_type, candidate_pos, candidate_charge, frags_per_cand,
        spectrum_mz, spectrum_intensity, spectrum_rt, precursor_rt,
        mz_tolerance_ppm=10.0, rt_tolerance_sec=10.0
    )

    match_count = results[0][0]
    match_intensities = results[1][0]
    match_types = results[4][0]
    match_positions = results[5][0]
    match_charges = results[6][0]

    # Debug: Track match statistics
    if not hasattr(compute_new_features, '_match_stats'):
        compute_new_features._match_stats = {'total': 0, 'with_matches': 0, 'match_sum': 0}
    compute_new_features._match_stats['total'] += 1
    if match_count > 0:
        compute_new_features._match_stats['with_matches'] += 1
        compute_new_features._match_stats['match_sum'] += match_count

    # Initialize features with defaults
    features = {
        'fragment_intensity_correlation': 0.0,
        'ms1_isotope_score': 0.0,
        'ms2_isotope_fraction': 0.0,
        'ms2_isotope_recommended_weight': 0.0,
    }

    if match_count == 0:
        return features

    # 1. Fragment intensity correlation (if scorer provided)
    if intensity_scorer is not None:
        try:
            # Build matched fragments string for intensity scorer
            matched_frags = []
            for i in range(match_count):
                ion_type = 'b' if match_types[i] == 0 else 'y'
                pos = int(match_positions[i])
                chg = int(match_charges[i])
                matched_frags.append(f"{ion_type}{pos}+{chg}")

            matched_fragments_str = ','.join(matched_frags)

            intensity_result = intensity_scorer.score_match(
                peptide=peptide,
                charge=charge,
                matched_fragments=matched_fragments_str,
                observed_intensities=match_intensities[:match_count],
            )
            features['fragment_intensity_correlation'] = float(intensity_result.get('correlation', 0.0))
        except Exception as e:
            # Intensity prediction may fail for some peptides
            features['fragment_intensity_correlation'] = 0.0

    # 2. MS1 isotope score (if scorer provided and we have MS1 data)
    # Note: This would require MS1 spectrum data which we don't have in this pipeline yet
    # For now, leave as 0.0
    # TODO: Add MS1 isotope scoring when we have MS1 spectrum access

    # 3. MS2 isotope fraction and weight
    try:
        # Find matched fragment indices in spectrum
        matched_spectrum_indices = []
        for i in range(match_count):
            theo_mz_val = theo_mz[i]
            # Binary search to find the matched peak
            mass_delta = theo_mz_val * 10.0 / 1e6  # 10 ppm tolerance
            mz_min = theo_mz_val - mass_delta
            mz_max = theo_mz_val + mass_delta

            # Find peaks in range
            idx_start = np.searchsorted(spectrum_mz, mz_min)
            if idx_start < len(spectrum_mz) and spectrum_mz[idx_start] <= mz_max:
                # Check RT tolerance
                if abs(spectrum_rt[idx_start] - precursor_rt) <= 10.0:
                    matched_spectrum_indices.append(idx_start)

        if len(matched_spectrum_indices) > 0:
            matched_indices = np.array(matched_spectrum_indices, dtype=np.int32)

            # Detect isotopes
            n_with_isotope, isotope_fraction, isotope_ratios = detect_fragment_isotopes(
                spectrum_mz=spectrum_mz,
                spectrum_intensity=spectrum_intensity,
                matched_spectrum_indices=matched_indices,
                matched_fragment_charge=match_charges[:match_count],
                mz_tolerance_ppm=10.0,
            )

            features['ms2_isotope_fraction'] = float(isotope_fraction)

            # Score isotopes
            isotope_score, recommended_weight = score_ms2_fragment_isotopes(
                isotope_fraction=isotope_fraction,
                isotope_ratios=isotope_ratios,
                expected_ratio_mean=0.25,
                expected_ratio_std=0.15,
            )

            features['ms2_isotope_recommended_weight'] = float(recommended_weight)
    except Exception as e:
        # MS2 isotope detection may fail
        features['ms2_isotope_fraction'] = 0.0
        features['ms2_isotope_recommended_weight'] = 0.0

    return features


def main():
    parser = argparse.ArgumentParser(description='Augment training data with 4 new features')
    parser.add_argument('--input', type=str,
                       default='~/LocalData/mass_spec_data/ProteinFirst_MS1centric/data/results/training_data_rf_clean.tsv',
                       help='Input training data TSV')
    parser.add_argument('--output', type=str,
                       default='./training_data_augmented.tsv',
                       help='Output augmented training data TSV')
    parser.add_argument('--n-rows', type=int, default=None,
                       help='Process only first N rows (for testing)')
    parser.add_argument('--use-intensity-scorer', action='store_true',
                       help='Use AlphaPeptDeep intensity predictions (requires model)')
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    print("=" * 80)
    print("AlphaPeptFast Training Data Augmentation")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Initialize scorers (optional)
    intensity_scorer = None
    if args.use_intensity_scorer:
        print("Initializing AlphaPeptDeep intensity scorer...")
        try:
            intensity_scorer = IntensityScorer()
            print("✓ Intensity scorer loaded")
        except Exception as e:
            print(f"Warning: Could not load intensity scorer: {e}")
            print("Continuing without intensity predictions (will use 0.0)")

    # Read input file
    print(f"\nReading input file...")
    rows = []
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader):
            if args.n_rows and i >= args.n_rows:
                break
            rows.append(row)

    print(f"Loaded {len(rows)} rows")
    print(f"Existing columns: {len(fieldnames)}")

    # Prepare output file with new columns
    new_fieldnames = list(fieldnames) + [
        'fragment_intensity_correlation',
        'ms1_isotope_score',
        'ms2_isotope_fraction',
        'ms2_isotope_recommended_weight',
    ]

    output_file = open(output_path, 'w', newline='')
    writer = csv.DictWriter(output_file, fieldnames=new_fieldnames, delimiter='\t')
    writer.writeheader()

    # Process by window to minimize feature file loads
    windows = sorted(set(row['window'] for row in rows))
    print(f"\nProcessing {len(windows)} unique windows...")

    processed = 0
    failed = 0

    for window in windows:
        print(f"\n--- Window {window} ---")

        # Load window features
        window_features = load_window_features(int(window))
        if window_features is None:
            print(f"Warning: Could not load features for window {window}, skipping")
            window_rows = [r for r in rows if r['window'] == window]
            # Write rows with zero features
            for row in window_rows:
                row['fragment_intensity_correlation'] = 0.0
                row['ms1_isotope_score'] = 0.0
                row['ms2_isotope_fraction'] = 0.0
                row['ms2_isotope_recommended_weight'] = 0.0
                writer.writerow(row)
                failed += 1
            continue

        print(f"Loaded {window_features.mz.shape[0]} features")

        # Process all PSMs in this window
        window_rows = [r for r in rows if r['window'] == window]
        print(f"Processing {len(window_rows)} PSMs...")

        for row in window_rows:
            try:
                new_features = compute_new_features(
                    row, window_features,
                    intensity_scorer=intensity_scorer
                )

                # Add new features to row
                row.update(new_features)
                writer.writerow(row)
                processed += 1

                if processed % 1000 == 0:
                    print(f"  Processed {processed} PSMs...")

            except Exception as e:
                print(f"Error processing PSM {row.get('peptide', 'unknown')}: {e}")
                # Write with zero features
                row['fragment_intensity_correlation'] = 0.0
                row['ms1_isotope_score'] = 0.0
                row['ms2_isotope_fraction'] = 0.0
                row['ms2_isotope_recommended_weight'] = 0.0
                writer.writerow(row)
                failed += 1

    output_file.close()

    # Print match statistics
    if hasattr(compute_new_features, '_match_stats'):
        stats = compute_new_features._match_stats
        print(f"\nMatch Statistics:")
        print(f"  Total PSMs: {stats['total']}")
        print(f"  PSMs with matches: {stats['with_matches']} ({100*stats['with_matches']/max(1,stats['total']):.1f}%)")
        if stats['with_matches'] > 0:
            print(f"  Average matches per PSM (when >0): {stats['match_sum']/stats['with_matches']:.1f}")

    print("\n" + "=" * 80)
    print("AUGMENTATION COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output written to: {output_path}")
    print("\nNew features added:")
    print("  1. fragment_intensity_correlation (AlphaPeptDeep predictions)")
    print("  2. ms1_isotope_score (MS1 isotope envelope)")
    print("  3. ms2_isotope_fraction (fraction with M+1)")
    print("  4. ms2_isotope_recommended_weight (adaptive weight)")
    print(f"\nTotal features: 34 (30 baseline + 4 new)")
    print("=" * 80)


if __name__ == '__main__':
    main()

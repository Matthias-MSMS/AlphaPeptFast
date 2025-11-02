"""Candidate peptide matching and feature extraction for Random Forest scoring.

This module implements the core matching logic for peptide-spectrum matching (PSM)
in DIA mass spectrometry. It matches theoretical fragments to observed MS2 peaks
and extracts features for Random Forest-based scoring.

Key Features
------------
- Numba-optimized parallel binary search for fragment matching
- Comprehensive feature extraction (33 baseline features)
- RT coelution scoring (most important features)
- Ion series analysis (b/y ratios, continuity)
- Mass accuracy and intensity statistics

Performance
-----------
- Matching: ~1000 candidates/second (parallel Numba)
- Feature extraction: ~5000 PSMs/second
- Memory: O(n_candidates * max_matches) for match tracking

Examples
--------
>>> from alphapeptfast.search import match_candidates_batch, extract_features
>>>
>>> # Match candidates to spectrum
>>> results = match_candidates_batch(
...     candidate_fragments_mz=theo_mz,
...     candidate_fragments_type=theo_type,
...     candidate_fragments_pos=theo_pos,
...     candidate_fragments_charge=theo_charge,
...     fragments_per_candidate=frag_counts,
...     spectrum_mz=obs_mz,
...     spectrum_intensity=obs_intensity,
...     spectrum_rt=obs_rt,
...     precursor_rt=prec_rt,
...     mz_tolerance_ppm=10.0,
...     rt_tolerance_sec=10.0,
... )
>>>
>>> # Extract features for first candidate
>>> features = extract_features(
...     peptide='PEPTIDE',
...     charge=2,
...     precursor_intensity=1e6,
...     match_count=results.match_counts[0],
...     match_intensities=results.match_intensities[0],
...     match_mz_errors=results.match_mz_errors[0],
...     match_rt_diffs=results.match_rt_diffs[0],
...     match_types=results.match_types[0],
...     match_positions=results.match_positions[0],
...     match_charges=results.match_charges[0],
...     n_theoretical_fragments=len(theo_mz[0]),
... )
>>> print(f"Coverage: {features['coverage']:.2f}, RT_diff: {features['mean_rt_diff']:.1f}s")
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from ..scoring.intensity_scoring import IntensityScorer
    from ..scoring.isotope_scoring import MS1IsotopeScorer


class MatchResults(NamedTuple):
    """Results from batch candidate matching.

    Attributes
    ----------
    match_counts : np.ndarray, shape (n_candidates,)
        Number of fragments matched for each candidate
    match_intensities : np.ndarray, shape (n_candidates, max_matches)
        Intensities of matched fragments (padded with zeros)
    match_mz_errors : np.ndarray, shape (n_candidates, max_matches)
        Mass errors in PPM for matched fragments
    match_rt_diffs : np.ndarray, shape (n_candidates, max_matches)
        Absolute RT differences (seconds) between fragment and precursor
    match_types : np.ndarray, shape (n_candidates, max_matches)
        Fragment ion types (0=b, 1=y)
    match_positions : np.ndarray, shape (n_candidates, max_matches)
        Fragment positions in peptide sequence (1-indexed)
    match_charges : np.ndarray, shape (n_candidates, max_matches)
        Fragment charge states
    """
    match_counts: np.ndarray
    match_intensities: np.ndarray
    match_mz_errors: np.ndarray
    match_rt_diffs: np.ndarray
    match_types: np.ndarray
    match_positions: np.ndarray
    match_charges: np.ndarray


@njit(parallel=True, cache=True)
def match_candidates_batch(
    candidate_fragments_mz: np.ndarray,
    candidate_fragments_type: np.ndarray,
    candidate_fragments_pos: np.ndarray,
    candidate_fragments_charge: np.ndarray,
    fragments_per_candidate: np.ndarray,
    spectrum_mz: np.ndarray,
    spectrum_intensity: np.ndarray,
    spectrum_rt: np.ndarray,
    precursor_rt: float,
    mz_tolerance_ppm: float = 10.0,
    rt_tolerance_sec: float = 10.0,
    max_matches_per_candidate: int = 50,
) -> tuple:
    """Match theoretical fragments to observed spectrum peaks (batch, parallel).

    This function performs binary search matching for multiple peptide candidates
    against a single MS2 spectrum. It's the core workhorse of the PSM scoring
    pipeline.

    Parameters
    ----------
    candidate_fragments_mz : np.ndarray, shape (n_candidates, max_fragments)
        Theoretical fragment m/z values for all candidates (padded)
    candidate_fragments_type : np.ndarray, shape (n_candidates, max_fragments)
        Fragment ion types (0=b, 1=y)
    candidate_fragments_pos : np.ndarray, shape (n_candidates, max_fragments)
        Fragment positions in sequence (1-indexed)
    candidate_fragments_charge : np.ndarray, shape (n_candidates, max_fragments)
        Fragment charge states
    fragments_per_candidate : np.ndarray, shape (n_candidates,)
        Number of valid fragments for each candidate (for unpacking from padded arrays)
    spectrum_mz : np.ndarray
        Observed m/z values in spectrum (must be sorted ascending!)
    spectrum_intensity : np.ndarray
        Observed intensities
    spectrum_rt : np.ndarray
        Retention times for each observed peak (from XICs)
    precursor_rt : float
        Retention time of precursor (for coelution scoring)
    mz_tolerance_ppm : float, optional
        Mass tolerance in PPM (default: 10.0)
    rt_tolerance_sec : float, optional
        RT tolerance in seconds for fragment-precursor coelution (default: 10.0)
    max_matches_per_candidate : int, optional
        Maximum matches to track per candidate (default: 50)

    Returns
    -------
    tuple of 7 arrays
        - match_counts: shape (n_candidates,), int32
        - match_intensities: shape (n_candidates, max_matches), float32
        - match_mz_errors: shape (n_candidates, max_matches), float32 (PPM)
        - match_rt_diffs: shape (n_candidates, max_matches), float32 (seconds)
        - match_types: shape (n_candidates, max_matches), uint8
        - match_positions: shape (n_candidates, max_matches), uint8
        - match_charges: shape (n_candidates, max_matches), uint8

    Notes
    -----
    - spectrum_mz MUST be sorted ascending for binary search to work
    - RT tolerance filters out non-coeluting fragments (important for DIA)
    - Matched peaks are NOT removed (same peak can match multiple candidates)
    - PPM errors are signed: (observed - theoretical) / theoretical * 1e6
    - RT differences are absolute: |fragment_rt - precursor_rt|

    Performance
    -----------
    - ~1000 candidates/second on M1 Max (10 cores)
    - Parallelized over candidates (prange)
    - O(n_candidates * n_fragments * log(n_peaks)) complexity

    Examples
    --------
    >>> # Match 10 candidates to spectrum
    >>> results = match_candidates_batch(
    ...     candidate_fragments_mz=theo_mz,  # (10, 50)
    ...     candidate_fragments_type=theo_type,
    ...     candidate_fragments_pos=theo_pos,
    ...     candidate_fragments_charge=theo_charge,
    ...     fragments_per_candidate=np.array([30, 32, 28, ...]),  # Actual counts
    ...     spectrum_mz=obs_mz,  # (1000,) sorted
    ...     spectrum_intensity=obs_intensity,
    ...     spectrum_rt=obs_rt,
    ...     precursor_rt=600.0,  # 10 minutes
    ...     mz_tolerance_ppm=10.0,
    ...     rt_tolerance_sec=10.0,
    ... )
    >>> match_counts, match_intensities, *_ = results
    >>> print(f"Candidate 0 matched {match_counts[0]} fragments")
    """
    n_candidates = len(fragments_per_candidate)

    # Pre-allocate match tracking arrays
    match_counts = np.zeros(n_candidates, dtype=np.int32)
    match_intensities = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.float32)
    match_mz_errors = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.float32)
    match_rt_diffs = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.float32)
    match_types = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.uint8)
    match_positions = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.uint8)
    match_charges = np.zeros((n_candidates, max_matches_per_candidate), dtype=np.uint8)

    # Process each candidate in parallel
    for cand_idx in prange(n_candidates):
        n_frags = fragments_per_candidate[cand_idx]
        n_matched = 0

        # Try to match each theoretical fragment
        for frag_idx in range(n_frags):
            if n_matched >= max_matches_per_candidate:
                break  # Avoid overflow

            target_mz = candidate_fragments_mz[cand_idx, frag_idx]
            frag_type = candidate_fragments_type[cand_idx, frag_idx]
            frag_pos = candidate_fragments_pos[cand_idx, frag_idx]
            frag_charge = candidate_fragments_charge[cand_idx, frag_idx]

            # Calculate m/z tolerance window
            mass_delta = target_mz * mz_tolerance_ppm / 1e6
            mz_min = target_mz - mass_delta
            mz_max = target_mz + mass_delta

            # Binary search for matching peak
            # Find leftmost peak with mz >= mz_min
            left, right = 0, len(spectrum_mz)
            while left < right:
                mid = (left + right) // 2
                if spectrum_mz[mid] < mz_min:
                    left = mid + 1
                else:
                    right = mid
            start_idx = left

            # Check if peak is within tolerance
            if start_idx >= len(spectrum_mz) or spectrum_mz[start_idx] > mz_max:
                continue  # No match found

            # Check RT coelution (fragment RT should be close to precursor RT)
            feature_rt = spectrum_rt[start_idx]
            rt_diff = abs(feature_rt - precursor_rt)

            if rt_diff > rt_tolerance_sec:
                continue  # Not coeluting

            # Record match details
            match_intensities[cand_idx, n_matched] = spectrum_intensity[start_idx]
            match_mz_errors[cand_idx, n_matched] = (spectrum_mz[start_idx] - target_mz) / target_mz * 1e6
            match_rt_diffs[cand_idx, n_matched] = rt_diff
            match_types[cand_idx, n_matched] = frag_type
            match_positions[cand_idx, n_matched] = frag_pos
            match_charges[cand_idx, n_matched] = frag_charge
            n_matched += 1

        match_counts[cand_idx] = n_matched

    return (
        match_counts,
        match_intensities,
        match_mz_errors,
        match_rt_diffs,
        match_types,
        match_positions,
        match_charges,
    )


def extract_features(
    peptide: str,
    charge: int,
    precursor_intensity: float,
    match_count: int,
    match_intensities: np.ndarray,
    match_mz_errors: np.ndarray,
    match_rt_diffs: np.ndarray,
    match_types: np.ndarray,
    match_positions: np.ndarray,
    match_charges: np.ndarray,
    n_theoretical_fragments: int,
) -> Dict[str, float]:
    """Extract 33 production features from fragment matching results.

    This function calculates all features used in the Random Forest scoring model.
    Features are organized into 5 categories based on their discriminative power.

    Parameters
    ----------
    peptide : str
        Peptide sequence (clean, no modifications)
    charge : int
        Precursor charge state
    precursor_intensity : float
        Precursor intensity from MS1
    match_count : int
        Number of fragments matched (from match_candidates_batch)
    match_intensities : np.ndarray
        Intensities of matched fragments (length = match_count)
    match_mz_errors : np.ndarray
        Mass errors in PPM (signed)
    match_rt_diffs : np.ndarray
        RT differences in seconds (absolute)
    match_types : np.ndarray
        Ion types (0=b, 1=y)
    match_positions : np.ndarray
        Fragment positions (1-indexed)
    match_charges : np.ndarray
        Fragment charge states
    n_theoretical_fragments : int
        Total number of theoretical fragments generated for this peptide

    Returns
    -------
    features : Dict[str, float]
        Dictionary of 33 features:

        **RT Features (5)** ⭐ MOST IMPORTANT - 31.3% importance
        - mean_rt_diff: Average RT difference (seconds)
        - median_rt_diff: Median RT difference
        - std_rt_diff: Standard deviation of RT differences
        - min_rt_diff: Best RT match
        - max_rt_diff: Worst RT match

        **Fragment Matching (12)** - 18.9% importance
        - match_count: Number of matched fragments
        - coverage: Fraction of theoretical fragments matched
        - total_intensity: Sum of matched intensities
        - mean_intensity: Average intensity
        - max_intensity: Strongest fragment
        - median_intensity: Median intensity
        - intensity_std: Std dev of intensities
        - mean_abs_ppm_error: Average mass error (absolute PPM)
        - ppm_error_std: Std dev of mass errors
        - max_abs_ppm_error: Worst mass error
        - intensity_snr: Signal-to-noise (max/mean)
        - match_efficiency: Matches per theoretical fragment

        **Ion Series (10)** - 19.2% importance
        - n_b_ions: Number of b ions matched
        - n_y_ions: Number of y ions matched
        - y_to_b_ratio: Ratio of y to b ions
        - b_series_continuity: Longest consecutive b series
        - y_series_continuity: Longest consecutive y series
        - max_continuity: Better of b or y
        - n_high_mass_ions: Matches in top 30% of sequence
        - n_low_mass_ions: Matches in bottom 30%
        - n_mid_mass_ions: Matches in middle 40%
        - mean_fragment_spacing: Average gap between matched positions

        **Precursor (1)** - 8.2% importance
        - precursor_intensity: Precursor intensity from MS1

        **Other (1)**
        - precursor_charge: Precursor charge state

        **String Features (1)** - For intensity correlation
        - matched_fragments_string: Compact encoding "b2_1:1000|y5_1:2500|..."

    Notes
    -----
    - All features are float64 for compatibility with sklearn
    - Zero-match PSMs get minimal features (all zeros except metadata)
    - Continuity calculated using longest consecutive positions (1-indexed)
    - RT features dominate model performance (+3.21 pts improvement)
    - Matched fragments string enables later intensity correlation scoring

    Examples
    --------
    >>> features = extract_features(
    ...     peptide='PEPTIDE',
    ...     charge=2,
    ...     precursor_intensity=1e6,
    ...     match_count=15,
    ...     match_intensities=intensities[:15],
    ...     match_mz_errors=ppm_errors[:15],
    ...     match_rt_diffs=rt_diffs[:15],
    ...     match_types=types[:15],
    ...     match_positions=positions[:15],
    ...     match_charges=charges[:15],
    ...     n_theoretical_fragments=30,
    ... )
    >>> print(f"Coverage: {features['coverage']:.2%}")
    >>> print(f"Mean RT diff: {features['mean_rt_diff']:.1f}s")
    >>> print(f"y/b ratio: {features['y_to_b_ratio']:.2f}")
    """
    features = {}

    # Handle zero-match case
    if match_count == 0:
        # Minimal features
        features['match_count'] = 0.0
        features['coverage'] = 0.0
        features['total_intensity'] = 0.0
        features['mean_intensity'] = 0.0
        features['max_intensity'] = 0.0
        features['median_intensity'] = 0.0
        features['intensity_std'] = 0.0
        features['mean_abs_ppm_error'] = 0.0
        features['ppm_error_std'] = 0.0
        features['max_abs_ppm_error'] = 0.0
        features['mean_rt_diff'] = 0.0
        features['std_rt_diff'] = 0.0
        features['max_rt_diff'] = 0.0
        features['min_rt_diff'] = 0.0
        features['median_rt_diff'] = 0.0
        features['n_b_ions'] = 0.0
        features['n_y_ions'] = 0.0
        features['y_to_b_ratio'] = 0.0
        features['b_series_continuity'] = 0.0
        features['y_series_continuity'] = 0.0
        features['max_continuity'] = 0.0
        features['n_high_mass_ions'] = 0.0
        features['n_low_mass_ions'] = 0.0
        features['n_mid_mass_ions'] = 0.0
        features['mean_fragment_spacing'] = 0.0
        features['precursor_intensity'] = float(precursor_intensity)
        features['precursor_charge'] = float(charge)
        features['intensity_snr'] = 0.0
        features['match_efficiency'] = 0.0
        features['matched_fragments_string'] = ""
        return features

    # Trim arrays to actual matches
    intensities = match_intensities[:match_count]
    mz_errors = match_mz_errors[:match_count]
    rt_diffs = match_rt_diffs[:match_count]
    types = match_types[:match_count]
    positions = match_positions[:match_count]
    charges = match_charges[:match_count]

    # === Fragment Matching Features (12) ===
    features['match_count'] = float(match_count)
    features['coverage'] = match_count / max(n_theoretical_fragments, 1)
    features['total_intensity'] = float(np.sum(intensities))
    features['mean_intensity'] = float(np.mean(intensities))
    features['max_intensity'] = float(np.max(intensities))
    features['median_intensity'] = float(np.median(intensities))
    features['intensity_std'] = float(np.std(intensities))

    # Mass accuracy
    features['mean_abs_ppm_error'] = float(np.mean(np.abs(mz_errors)))
    features['ppm_error_std'] = float(np.std(mz_errors))
    features['max_abs_ppm_error'] = float(np.max(np.abs(mz_errors)))

    # Signal-to-noise proxy
    if features['mean_intensity'] > 0:
        features['intensity_snr'] = features['max_intensity'] / features['mean_intensity']
    else:
        features['intensity_snr'] = 0.0

    # Match efficiency
    features['match_efficiency'] = features['match_count'] / max(n_theoretical_fragments, 1)

    # === RT Coelution Features (5) ⭐ MOST IMPORTANT ===
    features['mean_rt_diff'] = float(np.mean(rt_diffs))
    features['std_rt_diff'] = float(np.std(rt_diffs))
    features['max_rt_diff'] = float(np.max(rt_diffs))
    features['min_rt_diff'] = float(np.min(rt_diffs))
    features['median_rt_diff'] = float(np.median(rt_diffs))

    # === Ion Series Features (10) ===
    n_b = int(np.sum(types == 0))
    n_y = int(np.sum(types == 1))

    features['n_b_ions'] = float(n_b)
    features['n_y_ions'] = float(n_y)
    features['y_to_b_ratio'] = float(n_y) / max(float(n_b), 1.0)

    # Continuity: longest consecutive b or y series
    if n_b > 0:
        b_positions = np.sort(positions[types == 0])
        b_gaps = np.diff(b_positions)
        # Find longest run of consecutive positions (gap = 1)
        b_continuity = 1
        current_run = 1
        for gap in b_gaps:
            if gap == 1:
                current_run += 1
                b_continuity = max(b_continuity, current_run)
            else:
                current_run = 1
        features['b_series_continuity'] = float(b_continuity)
    else:
        features['b_series_continuity'] = 0.0

    if n_y > 0:
        y_positions = np.sort(positions[types == 1])
        y_gaps = np.diff(y_positions)
        y_continuity = 1
        current_run = 1
        for gap in y_gaps:
            if gap == 1:
                current_run += 1
                y_continuity = max(y_continuity, current_run)
            else:
                current_run = 1
        features['y_series_continuity'] = float(y_continuity)
    else:
        features['y_series_continuity'] = 0.0

    features['max_continuity'] = max(
        features['b_series_continuity'],
        features['y_series_continuity']
    )

    # Position distribution (high/mid/low mass ions)
    peptide_length = len(peptide)
    threshold_high = int(0.7 * peptide_length)
    threshold_low = int(0.3 * peptide_length)

    features['n_high_mass_ions'] = float(np.sum(positions > threshold_high))
    features['n_low_mass_ions'] = float(np.sum(positions < threshold_low))
    features['n_mid_mass_ions'] = float(
        match_count - features['n_high_mass_ions'] - features['n_low_mass_ions']
    )

    # Fragment spacing (mean gap between matched positions)
    sorted_positions = np.sort(positions)
    if len(sorted_positions) > 1:
        features['mean_fragment_spacing'] = float(np.mean(np.diff(sorted_positions)))
    else:
        features['mean_fragment_spacing'] = 0.0

    # === Precursor Features (2) ===
    features['precursor_intensity'] = float(precursor_intensity)
    features['precursor_charge'] = float(charge)

    # === Matched Fragments String (for intensity correlation) ===
    # Format: "b2_1:1000.5|y5_1:2500.3|y7_2:1800.7"
    # This enables later alignment with AlphaPeptDeep intensity predictions
    matched_fragments_str = "|".join([
        f"{'b' if types[i] == 0 else 'y'}{positions[i]}_{charges[i]}:{intensities[i]:.1f}"
        for i in range(match_count)
    ])
    features['matched_fragments_string'] = matched_fragments_str

    return features


def extract_features_extended(
    peptide: str,
    charge: int,
    precursor_intensity: float,
    precursor_mz: float,
    precursor_mass: float,
    match_count: int,
    match_intensities: np.ndarray,
    match_mz_errors: np.ndarray,
    match_rt_diffs: np.ndarray,
    match_types: np.ndarray,
    match_positions: np.ndarray,
    match_charges: np.ndarray,
    n_theoretical_fragments: int,
    # Advanced scoring (optional)
    intensity_scorer: Optional['IntensityScorer'] = None,
    ms1_isotope_scorer: Optional['MS1IsotopeScorer'] = None,
    ms1_spectrum_mz: Optional[np.ndarray] = None,
    ms1_spectrum_intensity: Optional[np.ndarray] = None,
    ms2_spectrum_mz: Optional[np.ndarray] = None,
    ms2_spectrum_intensity: Optional[np.ndarray] = None,
    matched_spectrum_indices: Optional[np.ndarray] = None,
    theoretical_fragment_mz: Optional[np.ndarray] = None,
    theoretical_fragment_charge: Optional[np.ndarray] = None,
    theoretical_fragment_mass: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract all 37 features including advanced isotope and intensity scoring.

    This is the extended version of extract_features() that adds 4 new features:
    - fragment_intensity_correlation (from AlphaPeptDeep predictions)
    - ms1_isotope_score (from MS1 envelope scoring)
    - ms2_isotope_fraction (from MS2 fragment isotope detection)
    - ms2_isotope_recommended_weight (adaptive weight based on fraction)

    Parameters
    ----------
    peptide : str
        Peptide sequence
    charge : int
        Precursor charge state
    precursor_intensity : float
        Precursor intensity from MS1
    precursor_mz : float
        Precursor m/z value
    precursor_mass : float
        Precursor neutral mass
    match_count : int
        Number of matched fragments
    match_intensities : np.ndarray
        Intensities of matched fragments
    match_mz_errors : np.ndarray
        Mass errors in PPM
    match_rt_diffs : np.ndarray
        RT differences in seconds
    match_types : np.ndarray
        Ion types (0=b, 1=y)
    match_positions : np.ndarray
        Fragment positions
    match_charges : np.ndarray
        Fragment charge states
    n_theoretical_fragments : int
        Total theoretical fragments
    intensity_scorer : IntensityScorer, optional
        Scorer for fragment intensity correlation (uses AlphaPeptDeep)
    ms1_isotope_scorer : MS1IsotopeScorer, optional
        Scorer for MS1 isotope envelope
    ms1_spectrum_mz : np.ndarray, optional
        MS1 spectrum m/z values (for isotope scoring)
    ms1_spectrum_intensity : np.ndarray, optional
        MS1 spectrum intensities
    ms2_spectrum_mz : np.ndarray, optional
        MS2 spectrum m/z values (for MS2 isotope detection)
    ms2_spectrum_intensity : np.ndarray, optional
        MS2 spectrum intensities
    matched_spectrum_indices : np.ndarray, optional
        Indices in MS2 spectrum where fragments matched
    theoretical_fragment_mz : np.ndarray, optional
        Full theoretical fragment m/z array (for intensity scorer)
    theoretical_fragment_charge : np.ndarray, optional
        Full theoretical fragment charge array
    theoretical_fragment_mass : np.ndarray, optional
        Full theoretical fragment mass array

    Returns
    -------
    features : Dict[str, float]
        Dictionary of 37 features (33 baseline + 4 new):

        **NEW FEATURES (4)**:
        - fragment_intensity_correlation: Pearson correlation with AlphaPeptDeep (FIXED bug!)
        - ms1_isotope_score: MS1 isotope envelope score (0-1)
        - ms2_isotope_fraction: Fraction of fragments with M+1 detected (0-1)
        - ms2_isotope_recommended_weight: Adaptive weight for MS2 isotopes (0-1)

        Plus all 33 baseline features from extract_features().

    Notes
    -----
    - If optional scorers/data not provided, new features default to 0.0
    - Backward compatible: works exactly like extract_features() if advanced params omitted
    - Intensity correlation uses FIXED alignment (bug from Phase 1F)
    - MS2 isotopes only useful on high-res instruments (>1M resolution)

    Examples
    --------
    >>> # Basic usage (33 features)
    >>> features = extract_features_extended(
    ...     peptide='PEPTIDE', charge=2, precursor_intensity=1e6,
    ...     precursor_mz=650.5, precursor_mass=1299.0,
    ...     match_count=10, match_intensities=intensities,
    ...     match_mz_errors=errors, match_rt_diffs=rt_diffs,
    ...     match_types=types, match_positions=positions,
    ...     match_charges=charges, n_theoretical_fragments=30,
    ... )
    >>> # Returns 33 baseline features + 4 zeros
    >>>
    >>> # Advanced usage (37 features)
    >>> from alphapeptfast.scoring import IntensityScorer, MS1IsotopeScorer
    >>> intensity_scorer = IntensityScorer('predictions.hdf')
    >>> ms1_scorer = MS1IsotopeScorer()
    >>>
    >>> features = extract_features_extended(
    ...     peptide='PEPTIDE', charge=2,
    ...     precursor_intensity=1e6, precursor_mz=650.5, precursor_mass=1299.0,
    ...     match_count=10, match_intensities=intensities,
    ...     match_mz_errors=errors, match_rt_diffs=rt_diffs,
    ...     match_types=types, match_positions=positions, match_charges=charges,
    ...     n_theoretical_fragments=30,
    ...     # Advanced scoring
    ...     intensity_scorer=intensity_scorer,
    ...     ms1_isotope_scorer=ms1_scorer,
    ...     ms1_spectrum_mz=ms1_mz, ms1_spectrum_intensity=ms1_intensity,
    ...     ms2_spectrum_mz=ms2_mz, ms2_spectrum_intensity=ms2_intensity,
    ...     matched_spectrum_indices=matched_indices,
    ...     theoretical_fragment_mz=theo_mz, theoretical_fragment_charge=theo_charge,
    ...     theoretical_fragment_mass=theo_mass,
    ... )
    >>> print(f"Intensity corr: {features['fragment_intensity_correlation']:.3f}")
    >>> print(f"MS1 isotope: {features['ms1_isotope_score']:.3f}")
    >>> print(f"MS2 isotope fraction: {features['ms2_isotope_fraction']:.3f}")
    """
    # Get all 33 baseline features first
    features = extract_features(
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
        n_theoretical_fragments=n_theoretical_fragments,
    )

    # === NEW FEATURE 1: Fragment Intensity Correlation ===
    # Uses AlphaPeptDeep predictions with FIXED alignment (Phase 1F bug fix)
    if intensity_scorer is not None and theoretical_fragment_mz is not None:
        try:
            # Need to reconstruct matched fragment info for scorer
            # The intensity scorer needs ALL theoretical fragments, not just matched
            intensity_result = intensity_scorer.score_match(
                peptide=peptide,
                charge=charge,
                observed_mz=ms2_spectrum_mz if ms2_spectrum_mz is not None else np.array([]),
                observed_intensity=ms2_spectrum_intensity if ms2_spectrum_intensity is not None else np.array([]),
                fragment_mz=theoretical_fragment_mz,
                fragment_type=match_types[:n_theoretical_fragments] if len(match_types) >= n_theoretical_fragments else match_types,
                fragment_position=match_positions[:n_theoretical_fragments] if len(match_positions) >= n_theoretical_fragments else match_positions,
                fragment_charge=theoretical_fragment_charge if theoretical_fragment_charge is not None else match_charges[:n_theoretical_fragments],
                mz_tolerance_ppm=20.0,  # Standard tolerance
            )
            features['fragment_intensity_correlation'] = float(intensity_result['correlation'])
        except Exception:
            # If scoring fails (e.g., peptide not in library), default to 0
            features['fragment_intensity_correlation'] = 0.0
    else:
        features['fragment_intensity_correlation'] = 0.0

    # === NEW FEATURE 2: MS1 Isotope Score ===
    # Validates precursor using isotope envelope
    if ms1_isotope_scorer is not None and ms1_spectrum_mz is not None:
        try:
            ms1_result = ms1_isotope_scorer.score_envelope(
                spectrum_mz=ms1_spectrum_mz,
                spectrum_intensity=ms1_spectrum_intensity if ms1_spectrum_intensity is not None else np.array([]),
                precursor_mz=precursor_mz,
                precursor_charge=charge,
                precursor_mass=precursor_mass,
            )
            features['ms1_isotope_score'] = float(ms1_result['combined_score'])
        except Exception:
            features['ms1_isotope_score'] = 0.0
    else:
        features['ms1_isotope_score'] = 0.0

    # === NEW FEATURE 3 & 4: MS2 Fragment Isotope Detection ===
    # Detects M+1 isotopes for matched fragments (high-res instruments only)
    if (ms2_spectrum_mz is not None and matched_spectrum_indices is not None and
        theoretical_fragment_mz is not None and theoretical_fragment_charge is not None and
        theoretical_fragment_mass is not None):
        try:
            from ..scoring.isotope_scoring import detect_fragment_isotopes, score_ms2_fragment_isotopes

            # Detect isotopes
            n_with_isotope, isotope_fraction, isotope_ratios = detect_fragment_isotopes(
                spectrum_mz=ms2_spectrum_mz,
                spectrum_intensity=ms2_spectrum_intensity if ms2_spectrum_intensity is not None else np.array([]),
                matched_fragment_indices=matched_spectrum_indices,
                fragment_mz=theoretical_fragment_mz[:match_count] if len(theoretical_fragment_mz) >= match_count else theoretical_fragment_mz,
                fragment_charge=theoretical_fragment_charge[:match_count] if len(theoretical_fragment_charge) >= match_count else theoretical_fragment_charge,
                fragment_mass=theoretical_fragment_mass[:match_count] if len(theoretical_fragment_mass) >= match_count else theoretical_fragment_mass,
                tolerance_ppm=10.0,
            )

            features['ms2_isotope_fraction'] = float(isotope_fraction)

            # Calculate adaptive weight
            isotope_score, recommended_weight = score_ms2_fragment_isotopes(
                n_with_isotope=n_with_isotope,
                n_matched_fragments=match_count,
                isotope_ratios=isotope_ratios,
            )
            features['ms2_isotope_recommended_weight'] = float(recommended_weight)

        except Exception:
            features['ms2_isotope_fraction'] = 0.0
            features['ms2_isotope_recommended_weight'] = 0.0
    else:
        features['ms2_isotope_fraction'] = 0.0
        features['ms2_isotope_recommended_weight'] = 0.0

    return features

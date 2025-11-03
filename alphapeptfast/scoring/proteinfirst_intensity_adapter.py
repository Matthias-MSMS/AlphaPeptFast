"""Adapter for ProteinFirst's pickle-based intensity predictions.

This adapter allows us to use ProteinFirst's pre-generated AlphaPeptDeep predictions
without requiring the HDF5 format loader.

Format: pickle file with dict['predictions'] = {(peptide, charge): {...intensities...}}
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from .intensity_scoring import normalize_intensities, pearson_correlation


def simple_binary_search_mz(spectrum_mz: np.ndarray, target_mz: float, tol_ppm: float) -> int:
    """Simple binary search for m/z matching (Python version for non-Numba code).

    Parameters
    ----------
    spectrum_mz : np.ndarray
        Sorted m/z array
    target_mz : float
        Target m/z to find
    tol_ppm : float
        PPM tolerance

    Returns
    -------
    int
        Index of best match, or -1 if no match within tolerance
    """
    if len(spectrum_mz) == 0:
        return -1

    # Calculate mass tolerance
    mass_tol = target_mz * tol_ppm / 1e6

    # Binary search for closest m/z
    left, right = 0, len(spectrum_mz)
    while left < right:
        mid = (left + right) // 2
        if spectrum_mz[mid] < target_mz:
            left = mid + 1
        else:
            right = mid

    # Check neighbors for best match within tolerance
    best_idx = -1
    best_error = mass_tol + 1.0

    for idx in [left - 1, left, left + 1]:
        if 0 <= idx < len(spectrum_mz):
            error = abs(spectrum_mz[idx] - target_mz)
            if error < best_error and error <= mass_tol:
                best_error = error
                best_idx = idx

    return best_idx


class ProteinFirstIntensityAdapter:
    """Adapter for ProteinFirst's intensity prediction format.

    Examples
    --------
    >>> adapter = ProteinFirstIntensityAdapter('fragment_intensity_predictions.pkl')
    >>> score = adapter.score_match(
    ...     peptide='PEPTIDE',
    ...     charge=2,
    ...     observed_mz=spectrum_mz,
    ...     observed_intensity=spectrum_intensity,
    ...     fragment_mz=theo_mz,
    ...     fragment_type=theo_type,
    ...     fragment_position=theo_pos,
    ...     fragment_charge=theo_charge,
    ... )
    """

    def __init__(self, pickle_path: Path | str):
        """Load predictions from ProteinFirst pickle file.

        Parameters
        ----------
        pickle_path : Path or str
            Path to fragment_intensity_predictions.pkl
        """
        self.pickle_path = Path(pickle_path)

        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)
            self.predictions = data['predictions']  # {(peptide, charge): {...}}
            self.metadata = data.get('metadata', {})

        print(f"Loaded {len(self.predictions)} intensity predictions from ProteinFirst")

    def score_match(
        self,
        peptide: str,
        charge: int,
        observed_mz: np.ndarray,
        observed_intensity: np.ndarray,
        fragment_mz: np.ndarray,
        fragment_type: np.ndarray,
        fragment_position: np.ndarray,
        fragment_charge: np.ndarray,
        mz_tolerance_ppm: float = 20.0,
    ) -> Dict[str, float]:
        """Score a peptide match using intensity correlation.

        Parameters
        ----------
        peptide : str
            Peptide sequence
        charge : int
            Precursor charge
        observed_mz : np.ndarray
            Observed m/z values from spectrum
        observed_intensity : np.ndarray
            Observed intensities
        fragment_mz : np.ndarray
            Theoretical fragment m/z (from generate_by_ions)
        fragment_type : np.ndarray
            Fragment types (0=b, 1=y)
        fragment_position : np.ndarray
            Fragment positions (1 to n-1)
        fragment_charge : np.ndarray
            Fragment charges
        mz_tolerance_ppm : float
            Mass tolerance for matching

        Returns
        -------
        dict
            Scoring results:
            - 'correlation': Overall intensity correlation
            - 'b_correlation': b-ion correlation
            - 'y_correlation': y-ion correlation
            - 'n_matched': Number of matched fragments
            - 'n_predicted': Number of predicted fragments
            - 'coverage': Fraction of predicted fragments matched
        """
        # Look up predictions
        key = (peptide, charge)
        if key not in self.predictions:
            return {
                'correlation': 0.0,
                'b_correlation': 0.0,
                'y_correlation': 0.0,
                'n_matched': 0,
                'n_predicted': 0,
                'coverage': 0.0,
            }

        pred_dict = self.predictions[key]

        # Extract prediction arrays (organized by position, 1-indexed)
        # pred_dict has: b_z1, b_z2, y_z1, y_z2 arrays (length = peptide_len - 1)
        pred_b_z1 = pred_dict['b_z1']  # shape: (n_positions,)
        pred_b_z2 = pred_dict['b_z2']
        pred_y_z1 = pred_dict['y_z1']
        pred_y_z2 = pred_dict['y_z2']

        # Match theoretical fragments to observed and extract predicted intensities
        matched_pred = []
        matched_obs = []
        matched_b_pred = []
        matched_b_obs = []
        matched_y_pred = []
        matched_y_obs = []

        n_predicted = 0  # Count fragments with non-zero predictions

        for i in range(len(fragment_mz)):
            ion_type = 'b' if fragment_type[i] == 0 else 'y'
            pos = int(fragment_position[i])  # 1-indexed
            frag_charge = int(fragment_charge[i])

            # Get predicted intensity (ProteinFirst uses 1-indexed positions)
            try:
                if ion_type == 'b' and frag_charge == 1:
                    pred_intensity = pred_b_z1[pos - 1]
                elif ion_type == 'b' and frag_charge == 2:
                    pred_intensity = pred_b_z2[pos - 1]
                elif ion_type == 'y' and frag_charge == 1:
                    pred_intensity = pred_y_z1[pos - 1]
                elif ion_type == 'y' and frag_charge == 2:
                    pred_intensity = pred_y_z2[pos - 1]
                else:
                    continue  # Unsupported fragment type/charge
            except IndexError:
                continue  # Position out of range

            # Skip if prediction is zero (AlphaPeptDeep predicts 0 for unrealistic fragments)
            if pred_intensity < 0.01:
                continue

            n_predicted += 1

            # Search for this fragment in observed spectrum
            obs_idx = simple_binary_search_mz(observed_mz, fragment_mz[i], mz_tolerance_ppm)

            if obs_idx == -1:
                continue  # Not observed

            obs_intensity = observed_intensity[obs_idx]

            # Store match
            matched_pred.append(pred_intensity)
            matched_obs.append(obs_intensity)

            if ion_type == 'b':
                matched_b_pred.append(pred_intensity)
                matched_b_obs.append(obs_intensity)
            else:
                matched_y_pred.append(pred_intensity)
                matched_y_obs.append(obs_intensity)

        # Calculate correlations
        result = {
            'correlation': 0.0,
            'b_correlation': 0.0,
            'y_correlation': 0.0,
            'n_matched': len(matched_pred),
            'n_predicted': n_predicted,
            'coverage': len(matched_pred) / n_predicted if n_predicted > 0 else 0.0,
        }

        if len(matched_pred) >= 3:
            # Normalize intensities before correlation
            pred_norm = normalize_intensities(np.array(matched_pred, dtype=np.float64))
            obs_norm = normalize_intensities(np.array(matched_obs, dtype=np.float64))
            result['correlation'] = float(pearson_correlation(pred_norm, obs_norm))

        if len(matched_b_pred) >= 3:
            pred_norm = normalize_intensities(np.array(matched_b_pred, dtype=np.float64))
            obs_norm = normalize_intensities(np.array(matched_b_obs, dtype=np.float64))
            result['b_correlation'] = float(pearson_correlation(pred_norm, obs_norm))

        if len(matched_y_pred) >= 3:
            pred_norm = normalize_intensities(np.array(matched_y_pred, dtype=np.float64))
            obs_norm = normalize_intensities(np.array(matched_y_obs, dtype=np.float64))
            result['y_correlation'] = float(pearson_correlation(pred_norm, obs_norm))

        return result

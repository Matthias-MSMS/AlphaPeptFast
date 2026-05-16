"""Hyperscore — log-factorial fragment-matching score for DDA search.

Two formulations exposed:

- **Additive** (legacy AlphaPeptLookup / v3c-v3i):
      hs_add = log(sum_matched_int + 1) + lnfact(n_b) + lnfact(n_y)

- **Multiplicative** (Sage / X!Tandem):
      hs_mul = log(sum_b_int + 1) + log(sum_y_int + 1) + lnfact(n_b) + lnfact(n_y)

In our E314_A1 campaign (May 2026) the additive form gave a higher PSM count
at strict 1% FDR when used as the per-spectrum top-1 ranker, because
multiplicative widens the decoy distribution proportionally. Multiplicative as
an *additional* feature for LDA rescoring did not help significantly.

Multi-charge fragment matching: when `max_frag_charge >= 2`, b/y ions are
matched at z=1 AND z=2. Sage default: `max_frag_charge = max(1, prec_charge - 1)`,
so charge-2 precursors stay z=1 and charge-3+ also try z=2. Returns separate
counts for z=1 and z=2 matches so a downstream classifier can weight them.

Public API
----------
- `log_factorial(n)`               — numba helper
- `match_one_mz(peak_mz, peak_intensity, target_mz, ppm_tolerance)`
                                   — closest-peak match
- `compute_hyperscore(...)`        — full b/y scoring with both forms +
                                     longest_b/y + multi-charge matches
"""
from __future__ import annotations
import numpy as np
from numba import njit


PROTON_MASS = 1.007276466621
H2O_MASS = 18.010564686


@njit(cache=True)
def log_factorial(n):
    """log(n!) via direct accumulation. Numba-jitted; n ≥ 0 expected."""
    result = 0.0
    for i in range(2, n + 1):
        result += np.log(float(i))
    return result


@njit(cache=True)
def match_one_mz(peak_mz, peak_intensity, target_mz, ppm_tolerance, n_peaks):
    """Binary-search the closest peak in [target ± ppm]. Returns (best_int, matched).

    Looks at the bisect_left position and ±1 neighbours, picks the one with
    smallest ppm error. Returns (intensity_of_match, True) if any candidate
    is within ppm_tolerance, else (0.0, False).

    Caller is responsible for ensuring peak_mz is sorted ascending.
    """
    lo, hi = 0, n_peaks
    while lo < hi:
        mid = (lo + hi) // 2
        if peak_mz[mid] < target_mz:
            lo = mid + 1
        else:
            hi = mid
    best_ppm = ppm_tolerance + 1.0
    best_int = 0.0
    for di in range(-1, 2):
        pi = lo + di
        if 0 <= pi < n_peaks:
            ppm_err = abs(peak_mz[pi] - target_mz) / target_mz * 1e6
            if ppm_err < best_ppm:
                best_ppm = ppm_err
                best_int = peak_intensity[pi]
    if best_ppm <= ppm_tolerance:
        return best_int, True
    return 0.0, False


@njit(cache=True)
def compute_hyperscore(peak_mz, peak_intensity, seq_codes, seq_len,
                      mod_position, mod_mass, aa_masses, ppm_tolerance,
                      max_frag_charge):
    """Full b/y hyperscore with multi-charge matching + both forms.

    Returns
    -------
    hs_add : float
        Additive hyperscore: log(sum_matched_int + 1) + lnfact(b) + lnfact(y).
    hs_mul : float
        Multiplicative hyperscore (Sage / X!Tandem): log(sum_b+1) + log(sum_y+1)
        + lnfact(b) + lnfact(y).
    n_matched : int
        Total matched fragments (z1 + z2, b + y).
    n_b_z1, n_y_z1, n_b_z2, n_y_z2 : int
        Per-charge counts.
    longest_b, longest_y : int
        Longest contiguous run of matched z=1 b/y ions (Sage feature).
    sum_matched_int : float
        Sum of intensities of all matched fragments.

    If matched < 1, both hs are 0.0.
    """
    n_ions = seq_len - 1
    n_peaks = len(peak_mz)
    if n_ions <= 0 or n_peaks == 0:
        return 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0

    total_mass = 0.0
    for k in range(seq_len):
        total_mass += aa_masses[seq_codes[k]]

    b_neutral = np.empty(n_ions, dtype=np.float64)
    y_neutral = np.empty(n_ions, dtype=np.float64)
    cumsum = 0.0
    for k in range(n_ions):
        cumsum += aa_masses[seq_codes[k]]
        b_neutral[k] = cumsum
        y_neutral[n_ions - 1 - k] = (total_mass - cumsum) + H2O_MASS

    n_b_z1 = 0; n_y_z1 = 0; n_b_z2 = 0; n_y_z2 = 0
    sum_b_int = 0.0; sum_y_int = 0.0
    longest_b = 0; current_b = 0
    longest_y = 0; current_y = 0

    for fi in range(n_ions):
        # Apply mod mass to b/y neutral mass when the mod position is
        # before/after the ion's break point.
        b_neutral_fi = b_neutral[fi] if fi < mod_position else b_neutral[fi] + mod_mass
        y_neutral_fi = y_neutral[fi] + mod_mass if mod_position >= seq_len - fi - 1 else y_neutral[fi]
        # b ion z=1
        b_z1_mz = b_neutral_fi + PROTON_MASS
        bint1, b1_match = match_one_mz(peak_mz, peak_intensity, b_z1_mz, ppm_tolerance, n_peaks)
        if b1_match:
            n_b_z1 += 1; sum_b_int += bint1
            current_b += 1
            if current_b > longest_b:
                longest_b = current_b
        else:
            current_b = 0
        if max_frag_charge >= 2:
            b_z2_mz = (b_neutral_fi + 2.0 * PROTON_MASS) / 2.0
            bint2, b2_match = match_one_mz(peak_mz, peak_intensity, b_z2_mz, ppm_tolerance, n_peaks)
            if b2_match:
                n_b_z2 += 1; sum_b_int += bint2
        # y ion z=1
        y_z1_mz = y_neutral_fi + PROTON_MASS
        yint1, y1_match = match_one_mz(peak_mz, peak_intensity, y_z1_mz, ppm_tolerance, n_peaks)
        if y1_match:
            n_y_z1 += 1; sum_y_int += yint1
            current_y += 1
            if current_y > longest_y:
                longest_y = current_y
        else:
            current_y = 0
        if max_frag_charge >= 2:
            y_z2_mz = (y_neutral_fi + 2.0 * PROTON_MASS) / 2.0
            yint2, y2_match = match_one_mz(peak_mz, peak_intensity, y_z2_mz, ppm_tolerance, n_peaks)
            if y2_match:
                n_y_z2 += 1; sum_y_int += yint2

    n_b = n_b_z1 + n_b_z2
    n_y = n_y_z1 + n_y_z2
    n_matched = n_b + n_y
    sum_matched_int = sum_b_int + sum_y_int
    if n_matched < 1:
        return 0.0, 0.0, 0, n_b_z1, n_y_z1, n_b_z2, n_y_z2, longest_b, longest_y, sum_matched_int

    lnf_b = log_factorial(n_b)
    lnf_y = log_factorial(n_y)
    hs_add = np.log(sum_matched_int + 1.0) + lnf_b + lnf_y
    hs_mul = np.log(sum_b_int + 1.0) + np.log(sum_y_int + 1.0) + lnf_b + lnf_y
    return hs_add, hs_mul, n_matched, n_b_z1, n_y_z1, n_b_z2, n_y_z2, longest_b, longest_y, sum_matched_int

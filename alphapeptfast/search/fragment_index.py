"""Inverted log-binned fragment index with IDF weighting + complementary lookup.

This is the core data structure behind AlphaPeptLookup (the fragment-index
search engine). Factored out of `dependent_search/core.py` (Mar 2026) so that
both round-1 (primary) and round-2 (variant / dependent) search engines can
share the same battle-tested primitives.

The architecture
----------------
1. **Logarithmic mass binning**: each fragment m/z is mapped to a bin index
   `floor(log(m/z) / log_bin_size)`. With `ppm_bin_size=5.0`, each bin holds
   fragments within ±5 ppm of one another. Bin number is constant in *ppm*
   space across the whole m/z range — a property linear binning cannot give.

2. **Inverted index**: for each bin, store the sorted list of peptide IDs
   whose fragments fall in that bin. Search becomes: for each peak, find its
   bin (and ±tolerance neighbours), accumulate hits per peptide.

3. **IDF weighting**: bins that are common across many peptides are
   down-weighted; bins that are specific (rare fragments) are up-weighted.
   This is the "discriminative fragments matter more" intuition formalised:
       bin_idf[b] = log(1 + N / (count[b] + 1))
   where N is total fragments in the index and count[b] is the bin's size.

4. **Complementary fragment lookup** (mirror): given precursor neutral
   mass M and a peak at m/z, the complementary b/y partner satisfies
       comp = M + 2*proton - peak
   Looking up *both* the peak and its complement doubles the matching
   capacity and — crucially — works at arbitrary precursor mass shift, so
   it enables open-modification search without enumerating mod hypotheses.

Public API
----------
- `build_fragment_index(frag_masses, frag_peptide_ids, ppm_bin_size=5.0)`
  → (index_peptide_ids, bin_starts, n_bins, log_bin_size, bin_idf)
- `search_spectrum_direct(...)`  — direct b/y matching within a precursor range
- `search_spectrum_complementary(...)` — direct + complementary (open mod)

All functions are numba-jitted and operate on flat numpy arrays.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def mass_to_bin(mass, log_bin_size):
    """Map a fragment m/z to its log bin index. Returns 0 for non-positive."""
    if mass <= 0.0:
        return 0
    return int(np.floor(np.log(mass) / log_bin_size))


@njit(cache=True)
def _bisect_left(arr, lo, hi, val):
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bisect_right(arr, lo, hi, val):
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _build_index_numba(frag_masses, frag_peptide_ids, n_bins, log_bin_size):
    n_frags = len(frag_masses)
    inv_lbs = 1.0 / log_bin_size
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    for i in range(n_frags):
        if frag_masses[i] > 0:
            b = int(np.floor(np.log(frag_masses[i]) * inv_lbs))
            if 0 <= b < n_bins:
                bin_counts[b] += 1
    bin_starts = np.empty(n_bins + 1, dtype=np.int32)
    bin_starts[0] = 0
    for i in range(n_bins):
        bin_starts[i + 1] = bin_starts[i] + bin_counts[i]
    total = bin_starts[n_bins]
    index_peptide_ids = np.empty(total, dtype=np.int32)
    bin_pos = bin_starts[:n_bins].copy()
    for i in range(n_frags):
        if frag_masses[i] > 0:
            b = int(np.floor(np.log(frag_masses[i]) * inv_lbs))
            if 0 <= b < n_bins:
                index_peptide_ids[bin_pos[b]] = frag_peptide_ids[i]
                bin_pos[b] += 1
    return index_peptide_ids, bin_starts


@njit(cache=True)
def _compute_bin_idf(bin_starts, n_bins, total_entries):
    bin_idf = np.empty(n_bins, dtype=np.float32)
    N = np.float32(total_entries)
    for b in range(n_bins):
        count = bin_starts[b + 1] - bin_starts[b]
        bin_idf[b] = np.float32(np.log(1.0 + N / (count + 1.0)))
    return bin_idf


def build_fragment_index(frag_masses, frag_peptide_ids, ppm_bin_size=5.0):
    """Build the log-binned inverted fragment index with IDF weights.

    Parameters
    ----------
    frag_masses : np.float32[N]
        Flat array of fragment m/z values from all peptides.
    frag_peptide_ids : np.int32[N]
        Peptide ID for each fragment (same length as frag_masses).
        IDs MUST be sorted within each peptide group; the function assumes
        the input is laid out so that peptide IDs end up monotonically
        non-decreasing within each bin (typical: emit fragments
        peptide-by-peptide in ID order).
    ppm_bin_size : float
        Bin width in ppm (default 5). Each bin holds fragments within
        ±ppm_bin_size of one another.

    Returns
    -------
    index_peptide_ids : np.int32[total]
        Concatenation of per-bin peptide-ID lists.
    bin_starts : np.int32[n_bins + 1]
        Cumulative start indices into `index_peptide_ids`. Bin `b` occupies
        `index_peptide_ids[bin_starts[b]:bin_starts[b+1]]`.
    n_bins : int
        Number of log bins.
    log_bin_size : float
        log(1 + ppm_bin_size / 1e6); used by the search routines.
    bin_idf : np.float32[n_bins]
        log(1 + N / (count[b] + 1)) per bin; used as a per-bin score weight.
    """
    log_bin_size = np.log(1.0 + ppm_bin_size / 1e6)
    valid = frag_masses > 0
    if valid.sum() == 0:
        return (np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                0, log_bin_size,
                np.empty(0, dtype=np.float32))
    max_mass = frag_masses[valid].max()
    n_bins = mass_to_bin(max_mass, log_bin_size) + 2
    index_peptide_ids, bin_starts = _build_index_numba(
        frag_masses, frag_peptide_ids, n_bins, log_bin_size)
    total_entries = len(index_peptide_ids)
    bin_idf = _compute_bin_idf(bin_starts, n_bins, total_entries)
    return index_peptide_ids, bin_starts, n_bins, log_bin_size, bin_idf


@njit(cache=True)
def search_spectrum_direct(peak_mz, peak_intensity,
                           index_peptide_ids, bin_starts, n_bins, log_bin_size,
                           bin_idf, precursor_lo, precursor_hi,
                           ppm_tolerance, top_k=50, min_matched=4):
    """Direct fragment-index search within a precursor-mass range.

    For each peak in the spectrum, look up matching peptides via the
    log-binned inverted index. Score each candidate by sqrt(intensity) *
    bin_idf summed over matched peaks. Returns top-K candidates with
    `count >= min_matched` matches.

    Parameters
    ----------
    peak_mz, peak_intensity : np.float64[n_peaks]
        Spectrum peaks (sorted by m/z).
    index_peptide_ids, bin_starts, n_bins, log_bin_size, bin_idf : returned
        from `build_fragment_index`.
    precursor_lo, precursor_hi : int
        Half-open peptide-ID range from precursor mass binary search.
        Only peptides with `precursor_lo <= peptide_id < precursor_hi`
        are considered.
    ppm_tolerance : float
        Fragment m/z tolerance in ppm.
    top_k : int
        Cap on number of candidates returned.
    min_matched : int
        Minimum number of matched peaks for a candidate to be returned.

    Returns
    -------
    out_ids : np.int32[k]
        Peptide IDs (within [precursor_lo, precursor_hi)).
    out_scores : np.float32[k]
        Sum of sqrt(intensity) * IDF across matched peaks.
    out_counts : np.int32[k]
        Number of matched peaks.
    """
    n_candidates = precursor_hi - precursor_lo
    if n_candidates <= 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    scores = np.zeros(n_candidates, dtype=np.float32)
    counts = np.zeros(n_candidates, dtype=np.int32)
    inv_lbs = 1.0 / log_bin_size

    for i in range(len(peak_mz)):
        mz = peak_mz[i]
        if mz <= 0.0:
            continue
        w = np.sqrt(peak_intensity[i]) if peak_intensity[i] > 0.0 else 0.0
        min_mz = mz * (1.0 - ppm_tolerance / 1e6)
        max_mz = mz * (1.0 + ppm_tolerance / 1e6)
        lo_bin = int(np.floor(np.log(min_mz) * inv_lbs))
        hi_bin = int(np.floor(np.log(max_mz) * inv_lbs)) + 1
        if lo_bin < 0:
            lo_bin = 0
        if hi_bin > n_bins:
            hi_bin = n_bins
        for b in range(lo_bin, hi_bin):
            bin_s = bin_starts[b]
            bin_e = bin_starts[b + 1]
            if bin_s >= bin_e:
                continue
            match_s = _bisect_left(index_peptide_ids, bin_s, bin_e, precursor_lo)
            match_e = _bisect_right(index_peptide_ids, match_s, bin_e, precursor_hi - 1)
            idf_w = w * bin_idf[b]
            for j in range(match_s, match_e):
                local = index_peptide_ids[j] - precursor_lo
                scores[local] += idf_w
                counts[local] += 1

    n_above = 0
    for i in range(n_candidates):
        if counts[i] >= min_matched:
            n_above += 1
    if n_above == 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    out_ids = np.empty(n_above, dtype=np.int32)
    out_scores = np.empty(n_above, dtype=np.float32)
    out_counts = np.empty(n_above, dtype=np.int32)
    pos = 0
    for i in range(n_candidates):
        if counts[i] >= min_matched:
            out_ids[pos] = precursor_lo + i
            out_scores[pos] = scores[i]
            out_counts[pos] = counts[i]
            pos += 1
    order = np.argsort(-out_scores)
    k = min(top_k, len(order))
    return out_ids[order[:k]], out_scores[order[:k]], out_counts[order[:k]]


@njit(cache=True)
def search_spectrum_complementary(peak_mz, peak_intensity,
                                    index_peptide_ids, bin_starts, n_bins, log_bin_size,
                                    bin_idf, precursor_lo, precursor_hi,
                                    observed_neutral_mass,
                                    ppm_tolerance, top_k=50, min_matched=4):
    """Direct + complementary (mirror) fragment-index search.

    For each peak `peak_mz[i]`, look up:
      (a) the peak itself, and
      (b) its complementary partner: `comp = M_obs + 2*proton - peak`,
    accumulating IDF-weighted scores per candidate from BOTH lookups.

    The complementary lookup is the key to open-modification search:
    when a peptide carries an unknown mass shift Δ, both b and y ions
    are shifted by Δ/2 individually but their sum still equals the
    *observed* precursor mass (not the theoretical). So pairing each
    peak with its complement against the observed precursor lets us
    match modified b/y partners without enumerating Δ.
    """
    n_candidates = precursor_hi - precursor_lo
    if n_candidates <= 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    scores = np.zeros(n_candidates, dtype=np.float32)
    counts = np.zeros(n_candidates, dtype=np.int32)
    inv_lbs = 1.0 / log_bin_size
    proton = 1.007276466621

    for i in range(len(peak_mz)):
        mz = peak_mz[i]
        if mz <= 0.0:
            continue
        w = np.sqrt(peak_intensity[i]) if peak_intensity[i] > 0.0 else 0.0
        for lookup in range(2):
            if lookup == 0:
                target_mz = mz
            else:
                target_mz = observed_neutral_mass + 2.0 * proton - mz
                if target_mz <= 0.0:
                    continue
            min_mz = target_mz * (1.0 - ppm_tolerance / 1e6)
            max_mz = target_mz * (1.0 + ppm_tolerance / 1e6)
            lo_bin = int(np.floor(np.log(min_mz) * inv_lbs))
            hi_bin = int(np.floor(np.log(max_mz) * inv_lbs)) + 1
            if lo_bin < 0:
                lo_bin = 0
            if hi_bin > n_bins:
                hi_bin = n_bins
            for b in range(lo_bin, hi_bin):
                bin_s = bin_starts[b]
                bin_e = bin_starts[b + 1]
                if bin_s >= bin_e:
                    continue
                match_s = _bisect_left(index_peptide_ids, bin_s, bin_e, precursor_lo)
                match_e = _bisect_right(index_peptide_ids, match_s, bin_e, precursor_hi - 1)
                idf_w = w * bin_idf[b]
                for j in range(match_s, match_e):
                    local = index_peptide_ids[j] - precursor_lo
                    scores[local] += idf_w
                    counts[local] += 1

    n_above = 0
    for i in range(n_candidates):
        if counts[i] >= min_matched:
            n_above += 1
    if n_above == 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    out_ids = np.empty(n_above, dtype=np.int32)
    out_scores = np.empty(n_above, dtype=np.float32)
    out_counts = np.empty(n_above, dtype=np.int32)
    pos = 0
    for i in range(n_candidates):
        if counts[i] >= min_matched:
            out_ids[pos] = precursor_lo + i
            out_scores[pos] = scores[i]
            out_counts[pos] = counts[i]
            pos += 1
    order = np.argsort(-out_scores)
    k = min(top_k, len(order))
    return out_ids[order[:k]], out_scores[order[:k]], out_counts[order[:k]]

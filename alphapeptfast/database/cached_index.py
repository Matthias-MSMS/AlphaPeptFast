"""Cached target+decoy peptide index for primary DDA/DIA search.

Building the search index from FASTA — digestion, decoy generation, mass
computation, sorting — is deterministic given (FASTA, MC, length range,
decoy method, fixed Cys mod, AA mass table). For a typical human SwissProt
search at MC=2, len 7-60 this is ~30s of work that is identical across every
run with the same parameters. Cache it.

The cache stores everything search engines need to start scoring spectra:
    sorted_masses    : np.float64[N]
    sorted_codes     : list[np.int32[L]]  (one per peptide; varying length)
    sorted_is_decoy  : np.bool_[N]
    sorted_peptides  : list[str]
    aa_mass_hash     : str (used to invalidate cache if mass table changes)

Cache layout: a single .npz with masses + is_decoy + sequences (str array)
plus a .json with the parameters and aa-mass hash. `sorted_codes` is rebuilt
from `sorted_peptides` on load (cheap: ~1s for 5M peptides).

Cache key (filename) is a short hash of:
    (fasta_path_resolved, fasta_size, fasta_mtime,
     missed_cleavages, min_length, max_length, decoy_method,
     cys_mod, aa_mass_hash)

We use mtime+size rather than full content hash so we don't re-read multi-MB
FASTAs to check freshness. If you copy a different FASTA over the same path
without changing mtime, you'll get a stale cache — that's a deliberate
ergonomic trade.

Typical use
-----------
>>> from alphapeptfast.database.cached_index import build_or_load_index
>>> idx = build_or_load_index(
...     fasta_path='human_swissprot.fasta',
...     missed_cleavages=2, min_length=7, max_length=60,
...     decoy_method='diann', cys_mod=57.021464,
... )
>>> idx.sorted_masses.shape, idx.sorted_is_decoy.sum()
((4961254,), 2480627)
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np

from .digestion import digest_fasta
from .decoys import generate_decoys


# 20 standard AAs — frozen here so a change to the table forces cache rebuild.
DEFAULT_AA_MASSES_DICT = {
    'G': 57.02146, 'A': 71.03711, 'V': 99.06841, 'L': 113.08406,
    'I': 113.08406, 'P': 97.05276, 'F': 147.06841, 'W': 186.07931,
    'M': 131.04049, 'S': 87.03203, 'T': 101.04768, 'C': 103.00919,
    'Y': 163.06333, 'H': 137.05891, 'D': 115.02694, 'E': 129.04259,
    'N': 114.04293, 'Q': 128.05858, 'K': 128.09496, 'R': 156.10111,
}
H2O_MASS = 18.010564686


def _aa_mass_hash(aa_dict: dict, cys_mod: float) -> str:
    items = sorted(aa_dict.items()) + [('__cys_mod__', cys_mod)]
    blob = json.dumps(items, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _fasta_signature(fasta_path: Path) -> tuple[int, int]:
    st = fasta_path.stat()
    return int(st.st_size), int(st.st_mtime)


def _make_aa_masses_array(aa_dict: dict, cys_mod: float) -> np.ndarray:
    masses = np.zeros(256, dtype=np.float64)
    for aa, m in aa_dict.items():
        masses[ord(aa)] = m
    masses[ord('C')] += cys_mod
    return masses


def _peptide_neutral_mass_array(seqs, aa_masses) -> np.ndarray:
    out = np.empty(len(seqs), dtype=np.float64)
    for i, s in enumerate(seqs):
        m = H2O_MASS
        for c in s:
            m += aa_masses[ord(c)]
        out[i] = m
    return out


def _peptides_to_codes(seqs) -> list:
    return [np.fromiter((ord(c) for c in s), dtype=np.int32, count=len(s)) for s in seqs]


def _peptides_to_flat_codes(seqs):
    """Build CSR-style flat-codes representation for numba-parallel consumers.

    Returns
    -------
    flat_codes : np.int32[total_length]
        Concatenation of all peptide codes.
    code_starts : np.int32[N+1]
        Cumulative start offsets; peptide ci occupies
        flat_codes[code_starts[ci]:code_starts[ci+1]].
    """
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
    code_starts = np.empty(len(seqs) + 1, dtype=np.int32)
    code_starts[0] = 0
    np.cumsum(lengths, out=code_starts[1:])
    flat_codes = np.empty(int(code_starts[-1]), dtype=np.int32)
    pos = 0
    for s in seqs:
        L = len(s)
        for i, c in enumerate(s):
            flat_codes[pos + i] = ord(c)
        pos += L
    return flat_codes, code_starts


@dataclass
class TargetDecoyIndex:
    sorted_masses: np.ndarray       # float64[N]
    sorted_codes: list              # list[int32[L]] length N
    sorted_is_decoy: np.ndarray     # bool[N]
    sorted_peptides: list           # list[str] length N
    n_targets: int
    n_decoys: int
    cache_path: Path | None = None
    cache_hit: bool = False
    build_seconds: float = 0.0
    # Flat (CSR) representation — built lazily on first access.
    # Required by numba-parallel scoring kernels.
    _flat_codes: np.ndarray | None = None
    _code_starts: np.ndarray | None = None

    def __repr__(self):
        return (f'TargetDecoyIndex(N={len(self.sorted_peptides):,}, '
                f'targets={self.n_targets:,}, decoys={self.n_decoys:,}, '
                f'cache_hit={self.cache_hit})')

    @property
    def flat_codes(self) -> np.ndarray:
        if self._flat_codes is None:
            self._flat_codes, self._code_starts = _peptides_to_flat_codes(self.sorted_peptides)
        return self._flat_codes

    @property
    def code_starts(self) -> np.ndarray:
        if self._code_starts is None:
            self._flat_codes, self._code_starts = _peptides_to_flat_codes(self.sorted_peptides)
        return self._code_starts


def _cache_key(
    fasta_path: Path,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
    decoy_method: str,
    cys_mod: float,
    aa_dict: dict,
) -> str:
    size, mtime = _fasta_signature(fasta_path)
    aa_hash = _aa_mass_hash(aa_dict, cys_mod)
    parts = [
        str(fasta_path.resolve()), str(size), str(mtime),
        str(missed_cleavages), str(min_length), str(max_length),
        decoy_method, f'{cys_mod:.6f}', aa_hash,
    ]
    blob = '|'.join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _default_cache_dir() -> Path:
    return Path(os.environ.get('APF_CACHE_DIR',
                                Path.home() / '.cache' / 'alphapeptfast' / 'index'))


def build_or_load_index(
    fasta_path: str | Path,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 60,
    decoy_method: str = 'diann',
    cys_mod: float = 57.021464,
    aa_masses_dict: dict | None = None,
    cache_dir: str | Path | None = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> TargetDecoyIndex:
    """Return a target+decoy index, building from FASTA on first call and
    loading from disk cache on subsequent calls with identical parameters.

    Parameters
    ----------
    fasta_path : str | Path
        Path to FASTA file.
    missed_cleavages : int
        Trypsin missed cleavages allowed.
    min_length, max_length : int
        Peptide-length range.
    decoy_method : str
        Passed to alphapeptfast.database.decoys.generate_decoys (default
        'diann' = reverse-keep-both-termini).
    cys_mod : float
        Fixed Cys modification mass (default 57.021464 = carbamidomethyl).
    aa_masses_dict : dict | None
        Override AA mass table. Default is the canonical 20-AA monoisotopic
        masses (DEFAULT_AA_MASSES_DICT).
    cache_dir : str | Path | None
        Where to store cached indices. Defaults to ~/.cache/alphapeptfast/index/
        (override with APF_CACHE_DIR env var).
    rebuild : bool
        If True, ignore any existing cache and rebuild.
    verbose : bool
        Print cache hit/miss + timing.
    """
    fasta_path = Path(fasta_path).resolve()
    if not fasta_path.exists():
        raise FileNotFoundError(fasta_path)
    aa_dict = aa_masses_dict if aa_masses_dict is not None else DEFAULT_AA_MASSES_DICT
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(fasta_path, missed_cleavages, min_length, max_length,
                      decoy_method, cys_mod, aa_dict)
    cache_npz = cache_dir / f'index_{key}.npz'
    cache_json = cache_dir / f'index_{key}.json'

    if not rebuild and cache_npz.exists() and cache_json.exists():
        t0 = time.perf_counter()
        # allow_pickle=True needed because peptides are stored as a numpy
        # object array (variable-length strings). Safe — we wrote the cache.
        data = np.load(cache_npz, allow_pickle=True)
        sorted_masses = data['masses']
        sorted_is_decoy = data['is_decoy']
        sorted_peptides = data['peptides'].tolist()
        sorted_codes = _peptides_to_codes(sorted_peptides)
        dt = time.perf_counter() - t0
        n_t = int((~sorted_is_decoy).sum())
        n_d = int(sorted_is_decoy.sum())
        if verbose:
            print(f'[apf cached_index] HIT  {cache_npz.name}  '
                  f'{len(sorted_peptides):,} peptides loaded in {dt:.1f}s')
        return TargetDecoyIndex(
            sorted_masses=sorted_masses,
            sorted_codes=sorted_codes,
            sorted_is_decoy=sorted_is_decoy,
            sorted_peptides=sorted_peptides,
            n_targets=n_t, n_decoys=n_d,
            cache_path=cache_npz, cache_hit=True, build_seconds=dt,
        )

    # ----- BUILD -----
    t0 = time.perf_counter()
    if verbose:
        print(f'[apf cached_index] MISS  building index for {fasta_path.name} '
              f'(MC={missed_cleavages}, len={min_length}-{max_length}, decoy={decoy_method})')
    unique_peptides, _peptide_to_proteins, _protein_db = digest_fasta(
        str(fasta_path), missed_cleavages=missed_cleavages,
        min_length=min_length, max_length=max_length,
    )
    target = list(set(unique_peptides))
    decoys = generate_decoys(target, method=decoy_method)
    all_pep = target + decoys
    is_decoy = np.array([False] * len(target) + [True] * len(decoys))
    aa_masses = _make_aa_masses_array(aa_dict, cys_mod)
    masses = _peptide_neutral_mass_array(all_pep, aa_masses)
    order = np.argsort(masses)
    sorted_masses = masses[order]
    sorted_peptides = [all_pep[i] for i in order]
    sorted_is_decoy = is_decoy[order]
    sorted_codes = _peptides_to_codes(sorted_peptides)
    dt = time.perf_counter() - t0
    if verbose:
        print(f'[apf cached_index] built {len(all_pep):,} peptides in {dt:.1f}s')

    # ----- WRITE CACHE -----
    t1 = time.perf_counter()
    pep_arr = np.array(sorted_peptides, dtype=object)
    np.savez(cache_npz, masses=sorted_masses, is_decoy=sorted_is_decoy,
              peptides=pep_arr)
    cache_json.write_text(json.dumps({
        'fasta_path': str(fasta_path),
        'fasta_size': fasta_path.stat().st_size,
        'fasta_mtime': fasta_path.stat().st_mtime,
        'missed_cleavages': missed_cleavages,
        'min_length': min_length, 'max_length': max_length,
        'decoy_method': decoy_method, 'cys_mod': cys_mod,
        'aa_mass_hash': _aa_mass_hash(aa_dict, cys_mod),
        'n_targets': int((~is_decoy).sum()),
        'n_decoys': int(is_decoy.sum()),
        'built_at': time.time(),
    }, indent=2))
    if verbose:
        print(f'[apf cached_index] wrote cache {cache_npz.name} '
              f'({cache_npz.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t1:.1f}s')

    return TargetDecoyIndex(
        sorted_masses=sorted_masses,
        sorted_codes=sorted_codes,
        sorted_is_decoy=sorted_is_decoy,
        sorted_peptides=sorted_peptides,
        n_targets=int((~is_decoy).sum()),
        n_decoys=int(is_decoy.sum()),
        cache_path=cache_npz, cache_hit=False, build_seconds=dt,
    )

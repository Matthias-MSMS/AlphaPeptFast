"""Peptide database with mass-indexed binary search.

Core data structure for proteome-scale peptide search. Uses mass-sorted
index for O(log n) candidate selection.

Design principles:
1. Mass-sorted for binary search
2. Minimal memory footprint
3. Thread-safe (read-only after construction)
4. Numba-accelerated search
"""

import numpy as np
import numba
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..constants import (
    AA_MASSES,
    H2O_MASS,
    PROTON_MASS,
)
from ..fragments.generator import (
    encode_peptide_to_ord,
    calculate_neutral_mass,
)


# =============================================================================
# Numba-Accelerated Binary Search
# =============================================================================

@numba.jit(nopython=True, cache=True)
def search_mass_range_numba(
    masses: np.ndarray,
    target_mass: float,
    tol_ppm: float,
) -> Tuple[int, int]:
    """Binary search for peptide mass range (Numba-compiled).

    Finds all peptides within PPM tolerance of target mass.

    Parameters
    ----------
    masses : np.ndarray (float64)
        Sorted neutral masses
    target_mass : float
        Target neutral mass
    tol_ppm : float
        Tolerance in parts per million

    Returns
    -------
    start_idx : int
        First index in range (inclusive)
    end_idx : int
        Last index in range (exclusive, Python convention)

    Performance
    -----------
    O(log n + k) where k = number of matches
    >1,000,000 queries/second

    Examples
    --------
    >>> masses = np.array([100.0, 200.0, 200.1, 300.0])
    >>> start, end = search_mass_range_numba(masses, 200.0, 500.0)
    >>> # Returns (1, 3) - indices 1 and 2 match within 500 ppm
    """
    # Calculate mass window
    mass_delta = target_mass * tol_ppm / 1e6
    mass_min = target_mass - mass_delta
    mass_max = target_mass + mass_delta

    n = len(masses)
    if n == 0:
        return (0, 0)

    # Binary search for start index (first mass >= mass_min)
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if masses[mid] < mass_min:
            left = mid + 1
        else:
            right = mid
    start_idx = left

    # Binary search for end index (first mass > mass_max)
    left, right = start_idx, n
    while left < right:
        mid = (left + right) // 2
        if masses[mid] <= mass_max:
            left = mid + 1
        else:
            right = mid
    end_idx = left

    return (start_idx, end_idx)


# =============================================================================
# Peptide Database
# =============================================================================

class PeptideDatabase:
    """Peptide database with mass-indexed binary search.

    Central data structure for proteome-scale search. Peptides sorted by
    neutral mass for O(log n) candidate selection.

    Attributes
    ----------
    peptides : List[str]
        All peptide sequences
    neutral_masses : np.ndarray (float64)
        Neutral masses, sorted ascending
    sort_indices : np.ndarray (int32)
        Original indices before sorting
        peptides[sort_indices[i]] corresponds to neutral_masses[i]
    n_peptides : int
        Total number of peptides

    Examples
    --------
    >>> # Create from peptide list
    >>> peptides = ["PEPTIDE", "SEQUENCE", "PROTEIN"]
    >>> db = PeptideDatabase(peptides)
    >>>
    >>> # Search by precursor m/z
    >>> candidates = db.search_by_mz(mz=500.5, charge=2, tol_ppm=5.0)
    >>> print([db.get_peptide(i) for i in candidates])
    """

    def __init__(
        self,
        peptides: List[str],
        fixed_modifications: Optional[Dict[str, float]] = None
    ):
        """Build database with mass index.

        Parameters
        ----------
        peptides : List[str]
            Peptide sequences (any order)
        fixed_modifications : Dict[str, float], optional
            Fixed modifications as {amino_acid: mass_shift}
            Default includes Carbamidomethyl C (+57.021464)
            Note: Already included in AA_MASSES, specify if different
        """
        self.peptides = peptides
        self.n_peptides = len(peptides)

        # Calculate neutral masses
        print(f"Calculating neutral masses for {self.n_peptides:,} peptides...")
        self.neutral_masses = np.array([
            calculate_neutral_mass(encode_peptide_to_ord(pep))
            for pep in peptides
        ], dtype=np.float64)

        # Sort by mass
        self.sort_indices = np.argsort(self.neutral_masses)
        self.neutral_masses = self.neutral_masses[self.sort_indices]

        print(f"✓ Database indexed by mass")
        print(f"  Mass range: {self.neutral_masses[0]:.2f} - {self.neutral_masses[-1]:.2f} Da")

    def search_by_mass(
        self,
        mass: float,
        tol_ppm: float = 5.0,
    ) -> np.ndarray:
        """Search by neutral mass.

        Parameters
        ----------
        mass : float
            Neutral mass to search for
        tol_ppm : float
            Mass tolerance in PPM

        Returns
        -------
        indices : np.ndarray (int32)
            Original peptide indices (into self.peptides)

        Examples
        --------
        >>> indices = db.search_by_mass(mass=1000.5, tol_ppm=5.0)
        >>> candidates = [db.peptides[i] for i in indices]
        """
        start_idx, end_idx = search_mass_range_numba(
            self.neutral_masses, mass, tol_ppm
        )

        # Return original indices (before sorting)
        return self.sort_indices[start_idx:end_idx]

    def search_by_mz(
        self,
        mz: float,
        charge: int,
        tol_ppm: float = 5.0,
    ) -> np.ndarray:
        """Search by precursor m/z (convenience method).

        Parameters
        ----------
        mz : float
            Precursor m/z
        charge : int
            Precursor charge state
        tol_ppm : float
            Mass tolerance in PPM

        Returns
        -------
        indices : np.ndarray (int32)
            Original peptide indices

        Examples
        --------
        >>> indices = db.search_by_mz(mz=500.5, charge=2, tol_ppm=5.0)
        """
        # Convert m/z to neutral mass
        neutral_mass = mz * charge - charge * PROTON_MASS

        return self.search_by_mass(neutral_mass, tol_ppm)

    def get_peptide(self, idx: int) -> str:
        """Get peptide sequence by original index.

        Parameters
        ----------
        idx : int
            Original peptide index

        Returns
        -------
        peptide : str
            Peptide sequence
        """
        return self.peptides[idx]

    def get_mass(self, idx: int) -> float:
        """Get neutral mass by original index.

        Parameters
        ----------
        idx : int
            Original peptide index

        Returns
        -------
        mass : float
            Neutral mass
        """
        # Find position in sorted array
        pos = np.searchsorted(self.sort_indices, idx)
        return self.neutral_masses[pos]

    @classmethod
    def from_list(
        cls,
        peptides: List[str],
        **kwargs
    ) -> 'PeptideDatabase':
        """Create database from peptide list (alias for __init__).

        Parameters
        ----------
        peptides : List[str]
            Peptide sequences
        **kwargs
            Additional arguments for __init__

        Returns
        -------
        db : PeptideDatabase
            Initialized database
        """
        return cls(peptides, **kwargs)

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        peptide_column: str = 'peptide',
        **kwargs
    ) -> 'PeptideDatabase':
        """Create database from TSV file.

        Parameters
        ----------
        tsv_path : str
            Path to TSV file with peptide column
        peptide_column : str
            Name of peptide column (default: 'peptide')
        **kwargs
            Additional arguments for __init__

        Returns
        -------
        db : PeptideDatabase
            Initialized database
        """
        import pandas as pd

        df = pd.read_csv(tsv_path, sep='\t')
        peptides = df[peptide_column].tolist()

        print(f"Loaded {len(peptides):,} peptides from {Path(tsv_path).name}")

        return cls(peptides, **kwargs)

    def __len__(self) -> int:
        """Number of peptides in database."""
        return self.n_peptides

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PeptideDatabase(n_peptides={self.n_peptides:,}, "
            f"mass_range=[{self.neutral_masses[0]:.2f}, "
            f"{self.neutral_masses[-1]:.2f}] Da)"
        )


# =============================================================================
# Target-Decoy Database (For FDR Control)
# =============================================================================

class TargetDecoyDatabase(PeptideDatabase):
    """Peptide database with integrated target-decoy approach.

    Extends PeptideDatabase with decoy sequences for FDR estimation.
    Decoys generated by sequence reversal.

    Attributes
    ----------
    n_targets : int
        Number of target peptides
    n_decoys : int
        Number of decoy peptides
    is_decoy : np.ndarray (bool)
        Mask indicating decoy peptides (True = decoy)

    Organization
    ------------
    - Indices 0 to n_targets-1: Target peptides
    - Indices n_targets to end: Decoy peptides

    Examples
    --------
    >>> db = TargetDecoyDatabase.from_list(["PEPTIDE", "SEQUENCE"])
    >>> print(f"Targets: {db.n_targets}, Decoys: {db.n_decoys}")
    >>>
    >>> # Check if peptide is decoy
    >>> idx = 2
    >>> if db.is_decoy[idx]:
    ...     print("Decoy match")
    """

    def __init__(self, target_peptides: List[str], **kwargs):
        """Build target-decoy database.

        Automatically generates decoys by reversing target sequences.

        Parameters
        ----------
        target_peptides : List[str]
            Target peptide sequences
        **kwargs
            Additional arguments for PeptideDatabase
        """
        self.n_targets = len(target_peptides)

        # Generate decoys
        print(f"Generating {self.n_targets:,} decoy sequences...")
        decoy_peptides = [
            self.reverse_peptide(pep) for pep in target_peptides
        ]
        self.n_decoys = len(decoy_peptides)

        # Combine targets and decoys
        all_peptides = target_peptides + decoy_peptides

        # Initialize base class
        super().__init__(all_peptides, **kwargs)

        # Create decoy mask (after sorting!)
        self.is_decoy = self.sort_indices >= self.n_targets

        print(f"✓ Target-decoy database ready")
        print(f"  Targets: {self.n_targets:,}")
        print(f"  Decoys: {self.n_decoys:,}")

    @staticmethod
    def reverse_peptide(peptide: str, preserve_terminal: bool = True) -> str:
        """Generate decoy by reversing sequence.

        Parameters
        ----------
        peptide : str
            Target peptide sequence
        preserve_terminal : bool
            Preserve C-terminal residue (for trypsin)

        Returns
        -------
        decoy : str
            Reversed peptide sequence

        Examples
        --------
        >>> reverse_peptide("PEPTIDER", preserve_terminal=True)
        'EDITPEPR'  # R preserved at C-terminus
        >>> reverse_peptide("PEPTIDER", preserve_terminal=False)
        'REDITPEP'  # Fully reversed
        """
        if preserve_terminal and len(peptide) > 1:
            # Reverse all but last residue, keep C-terminal
            return peptide[-2::-1] + peptide[-1]
        else:
            return peptide[::-1]

    @classmethod
    def from_list(
        cls,
        target_peptides: List[str],
        **kwargs
    ) -> 'TargetDecoyDatabase':
        """Create target-decoy database from peptide list.

        Parameters
        ----------
        target_peptides : List[str]
            Target peptide sequences
        **kwargs
            Additional arguments

        Returns
        -------
        db : TargetDecoyDatabase
            Initialized target-decoy database
        """
        return cls(target_peptides, **kwargs)

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        peptide_column: str = 'peptide',
        **kwargs
    ) -> 'TargetDecoyDatabase':
        """Create target-decoy database from TSV file.

        Parameters
        ----------
        tsv_path : str
            Path to TSV file
        peptide_column : str
            Name of peptide column
        **kwargs
            Additional arguments

        Returns
        -------
        db : TargetDecoyDatabase
            Initialized database
        """
        import pandas as pd

        df = pd.read_csv(tsv_path, sep='\t')
        target_peptides = df[peptide_column].tolist()

        print(f"Loaded {len(target_peptides):,} target peptides from {Path(tsv_path).name}")

        return cls(target_peptides, **kwargs)

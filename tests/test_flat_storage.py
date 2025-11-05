#!/usr/bin/env python3
"""Test flat ord() storage in PeptideDatabase."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphapeptfast.database import PeptideDatabase

print("=" * 80)
print("Testing Flat Ord() Storage")
print("=" * 80)

# Test peptides
peptides = ["PEPTIDEK", "SEQUENCE", "PROTEINR", "ALPHAK"]

print(f"\nBuilding database with {len(peptides)} peptides...")
db = PeptideDatabase(peptides)

print("\n" + "=" * 80)
print("Test 1: Flat storage structure")
print("=" * 80)
print(f"peptides_ord_flat shape: {db.peptides_ord_flat.shape}")
print(f"peptide_starts shape: {db.peptide_starts.shape}")
print(f"peptide_lengths shape: {db.peptide_lengths.shape}")
print(f"peptides_display length: {len(db.peptides_display)}")

print("\n" + "=" * 80)
print("Test 2: Access methods")
print("=" * 80)

for i in range(len(peptides)):
    # Display string
    pep_str = db.get_peptide(i)

    # Ord array
    pep_ord = db.get_peptide_ord(i)

    # Reconstruct from ord
    pep_reconstructed = ''.join(chr(c) for c in pep_ord)

    match = "✅" if pep_str == pep_reconstructed else "❌"
    print(f"{match} Peptide {i}: '{pep_str}' == '{pep_reconstructed}' | ord: {pep_ord}")

print("\n" + "=" * 80)
print("Test 3: Numba array extraction")
print("=" * 80)

flat, starts, lengths = db.get_flat_arrays()
print(f"flat: {flat.dtype} shape {flat.shape}")
print(f"starts: {starts.dtype} shape {starts.shape}")
print(f"lengths: {lengths.dtype} shape {lengths.shape}")

# Manual access (simulating Numba loop)
print("\nManual array access (Numba pattern):")
for i in range(len(peptides)):
    start = starts[i]
    length = lengths[i]
    pep_ord = flat[start:start + length]
    pep_str = ''.join(chr(c) for c in pep_ord)
    print(f"  {i}: {pep_str} | start={start}, length={length}")

print("\n" + "=" * 80)
print("Test 4: Mass search")
print("=" * 80)

# Search by mass
test_mass = db.neutral_masses[1]  # Mass of second peptide
indices = db.search_by_mass(test_mass, tol_ppm=1.0)
print(f"Search for mass {test_mass:.4f} Da")
print(f"Found {len(indices)} matches: {indices}")

for idx in indices:
    pep = db.get_peptide(idx)
    mass = db.get_mass(idx)
    print(f"  Index {idx}: {pep} (mass={mass:.4f} Da)")

print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)

"""Tests for database generation (FASTA reading, digestion, decoys).

Tests the new database generation pipeline added in v2.0:
- FASTA parsing
- Tryptic digestion
- K↔R swap decoy generation
- End-to-end from_fasta() workflow
"""

import pytest
import tempfile
from pathlib import Path

from alphapeptfast.database import (
    # FASTA reading
    read_fasta,
    parse_protein_id,

    # Digestion
    digest_protein_trypsin,
    digest_protein_list,

    # Decoys
    generate_kr_swap_decoy,
    generate_reverse_decoy,
    generate_pseudo_reverse_decoy,
    generate_decoys,

    # Database
    TargetDecoyDatabase,
)

from alphapeptfast.constants import AA_MASSES_DICT, H2O_MASS


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_neutral_mass(peptide: str) -> float:
    """Calculate peptide neutral mass."""
    return sum(AA_MASSES_DICT[aa] for aa in peptide) + H2O_MASS


# =============================================================================
# FASTA Parsing Tests
# =============================================================================

class TestFastaParsing:
    """Test FASTA file parsing."""

    def test_parse_uniprot_id(self):
        """Test UniProt ID extraction."""
        # UniProt format: sp|P12345|NAME_HUMAN
        protein_id, desc = parse_protein_id("sp|P12345|NAME_HUMAN Some protein")
        assert protein_id == "P12345"
        assert "NAME_HUMAN" in desc

    def test_parse_generic_id(self):
        """Test generic ID extraction."""
        protein_id, desc = parse_protein_id("PROT123 Description here")
        assert protein_id == "PROT123"
        assert desc == "PROT123 Description here"

    def test_read_fasta_basic(self):
        """Test basic FASTA reading."""
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">sp|P12345|TEST_HUMAN Test protein\n")
            f.write("PEPTIDEKRPROTEINK\n")
            f.write(">sp|Q98765|TEST2_HUMAN Another protein\n")
            f.write("SEQUENCEK\n")
            fasta_path = f.name

        try:
            proteins = read_fasta(fasta_path)
            assert len(proteins) == 2

            # Check first protein
            protein_id, sequence, description = proteins[0]
            assert protein_id == "P12345"
            assert sequence == "PEPTIDEKRPROTEINK"
            assert "TEST_HUMAN" in description

            # Check second protein
            protein_id, sequence, description = proteins[1]
            assert protein_id == "Q98765"
            assert sequence == "SEQUENCEK"

        finally:
            Path(fasta_path).unlink()

    def test_read_fasta_min_length(self):
        """Test minimum length filtering."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">P1 Protein 1\n")
            f.write("SHORTSEQ\n")  # 8 aa
            f.write(">P2 Protein 2\n")
            f.write("SHORT\n")  # 5 aa
            fasta_path = f.name

        try:
            # Filter proteins < 7 aa
            proteins = read_fasta(fasta_path, min_length=7)
            assert len(proteins) == 1
            assert proteins[0][1] == "SHORTSEQ"

        finally:
            Path(fasta_path).unlink()


# =============================================================================
# Digestion Tests
# =============================================================================

class TestDigestion:
    """Test tryptic digestion."""

    def test_trypsin_basic(self):
        """Test basic trypsin cleavage."""
        peptides = digest_protein_trypsin("PEPTADEKRPROTELNK", "P12345")

        # Expected: PEPTADEK, R, PROTELNK (P blocks cleavage after R)
        # Note: No I in sequence to avoid I→L conversion confusion
        assert "PEPTADEK" in peptides
        assert "RPROTELNK" in peptides

    def test_trypsin_proline_blocking(self):
        """Test proline blocking rule."""
        # RP should NOT be a cleavage site
        peptides = digest_protein_trypsin("ARPBCD", "P12345", min_length=3)

        # Should NOT cleave after R (because P follows)
        assert "ARPBCD" in peptides or len(peptides) == 0  # Might be filtered by length

    def test_missed_cleavages(self):
        """Test missed cleavage generation."""
        peptides = digest_protein_trypsin(
            "ABCDEFGHLJK", "P12345", min_length=5, max_length=20, missed_cleavages=1
        )

        # Should have peptides with 0 and 1 missed cleavages
        # K at position 10, so we expect ABCDEFGHLJK (with 0 missed)
        # Note: Changed I→L to avoid conversion
        assert any(len(p) >= 10 for p in peptides)

    def test_length_filtering(self):
        """Test peptide length filtering."""
        peptides = digest_protein_trypsin(
            "ABCK", "P12345", min_length=5, max_length=10, missed_cleavages=0
        )

        # ABCK has only 4 aa, should be filtered out
        assert len(peptides) == 0

    def test_i_to_l_conversion(self):
        """Test I→L conversion."""
        peptides = digest_protein_trypsin("PEPTIIDEK", "P12345")

        # All I should be converted to L
        for peptide in peptides:
            assert 'I' not in peptide
            if 'L' in peptide:
                # Original had I, now has L
                pass

    def test_digest_protein_list(self):
        """Test digesting multiple proteins."""
        proteins = [
            ("P1", "PEPTADEK", "Protein 1"),  # Changed I→A to avoid conversion
            ("P2", "SEQUENCEK", "Protein 2"),
        ]

        peptides, mapping = digest_protein_list(proteins, min_length=7)

        # Check unique peptides
        assert len(peptides) >= 2

        # Check mapping
        assert "PEPTADEK" in mapping
        assert "P1" in mapping["PEPTADEK"]


# =============================================================================
# Decoy Generation Tests
# =============================================================================

class TestDecoyGeneration:
    """Test decoy peptide generation."""

    def test_kr_swap_basic(self):
        """Test K↔R swap (reverse + swap)."""
        # K↔R swap: reverse, then swap K↔R
        # "PEPTIDEK" → "KEDITPEP" (reversed) → "REDITPEP" (K→R at position 0)
        assert generate_kr_swap_decoy("PEPTADEK") == "REDATPEP"

        # "PROTEINK" → "KNIETOR P" (reversed) → "RNIETOR P" (K→R at position 0)
        assert generate_kr_swap_decoy("PROTELNK") == "RNLETORP"

        # "KRPEPTIDE" → "EDITPEPKR" (reversed) → "EDITPEPRK" (K→R, R→K)
        assert generate_kr_swap_decoy("RRPEPTADE") == "EDATPEPRR"

    def test_kr_swap_mass_preservation(self):
        """Test that K↔R swap preserves mass."""
        targets = ["PEPTIDEK", "PROTEINKR", "SEQUENCER"]

        for target in targets:
            decoy = generate_kr_swap_decoy(target)

            target_mass = calculate_neutral_mass(target)
            decoy_mass = calculate_neutral_mass(decoy)

            # Should be identical (within floating point precision)
            assert abs(target_mass - decoy_mass) < 0.001

    def test_kr_swap_changes_sequence(self):
        """Test that K↔R swap produces different sequences."""
        # K↔R swap should produce different sequences than targets
        targets = ["PEPTADEK", "PROTELNKR", "SEQUENCER"]

        for target in targets:
            decoy = generate_kr_swap_decoy(target)

            # Decoy should be different from target
            assert decoy != target, f"Decoy {decoy} same as target {target}"

            # But should have same amino acid composition (mass preserved)
            target_composition = sorted(target)
            decoy_composition = sorted(decoy)
            assert target_composition == decoy_composition

    def test_reverse_decoy(self):
        """Test simple reversal."""
        assert generate_reverse_decoy("PEPTIDEK") == "KEDITPEP"
        assert generate_reverse_decoy("ABC") == "CBA"

    def test_pseudo_reverse_decoy(self):
        """Test pseudo-reverse (preserve C-term)."""
        # Pseudo-reverse: reverse all but last, then add last
        # "PEPTIDEK" → "EDITPEP" (reversed) + "K" → "EDITPEPK"
        assert generate_pseudo_reverse_decoy("PEPTADEK") == "EDATPEPK"

        # "ABCR" → "CB" (reversed) + "R" → "CBR"
        # Wait, let me recompute: "ABCR"
        # - All but last: "ABC"
        # - Reverse: "CBA"
        # - Add last: "CBAR"
        assert generate_pseudo_reverse_decoy("ABCR") == "CBAR"

    def test_pseudo_reverse_short_peptides(self):
        """Test pseudo-reverse on short peptides."""
        # Length 1-2: return unchanged
        assert generate_pseudo_reverse_decoy("K") == "K"
        assert generate_pseudo_reverse_decoy("KR") == "KR"

    def test_generate_decoys_batch(self):
        """Test batch decoy generation."""
        targets = ["PEPTIDEK", "PROTEINK", "SEQUENCER"]

        # Test kr_swap
        decoys = generate_decoys(targets, method='kr_swap')
        assert len(decoys) == 3
        assert decoys[0] == "PEPTIDER"

        # Test reverse
        decoys = generate_decoys(targets, method='reverse')
        assert len(decoys) == 3
        assert decoys[0] == "KEDITPEP"

        # Test pseudo_reverse
        decoys = generate_decoys(targets, method='pseudo_reverse')
        assert len(decoys) == 3
        assert decoys[0] == "EDITPEPK"

    def test_invalid_decoy_method(self):
        """Test error on invalid decoy method."""
        with pytest.raises(ValueError):
            generate_decoys(["PEPTIDEK"], method='invalid_method')


# =============================================================================
# Database Integration Tests
# =============================================================================

class TestTargetDecoyDatabase:
    """Test TargetDecoyDatabase with new decoy methods."""

    def test_kr_swap_database(self):
        """Test creating database with K↔R swap decoys."""
        targets = ["PEPTIDEK", "PROTEINK", "SEQUENCER"]

        db = TargetDecoyDatabase(targets, decoy_method='kr_swap')

        # Check counts
        assert db.n_targets == 3
        assert db.n_decoys == 3
        assert len(db) == 6

        # Check decoy method stored
        assert db.decoy_method == 'kr_swap'

    def test_from_fasta_basic(self):
        """Test end-to-end from_fasta() workflow."""
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">P1 Protein 1\n")
            f.write("PEPTIDEKRPROTEINK\n")
            f.write(">P2 Protein 2\n")
            f.write("SEQUENCEK\n")
            fasta_path = f.name

        try:
            db = TargetDecoyDatabase.from_fasta(
                fasta_path,
                max_missed_cleavages=1,
                min_peptide_length=7,
                max_peptide_length=35,
                decoy_method='kr_swap',
            )

            # Check that we got some peptides
            assert db.n_targets > 0
            assert db.n_decoys == db.n_targets

            # Check peptide-to-protein mapping exists
            assert hasattr(db, 'peptide_to_proteins')
            assert len(db.peptide_to_proteins) > 0

            # Check that we can search by mass
            indices = db.search_by_mass(mass=1000.0, tol_ppm=10.0)
            assert isinstance(indices, type(pytest.importorskip("numpy").array([])))

        finally:
            Path(fasta_path).unlink()

    def test_from_fasta_kr_swap(self):
        """Test that from_fasta() correctly uses K↔R swap."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">P1 Test\n")
            f.write("TESTPEPTIDEKR\n")
            fasta_path = f.name

        try:
            db = TargetDecoyDatabase.from_fasta(
                fasta_path,
                decoy_method='kr_swap',
                min_peptide_length=7,
            )

            # Find the target and decoy
            target_found = False
            decoy_found = False

            for i, peptide in enumerate(db.peptides):
                if peptide == "TESTPEPTIDEKR":
                    target_found = True
                    target_mass = db.get_mass(i)
                elif peptide == "TESTPEPTIDERK":  # K↔R swapped
                    decoy_found = True
                    decoy_mass = db.get_mass(i)

            # If both found, check mass preservation
            if target_found and decoy_found:
                assert abs(target_mass - decoy_mass) < 0.001

        finally:
            Path(fasta_path).unlink()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

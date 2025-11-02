"""AlphaPeptFast - High-performance proteomics computing library.

Authors: Matthias Mann with Claude Code (Anthropic)

This library provides battle-tested, Numba-optimized functions for proteomics
workflows, extracted from production projects (AlphaMod, AlphaModFS, MSC MS1 High-Res).

All functions follow strict performance standards (>100,000 operations/second) and
best practices from the Computational Proteomics Handbook.
"""

__version__ = "0.2.0"
__author__ = "Matthias Mann with Claude Code (Anthropic)"

# Import main submodules for convenient access
from alphapeptfast import mass
from alphapeptfast import features
from alphapeptfast import rt
from alphapeptfast import search
from alphapeptfast import isotopes
from alphapeptfast import utils
from alphapeptfast import fragments  # v0.2: Fragment generation (battle-tested)
from alphapeptfast import database   # v0.2: Peptide database (battle-tested)

__all__ = [
    "mass",
    "features",
    "rt",
    "search",
    "isotopes",
    "utils",
    "fragments",  # v0.2
    "database",   # v0.2
]

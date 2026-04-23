"""
paths.py — Canonical path constants for the Rosetta data layout.

All scripts should import from here rather than constructing paths locally.
"""

from pathlib import Path

ROSETTA_DATA    = Path.home() / "rosetta_data"
ROSETTA_MODELS  = ROSETTA_DATA / "models"
ROSETTA_RESULTS = ROSETTA_DATA / "results"

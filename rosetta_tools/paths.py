"""
paths.py — Canonical path constants for the Rosetta data layout.

All scripts should import from here rather than constructing paths locally.
"""

from pathlib import Path

ROSETTA_DATA      = Path.home() / "rosetta_data"
ROSETTA_DATA_ROOT = ROSETTA_DATA          # alias used by some scripts
ROSETTA_MODELS    = ROSETTA_DATA / "models"
ROSETTA_MODELS_SNAPSHOTS = ROSETTA_DATA / "model_snapshots"  # isolated per-paper extraction archives
ROSETTA_RESULTS   = ROSETTA_DATA / "results"

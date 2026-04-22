"""
rosetta_tools — Shared tooling for the Rosetta interpretability research program.

Modules
-------
gpu_utils   Device selection, dtype resolution, VRAM reporting, model teardown.
extraction  Model-agnostic layer activation extraction (HF AutoModel).
caz         CAZ metric computation — separation, coherence, velocity, boundary detection.
probes      Probe extraction from contrastive activations — layer selection, direction, threshold.
alignment   Orthogonal Procrustes alignment for cross-architecture vector comparison.
ablation    Directional ablation hooks for HF AutoModel (mid-stream hypothesis testing).
dataset     Dataset loading, validation, and pair iteration utilities.
            load_concept_pairs() is the primary entry point — resolves the
            canonical Rosetta_Concept_Pairs root, applies the fixed train/val
            split, and samples up to N pairs (default 200, clamped).
reporting   Pandas-based result loading — tidy DataFrames from JSON checkpoints.
viz         Matplotlib helpers — CAZ profiles, multi-concept overlays, peak heatmaps.
"""

__version__ = "0.1.0"
__author__ = "James Henry"
__email__ = "jamesrahenry@henrynet.ca"

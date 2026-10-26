"""
rosetta_tools — Shared tooling for the Rosetta interpretability research program.

Modules
-------
gpu_utils   Device selection, dtype resolution, VRAM reporting, model teardown.
extraction  Model-agnostic layer activation extraction (HF AutoModel) and the
            CAZ per-concept pipeline — extract_layer_wise_metrics() (per-layer
            S/C/velocity/DOM + peak detection + inline QA) and build_caz_record()
            (the caz_<concept>.json schema assembler).
caz         CAZ metric computation — separation, coherence, velocity, boundary detection.
probes      Probe extraction from contrastive activations — layer selection, direction, threshold.
            extract_probe() for raw/fisher/auroc layer selection; extract_gem_probe() for
            semantic concepts (GEM handoff layer + settled direction).  ProbeResult carries
            probe_type ("gem" | "raw" | "auroc" | "raw_fallback") and ConceptType annotation
            distinguishes semantic (CAZ-assembled) from tokenization-level concepts.
alignment   Orthogonal Procrustes alignment for cross-architecture vector comparison.
ablation    Directional ablation hooks for HF AutoModel (mid-stream hypothesis testing).
dataset     Dataset loading, validation, and pair iteration utilities.
            load_concept_pairs() is the primary entry point — resolves the
            canonical Rosetta_Concept_Pairs root, applies the fixed train/val
            split, and samples up to N pairs (default 200, clamped).
gem         Geometric Evolution Map (GEM) — reified CAZ assembly events for
            zone-level ablation.  EigThread, GEMNode, ConceptGEM dataclasses;
            build_concept_gem(), save_gem(), load_gem(), gem_diagnostics().
            Phase 2: extract_phase2_threads() (k>1 eigen-threads) and
            extract_settled_subspace() / attach_settled_subspace() (INLP rank-k
            settled structure; SettledSubspace dataclass).
models      YAML-backed model and concept registry (data/models.yaml,
            data/concepts.yaml) — all_models(), get_model(), instruct_pairs(),
            all_concepts(), concepts_by_pipeline(), ...
reporting   Pandas-based result loading — tidy DataFrames from JSON checkpoints.
viz         Matplotlib helpers — CAZ profiles, multi-concept overlays, peak heatmaps.
viz_style   Figure style constants and helpers shared by every Rosetta figure
            (no matplotlib backend import).
compound_monitor
            Vigilance thresholds — tighten an allocation threshold in proportion
            to cross-module anomaly evidence (ModuleEvidence, vigilance_threshold,
            apply_vigilance).
paths       Canonical data path resolution (ROSETTA_DATA, ROSETTA_MODELS,
            ROSETTA_RESULTS, ROSETTA_PAPER_N250) — identical on the dev machine
            and GPU hosts.
"""

__version__ = "2.1.0"  # keep in sync with pyproject.toml [project].version
__author__ = "James Henry"
__email__ = "jamesrahenry@henrynet.ca"

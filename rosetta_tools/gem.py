"""
gem.py — Geometric Evolution Map (GEM): reified CAZ assembly events
for zone-level ablation.

A CAZ GEM is a portable data structure representing what a Concept
Allocation Zone produced during concept assembly.  Instead of ablating
a concept direction at the geometric peak (single-layer ablation), the
GEM tracks how the concept direction evolves through the CAZ window,
identifies the settled product at the handoff point, and provides that
product as a surgical ablation target.

Design principles:
  1. Follow, don't fight, the rotation.  Don't ablate during assembly.
     Let the zone complete its work.  Ablate the settled product at the
     handoff layer (CAZ exit + 1).
  2. Eigenvector threading for precision.  Track top-k eigenvectors
     through the CAZ window via cosine matching.  Select the thread with
     highest cumulative eigenvalue x concept alignment.  Ablate only
     that thread's settled direction.
  3. Multimodal concepts require multiple ablations.  Independent
     branches each get their own GEM node.  Dependent chains only
     need upstream ablation.

Phases:
  Phase 1 (k=1): Use existing dom_vector from caz_*.json.  No new
     extraction needed.  Tests core hypothesis.
  Phase 2 (k=3+): Use deep dive eigenvectors (directions_L*.npy).
     Multi-thread selection for surgical precision.

Schema v2 (2026-10-15, all optional, v1 files load unchanged): per-layer
records at EVERY layer (LayerProfile / LayerRecord), attention-vs-MLP
attribution per layer (SublayerRecord), provenance, and a sidecar .npz for
large arrays (SidecarRef). Computation lives in rosetta_tools.gem_sublayer.
Design: Notes/gem_schema_redesign_2026-10-15.md (Rosetta_Program).

Usage:
    from geometric_evolution_map.gem import build_concept_gem, save_gem
    gem = build_concept_gem(caz_data, multimodal_data)
    save_gem(gem, Path("gem_causation.json"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from rosetta_tools.caz import (
    CAZProfile,
    CAZRegion,
    LayerMetrics,
    find_caz_regions_scored,
)

log = logging.getLogger(__name__)

from rosetta_tools.paths import ROSETTA_MODELS as _MODELS_ROOT


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def find_extraction_dir(model_id: str, models_root: Path | None = None) -> Path | None:
    """Return the extraction directory for a model, or None if not found."""
    root = models_root or _MODELS_ROOT
    d = root / _model_slug(model_id)
    return d if d.is_dir() and (d / "run_summary.json").exists() else None


def discover_concepts(extraction_dir: Path, source: str = "gem") -> list[str]:
    """Return sorted concept names from gem_*.json (source='gem') or caz_*.json (source='caz')."""
    prefix = f"{source}_"
    return sorted(p.stem[len(prefix):] for p in extraction_dir.glob(f"{prefix}*.json"))


def discover_all_models(models_root: Path | None = None) -> list[str]:
    """Return sorted model IDs for all models with extraction results (including instruct)."""
    root = models_root or _MODELS_ROOT
    models: list[str] = []
    if not root.exists():
        return models
    for d in root.iterdir():
        s = d / "run_summary.json"
        if not s.exists():
            continue
        try:
            mid = json.loads(s.read_text()).get("model_id", "")
            if mid:
                models.append(mid)
        except Exception:
            pass
    return sorted(models)


def discover_base_models(models_root: Path | None = None) -> list[str]:
    """Return sorted model IDs for all base (non-instruct) models with extraction results."""
    root = models_root or _MODELS_ROOT
    models: set[str] = set()
    if not root.exists():
        return []
    for d in root.iterdir():
        s = d / "run_summary.json"
        if not s.exists():
            continue
        try:
            mid = json.loads(s.read_text()).get("model_id", "")
            if mid and not any(t in mid for t in ["Instruct", "instruct", "-it"]):
                models.add(mid)
        except Exception:
            pass
    return sorted(models)



# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EigThread:
    """One eigenvector tracked through a CAZ window.

    In Phase 1 (k=1), this is the dom_vector (PC1 of contrastive PCA)
    at each layer in the CAZ window.  In Phase 2 (k>1), multiple
    threads are tracked and the best one selected by cumulative_score.

    Attributes
    ----------
    thread_id : int
        Index among threads in this node (0-based).
    layer_indices : list[int]
        Layers where this thread is alive.
    directions : list[list[float]]
        Unit vector at each layer (hidden_dim,).
    eigenvalues : list[float]
        Signal strength at each layer.  For Phase 1, this is
        separation_fisher (a proxy).  For Phase 2, this is the
        PCA eigenvalue from the deep dive.
    concept_alignments : list[float]
        |cos(direction, dom_vector)| at each layer.  For Phase 1
        this is always 1.0 (the thread IS the dom_vector).
    cos_chain : list[float]
        Cosine continuity between consecutive layers.  First entry
        is 1.0 (no predecessor).
    cumulative_score : float
        sum(eigenvalue_i * concept_alignment_i) across all layers.
    mean_eigenvalue : float
    mean_alignment : float
    """

    thread_id: int
    layer_indices: list[int]
    directions: list[list[float]]
    eigenvalues: list[float]
    concept_alignments: list[float]
    cos_chain: list[float]
    cumulative_score: float = 0.0
    mean_eigenvalue: float = 0.0
    mean_alignment: float = 0.0

    @property
    def settled_direction(self) -> list[float]:
        """Direction at the final tracked layer."""
        return self.directions[-1]

    @property
    def n_layers(self) -> int:
        return len(self.layer_indices)


@dataclass
class SettledSubspace:
    """Multi-directional decomposition at ONE layer (typically handoff_layer),
    found via iterative probe-and-remove (INLP). Distinct from EigThread's
    k>1 design (multiple CANDIDATE threads tracked across the CAZ window,
    with the single best one selected by cumulative_score, for surgical
    single-direction ablation targeting): every direction kept here is
    simultaneously concept-relevant, not competing alternatives, so none is
    discarded once found.

    Motivated by an empirical result (2026-09-23): across 51 concept x model
    combinations (17 concepts x 3 architectures), a single direction removed
    from the settled/handoff layer still leaves the concept ~70% decodable on
    held-out data after 5 rounds of further removal (vs ~57% at shallow
    layers) -- late/settled layers carry more concept-relevant structure than
    a rank-1 (single settled_direction) probe captures.

    Attributes
    ----------
    layer : int
        Which layer this was computed at (not assumed to be handoff_layer,
        so alternative layers can be compared cheaply during research).
    extraction_method : str
        "inlp" currently; "rlace" or raw top-k PCA are possible alternatives.
    directions : list[list[float]]
        k unit vectors, orthogonal by construction (each round's probe is
        fit in the null space of every direction removed before it).
    held_out_acc_curve, held_out_logloss_curve : list[float]
        Diagnostic curves, one entry per round attempted (including the
        round that triggered stopping, so the decay shape is visible even
        past k_effective).
    triage_pvalues : list[float]
        Cheap screening p-value per round: a one-sided binomial test of
        held-out accuracy against chance. Two earlier metrics were tried and
        failed empirically before this one: training-data log-likelihood
        deviance never detected exhaustion (a regularized fit can always
        explain training labels well when n << d, which is every round
        here); held-out log-loss deviance stopped too early because it is
        sensitive to probability calibration, not just discrimination, and
        penalized a genuinely-above-chance but under-confident fit.
    permutation_pvalues : list[float | None]
        Permutation-test p-value for rounds where the triage screen was
        ambiguous (not obviously significant nor obviously exhausted);
        None where the triage screen alone was decisive.
    k_effective : int
        Number of directions actually kept (<= len(directions)).
    stopping_reason : str
        "triage_clear_stop" | "permutation_stop" | "coef_collapsed" |
        "max_rounds".
    """

    layer: int
    extraction_method: str = "inlp"
    directions: list = field(default_factory=list)
    held_out_acc_curve: list = field(default_factory=list)
    held_out_logloss_curve: list = field(default_factory=list)
    triage_pvalues: list = field(default_factory=list)
    permutation_pvalues: list = field(default_factory=list)
    k_effective: int = 0
    stopping_reason: str = ""

    # Schema v2 (2026-10-15), optional. provenance: how/where this subspace
    # was computed. sublayer_decomposition: each kept direction's class-mean
    # separation split into attention / MLP writes (this layer's block and
    # every upstream block) -- see SubspaceSublayerDecomposition.
    provenance: Provenance | None = None
    sublayer_decomposition: SubspaceSublayerDecomposition | None = None


# ---------------------------------------------------------------------------
# Schema v2 (2026-10-15): per-layer and per-sublayer research records.
#
# Motivation: SettledSubspace covers ONE layer and EigThread only the CAZ
# window; the Freenor follow-ups (Notes/freenor_feedback_followups_2026-10-15
# §1, §2, §5) need data at EVERY layer, an attention-vs-MLP split of each
# block's write, and a gauge-corrected separation score with a null. Design
# doc: Notes/gem_schema_redesign_2026-10-15.md.
#
# Conventions:
#   - GEM layer l = output of decoder block l (= hidden_states[l+1], as in
#     extract_layer_wise_metrics). Block l reads GEM layer l-1 (embedding
#     output for l=0) and writes GEM layer l.
#   - u_in  = unit class-mean-difference direction of the stream block l
#     reads (GEM u_{l-1});  u_out = that of the stream it writes (GEM u_l).
#     Named by role, not index, because the Freenor note indexes by block
#     input (x_{l+1} = x_l + a_l + m_l) and is off by one from GEM's index.
#   - Everything is optional/None so schema-v1 records and k=1 consumers are
#     unaffected. Scalars live in the JSON; vectors and per-example arrays
#     live in a sidecar .npz referenced by ConceptGEM.sidecar.
# ---------------------------------------------------------------------------

GEM_SCHEMA_VERSION = "2"


@dataclass
class Provenance:
    """How a block of derived data was produced. Attached to the container
    (LayerProfile, SettledSubspace, SubspaceSublayerDecomposition) and covers
    every field inside it."""

    method: str
    code_version: str = ""
    created_utc: str = ""
    model_id: str = ""
    model_dtype: str | None = None      # dtype the forward pass ran in
    capture_dtype: str | None = None    # dtype captures were cast to (float32)
    stats_dtype: str | None = None      # dtype statistics computed in (float64)
    hook_points: dict | None = None     # role -> module path template
    pooling: str | None = None
    n_pos: int | None = None
    n_neg: int | None = None
    seed: int | None = None
    n_perm: int | None = None
    max_length: int | None = None
    batch_size: int | None = None
    device: str | None = None
    dataset: str | None = None          # corpus id / split, free text
    notes: list = field(default_factory=list)


@dataclass
class PermutationNull:
    """Label-permutation null for one separation statistic at one layer.
    The direction u is REFIT on every permutation (refitting is what the
    observed statistic does too; a fixed u would bias the null low)."""

    statistic: str                      # "separation_projected" | "separation_fisher"
    n_perm: int
    seed: int
    mean: float
    std: float
    q95: float
    max: float
    p_value: float                      # (#{null >= observed} + 1) / (n_perm + 1)


@dataclass
class SublayerRecord:
    """One sublayer's (attention or MLP) write at one layer, pooled per
    example. Projections are onto the class-mean-difference directions of
    the stream (u_in, u_out; see module conventions). "diff_proj" =
    u . (mean_pos - mean_neg) of the sublayer output: its concept write."""

    kind: str                           # "attn" | "mlp"
    hook_point: str                     # module path actually hooked
    diff_proj_u_in: float | None = None
    diff_proj_u_out: float | None = None
    pos_mean_proj_u_in: float | None = None
    neg_mean_proj_u_in: float | None = None
    pos_mean_proj_u_out: float | None = None
    neg_mean_proj_u_out: float | None = None
    var_proj_u_out: float | None = None              # pooled within-class variance
    separation_projected_u_out: float | None = None  # diff_proj_u_out / sqrt(var)
    # diff_proj_u_out(sublayer) / diff_proj_u_out(whole block delta). attn+mlp
    # sum to ~1 (exactly, minus the recon residual); can be <0 or >1 when the
    # two sublayers write in opposite directions.
    share_of_concept_write_u_out: float | None = None
    mean_diff_norm: float | None = None              # ||mean_pos - mean_neg||
    cos_mean_diff_u_out: float | None = None
    norm_mean: float | None = None                   # mean ||s||
    norm_frac_stream_mean: float | None = None       # mean ||s|| / ||x_in||
    norm_frac_delta_mean: float | None = None        # mean ||s|| / ||x_out - x_in||


@dataclass
class LayerRecord:
    """Per-layer research record (GEM layer `layer`; -1 = embedding output).
    Computed at every layer, not only the CAZ window."""

    layer: int
    # Separation: existing CAZ score and the gauge-corrected projected score
    # ||mu+ - mu-|| / sqrt(1/2 u^T (S+ + S-) u), u = unit mean difference.
    separation_fisher: float | None = None
    separation_projected: float | None = None
    raw_distance: float | None = None
    projected_pooled_std: float | None = None
    null_fisher: PermutationNull | None = None
    null_projected: PermutationNull | None = None
    cos_prev_direction: float | None = None          # signed cos(u_l, u_{l-1})
    stream_norm_mean: float | None = None
    # Whole-block write Delta = x_out - x_in (None for the embedding row)
    delta_diff_proj_u_in: float | None = None
    delta_diff_proj_u_out: float | None = None
    delta_norm_mean: float | None = None
    # Additivity check Delta == attn + mlp (pre-LN / post-sublayer-norm / parallel)
    recon_residual_max_abs: float | None = None
    recon_residual_max_rel_delta: float | None = None
    recon_residual_max_rel_stream: float | None = None
    sublayers: list[SublayerRecord] | None = None
    # INLP at this layer -- allowed by the schema, not computed by default
    # (expensive; see extract_settled_subspace caution).
    settled_subspace: SettledSubspace | None = None
    sidecar_row: int | None = None                   # row in stacked sidecar arrays
    # u_l inline. None by default (directions live in the sidecar); filled
    # only when saved with save_gem(..., inline_directions=True).
    direction: list | None = None


@dataclass
class ArchitectureFlags:
    """Architecture facts that decide whether Delta == attn + mlp is exact
    and what the sublayer hooks mean."""

    model_type: str = "unknown"
    decoder_class: str = ""
    layer_path: str = ""
    attn_hook: str | None = None
    mlp_hook: str | None = None
    # attn and mlp both read x_in (GPT-J, NeoX use_parallel_residual, Phi):
    # additive, but no stream point exists between the two sublayers.
    parallel_block: bool = False
    # Gemma2/3: sublayer output is normed before the add; hooks are AFTER
    # that norm, on what is actually added.
    sublayer_post_norm: bool = False
    # Post-LN (norm applied to x + sublayer): additivity does not hold.
    post_ln: bool = False
    # Mixture-of-experts MLP (Mixtral block_sparse_moe, Qwen2/3-MoE mlp): the
    # hooked "mlp" write is the router-weighted sum of experts (+ any shared
    # expert); still exactly additive, but not a single dense MLP.
    moe: bool = False
    # Module whose output is hidden_states[-1] (post-final-norm), if found.
    final_norm_hook: str | None = None
    additivity_expected_exact: bool = True
    notes: list = field(default_factory=list)


@dataclass
class LayerProfile:
    """Per-layer records across ALL layers for one concept x model."""

    n_layers: int
    hidden_dim: int
    layers: list[LayerRecord]
    embedding: LayerRecord | None = None
    # Post-final-norm stream at the LAST layer (= hidden_states[-1], what the
    # existing caz_*.json data holds at layer n_layers-1), alongside the raw
    # block output in layers[-1]. No sublayer fields (not a block write).
    final_norm: LayerRecord | None = None
    stream_source: str = "decoder_layer_output"      # or "hidden_states"
    direction_convention: str = (
        "GEM layer l = output of decoder block l; u_in = unit mean-diff of the "
        "stream block l reads (GEM l-1, embedding for l=0); u_out = that of "
        "the stream it writes (GEM l)."
    )
    architecture: ArchitectureFlags | None = None
    provenance: Provenance | None = None

    def record(self, layer: int) -> LayerRecord:
        return self.embedding if layer == -1 else self.layers[layer]


@dataclass
class SubspaceSublayerDecomposition:
    """Decomposes each SettledSubspace direction w_j's class-mean separation
    at layer L into the writes that built it. In pre-LN models, exactly:
        w.dmu(x_L) = w.dmu(embed) + sum_{l<=L} [w.dmu(attn_l) + w.dmu(mlp_l)]
    (dmu = mean_pos - mean_neg). All lists are indexed [j] or [j][l]."""

    layer: int
    attn_diff_proj: list = field(default_factory=list)          # [k], block L only
    mlp_diff_proj: list = field(default_factory=list)           # [k], block L only
    stream_diff_proj: list = field(default_factory=list)        # [k], w.dmu(x_L)
    embed_diff_proj: list = field(default_factory=list)         # [k]
    upstream_attn_diff_proj: list = field(default_factory=list) # [k][L+1]
    upstream_mlp_diff_proj: list = field(default_factory=list)  # [k][L+1]
    reconstruction_residual: list = field(default_factory=list) # [k]
    provenance: Provenance | None = None


@dataclass
class SidecarRef:
    """Pointer to the .npz holding a GEM's large arrays."""

    path: str                           # relative to the GEM JSON's directory
    format: str = "npz"
    sha256: str = ""
    arrays: dict = field(default_factory=dict)  # key -> {shape, dtype, axes, description}


@dataclass
class GEMExtractionConfig:
    """One switchboard for what a GEM build produces (2026-10-15).

    Defaults = schema v2 with everything on, so net-new generation records
    the per-layer / per-sublayer research data. GEMExtractionConfig.legacy()
    reproduces pre-v2 output exactly (k=1 nodes from caz_*.json only).

    layer_profile : build a LayerProfile at all layers.
    sublayer_capture : capture attention/MLP writes with forward hooks (needs
        the model + concept texts, i.e. a forward pass). False = stream-only
        profile from calibration_alllayer_{concept}.npy (no model needed).
    store_full_sublayer_activations : also put the raw pooled per-example
        attn/mlp writes (float16, [L, n, d]) in the sidecar. Large (about
        43 MB/concept at 0.5B, 147 MB at 3B).
    store_post_final_norm : also record the post-final-norm stream at the
        last layer (= hidden_states[-1], what existing CAZ data holds there).
    inline_directions : also write u_l inline in the JSON (sidecar always).
    n_permutations : label permutations per layer for the separation nulls.
    additivity_tol : None = ADDITIVITY_TOL[model dtype].
    allow_inexact : record (and flag) instead of raising when Delta != a + m.
    detection_score : "fisher" (existing CAZ detector, default) or
        "projected" (find_caz_regions_projected; research comparison path).
    """

    layer_profile: bool = True
    sublayer_capture: bool = True
    store_full_sublayer_activations: bool = True
    store_post_final_norm: bool = True
    inline_directions: bool = False
    n_permutations: int = 200
    permutation_seed: int = 0
    additivity_tol: float | None = None
    allow_inexact: bool = False
    pooling: str = "last"
    batch_size: int = 16
    max_length: int = 512
    detection_score: str = "fisher"

    @classmethod
    def legacy(cls) -> "GEMExtractionConfig":
        """Pre-v2 behaviour: no layer profile, nothing beyond caz_*.json."""
        return cls(layer_profile=False, sublayer_capture=False,
                   store_full_sublayer_activations=False, store_post_final_norm=False)


@dataclass
class GEMNode:
    """GEM node for a single CAZ assembly event.

    One concept typically has 2-4 GEMNodes per model (one per CAZ).
    Each node captures one assembly event: the trajectory through the
    zone, the eigenvector threads, and the settled product at the
    handoff point.

    Attributes
    ----------
    concept, model_id : str
        Identifiers.
    caz_index : int
        Index among this concept's CAZes (0-based, sorted by depth).
    caz_start, caz_peak, caz_end : int
        Layer boundaries of the CAZ region.
    caz_score : float
        Composite strength score from find_caz_regions_scored.
    attention_paradigm : str
        "mha", "gqa", "alternating", or "unknown".
    settled_direction : list[float]
        The concept thread's direction at the CAZ end layer.
        This is the assembly product — what the zone built.
    handoff_layer : int
        caz_end + 1, clamped to n_layers - 1.
    handoff_cosine : float
        |cos(settled_direction, dom_vector at handoff_layer)|.
        Measures whether the settled product persists past the
        CAZ boundary.  Values < 0.2 suggest the direction dies.
    threads : list[EigThread]
        Eigenvector threads through the window.
    concept_thread_id : int
        Which thread is the primary concept carrier.
    entry_exit_cosine : float
        |cos(direction[caz_start], direction[caz_end])|.
        Measures total rotation through the zone.
    max_rotation_per_layer : float
        Maximum angular change between consecutive layers.
    n_layers_total : int
        Total layers in the model (for depth % calculations).
    hidden_dim : int
    k : int
        Number of eigenvector threads (1 for Phase 1).
    phase : int
        1 or 2.
    """

    concept: str
    model_id: str
    caz_index: int

    # CAZ boundaries
    caz_start: int
    caz_peak: int
    caz_end: int
    caz_score: float
    attention_paradigm: str

    # Assembly product
    settled_direction: list[float]
    handoff_layer: int
    handoff_cosine: float

    # Threads
    threads: list[EigThread]
    concept_thread_id: int

    # Diagnostics
    entry_exit_cosine: float
    max_rotation_per_layer: float
    n_layers_total: int
    hidden_dim: int
    k: int
    phase: int = 1

    # Multi-directional structure at the settled layer (see SettledSubspace
    # docstring for how this differs from the threads/k>1 fields above).
    # Optional and None unless explicitly computed -- does not affect any
    # existing k=1 Phase 1 consumer.
    settled_subspace: SettledSubspace | None = None

    # Unknown keys found when loading (forward compat); preserved on re-save.
    extra: dict | None = None

    @property
    def concept_thread(self) -> EigThread:
        """The primary concept-carrying thread."""
        return self.threads[self.concept_thread_id]

    @property
    def depth_pct(self) -> float:
        """Handoff layer as % of model depth."""
        return round(100 * self.handoff_layer / max(self.n_layers_total - 1, 1), 1)


@dataclass
class ConceptGEM:
    """Full Geometric Evolution Map for one concept in one model.

    Contains one GEMNode per CAZ region, plus the dependency graph
    (from ablation_multimodal data) and the computed ablation plan.

    Attributes
    ----------
    concept, model_id : str
    n_nodes : int
    nodes : list[GEMNode]
        Sorted by depth (shallow to deep).
    interaction_matrix : list[list[float]] | None
        [i][j] = % of node j's separation retained when node i is
        ablated.  From ablation_multimodal_*.json.
    node_types : list[str] | None
        "independent", "upstream", or "downstream" per node.
    ablation_targets : list[int] | None
        Node indices to ablate (independent + upstream only).
    """

    concept: str
    model_id: str
    n_nodes: int
    nodes: list[GEMNode]

    # Dependency graph
    interaction_matrix: list[list[float]] | None = None
    node_types: list[str] | None = None
    ablation_targets: list[int] | None = None

    # Schema v2 (all optional). Files without schema_version load as "1".
    schema_version: str = GEM_SCHEMA_VERSION
    layer_profile: LayerProfile | None = None
    sidecar: SidecarRef | None = None
    # Which separation score drove CAZ region detection: None/"fisher" =
    # existing detector; "projected" = find_caz_regions_projected (parallel
    # research path, see build_concept_gem(detection_score=...)).
    caz_detection_score: str | None = None
    # Unknown top-level keys found when loading (e.g. the n_pairs existing
    # gem_*.json files carry) -- preserved on re-save instead of dropped.
    extra: dict | None = None

    @property
    def independent_nodes(self) -> list[GEMNode]:
        if self.node_types is None:
            return list(self.nodes)
        return [n for n, t in zip(self.nodes, self.node_types)
                if t == "independent"]

    @property
    def upstream_nodes(self) -> list[GEMNode]:
        if self.node_types is None:
            return []
        return [n for n, t in zip(self.nodes, self.node_types)
                if t == "upstream"]

    @property
    def target_nodes(self) -> list[GEMNode]:
        """Nodes that should be ablated (independent + upstream)."""
        if self.ablation_targets is None:
            return list(self.nodes)
        return [self.nodes[i] for i in self.ablation_targets]


# ---------------------------------------------------------------------------
# Dependency classification
# ---------------------------------------------------------------------------

def classify_node_dependencies(
    interaction_matrix: list[list[float]],
    independence_threshold: float = 80.0,
) -> tuple[list[str], list[int]]:
    """Classify nodes as independent, upstream, or downstream.

    Uses the N x N interaction matrix from ablation_multimodal data.
    interaction_matrix[i][j] = % of node j's separation retained when
    node i is ablated.

    A node is "downstream" if ablating a shallower node suppresses it
    (retained < threshold).  A node is "upstream" if ablating it
    suppresses a deeper node.  Independent nodes are neither.

    Ablation targets = independent + upstream.  Downstream nodes are
    never directly ablated (superposition risk).

    Parameters
    ----------
    interaction_matrix : list[list[float]]
        N x N matrix, rows = ablated node, cols = measured node.
        Nodes are sorted shallow to deep.
    independence_threshold : float
        retained_pct above this = independent relationship.

    Returns
    -------
    node_types : list[str]
        Classification per node.
    ablation_targets : list[int]
        Indices of nodes to ablate.
    """
    n = len(interaction_matrix)
    if n == 0:
        return [], []
    if n == 1:
        return ["independent"], [0]

    types = ["independent"] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            retained = interaction_matrix[i][j]
            if retained < independence_threshold:
                # Ablating i suppresses j.
                # Only count as dependency if i is shallower (forward pass).
                if i < j:
                    if types[j] != "downstream":
                        types[j] = "downstream"
                    if types[i] == "independent":
                        types[i] = "upstream"

    targets = [i for i in range(n) if types[i] != "downstream"]
    return types, targets


# ---------------------------------------------------------------------------
# Phase 1 builder: k=1, dom_vector only
# ---------------------------------------------------------------------------

def build_gem_node_k1(
    caz_data: dict,
    region: CAZRegion,
    caz_index: int,
    attention_paradigm: str = "unknown",
) -> GEMNode:
    """Build a Phase 1 GEM node from existing dom_vector data.

    No new extraction needed.  Reads dom_vector at each layer in the
    CAZ window, builds a single EigThread, identifies the settled
    direction at CAZ end.

    Parameters
    ----------
    caz_data : dict
        Loaded caz_*.json with layer_data.metrics containing
        dom_vector, separation_fisher, coherence per layer.
    region : CAZRegion
        The CAZ region to build a GEM node for.
    caz_index : int
        Index of this region among the concept's CAZes.
    attention_paradigm : str
        Architecture type for metadata.
    """
    metrics = caz_data["layer_data"]["metrics"]
    n_layers = int(caz_data["layer_data"]["n_layers"])
    hidden_dim = len(metrics[0]["dom_vector"])

    window_layers = list(range(region.start, min(region.end + 1, len(metrics))))
    if not window_layers:
        window_layers = [region.peak]

    # Extract and normalize dom_vectors within the window
    directions = []
    for li in window_layers:
        dv = np.array(metrics[li]["dom_vector"], dtype=np.float64)
        if dv.shape[0] != hidden_dim:
            raise ValueError(
                f"dom_vector dimension mismatch at layer {li}: "
                f"expected {hidden_dim}, got {dv.shape[0]} "
                f"(corrupted extraction data)"
            )
        norm = np.linalg.norm(dv)
        if norm > 1e-12:
            dv = dv / norm
        directions.append(dv)

    # Cosine chain (cross-layer continuity)
    cos_chain = [1.0]
    for i in range(1, len(directions)):
        cos_chain.append(float(np.abs(np.dot(directions[i], directions[i - 1]))))

    # Eigenvalue proxy: separation_fisher as signal strength
    eigenvalues = [float(metrics[li]["separation_fisher"]) for li in window_layers]

    # For k=1, the single thread IS the concept direction
    concept_alignments = [1.0] * len(window_layers)

    # Settled direction: dom_vector at CAZ end
    settled_direction = directions[-1]

    # Handoff layer
    handoff_layer = min(region.end + 1, n_layers - 1)

    # Handoff cosine: does the settled direction persist into the handoff?
    handoff_cosine = 0.0
    if handoff_layer < len(metrics):
        hoff_dv = np.array(metrics[handoff_layer]["dom_vector"], dtype=np.float64)
        hoff_norm = np.linalg.norm(hoff_dv)
        if hoff_norm > 1e-12:
            hoff_dv = hoff_dv / hoff_norm
            handoff_cosine = float(np.abs(np.dot(settled_direction, hoff_dv)))

    # Rotation diagnostics
    entry_exit_cosine = float(np.abs(np.dot(directions[0], directions[-1])))
    max_rotation = 0.0
    for c in cos_chain[1:]:
        angular_change = 1.0 - c
        max_rotation = max(max_rotation, angular_change)

    cum_score = sum(e * a for e, a in zip(eigenvalues, concept_alignments))

    thread = EigThread(
        thread_id=0,
        layer_indices=window_layers,
        directions=[d.tolist() for d in directions],
        eigenvalues=eigenvalues,
        concept_alignments=concept_alignments,
        cos_chain=cos_chain,
        cumulative_score=round(cum_score, 6),
        mean_eigenvalue=round(float(np.mean(eigenvalues)), 6),
        mean_alignment=1.0,
    )

    return GEMNode(
        concept=caz_data.get("concept", ""),
        model_id=caz_data.get("model_id", ""),
        caz_index=caz_index,
        caz_start=region.start,
        caz_peak=region.peak,
        caz_end=region.end,
        caz_score=region.caz_score,
        attention_paradigm=attention_paradigm,
        settled_direction=settled_direction.tolist(),
        handoff_layer=handoff_layer,
        handoff_cosine=round(handoff_cosine, 4),
        threads=[thread],
        concept_thread_id=0,
        entry_exit_cosine=round(entry_exit_cosine, 4),
        max_rotation_per_layer=round(max_rotation, 4),
        n_layers_total=n_layers,
        hidden_dim=hidden_dim,
        k=1,
        phase=1,
    )


def build_gem_node_phase2(
    caz_data: dict,
    region: CAZRegion,
    caz_index: int,
    h_pos_by_layer: dict[int, np.ndarray],
    h_neg_by_layer: dict[int, np.ndarray],
    k: int = 3,
    attention_paradigm: str = "unknown",
) -> GEMNode:
    """Build a Phase 2 GEM node: k candidate eigenvector threads tracked
    across the CAZ window via extract_phase2_threads, with the best one
    selected by cumulative_score (EigThread's own documented design).
    Requires raw per-example activations -- unlike build_gem_node_k1, this
    cannot be built from caz_*.json alone.
    """
    metrics = caz_data["layer_data"]["metrics"]
    n_layers = int(caz_data["layer_data"]["n_layers"])
    hidden_dim = len(metrics[0]["dom_vector"])

    window_layers = list(range(region.start, min(region.end + 1, len(metrics))))
    if not window_layers:
        window_layers = [region.peak]

    missing = [l for l in window_layers if l not in h_pos_by_layer or l not in h_neg_by_layer]
    if missing:
        raise ValueError(
            f"build_gem_node_phase2: h_pos_by_layer/h_neg_by_layer missing layers {missing} "
            f"needed for CAZ window {window_layers}"
        )

    dom_vectors_by_layer = {
        m["layer"]: np.asarray(m["dom_vector"], dtype=np.float64)
        for m in metrics if "dom_vector" in m
    }

    threads, best_thread_id = extract_phase2_threads(
        h_pos_by_layer, h_neg_by_layer, window_layers, k=k,
        dom_vectors_by_layer=dom_vectors_by_layer,
    )
    best = threads[best_thread_id]
    settled_direction = np.asarray(best.directions[-1], dtype=np.float64)

    handoff_layer = min(region.end + 1, n_layers - 1)
    handoff_cosine = 0.0
    if handoff_layer < len(metrics) and "dom_vector" in metrics[handoff_layer]:
        hoff_dv = np.asarray(metrics[handoff_layer]["dom_vector"], dtype=np.float64)
        hoff_norm = np.linalg.norm(hoff_dv)
        settled_norm = np.linalg.norm(settled_direction)
        if hoff_norm > 1e-12 and settled_norm > 1e-12:
            handoff_cosine = float(np.abs(np.dot(settled_direction / settled_norm, hoff_dv / hoff_norm)))

    entry_exit_cosine = float(np.abs(np.dot(
        np.asarray(best.directions[0], dtype=np.float64), settled_direction,
    )))
    max_rotation = 0.0
    for c in best.cos_chain[1:]:
        max_rotation = max(max_rotation, 1.0 - c)

    return GEMNode(
        concept=caz_data.get("concept", ""),
        model_id=caz_data.get("model_id", ""),
        caz_index=caz_index,
        caz_start=region.start,
        caz_peak=region.peak,
        caz_end=region.end,
        caz_score=region.caz_score,
        attention_paradigm=attention_paradigm,
        settled_direction=settled_direction.tolist(),
        handoff_layer=handoff_layer,
        handoff_cosine=round(handoff_cosine, 4),
        threads=threads,
        concept_thread_id=best_thread_id,
        entry_exit_cosine=round(entry_exit_cosine, 4),
        max_rotation_per_layer=round(max_rotation, 4),
        n_layers_total=n_layers,
        hidden_dim=hidden_dim,
        k=k,
        phase=2,
    )


# ---------------------------------------------------------------------------
# ConceptGEM builder
# ---------------------------------------------------------------------------

def _match_regions_to_multimodal(
    regions: list[CAZRegion],
    multimodal_data: dict,
) -> list[list[float]] | None:
    """Extract the interaction matrix aligned to the detected regions.

    The multimodal ablation data may have been run with a different
    detector pass.  We align by matching region peaks to the multimodal
    peaks via nearest-layer.  If alignment fails, returns None.
    """
    if not multimodal_data:
        return None

    mm_peaks = multimodal_data.get("interaction_peaks", [])
    mm_matrix = multimodal_data.get("interaction_matrix", [])
    if not mm_peaks or not mm_matrix:
        return None

    region_peaks = [r.peak for r in regions]
    n_regions = len(region_peaks)
    n_mm = len(mm_peaks)

    # Build mapping: region index -> multimodal index (nearest peak)
    region_to_mm = []
    for rp in region_peaks:
        best_idx = min(range(n_mm), key=lambda j: abs(mm_peaks[j] - rp))
        dist = abs(mm_peaks[best_idx] - rp)
        if dist <= 3:  # within 3 layers = same CAZ
            region_to_mm.append(best_idx)
        else:
            region_to_mm.append(None)

    # If we can't map at least 2 regions, interaction matrix is useless
    mapped = [i for i in region_to_mm if i is not None]
    if len(mapped) < 2:
        return None

    # Build aligned matrix (regions x regions)
    aligned = []
    for i in range(n_regions):
        row = []
        for j in range(n_regions):
            mi = region_to_mm[i]
            mj = region_to_mm[j]
            if mi is not None and mj is not None:
                row.append(mm_matrix[mi][mj])
            else:
                # Unknown interaction — assume independent
                row.append(100.0)
        aligned.append(row)

    return aligned


def build_concept_gem(
    caz_data: dict,
    multimodal_data: dict | None = None,
    attention_paradigm: str = "unknown",
    k: int = 1,
    h_pos_by_layer: dict[int, np.ndarray] | None = None,
    h_neg_by_layer: dict[int, np.ndarray] | None = None,
    compute_settled_subspace: bool = False,
    settled_subspace_kwargs: dict | None = None,
    layer_profile: LayerProfile | None = None,
    detection_score: str = "fisher",
) -> ConceptGEM:
    """Build the full ConceptGEM for one concept x model.

    Steps:
      1. Detect all CAZ regions via find_caz_regions_scored
      2. Build a GEMNode per region
      3. If multimodal_data provided, classify dependencies
      4. Compute ablation targets

    Parameters
    ----------
    caz_data : dict
        Loaded caz_*.json.
    multimodal_data : dict | None
        Loaded ablation_multimodal_*.json, if available.
    attention_paradigm : str
        Architecture type.
    k : int
        Number of eigenvector threads to track. k=1 (default) uses
        build_gem_node_k1 and needs nothing beyond caz_data. k>1 uses
        build_gem_node_phase2 and REQUIRES h_pos_by_layer/h_neg_by_layer
        (raw per-example activations, not derivable from caz_*.json alone
        -- e.g. loaded from calibration_alllayer_{concept}.npy).
    h_pos_by_layer, h_neg_by_layer : dict[int, np.ndarray] | None
        Layer -> (n_examples, hidden_dim) raw activations for the positive/
        negative contrastive examples. Optional; default (None) reproduces
        the exact pre-2026-09-25 k=1-only behavior. Required if k>1 or
        compute_settled_subspace=True.
    compute_settled_subspace : bool
        If True (and activations provided), additionally compute a
        SettledSubspace at each node's handoff_layer via INLP (see
        attach_settled_subspace). Expensive -- caution advised (2026-09-24
        validation on a 70M model took 30-90+ minutes for one layer when
        several removal rounds landed in the ambiguous triage band and
        needed a full permutation test). Off by default; not something to
        enable as a blanket default across a corpus without first checking
        the expected runtime.
    settled_subspace_kwargs : dict | None
        Passed through to attach_settled_subspace (e.g. max_rounds, n_perm)
        when compute_settled_subspace=True.
    layer_profile : LayerProfile | None
        Optional schema-v2 profile to attach (see rosetta_tools.gem_sublayer).
    detection_score : "fisher" | "projected"
        "fisher" (default) = the existing detector on separation_fisher,
        unchanged. "projected" = the SAME detector run on the gauge-corrected
        projected score from `layer_profile` (find_caz_regions_projected), a
        parallel path for comparing the two (Freenor, 2026-10-14). Node
        contents (dom_vector threads, eigenvalue proxy) still come from
        caz_data either way; only region boundaries change.
    """
    metrics_raw = caz_data["layer_data"]["metrics"]

    # Detect CAZ regions
    if detection_score == "fisher":
        layer_metrics = [
            LayerMetrics(
                m["layer"],
                m["separation_fisher"],
                m["coherence"],
                m["velocity"],
            )
            for m in metrics_raw
        ]
    elif detection_score == "projected":
        if layer_profile is None:
            raise ValueError("detection_score='projected' requires layer_profile")
        layer_metrics = projected_layer_metrics(caz_data, layer_profile)
    else:
        raise ValueError(f"detection_score must be 'fisher' or 'projected', got {detection_score!r}")
    profile = find_caz_regions_scored(
        layer_metrics,
        attention_paradigm=attention_paradigm,
    )

    if profile.n_regions == 0:
        return ConceptGEM(
            concept=caz_data.get("concept", ""),
            model_id=caz_data.get("model_id", ""),
            n_nodes=0,
            nodes=[],
            layer_profile=layer_profile,
            caz_detection_score=detection_score,
        )

    # Sort regions by depth (already sorted by find_caz_regions_scored)
    regions = sorted(profile.regions, key=lambda r: r.start)

    # Build GEM nodes
    nodes = []
    for idx, region in enumerate(regions):
        if k == 1:
            node = build_gem_node_k1(
                caz_data, region, idx, attention_paradigm,
            )
        else:
            if h_pos_by_layer is None or h_neg_by_layer is None:
                raise ValueError(
                    f"Phase 2 (k={k}) requires h_pos_by_layer/h_neg_by_layer "
                    "(raw per-example activations) -- caz_*.json alone isn't "
                    "enough. Load calibration_alllayer_{concept}.npy or pass k=1."
                )
            node = build_gem_node_phase2(
                caz_data, region, idx, h_pos_by_layer, h_neg_by_layer,
                k=k, attention_paradigm=attention_paradigm,
            )

        if compute_settled_subspace and h_pos_by_layer is not None and h_neg_by_layer is not None:
            hl = node.handoff_layer
            if hl in h_pos_by_layer and hl in h_neg_by_layer:
                node = attach_settled_subspace(
                    node, h_pos_by_layer[hl], h_neg_by_layer[hl],
                    **(settled_subspace_kwargs or {}),
                )

        nodes.append(node)

    # Dependency classification
    interaction_matrix = _match_regions_to_multimodal(regions, multimodal_data)
    node_types = None
    ablation_targets = None

    if interaction_matrix is not None and len(interaction_matrix) >= 2:
        node_types, ablation_targets = classify_node_dependencies(
            interaction_matrix
        )
    else:
        # No interaction data — treat all as independent
        node_types = ["independent"] * len(nodes)
        ablation_targets = list(range(len(nodes)))

    return ConceptGEM(
        concept=caz_data.get("concept", ""),
        model_id=caz_data.get("model_id", ""),
        n_nodes=len(nodes),
        nodes=nodes,
        interaction_matrix=interaction_matrix,
        node_types=node_types,
        ablation_targets=ablation_targets,
        layer_profile=layer_profile,
        caz_detection_score=detection_score,
    )


def projected_layer_metrics(caz_data: dict, layer_profile: LayerProfile) -> list[LayerMetrics]:
    """LayerMetrics with separation = the profile's separation_projected and
    velocity recomputed from it (compute_velocity, window=3, as extraction
    does); coherence is taken from caz_data unchanged. At the last layer the
    post-final-norm record is used when present, so every layer is in the
    same space as the caz_*.json data the Fisher path uses."""
    from rosetta_tools.caz import compute_velocity

    metrics_raw = caz_data["layer_data"]["metrics"]
    n = len(metrics_raw)
    if layer_profile.n_layers != n:
        raise ValueError(f"layer_profile has {layer_profile.n_layers} layers, caz_data {n}")
    seps = [r.separation_projected for r in layer_profile.layers]
    if layer_profile.final_norm is not None and layer_profile.final_norm.separation_projected is not None:
        seps[-1] = layer_profile.final_norm.separation_projected
    if any(s is None for s in seps):
        raise ValueError("layer_profile is missing separation_projected at some layer")
    vel = compute_velocity(seps, window=3)
    return [LayerMetrics(m["layer"], float(s), m["coherence"], float(v))
            for m, s, v in zip(metrics_raw, seps, vel)]


def find_caz_regions_projected(
    caz_data: dict,
    layer_profile: LayerProfile,
    attention_paradigm: str = "unknown",
    **detector_kwargs: Any,
) -> CAZProfile:
    """Parallel, opt-in CAZ boundary detection: the existing scored detector
    (find_caz_regions_scored, same thresholds) run over the gauge-corrected
    projected separation instead of separation_fisher. Research comparison
    path only -- the default detector is untouched. Note the detector's
    thresholds were tuned on the Fisher scale; the projected score has a
    different dynamic range, so differences in regions mix "better score"
    with "untuned thresholds"."""
    return find_caz_regions_scored(
        projected_layer_metrics(caz_data, layer_profile),
        attention_paradigm=attention_paradigm, **detector_kwargs,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class _GEMEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types in GEM data."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def gem_to_dict(gem: ConceptGEM) -> dict:
    """Serialize a ConceptGEM to a JSON-safe dict.

    `extra` (unknown keys preserved at load time) is written back at the
    level it came from rather than nested, so load -> save round-trips the
    original layout (e.g. top-level n_pairs)."""
    d = asdict(gem)
    for nd in d["nodes"]:
        node_extra = nd.pop("extra", None) or {}
        for k, v in node_extra.items():
            nd.setdefault(k, v)
    top_extra = d.pop("extra", None) or {}
    for k, v in top_extra.items():
        d.setdefault(k, v)
    return d


def _known(cls, d: dict, where: str) -> dict:
    """Keep only keys that are fields of `cls`; log (don't raise on) the rest.
    Used for the nested schema-v2 types, which have no `extra` slot."""
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        log.warning("gem_from_dict: dropping unknown keys %s in %s", unknown, where)
    return {k: v for k, v in d.items() if k in names}


def _split_extra(cls, d: dict, exclude: set[str] = frozenset()) -> tuple[dict, dict | None]:
    names = {f.name for f in fields(cls)} - {"extra"}
    known = {k: v for k, v in d.items() if k in names and k not in exclude}
    extra = {k: v for k, v in d.items() if k not in names and k != "extra"}
    if d.get("extra"):
        extra = {**d["extra"], **extra}
    return known, (extra or None)


def _provenance_from_dict(d: dict | None) -> Provenance | None:
    return Provenance(**_known(Provenance, d, "Provenance")) if d else None


def _null_from_dict(d: dict | None) -> PermutationNull | None:
    return PermutationNull(**_known(PermutationNull, d, "PermutationNull")) if d else None


def _decomp_from_dict(d: dict | None) -> SubspaceSublayerDecomposition | None:
    if not d:
        return None
    d = _known(SubspaceSublayerDecomposition, d, "SubspaceSublayerDecomposition")
    d["provenance"] = _provenance_from_dict(d.get("provenance"))
    return SubspaceSublayerDecomposition(**d)


def _subspace_from_dict(d: dict | None) -> SettledSubspace | None:
    if not d:
        return None
    d = _known(SettledSubspace, d, "SettledSubspace")
    d["provenance"] = _provenance_from_dict(d.get("provenance"))
    d["sublayer_decomposition"] = _decomp_from_dict(d.get("sublayer_decomposition"))
    return SettledSubspace(**d)


def _layer_record_from_dict(d: dict) -> LayerRecord:
    d = _known(LayerRecord, d, "LayerRecord")
    d["null_fisher"] = _null_from_dict(d.get("null_fisher"))
    d["null_projected"] = _null_from_dict(d.get("null_projected"))
    if d.get("sublayers") is not None:
        d["sublayers"] = [SublayerRecord(**_known(SublayerRecord, s, "SublayerRecord"))
                          for s in d["sublayers"]]
    d["settled_subspace"] = _subspace_from_dict(d.get("settled_subspace"))
    return LayerRecord(**d)


def _layer_profile_from_dict(d: dict | None) -> LayerProfile | None:
    if not d:
        return None
    d = _known(LayerProfile, d, "LayerProfile")
    d["layers"] = [_layer_record_from_dict(r) for r in d.get("layers", [])]
    if d.get("embedding"):
        d["embedding"] = _layer_record_from_dict(d["embedding"])
    if d.get("final_norm"):
        d["final_norm"] = _layer_record_from_dict(d["final_norm"])
    if d.get("architecture"):
        d["architecture"] = ArchitectureFlags(**_known(ArchitectureFlags, d["architecture"], "ArchitectureFlags"))
    d["provenance"] = _provenance_from_dict(d.get("provenance"))
    return LayerProfile(**d)


def gem_from_dict(data: dict) -> ConceptGEM:
    """Deserialize a ConceptGEM from a dict (schema v1 or v2).

    v1 records (no schema_version) load with every v2 field None and
    schema_version "1". Unknown keys at the top level or in a node are kept
    in `extra` (and written back by gem_to_dict) rather than raising, so
    newer files degrade gracefully in older code paths. Does not mutate
    `data`."""
    nodes = []
    for raw in data.get("nodes", []):
        nd, node_extra = _split_extra(GEMNode, raw, exclude={"threads", "settled_subspace"})
        threads = [EigThread(**_known(EigThread, td, "EigThread")) for td in raw.get("threads", [])]
        nodes.append(GEMNode(
            threads=threads,
            settled_subspace=_subspace_from_dict(raw.get("settled_subspace")),
            extra=node_extra,
            **nd,
        ))

    top, top_extra = _split_extra(
        ConceptGEM, data,
        exclude={"nodes", "layer_profile", "sidecar", "schema_version"},
    )
    sidecar = data.get("sidecar")
    return ConceptGEM(
        nodes=nodes,
        schema_version=str(data.get("schema_version", "1")),
        layer_profile=_layer_profile_from_dict(data.get("layer_profile")),
        sidecar=SidecarRef(**_known(SidecarRef, sidecar, "SidecarRef")) if sidecar else None,
        extra=top_extra,
        **top,
    )


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path_for(json_path: Path) -> Path:
    """gem_<concept>.json -> gem_<concept>.sidecar.npz (same directory)."""
    json_path = Path(json_path)
    return json_path.with_name(json_path.stem + ".sidecar.npz")


def save_gem(
    gem: ConceptGEM,
    path: Path,
    arrays: dict[str, np.ndarray] | None = None,
    array_docs: dict[str, dict] | None = None,
    inline_directions: bool = False,
) -> ConceptGEM:
    """Save a ConceptGEM to a JSON file (always as the current schema version).

    arrays : optional large arrays (e.g. from build_layer_profile). When
        given they are written to a sidecar .npz next to the JSON
        (sidecar_path_for) and indexed in ConceptGEM.sidecar (relative path,
        sha256, per-key shape/dtype/axes/description from array_docs). The
        JSON itself never holds them.

    inline_directions : also write each LayerRecord.direction (u_l) inline
        in the JSON (from arrays["profile.directions"]; plus the final_norm
        record's direction from "profile.final_norm_direction" if present).
        Off by default: the sidecar already holds them.

    Returns the ConceptGEM as written (with schema_version/sidecar set); the
    input object is not mutated. Existing callers that ignore the return
    value are unaffected.
    """
    from dataclasses import replace
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gem = replace(gem, schema_version=GEM_SCHEMA_VERSION)

    if inline_directions:
        if gem.layer_profile is None or not arrays or "profile.directions" not in arrays:
            raise ValueError("inline_directions=True needs a layer_profile and arrays['profile.directions']")
        dirs = np.asarray(arrays["profile.directions"], dtype=np.float64)
        lp = gem.layer_profile
        new_layers = [replace(r, direction=dirs[r.layer].tolist()) for r in lp.layers]
        fn = lp.final_norm
        if fn is not None and "profile.final_norm_direction" in arrays:
            fn = replace(fn, direction=np.asarray(arrays["profile.final_norm_direction"], np.float64).tolist())
        gem = replace(gem, layer_profile=replace(lp, layers=new_layers, final_norm=fn))

    if arrays:
        sc_path = sidecar_path_for(path)
        np.savez_compressed(sc_path, **{k: np.asarray(v) for k, v in arrays.items()})
        docs = array_docs or {}
        index = {
            k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype),
                **docs.get(k, {})}
            for k, v in arrays.items()
        }
        gem = replace(gem, sidecar=SidecarRef(
            path=sc_path.name, format="npz", sha256=_sha256(sc_path), arrays=index,
        ))

    with open(path, "w") as f:
        json.dump(gem_to_dict(gem), f, indent=2, cls=_GEMEncoder)
    return gem


def load_gem(path: Path) -> ConceptGEM:
    """Load a ConceptGEM from a JSON file (v1 or v2; never reads the sidecar)."""
    with open(path) as f:
        data = json.load(f)
    return gem_from_dict(data)


def load_gem_sidecar(json_path: Path, gem: ConceptGEM | None = None, verify: bool = True):
    """Open the sidecar .npz of a saved GEM (lazy np.load handle; use as a
    mapping key -> array, close when done). Raises FileNotFoundError if the
    GEM has no sidecar, ValueError on sha256 mismatch when verify=True."""
    json_path = Path(json_path)
    gem = gem or load_gem(json_path)
    if gem.sidecar is None:
        raise FileNotFoundError(f"{json_path} has no sidecar reference")
    sc = json_path.parent / gem.sidecar.path
    if verify and gem.sidecar.sha256 and _sha256(sc) != gem.sidecar.sha256:
        raise ValueError(f"sidecar sha256 mismatch for {sc} (stale or modified)")
    return np.load(sc, allow_pickle=False)


# ---------------------------------------------------------------------------
# Validation and diagnostics
# ---------------------------------------------------------------------------

def validate_gem_node(node: GEMNode) -> list[str]:
    """Return a list of warnings (empty = valid).

    Checks:
    - settled_direction is unit norm (tolerance 1e-3)
    - handoff_layer = caz_end + 1 (or n_layers_total - 1)
    - concept_thread_id indexes a valid thread
    - cos_chain values are all in [0, 1]
    - entry_exit_cosine is in [0, 1]
    """
    warnings = []

    # Unit norm check
    sd = np.array(node.settled_direction, dtype=np.float64)
    norm = np.linalg.norm(sd)
    if abs(norm - 1.0) > 1e-3:
        warnings.append(
            f"settled_direction norm={norm:.4f}, expected ~1.0"
        )

    # Handoff layer check
    expected_handoff = min(node.caz_end + 1, node.n_layers_total - 1)
    if node.handoff_layer != expected_handoff:
        warnings.append(
            f"handoff_layer={node.handoff_layer}, expected {expected_handoff}"
        )

    # Thread index check
    if node.concept_thread_id < 0 or node.concept_thread_id >= len(node.threads):
        warnings.append(
            f"concept_thread_id={node.concept_thread_id} out of range "
            f"[0, {len(node.threads)})"
        )

    # Cosine chain check
    for thread in node.threads:
        for i, c in enumerate(thread.cos_chain):
            if c < -0.01 or c > 1.01:
                warnings.append(
                    f"thread {thread.thread_id} cos_chain[{i}]={c:.4f} "
                    f"out of range [0, 1]"
                )
                break

    # Entry-exit cosine
    if node.entry_exit_cosine < -0.01 or node.entry_exit_cosine > 1.01:
        warnings.append(
            f"entry_exit_cosine={node.entry_exit_cosine:.4f} out of range"
        )

    return warnings


def gem_diagnostics(gem: ConceptGEM) -> dict:
    """Compute summary diagnostics for a ConceptGEM.

    Returns a dict with aggregate statistics useful for quick
    assessment of GEM quality.
    """
    if gem.n_nodes == 0:
        return {"n_nodes": 0, "status": "empty"}

    entry_exit_cosines = [n.entry_exit_cosine for n in gem.nodes]
    handoff_cosines = [n.handoff_cosine for n in gem.nodes]
    handoff_depths = [n.depth_pct for n in gem.nodes]
    caz_scores = [n.caz_score for n in gem.nodes]
    rotations = [n.max_rotation_per_layer for n in gem.nodes]

    n_independent = sum(
        1 for t in (gem.node_types or []) if t == "independent"
    )
    n_upstream = sum(
        1 for t in (gem.node_types or []) if t == "upstream"
    )
    n_downstream = sum(
        1 for t in (gem.node_types or []) if t == "downstream"
    )

    return {
        "concept": gem.concept,
        "model_id": gem.model_id,
        "n_nodes": gem.n_nodes,
        "n_ablation_targets": len(gem.ablation_targets or []),
        "n_independent": n_independent,
        "n_upstream": n_upstream,
        "n_downstream": n_downstream,
        "entry_exit_cosine_mean": round(float(np.mean(entry_exit_cosines)), 4),
        "entry_exit_cosine_std": round(float(np.std(entry_exit_cosines)), 4),
        "handoff_cosine_mean": round(float(np.mean(handoff_cosines)), 4),
        "handoff_cosine_std": round(float(np.std(handoff_cosines)), 4),
        "handoff_depth_pct_mean": round(float(np.mean(handoff_depths)), 1),
        "max_rotation_mean": round(float(np.mean(rotations)), 4),
        "caz_score_max": round(float(max(caz_scores)), 4),
        "caz_score_min": round(float(min(caz_scores)), 4),
    }


# ---------------------------------------------------------------------------
# CIA cross-pollination — routing_held, stage_detail
# Lifted from Concept_Integrity_Auditor/cia_trace/deep.py and trace.py
# and adapted for the Rosetta GEM vocabulary (CAZ=allocation, GEM=assembly).
# ---------------------------------------------------------------------------

_CAZ_TYPE_BREAKPOINTS = [
    (0.10, "embedding"),   # L0–10% of depth
    (0.35, "active"),      # 10–35%
    (0.70, "deep"),        # 35–70%
    (0.90, "functional"),  # 70–90%
    (1.01, "artifact"),    # 90–100%
]


def classify_caz_depth(layer: int, n_layers: int) -> str:
    """Return a depth-class label for a layer index.

    Classes follow the CAZ allocation vocabulary:
      embedding  — very early (tokenisation / position encoding region)
      active     — shallow assembly (syntactic / surface features)
      deep       — mid-network (semantic integration)
      functional — late network (task-specific routing)
      artifact   — near-final (output projection artefacts)
    """
    rel = layer / max(n_layers - 1, 1)
    for threshold, label in _CAZ_TYPE_BREAKPOINTS:
        if rel < threshold:
            return label
    return "artifact"


def routing_held_ratio(
    baseline_per_layer: dict[int, float],
    peak_layer: int,
    final_layer: int,
    degradation_factor: float = 0.95,
) -> tuple[float, bool]:
    """Population-level routing survival computed from ablation baseline data.

    Measures whether the concept-discriminative signal at the CAZ peak layer
    persists to the final layer, without any intervention.

    Uses the Fisher separation scores from the clean (no-ablation) baseline
    in ablation_gem_*.json.  This is a population-level proxy for the
    per-input cosine check in CIA's DeepModule._compute_routing.

    Parameters
    ----------
    baseline_per_layer : dict[int, float]
        Clean separation at each measured layer, keyed by layer index.
        Comes from ablation_gem_*.json handoff.per_layer[l]["baseline_sep"].
    peak_layer : int
        The CAZ peak layer for this node.
    final_layer : int
        The model's last transformer layer index.
    degradation_factor : float
        Threshold fraction: ratio >= this → routing held.  Default 0.95
        matches CIA's DeepModule.

    Returns
    -------
    (ratio, held) where ratio = baseline[final] / baseline[peak], and
    held = ratio >= degradation_factor.
    """
    peak_sep = baseline_per_layer.get(peak_layer, 0.0)
    final_sep = baseline_per_layer.get(final_layer, 0.0)
    if peak_sep <= 0:
        return (0.0, False)
    ratio = final_sep / peak_sep
    return (round(ratio, 4), ratio >= degradation_factor)


def routing_held_from_permutation(
    subset_results: dict[str, dict],
    node_idx: int,
    final_layer: int,
    baseline_final_sep: float,
    contribution_threshold: float = 0.05,
) -> bool:
    """Derive routing_held for a single node from permutation ablation results.

    A node's signal is considered to "route" to the final layer if ablating
    that node alone causes at least contribution_threshold fractional reduction
    in final-layer separation.  If ablating the node has no effect on the
    final layer, its assembled signal died before reaching the output.

    Parameters
    ----------
    subset_results : dict
        The "subsets" dict from ablate_gem_permutation output.
        Keys are comma-separated node indices, e.g. "0", "1", "0,1".
    node_idx : int
        Which node to evaluate.
    final_layer : int
        Final transformer layer index (used to look up reduction values).
    baseline_final_sep : float
        Baseline separation at the final layer (no ablation).
    contribution_threshold : float
        Minimum final-layer reduction fraction to count as routed.

    Returns
    -------
    bool — True if this node contributes to final-layer representation.
    """
    key = str(node_idx)
    if key not in subset_results:
        return False
    return subset_results[key]["final_reduction"] >= contribution_threshold


def stage_detail(
    nodes: list,
    n_layers: int,
    routing_held_per_node: dict[int, bool] | None = None,
) -> str:
    """Human-readable one-sentence stage description for a multi-node GEM concept.

    Adapted from CIA's CIATrace._make_stage_detail.  Vocabulary follows
    Rosetta's CAZ=allocation/GEM=assembly framing.

    Parameters
    ----------
    nodes : list[GEMNode] or list[dict]
        The GEM nodes, sorted shallow to deep (standard GEM order).
        Accepts both GEMNode objects and raw JSON dicts.
    n_layers : int
        Total transformer layer count.
    routing_held_per_node : dict[int, bool] | None
        Per-node routing_held result.  None = unknown.

    Returns
    -------
    Human-readable string, e.g.:
        "Multimodal (2 regions): dominant functional L28–30 (score=0.69),
         also embedding; routing held for node 1, lost for node 0."
    """
    def _get(node, attr, default=None):
        if isinstance(node, dict):
            return node.get(attr, default)
        return getattr(node, attr, default)

    if not nodes:
        return "No GEM nodes detected."

    if len(nodes) == 1:
        n = nodes[0]
        start = _get(n, "caz_start")
        end = _get(n, "caz_end")
        score = _get(n, "caz_score", 0.0)
        depth = classify_caz_depth(_get(n, "caz_peak", 0), n_layers)
        routing = ""
        if routing_held_per_node is not None:
            routing = "; routing held." if routing_held_per_node.get(0, True) \
                else "; routing lost before final layer."
        return (
            f"Allocated in {depth} region only "
            f"(L{start}–{end}, score={score:.2f}){routing}"
        )

    # Multimodal: find dominant node by caz_score
    scored = [(i, _get(n, "caz_score", 0.0), n) for i, n in enumerate(nodes)]
    dom_i, dom_score, dom_node = max(scored, key=lambda x: x[1])
    dom_depth = classify_caz_depth(_get(dom_node, "caz_peak", 0), n_layers)
    dom_start = _get(dom_node, "caz_start")
    dom_end = _get(dom_node, "caz_end")

    others = [
        classify_caz_depth(_get(n, "caz_peak", 0), n_layers)
        for i, _, n in scored if i != dom_i
    ]

    routing_parts = []
    if routing_held_per_node is not None:
        for i in range(len(nodes)):
            held = routing_held_per_node.get(i, None)
            if held is not None:
                routing_parts.append(f"n{i} {'held' if held else 'lost'}")
    routing_str = (
        "; routing: " + ", ".join(routing_parts) + "."
        if routing_parts else "."
    )

    return (
        f"Multimodal ({len(nodes)} regions): dominant {dom_depth} "
        f"L{dom_start}–{dom_end} (score={dom_score:.2f}), "
        f"also {', '.join(others)}{routing_str}"
    )


# ---------------------------------------------------------------------------
# Phase 2 extraction: two distinct capabilities, both requiring raw
# per-example activations (unlike build_gem_node_k1 above, which only reads
# pre-computed caz_*.json summaries). Added 2026-09-24, motivated by an
# empirical result: across 51 concept x model combinations (17 concepts x 3
# architectures: pythia-70m, pythia-160m, Qwen2.5-0.5B-Instruct), a single
# direction removed from the settled/handoff layer still leaves the concept
# ~70% decodable on held-out data after 5 further rounds of removal (vs ~57%
# at shallow layers, Wilcoxon p=8.6e-09) -- late/settled layers carry more
# concept-relevant structure than a rank-1 settled_direction probe captures.
#
#   1. extract_phase2_threads -- completes the ORIGINAL Phase 2 design named
#      in EigThread's own docstring: several candidate rotating eigenvector
#      threads tracked layer-to-layer across the CAZ window, with the single
#      best one selected by cumulative_score, for surgical single-direction
#      ablation targeting. Produces real EigThread objects (not a separate
#      type), since GEMNode.threads is already typed for exactly this.
#
#   2. extract_settled_subspace -- a DIFFERENT question this original design
#      doesn't answer: several directions that are all SIMULTANEOUSLY
#      concept-relevant at one layer (typically handoff_layer), none
#      discarded once found (not competing candidates for "the" thread).
#      Populates GEMNode.settled_subspace, not threads.
# ---------------------------------------------------------------------------


def _topk_eigvecs(hp: np.ndarray, hn: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvectors/eigenvalues of the paired-difference second moment
    cov(d)+mu*mu^T at one layer (same object delta-PCA's leading eigenvector
    already summarizes -- see the GEM paper's delta-PCA comparison)."""
    if hp.shape != hn.shape:
        raise ValueError(
            f"_topk_eigvecs expects paired pos/neg rows of equal shape, got {hp.shape} vs {hn.shape}"
        )
    diffs = hp - hn
    _, s, vt = np.linalg.svd(diffs, full_matrices=False)
    eigvals = (s ** 2) / diffs.shape[0]
    k = min(k, len(eigvals))  # cannot track more threads than the difference matrix has rank
    order = np.argsort(eigvals)[::-1][:k]
    return vt[order], eigvals[order]  # (k, d), (k,)


def extract_phase2_threads(
    h_pos_by_layer: dict[int, np.ndarray],
    h_neg_by_layer: dict[int, np.ndarray],
    layer_indices: list[int],
    k: int = 3,
    dom_vectors_by_layer: dict[int, np.ndarray] | None = None,
) -> tuple[list[EigThread], int]:
    """Track k candidate eigenvector threads across a CAZ window and select
    the best one by cumulative_score, per EigThread's own docstring.

    h_pos_by_layer / h_neg_by_layer : dict layer -> (n, d) raw activations.
    dom_vectors_by_layer : per-layer dom_vector (from caz_*.json), used to
        compute concept_alignments = |cos(direction, dom_vector)| as
        EigThread's docstring specifies. Without it, alignment can't be
        scored and every thread gets alignment 1.0 (cumulative_score then
        just ranks threads by raw eigenvalue).

    Matching is a greedy bipartite assignment: for each existing thread, the
    available top-k candidate at the next layer with maximum |cosine| to its
    previous direction extends it. Validated 2026-09-24 on pythia-70m/
    negation: the selected best thread matched the independently-computed
    k=1 dom_vector trajectory at 0.93-0.97 cosine across all 6 layers, and
    had visibly higher layer-to-layer continuity (cos_chain 0.33-0.69) than
    the two rejected candidates (0.04-0.29) -- the selection criterion
    correctly separates the real thread from noise.
    """
    threads: list[EigThread] | None = None
    doms: dict[int, np.ndarray] = {}

    for layer in layer_indices:
        vecs, eigvals = _topk_eigvecs(h_pos_by_layer[layer], h_neg_by_layer[layer], k)
        k = len(eigvals)  # may have been clamped to the available rank
        if dom_vectors_by_layer is not None and layer in dom_vectors_by_layer:
            dom = np.asarray(dom_vectors_by_layer[layer], dtype=np.float64)
            doms[layer] = dom / np.linalg.norm(dom)

        if threads is None:
            threads = [
                EigThread(thread_id=i, layer_indices=[layer], directions=[vecs[i].tolist()],
                          eigenvalues=[float(eigvals[i])], concept_alignments=[], cos_chain=[1.0])
                for i in range(k)
            ]
            continue

        used: set[int] = set()
        for th in threads:
            last = np.array(th.directions[-1])
            sims = [abs(np.dot(last, vecs[j])) if j not in used else -1.0 for j in range(k)]
            j_best = int(np.argmax(sims))
            used.add(j_best)
            sign = np.sign(np.dot(last, vecs[j_best])) or 1.0
            th.directions.append((sign * vecs[j_best]).tolist())
            th.layer_indices.append(layer)
            th.eigenvalues.append(float(eigvals[j_best]))
            th.cos_chain.append(float(sims[j_best]))

    assert threads is not None
    for th in threads:
        if doms:
            alignments = [
                abs(float(np.dot(np.array(d), doms[l]))) if l in doms else float("nan")
                for d, l in zip(th.directions, th.layer_indices)
            ]
        else:
            alignments = [1.0] * len(th.directions)
        th.concept_alignments = alignments
        valid_mask = [not np.isnan(a) for a in alignments]
        valid = np.array([a for a, m in zip(alignments, valid_mask) if m])
        valid_eig = np.array([e for e, m in zip(th.eigenvalues, valid_mask) if m])
        th.cumulative_score = float(np.sum(valid_eig * valid)) if len(valid) else 0.0
        th.mean_eigenvalue = float(np.mean(th.eigenvalues))
        th.mean_alignment = float(np.mean(valid)) if len(valid) else 0.0

    best = max(threads, key=lambda t: t.cumulative_score)
    return threads, best.thread_id


def _project_out(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / np.linalg.norm(w)
    return X - np.outer(X @ w, w)


def _held_out_logloss(clf, X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(clf.predict_proba(X)[:, 1], eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _triage_binomial_pvalue(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Cheap triage screen: one-sided binomial test of held-out ACCURACY
    against chance (0.5). Two earlier metrics were tried and failed
    empirically (2026-09-24), recorded here rather than silently dropped:
      - training-data log-likelihood deviance vs. chi2(1): with n<<d (every
        round here), a regularized fit can always explain the training
        labels well after several rounds of removal, so it never detected
        exhaustion -- confirmed empirically: it stayed ~0 for 10 straight
        rounds on a layer whose held-out accuracy had already decayed to
        chance by round 8.
      - held-out log-loss deviance: fixed the training-set problem, but
        log-loss is sensitive to *calibrated* probability confidence, not
        just discrimination -- it called a layer "exhausted" at 73.8%
        held-out accuracy because the fit's probabilities were under-
        confident despite being reliably correct.
    A binomial test on hard accuracy sidesteps both failure modes: immune to
    training-set overfitting (only looks at held-out predictions) and immune
    to calibration (only cares about right/wrong, not confidence)."""
    from scipy.stats import binomtest
    n = len(y_test)
    correct = int(round(clf.score(X_test, y_test) * n))
    # Null = the majority-class classifier. Equals 0.5 for a balanced test set;
    # for an imbalanced one p=0.5 would call a constant predictor "significant".
    frac_pos = float(np.mean(y_test))
    p_null = max(frac_pos, 1.0 - frac_pos)
    return float(binomtest(correct, n, p=p_null, alternative="greater").pvalue)


def _permutation_pvalue(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    real_logloss: float, n_perm: int = 200, C: float = 1.0, rng=None,
) -> float:
    """Shuffle TRAINING labels, refit, score on the REAL held-out set, build
    a null distribution of held-out log-loss under label-shuffled fits.
    Returns the fraction of shuffled fits that do at least as well as the
    real fit (lower log-loss => more significant => small p). Expensive
    (n_perm refits) -- only called when the triage screen is ambiguous."""
    from sklearn.linear_model import LogisticRegression
    rng = rng or np.random.default_rng(0)
    null_losses = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(ytr)
        clf = LogisticRegression(max_iter=2000, C=C)
        try:
            clf.fit(Xtr, y_shuf)
        except Exception:
            continue
        null_losses.append(_held_out_logloss(clf, Xte, yte))
    null_losses = np.array(null_losses)
    p = (np.sum(null_losses <= real_logloss) + 1) / (len(null_losses) + 1)
    return float(p)


def extract_settled_subspace(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    layer: int,
    max_rounds: int = 10,
    stop_alpha: float = 0.05,
    triage_clear_significant: float = 1e-4,
    triage_clear_null: float = 0.5,
    n_perm: int = 200,
    C: float = 1.0,
    rng=None,
) -> SettledSubspace:
    """Iteratively fit-and-remove (INLP) at one layer. Each round, a triage
    binomial p-value decides stop/continue cheaply; the expensive permutation
    test only runs when that triage p-value is ambiguous -- not solidly in
    the "obviously still significant" (<triage_clear_significant) or
    "obviously exhausted" (>triage_clear_null) extremes. Given n<<d in every
    round (X has as many or more feature dimensions as training examples),
    the triage number is a screen, not a rigorous test on its own; the
    permutation test is the trustworthy fallback.

    **Status: experimental (v2.1.0).** ``k_effective`` is an *upper bound* on
    the settled rank, not an estimate of it. With n << d the fitted logistic
    direction is a noisy estimate of the true one; projecting it out leaves
    residual signal ||dmu|| * sin(theta) that the next round re-detects. On
    synthetic exactly-rank-1 data (d=512, n=250/class, 3-sigma shift) this
    returns k_effective = 2; at d=64 it returns 3. Any claim that a settled
    representation is "rank > 1" needs a matched rank-1 control run through
    this same procedure (same n, d, C, split) before it means anything. Pure
    noise correctly returns k_effective = 0.

    Caution (2026-09-24): the permutation test is the accuracy bottleneck --
    each ambiguous round costs n_perm refits. On a 70M-parameter model this
    took 30-90+ minutes for one layer when several rounds landed in the
    ambiguous band. Not yet fast enough to run across a full corpus; a
    production version would want fewer permutations, a faster-converging
    solver, or warm-starting from the previous round's coefficients.
    """
    rng = rng or np.random.default_rng(0)
    from sklearn.linear_model import LogisticRegression

    sub = SettledSubspace(layer=layer, extraction_method="inlp")
    Xtr, Xte = X_train.copy(), X_test.copy()

    for r in range(max_rounds):
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(Xtr, y_train)
        acc = clf.score(Xte, y_test)
        logloss = _held_out_logloss(clf, Xte, y_test)
        w = clf.coef_[0]
        w_norm = np.linalg.norm(w)

        sub.held_out_acc_curve.append(float(acc))
        sub.held_out_logloss_curve.append(logloss)

        if w_norm < 1e-8:
            sub.triage_pvalues.append(1.0)
            sub.permutation_pvalues.append(None)
            sub.stopping_reason = "coef_collapsed"
            break

        triage_p = _triage_binomial_pvalue(clf, Xte, y_test)
        sub.triage_pvalues.append(triage_p)

        if triage_p < triage_clear_significant:
            sub.permutation_pvalues.append(None)
            sub.directions.append((w / w_norm).tolist())
            sub.k_effective = r + 1
            Xtr = _project_out(Xtr, w)
            Xte = _project_out(Xte, w)
            continue

        if triage_p > triage_clear_null:
            sub.permutation_pvalues.append(None)
            sub.stopping_reason = "triage_clear_stop"
            break

        perm_p = _permutation_pvalue(Xtr, y_train, Xte, y_test, logloss, n_perm=n_perm, C=C, rng=rng)
        sub.permutation_pvalues.append(perm_p)
        if perm_p > stop_alpha:
            sub.stopping_reason = "permutation_stop"
            break

        sub.directions.append((w / w_norm).tolist())
        sub.k_effective = r + 1
        Xtr = _project_out(Xtr, w)
        Xte = _project_out(Xte, w)
    else:
        sub.stopping_reason = "max_rounds"

    return sub


def attach_settled_subspace(
    node: GEMNode,
    h_pos: np.ndarray,
    h_neg: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
    **extract_kwargs: Any,
) -> GEMNode:
    """Compute a SettledSubspace at `node.handoff_layer` from raw per-example
    activations and return a new GEMNode with it attached. Non-mutating
    (dataclasses.replace), matching the functional style of the builders
    above. h_pos, h_neg: (n_examples, hidden_dim) raw activations at
    node.handoff_layer for the positive/negative contrastive examples --
    NOT derivable from caz_*.json alone, must come from the extraction
    pipeline that produced the original activations.
    """
    from dataclasses import replace
    from sklearn.model_selection import train_test_split

    X = np.concatenate([h_pos, h_neg], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(h_pos.shape[0]), np.zeros(h_neg.shape[0])])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=node.handoff_layer, **extract_kwargs)
    return replace(node, settled_subspace=sub)


# ---------------------------------------------------------------------------
# Schema-v2 computation lives in rosetta_tools.gem_sublayer (torch imported
# lazily there). Re-exported here lazily (PEP 562) so `from rosetta_tools.gem
# import build_layer_profile` works without a circular import.
# ---------------------------------------------------------------------------

_SUBLAYER_EXPORTS = {
    "build_layer_profile", "projected_separation", "attach_layer_profile",
    "decompose_settled_subspace", "attach_layer_settled_subspaces",
    "capture_sublayer_activations", "extract_concept_layer_profile",
    "resolve_architecture", "find_decoder_layers", "SublayerCapture",
    "SIDECAR_DOCS", "ADDITIVITY_TOL",
}


def __getattr__(name: str) -> Any:
    if name in _SUBLAYER_EXPORTS:
        from rosetta_tools import gem_sublayer
        return getattr(gem_sublayer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

Usage:
    from geometric_evolution_map.gem import build_concept_gem, save_gem
    gem = build_concept_gem(caz_data, multimodal_data)
    save_gem(gem, Path("gem_causation.json"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
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
        Number of eigenvector threads to track.  Only k=1 is
        implemented in Phase 1.
    """
    metrics_raw = caz_data["layer_data"]["metrics"]

    # Detect CAZ regions
    layer_metrics = [
        LayerMetrics(
            m["layer"],
            m["separation_fisher"],
            m["coherence"],
            m["velocity"],
        )
        for m in metrics_raw
    ]
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
            raise NotImplementedError(
                f"Phase 2 (k={k}) not yet implemented.  Use k=1."
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
    """Serialize a ConceptGEM to a JSON-safe dict."""
    return asdict(gem)


def gem_from_dict(data: dict) -> ConceptGEM:
    """Deserialize a ConceptGEM from a dict."""
    nodes = []
    for nd in data.get("nodes", []):
        threads = [EigThread(**td) for td in nd.pop("threads", [])]
        nodes.append(GEMNode(threads=threads, **nd))

    return ConceptGEM(
        concept=data["concept"],
        model_id=data["model_id"],
        n_nodes=data["n_nodes"],
        nodes=nodes,
        interaction_matrix=data.get("interaction_matrix"),
        node_types=data.get("node_types"),
        ablation_targets=data.get("ablation_targets"),
    )


def save_gem(gem: ConceptGEM, path: Path) -> None:
    """Save a ConceptGEM to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gem_to_dict(gem), f, indent=2, cls=_GEMEncoder)


def load_gem(path: Path) -> ConceptGEM:
    """Load a ConceptGEM from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return gem_from_dict(data)


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

"""
feature_tracker.py — Cross-layer feature tracking for unsupervised CAZ discovery.

Given principal component directions at each layer (from manifold_detector),
tracks which directions persist across layers, which appear/disappear, and
which correspond to known concept probes.

A "feature" is a direction in activation space that persists across multiple
consecutive layers.  Each feature has:
  - birth layer (where it first appears)
  - death layer (where it fades below threshold)
  - lifespan (number of layers it persists)
  - strength trajectory (eigenvalue at each layer)
  - concept alignment (cosine² with known concept directions)

The algorithm:
  1. At each layer, take the top N significant PCs.
  2. Match each PC to the best-matching PC at the next layer via cosine similarity.
  3. If cos > threshold, they're the same feature continuing.
  4. Unmatched PCs at layer L+1 are feature births.
  5. Unmatched PCs at layer L are feature deaths.
  6. Build feature trajectories from these chains.

This gives us the full "feature ontology" of a model — every organized direction
tracked from birth to death, without requiring any human labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Feature:
    """A single tracked feature (persistent direction across layers)."""

    feature_id: int
    birth_layer: int               # first layer where this direction appears
    death_layer: int               # last layer where it's tracked
    lifespan: int                  # death - birth + 1

    # Per-layer data (indexed by layer - birth_layer)
    layer_indices: list[int]       # which layers this feature appears in
    pc_indices: list[int]          # which PC index it was at each layer
    eigenvalues: list[float]       # eigenvalue (variance) at each layer
    cos_chain: list[float]         # cosine similarity with previous layer's match

    # Peak info
    peak_layer: int                # layer of maximum eigenvalue
    peak_eigenvalue: float         # eigenvalue at peak
    peak_depth_pct: float          # peak as % of model depth
    mean_eigenvalue: float         # mean eigenvalue across lifespan

    # Concept alignment (computed post-hoc)
    concept_alignment: dict[str, float] = field(default_factory=dict)
    # concept → max cos² across all layers of this feature's life

    @property
    def is_transient(self) -> bool:
        """True if feature lives for only 1-2 layers."""
        return self.lifespan <= 2

    @property
    def is_persistent(self) -> bool:
        """True if feature lives for 5+ layers."""
        return self.lifespan >= 5


@dataclass
class FeatureMap:
    """Complete feature map for a model."""

    model_id: str
    n_layers: int
    hidden_dim: int
    n_features: int
    features: list[Feature]

    # Summary
    n_persistent: int              # features with lifespan >= 5
    n_transient: int               # features with lifespan <= 2
    n_labeled: int                 # features matching a known concept (cos² > 0.3)
    n_unlabeled: int               # features NOT matching any known concept
    max_concurrent: int            # max features alive at any one layer

    def features_at_layer(self, layer: int) -> list[Feature]:
        """Return all features alive at a given layer."""
        return [f for f in self.features if f.birth_layer <= layer <= f.death_layer]

    def persistent_features(self) -> list[Feature]:
        return [f for f in self.features if f.is_persistent]

    def unlabeled_features(self) -> list[Feature]:
        return [f for f in self.features if not any(
            v > 0.3 for v in f.concept_alignment.values()
        )]


def track_features(
    layer_directions: list[NDArray],
    layer_eigenvalues: list[list[float]],
    n_layers_total: int,
    cos_threshold: float = 0.5,
    min_eigenvalue_frac: float = 0.01,
    concept_directions: dict[str, NDArray] | None = None,
    model_id: str = "",
) -> FeatureMap:
    """Track features across layers.

    Parameters
    ----------
    layer_directions:
        List of arrays, one per layer.  Each has shape
        ``[n_pcs, hidden_dim]`` — the top PC directions at that layer.
    layer_eigenvalues:
        List of eigenvalue lists, one per layer.  Must be same length
        as layer_directions, with matching PC ordering.
    n_layers_total:
        Total layers in model (for depth percentage computation).
    cos_threshold:
        Minimum cosine similarity to consider two PCs at adjacent
        layers as the "same" feature.  Default 0.5.
    min_eigenvalue_frac:
        Minimum eigenvalue as fraction of layer total to consider
        a PC worth tracking.  Filters noise PCs.
    concept_directions:
        Optional dict of known concept directions for alignment.
    model_id:
        Model identifier for the result.

    Returns
    -------
    FeatureMap
        Complete feature map with all tracked features.
    """
    n_layers = len(layer_directions)
    concept_dirs = concept_directions or {}

    # Normalize concept directions
    concept_units = {}
    for name, vec in concept_dirs.items():
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            concept_units[name] = vec / norm

    # ── Build active PCs per layer ──
    # Filter to PCs above min eigenvalue threshold
    active_pcs = []  # per layer: list of (pc_index, direction, eigenvalue)
    for layer_idx in range(n_layers):
        dirs = layer_directions[layer_idx]
        eigs = layer_eigenvalues[layer_idx]
        total_var = sum(eigs) if eigs else 1.0

        layer_pcs = []
        for pc_idx in range(min(len(dirs), len(eigs))):
            if eigs[pc_idx] / total_var >= min_eigenvalue_frac:
                layer_pcs.append((pc_idx, dirs[pc_idx], eigs[pc_idx]))
        active_pcs.append(layer_pcs)

    # ── Track features via greedy cosine matching ──
    # Each "open track" is a feature being tracked
    open_tracks: list[dict] = []  # {layers, pc_indices, eigenvalues, cos_chain, direction}
    finished_tracks: list[dict] = []
    next_id = 0

    for layer_idx in range(n_layers):
        layer_pcs = active_pcs[layer_idx]
        if not layer_pcs:
            # Close all open tracks
            finished_tracks.extend(open_tracks)
            open_tracks = []
            continue

        # Build cosine similarity matrix between open tracks and current PCs
        matched_tracks = set()
        matched_pcs = set()

        if open_tracks and layer_pcs:
            track_dirs = np.array([t["direction"] for t in open_tracks])
            pc_dirs = np.array([pc[1] for pc in layer_pcs])

            # Cosine similarity: |cos| because direction sign is arbitrary in PCA
            cos_matrix = np.abs(track_dirs @ pc_dirs.T)  # [n_tracks, n_pcs]

            # Greedy matching: best pairs first
            while True:
                if cos_matrix.size == 0:
                    break
                best_idx = np.unravel_index(np.argmax(cos_matrix), cos_matrix.shape)
                best_cos = cos_matrix[best_idx]

                if best_cos < cos_threshold:
                    break

                track_idx, pc_idx = best_idx
                if track_idx in matched_tracks or pc_idx in matched_pcs:
                    cos_matrix[best_idx] = -1
                    continue

                # Match: extend the track
                track = open_tracks[track_idx]
                pc = layer_pcs[pc_idx]
                track["layers"].append(layer_idx)
                track["pc_indices"].append(pc[0])
                track["eigenvalues"].append(pc[2])
                track["cos_chain"].append(float(best_cos))
                track["direction"] = pc[1]  # update to current direction

                matched_tracks.add(track_idx)
                matched_pcs.add(pc_idx)
                cos_matrix[track_idx, :] = -1
                cos_matrix[:, pc_idx] = -1

        # Close unmatched tracks
        new_open = []
        for i, track in enumerate(open_tracks):
            if i in matched_tracks:
                new_open.append(track)
            else:
                finished_tracks.append(track)
        open_tracks = new_open

        # Birth new tracks for unmatched PCs
        for pc_idx_local, (pc_idx, direction, eigenvalue) in enumerate(layer_pcs):
            if pc_idx_local not in matched_pcs:
                open_tracks.append({
                    "id": next_id,
                    "layers": [layer_idx],
                    "pc_indices": [pc_idx],
                    "eigenvalues": [eigenvalue],
                    "cos_chain": [1.0],  # first appearance
                    "direction": direction,
                })
                next_id += 1

    # Close remaining tracks
    finished_tracks.extend(open_tracks)

    # ── Build Feature objects ──
    features = []
    hidden_dim = layer_directions[0].shape[1] if layer_directions and len(layer_directions[0]) > 0 else 0

    for track in finished_tracks:
        layers = track["layers"]
        eigs = track["eigenvalues"]
        peak_idx = int(np.argmax(eigs))

        birth = layers[0]
        death = layers[-1]
        lifespan = death - birth + 1

        # Concept alignment: check direction at peak layer against known concepts
        peak_direction = None
        # Get the direction at the peak
        peak_layer_pos = layers.index(layers[peak_idx]) if peak_idx < len(layers) else 0
        for layer_idx_check in range(n_layers):
            if layer_idx_check == layers[peak_idx]:
                pc_target = track["pc_indices"][peak_idx]
                dirs_at_layer = layer_directions[layer_idx_check]
                if pc_target < len(dirs_at_layer):
                    peak_direction = dirs_at_layer[pc_target]
                break

        concept_align = {}
        if peak_direction is not None:
            peak_unit = peak_direction / (np.linalg.norm(peak_direction) + 1e-12)
            for c_name, c_unit in concept_units.items():
                cos_sq = float(np.dot(peak_unit, c_unit)) ** 2
                concept_align[c_name] = round(cos_sq, 4)

        f = Feature(
            feature_id=track["id"],
            birth_layer=birth,
            death_layer=death,
            lifespan=lifespan,
            layer_indices=layers,
            pc_indices=track["pc_indices"],
            eigenvalues=[round(e, 4) for e in eigs],
            cos_chain=[round(c, 4) for c in track["cos_chain"]],
            peak_layer=layers[peak_idx],
            peak_eigenvalue=round(eigs[peak_idx], 4),
            peak_depth_pct=round(100 * layers[peak_idx] / n_layers_total, 1),
            mean_eigenvalue=round(float(np.mean(eigs)), 4),
            concept_alignment=concept_align,
        )
        features.append(f)

    # Sort by peak eigenvalue descending
    features.sort(key=lambda f: -f.peak_eigenvalue)

    # ── Summary stats ──
    n_persistent = sum(1 for f in features if f.is_persistent)
    n_transient = sum(1 for f in features if f.is_transient)
    n_labeled = sum(1 for f in features if any(v > 0.3 for v in f.concept_alignment.values()))
    n_unlabeled = len(features) - n_labeled

    # Max concurrent features at any layer
    max_concurrent = 0
    for l in range(n_layers):
        concurrent = sum(1 for f in features if f.birth_layer <= l <= f.death_layer)
        max_concurrent = max(max_concurrent, concurrent)

    return FeatureMap(
        model_id=model_id,
        n_layers=n_layers_total,
        hidden_dim=hidden_dim,
        n_features=len(features),
        features=features,
        n_persistent=n_persistent,
        n_transient=n_transient,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
        max_concurrent=max_concurrent,
    )

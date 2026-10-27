"""
conftest.py — Shared synthetic-data helpers for the rosetta_tools test suite.

Plain functions, not pytest fixtures — imported directly by test modules
(``from .conftest import make_cloud``).  Consolidates helpers that were
previously duplicated verbatim across test_caz.py and test_probes.py, and
adds new helpers needed by test_gem.py.

No GPU, no model loading — every helper here builds purely synthetic
numpy/dict data with known geometric structure.
"""

from __future__ import annotations

import numpy as np

from rosetta_tools.caz import LayerMetrics, compute_velocity


def make_cloud(
    center: np.ndarray,
    n: int = 50,
    std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian cloud around center, shape [n, len(center)]."""
    rng = np.random.default_rng(seed)
    return center + rng.standard_normal((n, len(center))) * std


def make_layer_acts(
    n_layers: int,
    hidden_dim: int,
    peak_layer: int,
    n: int = 50,
    sep: float = 5.0,
    noise: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Synthetic contrastive activations with one clearly separating layer.

    At `peak_layer`, pos is centered at +sep along dim 0, neg at origin.
    All other layers have overlapping distributions (no separation).
    """
    rng = np.random.default_rng(42)
    layer_acts = []
    for li in range(n_layers):
        if li == peak_layer:
            pos_center = np.zeros(hidden_dim)
            pos_center[0] = sep
            pos = make_cloud(pos_center, n=n, std=noise, seed=li)
            neg = make_cloud(np.zeros(hidden_dim), n=n, std=noise, seed=li + 1000)
        else:
            cloud = rng.standard_normal((2 * n, hidden_dim)) * noise
            pos = cloud[:n]
            neg = cloud[n:]
        layer_acts.append((pos.astype(np.float32), neg.astype(np.float32)))
    return layer_acts


def make_layer_metrics_from_curve(
    separations: list[float],
    coherence: float | list[float] = 0.5,
    velocity_window: int = 1,
) -> list[LayerMetrics]:
    """Build a list[LayerMetrics] from an explicit separation curve.

    Velocity is auto-derived via compute_velocity so the resulting metrics
    are internally consistent, the way a real extraction pipeline would
    produce them. `coherence` may be a scalar (applied to every layer) or
    a per-layer list.
    """
    n = len(separations)
    cohs = coherence if isinstance(coherence, list) else [coherence] * n
    vels = compute_velocity(separations, window=velocity_window)
    return [
        LayerMetrics(
            layer=i,
            separation=float(separations[i]),
            coherence=float(cohs[i]),
            velocity=float(vels[i]),
        )
        for i in range(n)
    ]


def make_caz_data(
    directions: list[np.ndarray],
    separations: list[float] | None = None,
    coherences: list[float] | None = None,
    velocities: list[float] | None = None,
    concept: str = "test_concept",
    model_id: str = "test_model",
) -> dict:
    """Build a synthetic caz_data dict matching the real extraction pipeline's shape.

    Matches the input build_gem_node_k1/build_concept_gem expect:
    {concept, model_id, layer_data: {n_layers, metrics: [{layer,
    separation_fisher, coherence, velocity, dom_vector}, ...]}}
    (see Rosetta_Full_Tutorial.ipynb cell 18 for the real-pipeline equivalent).

    `directions` need not be unit norm — build_gem_node_k1 normalizes on read.
    One entry per layer; defines n_layers and hidden_dim.
    """
    n_layers = len(directions)
    seps = separations if separations is not None else [1.0] * n_layers
    cohs = coherences if coherences is not None else [0.5] * n_layers
    vels = velocities if velocities is not None else [0.0] * n_layers

    metrics = [
        {
            "layer": i,
            "separation_fisher": float(seps[i]),
            "coherence": float(cohs[i]),
            "velocity": float(vels[i]),
            "dom_vector": [float(x) for x in directions[i]],
        }
        for i in range(n_layers)
    ]
    return {
        "concept": concept,
        "model_id": model_id,
        "layer_data": {"n_layers": n_layers, "metrics": metrics},
    }

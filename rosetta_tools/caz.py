"""
caz.py — Concept Assembly Zone metric computation.

Implements the three layer-wise metrics that characterize the CAZ:

  S(l)  Separation  — Fisher-normalized centroid distance between concept classes
  C(l)  Coherence   — explained variance of the primary PCA component
  v(l)  Velocity    — rate of change of separation (dS/dl, smoothed)

And the boundary detection algorithm that identifies:
  caz_start  — layer where velocity crosses the sustained threshold
  caz_peak   — layer of maximum separation
  caz_end    — layer where velocity turns consistently negative

All functions operate on numpy arrays and have no dependency on any
specific model library (HuggingFace, TransformerLens, etc.).  Feed in
activation matrices, get metrics back.

Typical usage
-------------
    import numpy as np
    from rosetta_tools.caz import (
        compute_separation,
        compute_coherence,
        compute_velocity,
        compute_layer_metrics,
        find_caz_boundary,
    )

    # activations_pos: [n_samples, hidden_dim] for positive class at one layer
    # activations_neg: [n_samples, hidden_dim] for negative class at one layer
    S = compute_separation(activations_pos, activations_neg)
    C = compute_coherence(activations_pos, activations_neg)

    # For a full model run:
    # layer_acts: list of (pos_acts, neg_acts) tuples, one per layer
    metrics = compute_layer_metrics(layer_acts)
    boundary = find_caz_boundary(metrics)
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Per-layer metrics
# ---------------------------------------------------------------------------


def compute_separation(
    pos: NDArray,
    neg: NDArray,
    eps: float = 1e-8,
) -> float:
    """Fisher-normalized centroid distance between two activation distributions.

    Parameters
    ----------
    pos:
        Activations for the positive class — shape ``[n_pos, hidden_dim]``.
        Must be float32 or float64 (cast fp16 activations before calling).
    neg:
        Activations for the negative class — shape ``[n_neg, hidden_dim]``.
    eps:
        Small constant added to the denominator to prevent division by zero.

    Returns
    -------
    float
        Fisher-normalized separation S(l).  Higher means more discriminable.
        Returns 0.0 if either class has fewer than 2 samples.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    if len(pos) < 2 or len(neg) < 2:
        return 0.0

    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)

    var_pos = pos.var(axis=0)
    var_neg = neg.var(axis=0)

    centroid_dist = float(np.linalg.norm(mu_pos - mu_neg))
    within_scatter = float(np.sqrt((var_pos + var_neg).mean())) + eps

    return centroid_dist / within_scatter


def compute_coherence(
    pos: NDArray,
    neg: NDArray,
    n_components: int = 1,
) -> float:
    """Explained variance ratio of the primary PCA component across both classes.

    Measures how geometrically concentrated the concept direction is —
    a high coherence means the concept is well-represented by a single
    linear direction in the residual stream.

    Parameters
    ----------
    pos:
        Activations for the positive class — shape ``[n_pos, hidden_dim]``.
    neg:
        Activations for the negative class — shape ``[n_neg, hidden_dim]``.
    n_components:
        Number of PCA components to fit (default: 1 — only the primary).

    Returns
    -------
    float
        Explained variance ratio of the first principal component.
        Range [0, 1]; higher means more geometrically concentrated.
        Returns 0.0 if there are fewer than 2 total samples.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    all_acts = np.concatenate([pos, neg], axis=0)
    if len(all_acts) < 2:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=min(n_components, all_acts.shape[1]))
        pca.fit(all_acts)

    return float(pca.explained_variance_ratio_[0])


def compute_velocity(
    separations: list[float] | NDArray,
    window: int = 3,
) -> NDArray[np.float64]:
    """Rate of change of separation across layers (smoothed first derivative).

    Parameters
    ----------
    separations:
        Sequence of S(l) values, one per layer.
    window:
        Smoothing window for the gradient computation (default: 3).

    Returns
    -------
    NDArray[np.float64]
        Velocity v(l) for each layer.  Same length as ``separations``.
        First element is always 0.0 (no previous layer to diff against).
    """
    seps = np.asarray(separations, dtype=np.float64)
    if len(seps) < 2:
        return np.zeros_like(seps)

    # Smooth separations before differencing to reduce noise
    if window > 1 and len(seps) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(seps, kernel, mode="same")
    else:
        smoothed = seps

    vel = np.zeros_like(smoothed)
    vel[1:] = np.diff(smoothed)
    return vel


# ---------------------------------------------------------------------------
# Full layer-sweep computation
# ---------------------------------------------------------------------------


class LayerMetrics(NamedTuple):
    """Metrics at a single transformer layer."""

    layer: int
    separation: float
    coherence: float
    velocity: float


def compute_layer_metrics(
    layer_activations: list[tuple[NDArray, NDArray]],
    velocity_window: int = 3,
) -> list[LayerMetrics]:
    """Compute S, C, v for every layer in a model run.

    Parameters
    ----------
    layer_activations:
        List of ``(pos_acts, neg_acts)`` tuples, one per layer
        (including the embedding layer if captured).
        Each array is ``[n_samples, hidden_dim]``, float32 or float64.
    velocity_window:
        Smoothing window for velocity computation.

    Returns
    -------
    list[LayerMetrics]
        One ``LayerMetrics`` per layer, in order.
    """
    separations = []
    coherences = []

    for pos, neg in layer_activations:
        separations.append(compute_separation(pos, neg))
        coherences.append(compute_coherence(pos, neg))

    velocities = compute_velocity(separations, window=velocity_window)

    return [
        LayerMetrics(
            layer=i,
            separation=separations[i],
            coherence=coherences[i],
            velocity=float(velocities[i]),
        )
        for i in range(len(layer_activations))
    ]


# ---------------------------------------------------------------------------
# CAZ boundary detection
# ---------------------------------------------------------------------------


class CAZBoundary(NamedTuple):
    """CAZ boundary locations for a single concept × model run."""

    caz_start: int
    caz_peak: int
    caz_end: int
    caz_width: int
    peak_separation: float
    threshold: float


def find_caz_boundary(
    metrics: list[LayerMetrics],
    threshold_factor: float = 0.5,
    min_sustained: int = 2,
) -> CAZBoundary:
    """Identify the CAZ boundaries from layer-wise metrics.

    The CAZ is defined as the contiguous region where velocity exceeds
    a sustained threshold.  The peak is the layer of maximum separation.

    Parameters
    ----------
    metrics:
        Output of ``compute_layer_metrics()``.
    threshold_factor:
        Velocity threshold as a fraction of the maximum velocity.
        Default 0.5 — the CAZ begins where velocity exceeds 50% of its peak.
    min_sustained:
        Minimum number of consecutive layers above threshold to count
        as a genuine CAZ onset (reduces noise sensitivity).

    Returns
    -------
    CAZBoundary
        Named tuple with start, peak, end, width, peak separation, threshold.
        If no clear boundary is found, returns the full model as the CAZ.
    """
    if not metrics:
        raise ValueError("metrics list is empty")

    seps = np.array([m.separation for m in metrics])
    vels = np.array([m.velocity for m in metrics])

    peak_layer = int(np.argmax(seps))
    peak_sep = float(seps[peak_layer])

    max_vel = float(vels.max())
    threshold = threshold_factor * max_vel

    # Find CAZ start — first layer where velocity is sustained above threshold
    caz_start = 0
    if max_vel > 0 and threshold > 0:
        for i in range(len(vels)):
            window = vels[i : i + min_sustained]
            if len(window) >= min_sustained and (window > threshold).all():
                caz_start = i
                break

    # Find CAZ end — last layer before velocity turns consistently negative
    caz_end = len(metrics) - 1
    for i in range(peak_layer, len(vels)):
        window = vels[i : i + min_sustained]
        if len(window) >= min_sustained and (window < 0).all():
            caz_end = max(i - 1, peak_layer)
            break

    caz_width = caz_end - caz_start + 1

    return CAZBoundary(
        caz_start=caz_start,
        caz_peak=peak_layer,
        caz_end=caz_end,
        caz_width=caz_width,
        peak_separation=peak_sep,
        threshold=threshold,
    )

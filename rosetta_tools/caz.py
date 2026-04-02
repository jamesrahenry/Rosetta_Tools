"""
caz.py — Concept Assembly Zone metric computation.

Implements the three layer-wise metrics that characterize the CAZ:

  S(l)  Separation  — Fisher-normalized centroid distance between concept classes
  C(l)  Coherence   — explained variance of the primary PCA component
  v(l)  Velocity    — rate of change of separation (dS/dl, smoothed)

And two levels of boundary detection:

  find_caz_boundary   — legacy single-peak detector (wraps the global max)
  find_caz_regions    — multi-modal detector that finds ALL significant
                        separation peaks and returns a CAZProfile describing
                        the full shape of concept assembly

The single-peak API is preserved for backward compatibility.  New analysis
code should use ``find_caz_regions`` to avoid the assumption that each
concept assembles at exactly one location.

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
        find_caz_boundary,     # legacy single-peak
        find_caz_regions,      # multi-modal shape detection
    )

    # activations_pos: [n_samples, hidden_dim] for positive class at one layer
    # activations_neg: [n_samples, hidden_dim] for negative class at one layer
    S = compute_separation(activations_pos, activations_neg)
    C = compute_coherence(activations_pos, activations_neg)

    # For a full model run:
    # layer_acts: list of (pos_acts, neg_acts) tuples, one per layer
    metrics = compute_layer_metrics(layer_acts)

    # Legacy (single peak):
    boundary = find_caz_boundary(metrics)

    # Full shape (multi-modal):
    profile = find_caz_regions(metrics)
    print(f"{profile.n_regions} assembly regions, dominant at L{profile.dominant.peak}")
    for region in profile.regions:
        print(f"  L{region.start}–L{region.end}: peak S={region.peak_separation:.3f}")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.signal import find_peaks
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

    Implements the separation metric from the CAZ framework (Henry 2026):

        S(l) = ‖h̄_A^(l) − h̄_B^(l)‖₂ / √[ (1/2)( tr(Σ_A^(l)) + tr(Σ_B^(l)) ) ]

    The denominator is the square root of the average trace of the two
    within-class covariance matrices.  tr(Σ) = sum of per-dimension variances,
    which is the total within-class spread in all directions.  Dividing by the
    square root of the average gives the Fisher-normalized criterion that
    corrects for layer-wise variation in activation dispersion.

    Mahalanobis distance would account for full covariance structure but is
    numerically unstable without regularization in high-dimensional spaces.
    This formulation is the principled tradeoff described in the CAZ paper.

    Parameters
    ----------
    pos:
        Activations for the positive class — shape ``[n_pos, hidden_dim]``.
        Must be float32 or float64 (cast fp16 activations before calling).
    neg:
        Activations for the negative class — shape ``[n_neg, hidden_dim]``.
    eps:
        Small constant added to the denominator to prevent division by zero
        in degenerate cases (zero within-class variance).

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

    # Drop rows containing NaN before computing statistics.
    # NaN propagates through mean/var and produces NaN results silently —
    # the kind of silent corruption that produced the fp16 overflow failure.
    pos = pos[np.isfinite(pos).all(axis=1)]
    neg = neg[np.isfinite(neg).all(axis=1)]

    if len(pos) < 2 or len(neg) < 2:
        return 0.0

    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)

    # tr(Σ) = sum of per-dimension variances (ddof=1 for unbiased estimate)
    trace_pos = float(pos.var(axis=0, ddof=1).sum())
    trace_neg = float(neg.var(axis=0, ddof=1).sum())

    centroid_dist = float(np.linalg.norm(mu_pos - mu_neg))
    # √[ (1/2)(tr(Σ_A) + tr(Σ_B)) ] — exact formula from CAZ paper §3.2
    within_scatter = float(np.sqrt(0.5 * (trace_pos + trace_neg))) + eps

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

    # Drop rows containing NaN or Inf — sklearn PCA raises on non-finite input.
    all_acts = all_acts[np.isfinite(all_acts).all(axis=1)]
    if len(all_acts) < 2:
        return 0.0

    # Constant arrays have zero variance — PCA returns NaN explained_variance_ratio.
    # Guard: if all rows are identical, there is no dominant direction → 0.0.
    if np.all(all_acts == all_acts[0]):
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=min(n_components, all_acts.shape[1]))
        pca.fit(all_acts)

    ratio = float(pca.explained_variance_ratio_[0])
    # PCA can return NaN when variance is numerically zero — return 0.0 in that case
    return ratio if np.isfinite(ratio) else 0.0


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


# ---------------------------------------------------------------------------
# Regional statistics
# ---------------------------------------------------------------------------


def compute_caz_statistics(
    metrics: list[LayerMetrics],
    boundary: CAZBoundary,
) -> dict:
    """Compute per-region summary statistics for a CAZ result.

    Splits the model into pre-CAZ, CAZ, and post-CAZ regions and reports
    mean/max separation and coherence for each.

    Parameters
    ----------
    metrics:
        Output of ``compute_layer_metrics()``.
    boundary:
        Output of ``find_caz_boundary()``.

    Returns
    -------
    dict
        Keys: ``pre_caz``, ``caz``, ``post_caz`` — each a sub-dict with
        ``mean_separation``, ``std_separation``, ``n_layers``, and
        (for the CAZ region) ``max_separation`` and ``mean_coherence``.
    """
    seps = np.array([m.separation for m in metrics])
    cohs = np.array([m.coherence for m in metrics])
    s, e = boundary.caz_start, boundary.caz_end

    pre = seps[:s] if s > 0 else np.array([])
    caz = seps[s : e + 1]
    post = seps[e + 1 :] if e < len(seps) - 1 else np.array([])

    def _region(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"mean_separation": 0.0, "std_separation": 0.0, "n_layers": 0}
        return {
            "mean_separation": float(arr.mean()),
            "std_separation": float(arr.std()),
            "n_layers": len(arr),
        }

    stats = {
        "pre_caz": _region(pre),
        "caz": {
            **_region(caz),
            "max_separation": float(caz.max()) if len(caz) else 0.0,
            "mean_coherence": float(cohs[s : e + 1].mean()) if len(caz) else 0.0,
        },
        "post_caz": _region(post),
    }
    return stats


# ---------------------------------------------------------------------------
# Multi-modal CAZ detection
# ---------------------------------------------------------------------------


@dataclass
class CAZRegion:
    """A single assembly region within the S(l) curve.

    One concept may have multiple regions (e.g. a shallow lexical peak
    and a deep compositional peak).  Each region has its own boundaries,
    peak, and summary statistics.

    The ``caz_score`` is a composite strength metric that combines
    separation prominence with coherence, allowing downstream code
    to rank regions by importance without imposing hard thresholds.
    """

    start: int                     # first layer of this region
    peak: int                      # layer of maximum separation within region
    end: int                       # last layer of this region
    width: int                     # end - start + 1
    peak_separation: float         # S(l) at peak
    peak_coherence: float          # C(l) at peak
    mean_separation: float         # mean S(l) across region
    mean_coherence: float          # mean C(l) across region
    prominence: float              # scipy peak prominence (height above saddle)
    depth_pct: float               # peak as % of model depth
    width_pct: float               # width as % of model depth

    # Asymmetry within this region
    rise_span: int                 # layers from start to peak
    fall_span: int                 # layers from peak to end

    # Composite strength score — higher means stronger assembly signal.
    # Combines prominence (how much does S stand out from neighbors)
    # with coherence (how geometrically organized is the direction).
    # Range: 0+ (unbounded, but typically 0–1 for most regions).
    caz_score: float = 0.0


@dataclass
class CAZProfile:
    """Full shape description of concept assembly across all layers.

    Replaces the single-peak CAZBoundary with a multi-region view.
    The ``dominant`` property points to the tallest region, preserving
    backward compatibility with code that expects a single peak.
    """

    n_layers: int
    regions: list[CAZRegion]
    n_regions: int

    # Global shape descriptors
    global_peak_layer: int         # layer of absolute max S(l)
    global_peak_separation: float  # S at that layer
    global_mean_separation: float  # mean S(l) across all layers
    is_multimodal: bool            # True if 2+ significant regions

    @property
    def dominant(self) -> CAZRegion:
        """The region with the highest peak separation."""
        return max(self.regions, key=lambda r: r.peak_separation)

    @property
    def secondary(self) -> CAZRegion | None:
        """Second-tallest region, or None if unimodal."""
        if len(self.regions) < 2:
            return None
        sorted_r = sorted(self.regions, key=lambda r: r.peak_separation, reverse=True)
        return sorted_r[1]

    def to_legacy_boundary(self) -> CAZBoundary:
        """Convert the dominant region to a legacy CAZBoundary for compat."""
        d = self.dominant
        return CAZBoundary(
            caz_start=d.start,
            caz_peak=d.peak,
            caz_end=d.end,
            caz_width=d.width,
            peak_separation=d.peak_separation,
            threshold=0.0,
        )


def find_caz_regions(
    metrics: list[LayerMetrics],
    min_prominence_frac: float = 0.10,
    min_peak_distance: int = 2,
) -> CAZProfile:
    """Detect all significant assembly regions in the S(l) curve.

    This is the legacy interface with a hard prominence threshold.
    For inclusive detection with scoring, use ``find_caz_regions_scored``.

    Parameters
    ----------
    metrics:
        Output of ``compute_layer_metrics()``.
    min_prominence_frac:
        Minimum peak prominence as a fraction of the global max separation.
        Peaks below this threshold are treated as noise.  Default 0.10
        (10% of max separation).
    min_peak_distance:
        Minimum distance (in layers) between adjacent peaks.
        Prevents detecting noise ripples as separate regions.

    Returns
    -------
    CAZProfile
        Full shape description with all detected regions.
    """
    return find_caz_regions_scored(
        metrics,
        min_prominence_frac=min_prominence_frac,
        min_peak_distance=min_peak_distance,
    )


def find_caz_regions_scored(
    metrics: list[LayerMetrics],
    min_prominence_frac: float = 0.005,
    min_peak_distance: int = 2,
) -> CAZProfile:
    """Detect assembly regions with composite CAZ scoring.

    Philosophy: cast a wide net, score everything, filter later.
    Any bump in the separation curve that shows geometric organization
    (coherence) is a candidate CAZ.  The ``caz_score`` on each region
    lets downstream code decide what's strong enough to care about.

    Detection
    ---------
    Uses scipy's ``find_peaks`` with a very low prominence floor
    (default 0.5% of max separation) to catch even subtle assembly
    events.  Every detected peak gets scored and returned.

    Scoring
    -------
    The CAZ score combines three signals:

        caz_score = prominence_norm × coherence_boost × width_factor

    Where:
      - ``prominence_norm`` = prominence / global_mean_separation
        How much does the peak stand out, relative to baseline?
      - ``coherence_boost`` = 1 + peak_coherence / mean_coherence
        Does this peak have above-average geometric organization?
        A peak with coherence at the model-wide mean scores 2.0;
        above-mean scores higher.  This ensures a low-prominence
        peak with strong coherence still gets a meaningful score.
      - ``width_factor`` = sqrt(width / n_layers)
        Wider regions are more sustained and less likely to be noise.
        Square root prevents very wide regions from dominating.

    The score is NOT a probability — it's an importance ranking.
    Strong, coherent, sustained peaks score highest.  Weak, narrow,
    incoherent bumps score near zero but are still reported.

    Region boundaries
    -----------------
    Each region extends from one saddle point (local minimum between
    peaks) to the next.  First region starts at layer 0; last ends
    at the final layer.

    Parameters
    ----------
    metrics:
        Output of ``compute_layer_metrics()``.
    min_prominence_frac:
        Minimum peak prominence as a fraction of global max separation.
        Default 0.005 (0.5%) — intentionally very low to be inclusive.
        Use ``caz_score`` to filter rather than raising this.
    min_peak_distance:
        Minimum distance (in layers) between adjacent peaks.

    Returns
    -------
    CAZProfile
        All detected regions with ``caz_score`` populated.
        Regions are sorted by layer index (start to end of model).
    """
    if not metrics:
        raise ValueError("metrics list is empty")

    seps = np.array([m.separation for m in metrics], dtype=np.float64)
    cohs = np.array([m.coherence for m in metrics], dtype=np.float64)
    n_layers = len(metrics)

    global_peak = int(np.argmax(seps))
    global_peak_sep = float(seps[global_peak])
    global_mean_sep = float(seps.mean())
    global_mean_coh = float(cohs.mean()) if cohs.mean() > 0 else 1e-6

    # Find all peaks — very low prominence floor to be inclusive
    min_prominence = max(min_prominence_frac * global_peak_sep, 1e-6)
    peak_indices, properties = find_peaks(
        seps,
        prominence=min_prominence,
        distance=min_peak_distance,
    )

    # If find_peaks returns nothing (monotonic or flat), use the global max
    if len(peak_indices) == 0:
        peak_indices = np.array([global_peak])
        properties = {"prominences": np.array([global_peak_sep])}

    # Sort peaks by layer index
    sort_order = np.argsort(peak_indices)
    peak_indices = peak_indices[sort_order]
    prominences = properties["prominences"][sort_order]

    # Find saddle points (local minima between consecutive peaks)
    saddles = []
    for i in range(len(peak_indices) - 1):
        segment = seps[peak_indices[i]:peak_indices[i + 1] + 1]
        saddle_offset = int(np.argmin(segment))
        saddle_layer = peak_indices[i] + saddle_offset
        saddles.append(saddle_layer)

    # Build regions: each extends from one saddle to the next
    region_starts = [0] + [s + 1 for s in saddles]
    region_ends = [s for s in saddles] + [n_layers - 1]

    regions = []
    for i, pk_idx in enumerate(peak_indices):
        pk_sep = float(seps[pk_idx])
        pk_coh = float(cohs[pk_idx])
        prominence = float(prominences[i])

        start = region_starts[i]
        end = region_ends[i]
        width = end - start + 1
        rise_span = max(pk_idx - start, 1)
        fall_span = max(end - pk_idx, 1)

        region_seps = seps[start:end + 1]
        region_cohs = cohs[start:end + 1]

        # ── CAZ score ──
        prominence_norm = prominence / global_mean_sep if global_mean_sep > 0 else 0.0
        coherence_boost = 1.0 + pk_coh / global_mean_coh
        width_factor = np.sqrt(width / n_layers) if n_layers > 0 else 0.0
        caz_score = prominence_norm * coherence_boost * width_factor

        regions.append(CAZRegion(
            start=start,
            peak=pk_idx,
            end=end,
            width=width,
            peak_separation=pk_sep,
            peak_coherence=pk_coh,
            mean_separation=float(region_seps.mean()),
            mean_coherence=float(region_cohs.mean()),
            prominence=prominence,
            depth_pct=100.0 * pk_idx / n_layers if n_layers > 0 else 0.0,
            width_pct=100.0 * width / n_layers if n_layers > 0 else 0.0,
            rise_span=rise_span,
            fall_span=fall_span,
            caz_score=round(caz_score, 6),
        ))

    return CAZProfile(
        n_layers=n_layers,
        regions=regions,
        n_regions=len(regions),
        global_peak_layer=global_peak,
        global_peak_separation=global_peak_sep,
        global_mean_separation=global_mean_sep,
        is_multimodal=len(regions) >= 2,
    )

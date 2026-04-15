"""
probes.py — Probe extraction from contrastive activations.

Given per-layer contrastive activations (from extract_contrastive_activations),
finds the optimal layer, extracts a difference-of-means direction, and calibrates
a detection threshold.

Three layer-selection methods:
    "raw"    — raw projected separation (centroid gap along DoM direction).
               Best for building working probes, especially on small/medium models.
    "fisher" — Fisher-normalized separation via compute_layer_metrics.
               Best for cross-model comparison and architecture analysis.
    "auroc"  — AUROC on held-out eval split.  Most rigorous for production probes.

Two threshold strategies:
    "midpoint"   — midpoint of mean positive/negative scores.  Simple, demo-friendly.
    "target_tpr" — threshold achieving a target true positive rate on eval data.

Usage
-----
    from rosetta_tools.extraction import extract_contrastive_activations
    from rosetta_tools.probes import extract_probe

    layer_acts = extract_contrastive_activations(
        model, tokenizer, pos_texts, neg_texts, device=device
    )[1:]  # skip embedding layer

    probe = extract_probe(layer_acts, method="raw")
    print(f"Layer {probe.layer}, threshold {probe.threshold}, AUROC {probe.auroc}")

    # Score new text
    from rosetta_tools.probes import score_direction
    new_acts = ...  # [n_texts, hidden_dim] at probe.layer
    scores = score_direction(new_acts, probe.direction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

PeakMethod = Literal["raw", "fisher", "auroc"]
ThresholdStrategy = Literal["midpoint", "target_tpr"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of probe extraction at a single concept's optimal layer."""

    concept: str = ""
    layer: int = 0
    direction: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    threshold: float = 0.5
    sep_curve: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    pos_mean: float = 0.5
    neg_mean: float = 0.5
    auroc: float = 0.0
    n_pos: int = 0
    n_neg: int = 0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def score_direction(
    acts: NDArray,
    direction: NDArray,
) -> NDArray[np.float64]:
    """Cosine similarity of activations against a direction vector, mapped to [0, 1].

    Parameters
    ----------
    acts:
        Activation matrix, shape ``[n_examples, hidden_dim]``.
    direction:
        Unit direction vector, shape ``[hidden_dim]``.

    Returns
    -------
    NDArray
        Scores in [0, 1].  1.0 = perfectly aligned with direction,
        0.0 = perfectly anti-aligned.
    """
    acts = np.asarray(acts, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    cosines = (acts / norms) @ direction
    return (cosines + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dom_direction(pos: NDArray, neg: NDArray) -> NDArray:
    """Difference-of-means direction, unit-normalized."""
    d = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return d
    return d / norm


def _raw_separation(pos: NDArray, neg: NDArray) -> float:
    """Raw projected separation: centroid gap along the DoM direction."""
    d = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return 0.0
    d = d / norm
    return float((pos @ d).mean() - (neg @ d).mean())


def _split_indices(
    n_pos: int,
    n_neg: int,
    eval_frac: float,
    seed: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Random train/eval index split for each class."""
    rng = np.random.RandomState(seed)
    n_pos_eval = max(1, int(n_pos * eval_frac))
    n_neg_eval = max(1, int(n_neg * eval_frac))

    pos_perm = rng.permutation(n_pos)
    neg_perm = rng.permutation(n_neg)

    return (
        pos_perm[n_pos_eval:],   # pos_train
        neg_perm[n_neg_eval:],   # neg_train
        pos_perm[:n_pos_eval],   # pos_eval
        neg_perm[:n_neg_eval],   # neg_eval
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_probe(
    layer_activations: list[tuple[NDArray, NDArray]],
    method: PeakMethod = "raw",
    threshold_strategy: ThresholdStrategy = "midpoint",
    target_tpr: float = 0.85,
    eval_frac: float = 0.0,
    seed: int = 42,
    concept: str = "",
) -> ProbeResult:
    """Extract a probe direction and threshold from contrastive layer activations.

    Parameters
    ----------
    layer_activations:
        List of ``(pos_acts, neg_acts)`` tuples, one per layer.  Each array
        has shape ``[n_texts, hidden_dim]``.  Typically the output of
        ``extract_contrastive_activations()[1:]`` (skip embedding).
    method:
        Layer selection strategy:

        ``"raw"`` — raw projected separation (centroid gap along the
        difference-of-means direction).  No variance correction.  Finds the
        layer with the strongest absolute signal.  Best for building probes
        that work on small/medium models.

        ``"fisher"`` — Fisher-normalized separation from
        ``compute_layer_metrics``.  Divides by within-class variance.  Best
        for cross-model comparison and architecture analysis.

        ``"auroc"`` — AUROC on a held-out eval split.  Most rigorous for
        production probes.  Requires ``eval_frac > 0``.
    threshold_strategy:
        ``"midpoint"`` — midpoint of mean positive/negative scores at the
        peak layer.  Simple, good for demos and tutorials.

        ``"target_tpr"`` — threshold achieving ``target_tpr`` on eval data
        (via ROC curve).  Use for production probes.
    target_tpr:
        Target true positive rate for ``"target_tpr"`` strategy.
    eval_frac:
        Fraction of examples to hold out for evaluation.  Required when
        ``method="auroc"``.  Optional but recommended for
        ``threshold_strategy="target_tpr"``.  When 0, scores are computed
        on training data (biased but functional for demos).
    seed:
        Random seed for reproducible train/eval splits.
    concept:
        Optional concept name stored in the result.

    Returns
    -------
    ProbeResult
        Probe with layer, direction, threshold, separation curve, and
        quality metrics.

    Raises
    ------
    ValueError
        If ``layer_activations`` is empty or ``method="auroc"`` with
        ``eval_frac <= 0``.
    """
    n_layers = len(layer_activations)
    if n_layers == 0:
        raise ValueError("layer_activations is empty")

    n_pos = len(layer_activations[0][0])
    n_neg = len(layer_activations[0][1])

    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"Need at least 2 examples per class, got {n_pos} pos / {n_neg} neg")

    if method == "auroc" and eval_frac <= 0:
        raise ValueError("method='auroc' requires eval_frac > 0")

    # ── Train / eval split ────────────────────────────────────────────────
    if eval_frac > 0:
        pos_train, neg_train, pos_eval, neg_eval = _split_indices(
            n_pos, n_neg, eval_frac, seed
        )
    else:
        pos_train = np.arange(n_pos)
        neg_train = np.arange(n_neg)
        pos_eval = pos_train
        neg_eval = neg_train

    # ── Layer selection ───────────────────────────────────────────────────
    if method == "raw":
        sep_curve = np.array([
            _raw_separation(pos[pos_train], neg[neg_train])
            for pos, neg in layer_activations
        ])
        best_layer = int(np.argmax(sep_curve))

    elif method == "fisher":
        from rosetta_tools.caz import compute_layer_metrics

        train_acts = [
            (pos[pos_train], neg[neg_train])
            for pos, neg in layer_activations
        ]
        metrics = compute_layer_metrics(train_acts)
        sep_curve = np.array([m.separation for m in metrics])
        best_layer = int(np.argmax(sep_curve))

    elif method == "auroc":
        from sklearn.metrics import roc_auc_score

        sep_curve = np.zeros(n_layers)
        for li, (pos, neg) in enumerate(layer_activations):
            d = _dom_direction(pos[pos_train], neg[neg_train])
            eval_acts = np.concatenate([pos[pos_eval], neg[neg_eval]])
            eval_labels = np.array([1] * len(pos_eval) + [0] * len(neg_eval))
            scores = score_direction(eval_acts, d)
            try:
                sep_curve[li] = roc_auc_score(eval_labels, scores)
            except ValueError:
                sep_curve[li] = 0.5
        best_layer = int(np.argmax(sep_curve))

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # ── Direction at peak layer ───────────────────────────────────────────
    pos_acts, neg_acts = layer_activations[best_layer]
    direction = _dom_direction(pos_acts[pos_train], neg_acts[neg_train])

    # ── Scores at peak layer (on eval data) ───────────────────────────────
    pos_scores = score_direction(pos_acts[pos_eval], direction)
    neg_scores = score_direction(neg_acts[neg_eval], direction)
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))

    # ── AUROC at peak layer ───────────────────────────────────────────────
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(all_labels, all_scores))
    except (ValueError, ImportError):
        auroc = 0.0

    # ── Threshold ─────────────────────────────────────────────────────────
    if threshold_strategy == "midpoint":
        threshold = (pos_mean + neg_mean) / 2.0

    elif threshold_strategy == "target_tpr":
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        threshold = float(thresholds[-1])
        for thr_val, tp in zip(thresholds, tpr):
            if tp >= target_tpr:
                threshold = float(thr_val)
                break
    else:
        raise ValueError(f"Unknown threshold_strategy: {threshold_strategy!r}")

    log.info(
        "  %s  L%d  |  threshold %.3f  (pos %.3f / neg %.3f)  AUROC %.3f",
        concept or "(unnamed)", best_layer, threshold, pos_mean, neg_mean, auroc,
    )

    return ProbeResult(
        concept=concept,
        layer=best_layer,
        direction=direction.astype(np.float32),
        threshold=round(threshold, 4),
        sep_curve=sep_curve,
        pos_mean=round(pos_mean, 4),
        neg_mean=round(neg_mean, 4),
        auroc=round(auroc, 4),
        n_pos=n_pos,
        n_neg=n_neg,
    )

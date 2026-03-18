"""
alignment.py — Cross-architecture vector alignment for PRH experiments.

Implements Orthogonal Procrustes alignment to compare semantic representations
across models with different latent space orientations.

The Platonic Representation Hypothesis predicts that architecturally diverse
models converge on equivalent semantic directions. Comparing raw vectors fails
because each model's residual stream has an arbitrary orientation — two models
can represent the same concept with vectors pointing in completely different
directions in their respective spaces. Procrustes finds the best rotation to
bring one space into alignment with another before comparing.

Typical usage
-------------
    import numpy as np
    from rosetta_tools.alignment import align_and_score, pairwise_alignment_df

    # acts_a, acts_b: calibration activations [n_texts, hidden_dim] from each model
    # vec_a, vec_b: dominant concept vectors [hidden_dim] from each model
    result = align_and_score(vec_a, vec_b, acts_a, acts_b)
    print(result["raw_cosine"], result["aligned_cosine"])

    # Multi-model pairwise comparison → DataFrame
    df = pairwise_alignment_df(vectors, activations)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes


# ---------------------------------------------------------------------------
# Core alignment math
# ---------------------------------------------------------------------------


def cosine_similarity(v1: NDArray, v2: NDArray) -> float:
    """Cosine similarity between two vectors."""
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def compute_procrustes_rotation(
    source_acts: NDArray,
    target_acts: NDArray,
) -> NDArray:
    """Find the orthogonal rotation R that maps target space → source space.

    Solves the Orthogonal Procrustes problem:
        minimize  ‖source_centered − target_centered @ R‖_F
        subject to  R^T R = I

    Parameters
    ----------
    source_acts:
        Calibration activations from the reference model — shape [n, d_src].
        Texts must be in the same order as target_acts.
    target_acts:
        Calibration activations from the model being aligned — shape [n, d_tgt].
        d_src need not equal d_tgt; if they differ, the rotation maps
        from d_tgt to d_src via a rectangular orthogonal matrix.

    Returns
    -------
    NDArray
        Rotation matrix R of shape [d_tgt, d_src] such that
        ``target_centered @ R ≈ source_centered``.
    """
    source = np.asarray(source_acts, dtype=np.float64)
    target = np.asarray(target_acts, dtype=np.float64)

    source -= source.mean(axis=0)
    target -= target.mean(axis=0)

    # orthogonal_procrustes(A, B) finds R minimising ||A @ R - B||
    # We want R s.t. target @ R ≈ source, so A=target, B=source
    R, _ = orthogonal_procrustes(target, source)
    return R


def apply_rotation(vector: NDArray, R: NDArray) -> NDArray:
    """Apply a Procrustes rotation to a concept vector.

    Parameters
    ----------
    vector:
        Concept direction from the target model — shape [d_tgt].
    R:
        Rotation matrix from ``compute_procrustes_rotation`` — shape [d_tgt, d_src].

    Returns
    -------
    NDArray
        Rotated vector in source model space — shape [d_src].
    """
    v = np.asarray(vector, dtype=np.float64).ravel()
    return v @ R


def align_and_score(
    source_vec: NDArray,
    target_vec: NDArray,
    source_acts: NDArray,
    target_acts: NDArray,
) -> dict:
    """Align target → source and report raw and aligned cosine similarity.

    Parameters
    ----------
    source_vec:
        Dominant concept direction from the source (reference) model.
    target_vec:
        Dominant concept direction from the target model.
    source_acts:
        Calibration activations from the source model [n_texts, hidden_dim].
    target_acts:
        Calibration activations from the target model [n_texts, hidden_dim].

    Returns
    -------
    dict
        ``raw_cosine``    — cosine similarity before alignment
        ``aligned_cosine`` — cosine similarity after Procrustes alignment
        ``alignment_gain`` — improvement (aligned − raw)
    """
    raw = cosine_similarity(source_vec, target_vec)

    R = compute_procrustes_rotation(source_acts, target_acts)
    aligned_target = apply_rotation(target_vec, R)
    aligned = cosine_similarity(source_vec, aligned_target)

    return {
        "raw_cosine": raw,
        "aligned_cosine": aligned,
        "alignment_gain": aligned - raw,
    }


# ---------------------------------------------------------------------------
# Multi-model pairwise comparison
# ---------------------------------------------------------------------------


def pairwise_alignment_df(
    vectors: dict[str, NDArray],
    activations: dict[str, NDArray],
) -> pd.DataFrame:
    """Compute pairwise Procrustes-aligned cosine similarities across models.

    Parameters
    ----------
    vectors:
        Mapping from model_id → dominant concept vector [hidden_dim].
    activations:
        Mapping from model_id → calibration activations [n_texts, hidden_dim].
        Same text order across all models.

    Returns
    -------
    pd.DataFrame
        Long-form with columns: source_model, target_model,
        raw_cosine, aligned_cosine, alignment_gain.
        Contains all ordered pairs (source ≠ target).
    """
    model_ids = list(vectors.keys())
    rows = []

    for src in model_ids:
        for tgt in model_ids:
            if src == tgt:
                continue
            result = align_and_score(
                vectors[src],
                vectors[tgt],
                activations[src],
                activations[tgt],
            )
            rows.append(
                {
                    "source_model": src,
                    "target_model": tgt,
                    **result,
                }
            )

    return pd.DataFrame(rows)

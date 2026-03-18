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
    n_components: int | None = None,
) -> NDArray:
    """Find the orthogonal rotation R that maps target space → source space.

    Solves the Orthogonal Procrustes problem:
        minimize  ‖source_proj − target_proj @ R‖_F
        subject to  R^T R = I

    When source and target have different hidden dimensions (common in
    cross-architecture comparison, e.g. GPT-2 at 768 vs Llama at 4096),
    both activation matrices are projected to a shared subspace via PCA
    before fitting the rotation. The number of components defaults to
    min(d_src, d_tgt, n_texts).

    Parameters
    ----------
    source_acts:
        Calibration activations from the reference model — shape [n, d_src].
        Texts must be in the same order as target_acts.
    target_acts:
        Calibration activations from the model being aligned — shape [n, d_tgt].
    n_components:
        Dimensionality of the shared PCA subspace. Only used when d_src ≠ d_tgt.
        Defaults to min(d_src, d_tgt, n_texts).

    Returns
    -------
    NDArray
        Rotation matrix R of shape [k, k] (where k = n_components or d if same-dim)
        such that ``target_proj @ R ≈ source_proj``.
    """
    from sklearn.decomposition import PCA

    source = np.asarray(source_acts, dtype=np.float64)
    target = np.asarray(target_acts, dtype=np.float64)

    source -= source.mean(axis=0)
    target -= target.mean(axis=0)

    if source.shape[1] != target.shape[1]:
        k = n_components or min(source.shape[1], target.shape[1], source.shape[0])
        source = PCA(n_components=k).fit_transform(source)
        target = PCA(n_components=k).fit_transform(target)

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
    n_components: int | None = None,
) -> dict:
    """Align target → source and report raw and aligned cosine similarity.

    When source and target have different hidden dimensions, both activation
    matrices and concept vectors are projected to a shared PCA subspace
    before Procrustes alignment. The raw cosine similarity is computed in
    the original spaces (or as close as possible — if dims differ, raw cosine
    is reported as NaN since the spaces are not directly comparable).

    Parameters
    ----------
    source_vec:
        Dominant concept direction from the source (reference) model.
    target_vec:
        Dominant concept direction from the target model.
    source_acts:
        Calibration activations from the source model [n_texts, d_src].
    target_acts:
        Calibration activations from the target model [n_texts, d_tgt].
    n_components:
        PCA components for cross-dim alignment. See compute_procrustes_rotation.

    Returns
    -------
    dict
        ``raw_cosine``     — cosine similarity before alignment (NaN if dims differ)
        ``aligned_cosine`` — cosine similarity after Procrustes alignment
        ``alignment_gain`` — improvement (aligned − raw; NaN if dims differ)
        ``same_dim``       — whether source and target share hidden dimension
    """
    from sklearn.decomposition import PCA

    src_v = np.asarray(source_vec, dtype=np.float64)
    tgt_v = np.asarray(target_vec, dtype=np.float64)
    src_a = np.asarray(source_acts, dtype=np.float64)
    tgt_a = np.asarray(target_acts, dtype=np.float64)

    same_dim = src_v.shape[0] == tgt_v.shape[0]
    raw = cosine_similarity(src_v, tgt_v) if same_dim else float("nan")

    # Project to shared subspace when dims differ
    if not same_dim:
        k = n_components or min(src_a.shape[1], tgt_a.shape[1], src_a.shape[0])
        src_pca = PCA(n_components=k).fit(src_a - src_a.mean(axis=0))
        tgt_pca = PCA(n_components=k).fit(tgt_a - tgt_a.mean(axis=0))
        src_a_proj = src_pca.transform(src_a - src_a.mean(axis=0))
        tgt_a_proj = tgt_pca.transform(tgt_a - tgt_a.mean(axis=0))
        src_v_proj = src_pca.transform(src_v.reshape(1, -1))[0]
        tgt_v_proj = tgt_pca.transform(tgt_v.reshape(1, -1))[0]
    else:
        src_a_proj, tgt_a_proj = src_a, tgt_a
        src_v_proj, tgt_v_proj = src_v, tgt_v

    R = compute_procrustes_rotation(src_a_proj, tgt_a_proj)
    aligned_tgt = apply_rotation(tgt_v_proj, R)
    aligned = cosine_similarity(src_v_proj, aligned_tgt)

    return {
        "raw_cosine": raw,
        "aligned_cosine": aligned,
        "alignment_gain": aligned - raw if same_dim else float("nan"),
        "same_dim": same_dim,
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

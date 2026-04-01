"""
test_alignment.py — Tests for rosetta_tools.alignment.

Tests Procrustes rotation, cosine similarity, align_and_score, and
pairwise alignment using synthetic data with known geometric properties.
No GPU, no model loading.
"""

import numpy as np
import pytest

from rosetta_tools.alignment import (
    align_and_score,
    apply_rotation,
    compute_procrustes_rotation,
    cosine_similarity,
    pairwise_alignment_df,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vector(self):
        v = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert cosine_similarity(v, z) == 0.0

    def test_different_magnitudes(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([100.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_procrustes_rotation
# ---------------------------------------------------------------------------


class TestProcrustesRotation:

    def test_identity_when_identical(self):
        """If source == target, rotation should be close to identity."""
        np.random.seed(42)
        acts = np.random.randn(50, 8)
        R = compute_procrustes_rotation(acts, acts)
        np.testing.assert_allclose(R, np.eye(8), atol=1e-10)

    def test_recovers_known_rotation(self):
        """Apply a known rotation, then recover it with Procrustes."""
        np.random.seed(42)
        source = np.random.randn(100, 4)

        # Create a random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(4, 4))
        target = source @ Q  # rotate source to get target

        R = compute_procrustes_rotation(source, target)
        # target @ R should recover source (up to centering)
        recovered = (target - target.mean(axis=0)) @ R
        original = source - source.mean(axis=0)
        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_orthogonality_of_result(self):
        """R should be an orthogonal matrix (R^T R = I)."""
        np.random.seed(42)
        source = np.random.randn(50, 6)
        target = np.random.randn(50, 6)
        R = compute_procrustes_rotation(source, target)
        np.testing.assert_allclose(R.T @ R, np.eye(6), atol=1e-10)

    def test_cross_dim_pca_projection(self):
        """Should handle different source and target dimensions via PCA."""
        np.random.seed(42)
        source = np.random.randn(50, 16)  # dim 16
        target = np.random.randn(50, 8)   # dim 8
        R = compute_procrustes_rotation(source, target)
        # R should be square: min(16, 8, 50) = 8
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] <= 8


# ---------------------------------------------------------------------------
# apply_rotation
# ---------------------------------------------------------------------------


class TestApplyRotation:

    def test_identity_rotation(self):
        v = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)
        rotated = apply_rotation(v, R)
        np.testing.assert_allclose(rotated, v)

    def test_90_degree_rotation(self):
        """90-degree rotation in 2D."""
        v = np.array([1.0, 0.0])
        R = np.array([[0.0, 1.0], [-1.0, 0.0]])  # 90° CCW
        rotated = apply_rotation(v, R)
        np.testing.assert_allclose(rotated, [0.0, 1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# align_and_score
# ---------------------------------------------------------------------------


class TestAlignAndScore:

    def test_identical_spaces(self):
        """Same model aligned with itself → aligned_cosine ≈ 1.0."""
        np.random.seed(42)
        acts = np.random.randn(50, 8)
        vec = np.random.randn(8)
        result = align_and_score(vec, vec, acts, acts)
        assert result["raw_cosine"] == pytest.approx(1.0)
        assert result["aligned_cosine"] == pytest.approx(1.0, abs=1e-6)
        assert result["same_dim"] is True

    def test_rotated_spaces_recoverable(self):
        """Known rotation should give aligned_cosine ≈ 1.0."""
        np.random.seed(42)
        source_acts = np.random.randn(100, 8)
        source_vec = np.random.randn(8)
        source_vec = source_vec / np.linalg.norm(source_vec)

        # Rotate everything by Q
        Q, _ = np.linalg.qr(np.random.randn(8, 8))
        target_acts = source_acts @ Q
        target_vec = source_vec @ Q

        result = align_and_score(source_vec, target_vec, source_acts, target_acts)
        assert result["aligned_cosine"] == pytest.approx(1.0, abs=1e-6)
        # Raw cosine should be low (random rotation)
        assert abs(result["raw_cosine"]) < 0.8  # not perfectly aligned by chance

    def test_cross_dim_returns_nan_raw(self):
        """Different hidden dims → raw_cosine is NaN, aligned_cosine is computed."""
        np.random.seed(42)
        src_acts = np.random.randn(50, 16)
        tgt_acts = np.random.randn(50, 8)
        src_vec = np.random.randn(16)
        tgt_vec = np.random.randn(8)

        result = align_and_score(src_vec, tgt_vec, src_acts, tgt_acts)
        assert np.isnan(result["raw_cosine"])
        assert result["same_dim"] is False
        # aligned_cosine should be a real number
        assert not np.isnan(result["aligned_cosine"])

    def test_alignment_gain(self):
        """alignment_gain should equal aligned - raw when same_dim."""
        np.random.seed(42)
        acts_a = np.random.randn(50, 8)
        acts_b = np.random.randn(50, 8)
        vec_a = np.random.randn(8)
        vec_b = np.random.randn(8)

        result = align_and_score(vec_a, vec_b, acts_a, acts_b)
        expected_gain = result["aligned_cosine"] - result["raw_cosine"]
        assert result["alignment_gain"] == pytest.approx(expected_gain)


# ---------------------------------------------------------------------------
# pairwise_alignment_df
# ---------------------------------------------------------------------------


class TestPairwiseAlignmentDf:

    def test_output_shape(self):
        np.random.seed(42)
        vectors = {
            "model_a": np.random.randn(8),
            "model_b": np.random.randn(8),
            "model_c": np.random.randn(8),
        }
        activations = {
            "model_a": np.random.randn(30, 8),
            "model_b": np.random.randn(30, 8),
            "model_c": np.random.randn(30, 8),
        }
        df = pairwise_alignment_df(vectors, activations)
        # 3 models → 3*2 = 6 ordered pairs
        assert len(df) == 6
        assert "source_model" in df.columns
        assert "target_model" in df.columns
        assert "aligned_cosine" in df.columns

    def test_no_self_pairs(self):
        np.random.seed(42)
        vectors = {"a": np.random.randn(4), "b": np.random.randn(4)}
        activations = {"a": np.random.randn(20, 4), "b": np.random.randn(20, 4)}
        df = pairwise_alignment_df(vectors, activations)
        assert not any(df["source_model"] == df["target_model"])

    def test_symmetric_raw_cosine(self):
        """Raw cosine should be symmetric: cos(a,b) == cos(b,a)."""
        np.random.seed(42)
        vectors = {"a": np.random.randn(4), "b": np.random.randn(4)}
        activations = {"a": np.random.randn(20, 4), "b": np.random.randn(20, 4)}
        df = pairwise_alignment_df(vectors, activations)
        ab = df[(df.source_model == "a") & (df.target_model == "b")].raw_cosine.values[0]
        ba = df[(df.source_model == "b") & (df.target_model == "a")].raw_cosine.values[0]
        assert ab == pytest.approx(ba)

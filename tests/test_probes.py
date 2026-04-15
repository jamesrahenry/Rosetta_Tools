"""
test_probes.py — Tests for rosetta_tools.probes probe extraction.

Synthetic data with known separation structure.  No model loading, no GPU.

Test philosophy:
- Perfect separation: two non-overlapping clouds → probe finds the right layer,
  direction points from neg to pos, AUROC ≈ 1.0
- Degenerate cases: identical distributions, single layer, too few examples
- Method equivalence: raw/fisher/auroc all find the same peak when one layer
  clearly dominates
- Threshold strategies: midpoint bisects class means, target_tpr achieves target
- score_direction: known geometry → known scores
"""

import numpy as np
import pytest

from rosetta_tools.probes import (
    ProbeResult,
    extract_probe,
    score_direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cloud(center, n=50, std=0.1, seed=0):
    """Gaussian cloud around center, shape [n, len(center)]."""
    rng = np.random.default_rng(seed)
    return center + rng.standard_normal((n, len(center))) * std


def make_layer_acts(n_layers, hidden_dim, peak_layer, n=50, sep=5.0, noise=0.1):
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
            # Overlapping — no separation
            cloud = rng.standard_normal((2 * n, hidden_dim)) * noise
            pos = cloud[:n]
            neg = cloud[n:]
        layer_acts.append((pos.astype(np.float32), neg.astype(np.float32)))
    return layer_acts


# ---------------------------------------------------------------------------
# score_direction
# ---------------------------------------------------------------------------


class TestScoreDirection:
    def test_aligned_vectors_score_one(self):
        direction = np.array([1.0, 0.0, 0.0])
        acts = np.array([[3.0, 0.0, 0.0]])
        scores = score_direction(acts, direction)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_anti_aligned_vectors_score_zero(self):
        direction = np.array([1.0, 0.0, 0.0])
        acts = np.array([[-3.0, 0.0, 0.0]])
        scores = score_direction(acts, direction)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_score_half(self):
        direction = np.array([1.0, 0.0, 0.0])
        acts = np.array([[0.0, 5.0, 0.0]])
        scores = score_direction(acts, direction)
        assert scores[0] == pytest.approx(0.5, abs=1e-6)

    def test_batch_output_shape(self):
        direction = np.array([1.0, 0.0])
        acts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        scores = score_direction(acts, direction)
        assert scores.shape == (3,)

    def test_zero_vector_does_not_crash(self):
        direction = np.array([1.0, 0.0])
        acts = np.array([[0.0, 0.0]])
        scores = score_direction(acts, direction)
        assert 0.0 <= scores[0] <= 1.0


# ---------------------------------------------------------------------------
# extract_probe — raw method
# ---------------------------------------------------------------------------


class TestExtractProbeRaw:
    def test_finds_correct_peak_layer(self):
        acts = make_layer_acts(n_layers=10, hidden_dim=32, peak_layer=6)
        probe = extract_probe(acts, method="raw")
        assert probe.layer == 6

    def test_direction_points_from_neg_to_pos(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=3, sep=10.0)
        probe = extract_probe(acts, method="raw")
        # Direction should have a strong positive component along dim 0
        assert probe.direction[0] > 0.5

    def test_pos_mean_greater_than_neg_mean(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=2, sep=5.0)
        probe = extract_probe(acts, method="raw")
        assert probe.pos_mean > probe.neg_mean

    def test_threshold_between_class_means(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=2)
        probe = extract_probe(acts, method="raw")
        assert probe.neg_mean < probe.threshold < probe.pos_mean

    def test_auroc_near_one_for_separable_data(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=32, peak_layer=3, sep=10.0)
        probe = extract_probe(acts, method="raw")
        assert probe.auroc > 0.95

    def test_sep_curve_peaks_at_correct_layer(self):
        acts = make_layer_acts(n_layers=8, hidden_dim=16, peak_layer=5)
        probe = extract_probe(acts, method="raw")
        assert int(np.argmax(probe.sep_curve)) == 5

    def test_concept_name_stored(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=8, peak_layer=1)
        probe = extract_probe(acts, method="raw", concept="authorization")
        assert probe.concept == "authorization"

    def test_counts_stored(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=8, peak_layer=1, n=30)
        probe = extract_probe(acts, method="raw")
        assert probe.n_pos == 30
        assert probe.n_neg == 30


# ---------------------------------------------------------------------------
# extract_probe — fisher method
# ---------------------------------------------------------------------------


class TestExtractProbeFisher:
    def test_finds_correct_peak_layer(self):
        acts = make_layer_acts(n_layers=10, hidden_dim=32, peak_layer=6, sep=10.0)
        probe = extract_probe(acts, method="fisher")
        assert probe.layer == 6

    def test_auroc_high_for_separable_data(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=32, peak_layer=2, sep=10.0)
        probe = extract_probe(acts, method="fisher")
        assert probe.auroc > 0.9


# ---------------------------------------------------------------------------
# extract_probe — auroc method
# ---------------------------------------------------------------------------


class TestExtractProbeAuroc:
    def test_finds_correct_peak_layer(self):
        acts = make_layer_acts(n_layers=10, hidden_dim=32, peak_layer=6, n=80, sep=10.0)
        probe = extract_probe(acts, method="auroc", eval_frac=0.25)
        assert probe.layer == 6

    def test_requires_eval_frac(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=8, peak_layer=1)
        with pytest.raises(ValueError, match="eval_frac"):
            extract_probe(acts, method="auroc", eval_frac=0.0)

    def test_sep_curve_contains_auroc_values(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=3, n=60, sep=8.0)
        probe = extract_probe(acts, method="auroc", eval_frac=0.2)
        # AUROC values should be between 0 and 1
        assert np.all(probe.sep_curve >= 0.0)
        assert np.all(probe.sep_curve <= 1.0)
        # Peak layer should have high AUROC
        assert probe.sep_curve[3] > 0.8


# ---------------------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------------------


class TestThresholdStrategies:
    def test_midpoint_bisects_class_means(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=16, peak_layer=1, sep=5.0)
        probe = extract_probe(acts, method="raw", threshold_strategy="midpoint")
        expected = (probe.pos_mean + probe.neg_mean) / 2
        assert probe.threshold == pytest.approx(expected, abs=0.01)

    def test_target_tpr_achieves_target(self):
        """target_tpr threshold should classify at least target_tpr of positives correctly."""
        acts = make_layer_acts(n_layers=5, hidden_dim=32, peak_layer=2, n=80, sep=3.0)
        probe = extract_probe(
            acts, method="raw", threshold_strategy="target_tpr",
            target_tpr=0.85, eval_frac=0.2,
        )
        # Score all positives at the peak layer against the probe direction
        pos_acts = acts[probe.layer][0]
        pos_scores = score_direction(pos_acts, probe.direction)
        actual_tpr = float(np.mean(pos_scores >= probe.threshold))
        assert actual_tpr >= 0.80  # allow small margin from eval/full split difference


# ---------------------------------------------------------------------------
# Eval split
# ---------------------------------------------------------------------------


class TestEvalSplit:
    def test_eval_frac_reduces_training_data(self):
        """With eval_frac, the probe still works but uses fewer training examples."""
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=3, n=60, sep=8.0)
        probe = extract_probe(acts, method="raw", eval_frac=0.2)
        assert probe.layer == 3
        assert probe.auroc > 0.9

    def test_reproducible_with_same_seed(self):
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=2, n=40, sep=5.0)
        p1 = extract_probe(acts, method="raw", eval_frac=0.2, seed=99)
        p2 = extract_probe(acts, method="raw", eval_frac=0.2, seed=99)
        assert p1.layer == p2.layer
        assert p1.threshold == p2.threshold

    def test_different_seeds_may_differ(self):
        """Smoke test — different seeds produce different splits."""
        acts = make_layer_acts(n_layers=5, hidden_dim=16, peak_layer=2, n=40, sep=2.0)
        p1 = extract_probe(acts, method="raw", eval_frac=0.3, seed=1)
        p2 = extract_probe(acts, method="raw", eval_frac=0.3, seed=2)
        # Thresholds may differ slightly due to different eval sets
        # (just check it doesn't crash — both should still find layer 2)
        assert p1.layer == 2
        assert p2.layer == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_activations_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract_probe([])

    def test_too_few_examples_raises(self):
        pos = np.array([[1.0, 0.0]])  # 1 example
        neg = np.array([[0.0, 1.0]])  # 1 example
        with pytest.raises(ValueError, match="at least 2"):
            extract_probe([(pos, neg)])

    def test_single_layer(self):
        """Single layer should work — it's the only choice."""
        pos = make_cloud(np.array([5.0, 0.0]), n=20, std=0.1, seed=1)
        neg = make_cloud(np.array([0.0, 0.0]), n=20, std=0.1, seed=2)
        probe = extract_probe([(pos, neg)], method="raw")
        assert probe.layer == 0
        assert probe.direction[0] > 0.5

    def test_unknown_method_raises(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=8, peak_layer=1)
        with pytest.raises(ValueError, match="Unknown method"):
            extract_probe(acts, method="nonsense")

    def test_unknown_threshold_raises(self):
        acts = make_layer_acts(n_layers=3, hidden_dim=8, peak_layer=1)
        with pytest.raises(ValueError, match="Unknown threshold_strategy"):
            extract_probe(acts, threshold_strategy="nonsense")

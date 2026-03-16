"""
test_caz.py — Tests for rosetta_tools.caz metric computation.

All tests use synthetic data with known structure so that assertions are
mathematically grounded, not just "it ran without crashing."  No model
loading, no GPU required.  These should pass in any CI environment.

Test philosophy:
- Perfect separation: two non-overlapping clouds → known S, C=1.0
- Zero separation: identical distributions → S≈0
- Formula fidelity: verify compute_separation matches the CAZ paper formula
  exactly (§3.2: S(l) = ‖μ_A − μ_B‖₂ / √[(1/2)(tr(Σ_A) + tr(Σ_B))])
- Velocity: direction and magnitude are correct for monotone and peaked S curves
- Boundary detection: peak found correctly, width is sane
- Full pipeline: compute_layer_metrics → find_caz_boundary round-trip
"""

import numpy as np
import pytest

from rosetta_tools.caz import (
    CAZBoundary,
    LayerMetrics,
    compute_coherence,
    compute_layer_metrics,
    compute_separation,
    compute_velocity,
    find_caz_boundary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cloud(
    center: np.ndarray,
    n: int = 50,
    std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian cloud around center, shape [n, len(center)]."""
    rng = np.random.default_rng(seed)
    return center + rng.standard_normal((n, len(center))) * std


# ---------------------------------------------------------------------------
# compute_separation
# ---------------------------------------------------------------------------


class TestComputeSeparation:
    def test_perfect_separation_is_large(self):
        """Non-overlapping clouds with small within-class variance → large S."""
        d = 64
        pos = make_cloud(np.ones(d) * 5.0, n=50, std=0.1, seed=1)
        neg = make_cloud(np.zeros(d), n=50, std=0.1, seed=2)
        S = compute_separation(pos, neg)
        assert S > 10.0, f"Expected S >> 10 for non-overlapping clouds, got {S:.4f}"

    def test_zero_separation_identical_distributions(self):
        """Identical distributions → S ≈ 0."""
        d = 32
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((100, d))
        # Split the same cloud into two halves — centroids near-identical
        S = compute_separation(cloud[:50], cloud[50:])
        assert S < 0.5, f"Expected S ≈ 0 for same distribution, got {S:.4f}"

    def test_matches_paper_formula_exactly(self):
        """
        Verify compute_separation implements the CAZ paper formula:

            S = ‖μ_A − μ_B‖₂ / √[ (1/2)(tr(Σ_A) + tr(Σ_B)) ]

        where tr(Σ) = sum of per-dimension variances (unbiased, ddof=1).
        """
        rng = np.random.default_rng(7)
        d = 16
        pos = rng.standard_normal((30, d)) + 2.0
        neg = rng.standard_normal((30, d))

        pos64 = pos.astype(np.float64)
        neg64 = neg.astype(np.float64)

        mu_pos = pos64.mean(axis=0)
        mu_neg = neg64.mean(axis=0)
        centroid_dist = np.linalg.norm(mu_pos - mu_neg)
        trace_pos = pos64.var(axis=0, ddof=1).sum()
        trace_neg = neg64.var(axis=0, ddof=1).sum()
        expected = centroid_dist / (np.sqrt(0.5 * (trace_pos + trace_neg)) + 1e-8)

        actual = compute_separation(pos, neg)

        assert abs(actual - expected) < 1e-10, (
            f"compute_separation diverges from paper formula: "
            f"expected {expected:.8f}, got {actual:.8f}"
        )

    def test_separation_increases_with_distance(self):
        """S should be monotonically larger as clouds are pushed further apart."""
        d = 32
        separations = []
        for gap in [0.5, 1.0, 2.0, 4.0, 8.0]:
            pos = make_cloud(np.ones(d) * gap, n=40, std=0.3, seed=10)
            neg = make_cloud(np.zeros(d), n=40, std=0.3, seed=11)
            separations.append(compute_separation(pos, neg))
        assert separations == sorted(separations), (
            f"S not monotone with distance: {separations}"
        )

    def test_returns_zero_for_single_sample(self):
        """Fewer than 2 samples per class → 0.0 (can't compute variance)."""
        pos = np.array([[1.0, 2.0]])
        neg = np.array([[0.0, 0.0], [0.1, 0.1]])
        assert compute_separation(pos, neg) == 0.0

    def test_float64_input_unchanged(self):
        """float64 input should work identically — no double-cast penalty."""
        rng = np.random.default_rng(99)
        pos = rng.standard_normal((20, 8)).astype(np.float64) + 1.0
        neg = rng.standard_normal((20, 8)).astype(np.float64)
        S = compute_separation(pos, neg)
        assert S > 0.0


# ---------------------------------------------------------------------------
# compute_coherence
# ---------------------------------------------------------------------------


class TestComputeCoherence:
    def test_high_coherence_for_1d_signal(self):
        """If all variance is in one dimension, coherence should be near 1."""
        rng = np.random.default_rng(3)
        n, d = 60, 32
        # All signal in dimension 0, noise elsewhere
        pos = rng.standard_normal((n, d)) * 0.01
        neg = rng.standard_normal((n, d)) * 0.01
        pos[:, 0] += 5.0
        neg[:, 0] -= 5.0

        C = compute_coherence(pos, neg)
        assert C > 0.8, f"Expected C > 0.8 for 1D signal, got {C:.4f}"

    def test_low_coherence_for_isotropic_noise(self):
        """Isotropic noise has no dominant direction → low coherence."""
        rng = np.random.default_rng(4)
        pos = rng.standard_normal((50, 64))
        neg = rng.standard_normal((50, 64))
        C = compute_coherence(pos, neg)
        # With 64 dims and isotropic noise, first PC explains ~1/64 ≈ 1.5%
        assert C < 0.1, f"Expected C < 0.1 for isotropic noise, got {C:.4f}"

    def test_coherence_in_unit_interval(self):
        """Coherence must always be in [0, 1]."""
        rng = np.random.default_rng(5)
        for _ in range(10):
            pos = rng.standard_normal((30, 16))
            neg = rng.standard_normal((30, 16))
            C = compute_coherence(pos, neg)
            assert 0.0 <= C <= 1.0, f"Coherence {C} outside [0, 1]"

    def test_returns_zero_for_empty_input(self):
        # The guard triggers when total pooled samples < 2
        # 1 sample per class = 2 total — PCA finds a perfect direction → 1.0
        # 0 samples per class = 0 total → 0.0
        assert compute_coherence(np.empty((0, 4)), np.empty((0, 4))) == 0.0


# ---------------------------------------------------------------------------
# compute_velocity
# ---------------------------------------------------------------------------


class TestComputeVelocity:
    def test_monotone_increasing_is_positive(self):
        """Monotone-increasing S → all velocities ≥ 0 (after index 0)."""
        seps = [0.0, 1.0, 2.0, 3.0, 4.0]
        v = compute_velocity(seps, window=1)
        assert all(x >= 0 for x in v[1:]), f"Expected non-negative velocity: {v}"

    def test_peak_then_decline_has_sign_change(self):
        """S rising then falling → velocity sign changes at peak."""
        seps = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
        v = compute_velocity(seps, window=1)
        # Should have positive velocities before peak, negative after
        assert v[3] >= 0, "Velocity at peak should be non-negative"
        assert v[-1] < 0, "Velocity at end of decline should be negative"

    def test_constant_separation_is_zero_velocity(self):
        """Flat S curve → all velocities ≈ 0."""
        seps = [2.5] * 10
        v = compute_velocity(seps, window=1)
        assert np.allclose(v, 0.0), f"Expected zero velocity for flat S: {v}"

    def test_length_preserved(self):
        """Output length must equal input length."""
        for n in [1, 5, 48]:
            seps = list(range(n))
            v = compute_velocity(seps, window=3)
            assert len(v) == n, f"Expected {n} velocities, got {len(v)}"

    def test_first_element_always_zero(self):
        """No previous layer to diff against → v[0] = 0."""
        v = compute_velocity([1.0, 2.0, 3.0], window=1)
        assert v[0] == 0.0


# ---------------------------------------------------------------------------
# compute_layer_metrics
# ---------------------------------------------------------------------------


class TestComputeLayerMetrics:
    def _make_layer_activations(
        self, n_layers: int = 12, d: int = 32, n: int = 40
    ) -> list:
        """Synthetic activations where separation grows with layer depth."""
        rng = np.random.default_rng(42)
        result = []
        for layer in range(n_layers):
            gap = layer * 0.2  # linearly increasing separation
            pos = make_cloud(np.ones(d) * gap, n=n, std=0.5, seed=layer)
            neg = make_cloud(np.zeros(d), n=n, std=0.5, seed=layer + 100)
            result.append((pos, neg))
        return result

    def test_returns_correct_length(self):
        acts = self._make_layer_activations(n_layers=8)
        metrics = compute_layer_metrics(acts)
        assert len(metrics) == 8

    def test_layer_indices_are_correct(self):
        acts = self._make_layer_activations(n_layers=6)
        metrics = compute_layer_metrics(acts)
        for i, m in enumerate(metrics):
            assert m.layer == i, f"Layer index mismatch: expected {i}, got {m.layer}"

    def test_separation_grows_with_signal(self):
        """S values should increase as the synthetic gap grows with layer."""
        acts = self._make_layer_activations(n_layers=12)
        metrics = compute_layer_metrics(acts)
        seps = [m.separation for m in metrics]
        # Allow some noise but the trend should be upward
        assert seps[-1] > seps[0], "Separation should grow with linearly increasing gap"

    def test_all_metrics_non_negative(self):
        acts = self._make_layer_activations(n_layers=10)
        metrics = compute_layer_metrics(acts)
        for m in metrics:
            assert m.separation >= 0.0
            assert m.coherence >= 0.0

    def test_named_tuple_fields(self):
        acts = self._make_layer_activations(n_layers=4)
        metrics = compute_layer_metrics(acts)
        m = metrics[0]
        assert hasattr(m, "layer")
        assert hasattr(m, "separation")
        assert hasattr(m, "coherence")
        assert hasattr(m, "velocity")


# ---------------------------------------------------------------------------
# find_caz_boundary
# ---------------------------------------------------------------------------


class TestFindCAZBoundary:
    def _peaked_metrics(
        self, peak_at: int = 8, n_layers: int = 16
    ) -> list[LayerMetrics]:
        """Synthetic metrics with a clear separation peak at a known layer."""
        # S rises to peak, then falls; velocities derived
        seps = [float(max(0, 5 - abs(i - peak_at))) for i in range(n_layers)]
        vels_arr = compute_velocity(seps, window=1)
        return [
            LayerMetrics(
                layer=i,
                separation=seps[i],
                coherence=0.5,
                velocity=float(vels_arr[i]),
            )
            for i in range(n_layers)
        ]

    def test_peak_layer_identified_correctly(self):
        metrics = self._peaked_metrics(peak_at=8, n_layers=16)
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_peak == 8, (
            f"Expected peak at layer 8, got {boundary.caz_peak}"
        )

    def test_peak_separation_value_correct(self):
        metrics = self._peaked_metrics(peak_at=8, n_layers=16)
        boundary = find_caz_boundary(metrics)
        assert boundary.peak_separation == 5.0

    def test_width_is_positive(self):
        metrics = self._peaked_metrics(peak_at=10, n_layers=20)
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_width > 0

    def test_start_le_peak_le_end(self):
        metrics = self._peaked_metrics(peak_at=6, n_layers=12)
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_start <= boundary.caz_peak <= boundary.caz_end, (
            f"Boundary ordering violated: {boundary}"
        )

    def test_returns_caz_boundary_type(self):
        metrics = self._peaked_metrics()
        boundary = find_caz_boundary(metrics)
        assert isinstance(boundary, CAZBoundary)

    def test_raises_on_empty_metrics(self):
        with pytest.raises(ValueError, match="empty"):
            find_caz_boundary([])

    def test_monotone_increasing_returns_full_width(self):
        """Monotone S with no decline → caz_end at last layer."""
        seps = list(range(10))
        vels_arr = compute_velocity(seps, window=1)
        metrics = [
            LayerMetrics(
                layer=i,
                separation=float(seps[i]),
                coherence=0.5,
                velocity=float(vels_arr[i]),
            )
            for i in range(10)
        ]
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_end == 9, (
            f"Expected end at last layer for monotone S, got {boundary.caz_end}"
        )


# ---------------------------------------------------------------------------
# Full pipeline round-trip
# ---------------------------------------------------------------------------


class TestFullPipelineRoundTrip:
    """End-to-end: synthetic activations → metrics → boundary."""

    def test_round_trip_produces_sensible_boundary(self):
        rng = np.random.default_rng(77)
        n_layers, d, n = 24, 64, 60

        layer_acts = []
        for layer in range(n_layers):
            # Gap peaks at layer 16 (2/3 depth — roughly where CAZ theory predicts)
            gap = float(max(0, 6 - abs(layer - 16)))
            pos = rng.standard_normal((n, d)) * 0.5 + gap
            neg = rng.standard_normal((n, d)) * 0.5
            layer_acts.append((pos, neg))

        metrics = compute_layer_metrics(layer_acts)
        boundary = find_caz_boundary(metrics)

        assert isinstance(boundary, CAZBoundary)
        assert boundary.caz_start <= boundary.caz_peak <= boundary.caz_end
        assert boundary.peak_separation > 0
        # Peak should be near layer 16 (allow ±3 given smoothing)
        assert abs(boundary.caz_peak - 16) <= 3, (
            f"Peak at {boundary.caz_peak}, expected near 16"
        )

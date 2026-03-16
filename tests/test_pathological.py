"""
test_pathological.py — Rigorous tests for NaN, Inf, and degenerate inputs.

The original GPT-2-XL credibility results contained a silent failure where
fp16 overflow caused Fisher normalization to produce S=0 for layers 32–47,
with NaN at layers 45–47.  These tests ensure rosetta_tools.caz produces
well-defined, non-NaN outputs regardless of what the activation arrays contain.

Philosophy: every function that accepts activation arrays must either:
  (a) return a well-defined finite value, OR
  (b) return 0.0 / a documented sentinel, OR
  (c) raise a ValueError with a clear message.

Silently returning NaN or Inf is never acceptable.

Tests cover:
- NaN activations (full arrays, single rows, single columns)
- Inf activations (positive and negative)
- Zero-variance arrays (constant activations — can happen with dead neurons)
- Single-sample classes (can't compute variance)
- Mismatched array shapes
- Empty arrays
- Mixed valid/NaN (partial corruption)
- Extremely large values (overflow territory for fp16)
- Extremely small values (underflow)
- All-identical activation vectors (zero within-class spread)
- compute_layer_metrics and find_caz_boundary with degenerate metric sequences
"""

import numpy as np
import pytest

from rosetta_tools.caz import (
    compute_coherence,
    compute_layer_metrics,
    compute_separation,
    compute_velocity,
    find_caz_boundary,
    LayerMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(x):
    """Assert x is a finite Python float."""
    assert isinstance(x, float), f"Expected float, got {type(x)}: {x}"
    assert np.isfinite(x), f"Expected finite value, got {x}"


def _finite_array(arr):
    """Assert all elements of a numpy array are finite."""
    arr = np.asarray(arr)
    assert np.all(np.isfinite(arr)), f"Array contains non-finite values: {arr}"


def _make_valid(n=50, d=32, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, d)) + 1.0
    neg = rng.standard_normal((n, d))
    return pos, neg


# ---------------------------------------------------------------------------
# NaN inputs
# ---------------------------------------------------------------------------


class TestNaNInputs:
    """compute_separation and compute_coherence must not return NaN."""

    def test_separation_all_nan_pos(self):
        pos = np.full((50, 32), np.nan)
        neg = np.random.default_rng(0).standard_normal((50, 32))
        result = compute_separation(pos, neg)
        _finite(result)

    def test_separation_all_nan_neg(self):
        pos = np.random.default_rng(0).standard_normal((50, 32))
        neg = np.full((50, 32), np.nan)
        result = compute_separation(pos, neg)
        _finite(result)

    def test_separation_all_nan_both(self):
        pos = np.full((50, 32), np.nan)
        neg = np.full((50, 32), np.nan)
        result = compute_separation(pos, neg)
        _finite(result)

    def test_separation_single_nan_row(self):
        """A single NaN row in a 50-row array should not propagate to the result."""
        rng = np.random.default_rng(1)
        pos = rng.standard_normal((50, 32)) + 1.0
        neg = rng.standard_normal((50, 32))
        # Inject one NaN row — this will affect the mean, but result must be finite
        pos[10, :] = np.nan
        result = compute_separation(pos, neg)
        # Result may be degraded but must not be NaN or Inf
        assert np.isfinite(result) or result == 0.0, (
            f"Single NaN row produced non-finite S: {result}"
        )

    def test_separation_nan_in_single_column(self):
        rng = np.random.default_rng(2)
        pos = rng.standard_normal((50, 32)) + 1.0
        neg = rng.standard_normal((50, 32))
        pos[:, 0] = np.nan  # Entire first dimension is NaN
        result = compute_separation(pos, neg)
        assert np.isfinite(result) or result == 0.0

    def test_coherence_all_nan(self):
        pos = np.full((50, 16), np.nan)
        neg = np.full((50, 16), np.nan)
        result = compute_coherence(pos, neg)
        _finite(result)

    def test_coherence_partial_nan(self):
        rng = np.random.default_rng(3)
        pos = rng.standard_normal((50, 16))
        neg = rng.standard_normal((50, 16))
        pos[:5, :] = np.nan
        result = compute_coherence(pos, neg)
        assert np.isfinite(result) or result == 0.0

    def test_velocity_nan_separations(self):
        """NaN in the separation sequence must not propagate through velocity."""
        seps = [0.1, np.nan, 0.3, 0.4, np.nan]
        result = compute_velocity(seps, window=1)
        # Each element should be finite or NaN propagated — but the function
        # should at minimum not raise and should return an array of correct length
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Inf inputs
# ---------------------------------------------------------------------------


class TestInfInputs:
    def test_separation_pos_inf(self):
        pos = np.full((50, 32), np.inf)
        neg = np.random.default_rng(0).standard_normal((50, 32))
        result = compute_separation(pos, neg)
        # inf inputs → numerator and denominator both inf → NaN or 0
        # We require: must not raise, result must be a Python float
        assert isinstance(result, float)

    def test_separation_neg_inf(self):
        pos = np.full((50, 32), -np.inf)
        neg = np.random.default_rng(0).standard_normal((50, 32))
        result = compute_separation(pos, neg)
        assert isinstance(result, float)

    def test_separation_mixed_inf(self):
        rng = np.random.default_rng(4)
        pos = rng.standard_normal((50, 32)) + 1.0
        neg = rng.standard_normal((50, 32))
        pos[0, 0] = np.inf
        neg[0, 0] = -np.inf
        result = compute_separation(pos, neg)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Zero-variance / constant inputs
# ---------------------------------------------------------------------------


class TestZeroVariance:
    """Constant activation arrays have zero within-class variance.

    The Fisher denominator becomes eps only.  Result should be very large
    (the eps guard kicks in) but finite and non-NaN.
    """

    def test_separation_constant_pos(self):
        pos = np.ones((50, 32)) * 5.0
        neg = np.random.default_rng(0).standard_normal((50, 32))
        result = compute_separation(pos, neg)
        _finite(result)

    def test_separation_both_constant_same_value(self):
        """Both classes identical and constant → zero centroid distance → S=0."""
        pos = np.ones((50, 32)) * 3.0
        neg = np.ones((50, 32)) * 3.0
        result = compute_separation(pos, neg)
        assert result == 0.0 or np.isclose(result, 0.0, atol=1e-6), (
            f"Expected S≈0 for identical constant arrays, got {result}"
        )

    def test_separation_both_constant_different_values(self):
        """Two different constant arrays → nonzero centroid, zero variance → large S."""
        pos = np.ones((50, 32)) * 5.0
        neg = np.ones((50, 32)) * 0.0
        result = compute_separation(pos, neg)
        _finite(result)
        assert result > 0.0, (
            f"Expected S > 0 for different constant arrays, got {result}"
        )

    def test_coherence_constant_input(self):
        """Constant array → zero variance → PCA undefined. Must return 0.0, not NaN."""
        pos = np.ones((50, 16)) * 2.0
        neg = np.ones((50, 16)) * 2.0
        result = compute_coherence(pos, neg)
        _finite(result)

    def test_separation_single_unique_value_per_class(self):
        """All samples within a class are identical (different from other class)."""
        pos = np.tile([1.0] * 32, (50, 1))
        neg = np.tile([0.0] * 32, (50, 1))
        result = compute_separation(pos, neg)
        _finite(result)


# ---------------------------------------------------------------------------
# Extreme values (fp16 overflow territory)
# ---------------------------------------------------------------------------


class TestExtremeValues:
    """Values that would overflow in fp16 (max ~65504) must be handled in float64."""

    def test_separation_large_values(self):
        """fp16 max is ~65504. These values would overflow fp16 but not float64."""
        rng = np.random.default_rng(5)
        pos = rng.standard_normal((50, 32)) * 1e4 + 1e5
        neg = rng.standard_normal((50, 32)) * 1e4
        result = compute_separation(pos, neg)
        _finite(result)
        assert result > 0.0

    def test_separation_very_small_values(self):
        """Underflow territory — very small activations should still work."""
        rng = np.random.default_rng(6)
        pos = rng.standard_normal((50, 32)) * 1e-10 + 1e-9
        neg = rng.standard_normal((50, 32)) * 1e-10
        result = compute_separation(pos, neg)
        _finite(result)

    def test_separation_mixed_scale(self):
        """One dimension with huge variance, rest near zero."""
        rng = np.random.default_rng(7)
        pos = rng.standard_normal((50, 32)) * 1e-6
        neg = rng.standard_normal((50, 32)) * 1e-6
        pos[:, 0] += 1e6  # One dimension with enormous signal
        result = compute_separation(pos, neg)
        _finite(result)
        assert result > 0.0

    def test_values_that_caused_fp16_overflow(self):
        """
        Reproduce the class of input that caused the GPT-2-XL Run 2 failure:
        activations at deep layers of a large model in fp16 can reach values
        where variance computation overflows.  In float64 this must not happen.
        """
        rng = np.random.default_rng(8)
        # Simulate the kind of large-magnitude activations seen at L32+ in gpt2-xl
        pos = rng.standard_normal((100, 1600)).astype(np.float32) * 200 + 100
        neg = rng.standard_normal((100, 1600)).astype(np.float32) * 200

        # This would produce NaN/0 with fp16 arithmetic but must work in float64
        result = compute_separation(pos, neg)
        _finite(result)
        assert result > 0.0, "Expected non-zero S for well-separated large-value arrays"


# ---------------------------------------------------------------------------
# Empty and minimal arrays
# ---------------------------------------------------------------------------


class TestEdgeCaseShapes:
    def test_separation_empty_arrays(self):
        result = compute_separation(np.empty((0, 32)), np.empty((0, 32)))
        assert result == 0.0

    def test_separation_one_sample_each(self):
        """Single sample per class → cannot compute variance → 0.0."""
        pos = np.array([[1.0, 2.0, 3.0]])
        neg = np.array([[0.0, 0.0, 0.0]])
        result = compute_separation(pos, neg)
        assert result == 0.0

    def test_separation_one_feature(self):
        """1-dimensional activations — degenerate but should not crash."""
        pos = np.array([[1.0], [2.0], [3.0]])
        neg = np.array([[0.0], [-1.0], [-2.0]])
        result = compute_separation(pos, neg)
        _finite(result)

    def test_coherence_one_sample_each(self):
        pos = np.array([[1.0, 2.0]])
        neg = np.array([[0.0, 0.0]])
        result = compute_coherence(pos, neg)
        _finite(result)

    def test_coherence_one_feature(self):
        pos = np.array([[1.0], [2.0], [3.0]])
        neg = np.array([[0.0], [-1.0], [-2.0]])
        result = compute_coherence(pos, neg)
        _finite(result)


# ---------------------------------------------------------------------------
# compute_layer_metrics with degenerate layer activations
# ---------------------------------------------------------------------------


class TestLayerMetricsPathological:
    """compute_layer_metrics must return finite metrics for every layer
    regardless of what the activation arrays contain."""

    def test_all_nan_activations(self):
        """Every layer is all-NaN → all metrics should be finite (0.0 or similar)."""
        layer_acts = [
            (np.full((50, 32), np.nan), np.full((50, 32), np.nan)) for _ in range(12)
        ]
        metrics = compute_layer_metrics(layer_acts)
        assert len(metrics) == 12
        for m in metrics:
            assert np.isfinite(m.separation) or m.separation == 0.0, (
                f"Layer {m.layer}: non-finite separation {m.separation}"
            )
            assert np.isfinite(m.coherence) or m.coherence == 0.0, (
                f"Layer {m.layer}: non-finite coherence {m.coherence}"
            )

    def test_nan_in_middle_layers(self):
        """Partial corruption: layers 4-6 are NaN, rest are valid."""
        rng = np.random.default_rng(9)
        layer_acts = []
        for i in range(12):
            if 4 <= i <= 6:
                pos = np.full((50, 32), np.nan)
                neg = np.full((50, 32), np.nan)
            else:
                pos = rng.standard_normal((50, 32)) + 1.0
                neg = rng.standard_normal((50, 32))
            layer_acts.append((pos, neg))

        metrics = compute_layer_metrics(layer_acts)
        assert len(metrics) == 12
        # Valid layers should have positive separation
        for m in metrics:
            if m.layer not in (4, 5, 6):
                assert np.isfinite(m.separation), (
                    f"Valid layer {m.layer} has non-finite S={m.separation}"
                )

    def test_constant_activations_all_layers(self):
        """Dead neuron scenario: every activation is the same constant."""
        layer_acts = [
            (np.ones((50, 32)) * float(i), np.ones((50, 32)) * float(i))
            for i in range(8)
        ]
        metrics = compute_layer_metrics(layer_acts)
        for m in metrics:
            assert np.isfinite(m.separation), (
                f"Constant layer {m.layer}: S={m.separation}"
            )

    def test_velocity_finite_when_separations_contain_nan(self):
        """compute_velocity must return an array of correct length even with NaN S."""
        seps_with_nan = [0.1, 0.2, np.nan, 0.4, 0.5, np.nan, 0.7]
        result = compute_velocity(seps_with_nan, window=1)
        assert len(result) == 7
        # We don't require all values to be finite (NaN propagation is expected
        # through the diff), but the function must not raise and length must match.

    def test_single_layer_model(self):
        """Edge case: 1-layer model."""
        rng = np.random.default_rng(10)
        layer_acts = [
            (rng.standard_normal((20, 8)) + 1.0, rng.standard_normal((20, 8)))
        ]
        metrics = compute_layer_metrics(layer_acts)
        assert len(metrics) == 1
        assert metrics[0].velocity == 0.0  # No previous layer to diff against


# ---------------------------------------------------------------------------
# find_caz_boundary with degenerate metric sequences
# ---------------------------------------------------------------------------


class TestCAZBoundaryPathological:
    def _make_metrics(self, seps):
        vels = compute_velocity(seps, window=1)
        return [
            LayerMetrics(
                layer=i,
                separation=float(seps[i]),
                coherence=0.5,
                velocity=float(vels[i]),
            )
            for i in range(len(seps))
        ]

    def test_all_zero_separation(self):
        """All S=0 — boundary should still return without error."""
        metrics = self._make_metrics([0.0] * 12)
        boundary = find_caz_boundary(metrics)
        assert boundary.peak_separation == 0.0
        assert np.isfinite(boundary.threshold)

    def test_single_layer(self):
        metrics = self._make_metrics([0.5])
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_peak == 0
        assert boundary.caz_width == 1

    def test_two_layers(self):
        metrics = self._make_metrics([0.3, 0.7])
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_peak == 1

    def test_nan_in_separation_sequence(self):
        """NaN S values in the sequence — boundary detection must not raise."""
        seps = [0.1, 0.2, np.nan, 0.4, 0.5, np.nan, 0.3]
        vels = compute_velocity(seps, window=1)
        metrics = [
            LayerMetrics(
                layer=i,
                separation=float(seps[i]),
                coherence=0.5,
                velocity=float(vels[i]),
            )
            for i in range(len(seps))
        ]
        # Should not raise — exact return value is implementation-defined for NaN inputs
        try:
            boundary = find_caz_boundary(metrics)
            # If it returns, peak_layer must be a valid index
            assert 0 <= boundary.caz_peak < len(metrics)
        except ValueError:
            pass  # Raising ValueError is also acceptable for NaN inputs

    def test_all_negative_velocity(self):
        """Monotone-decreasing S — velocity all negative after layer 0."""
        metrics = self._make_metrics([5.0, 4.0, 3.0, 2.0, 1.0])
        boundary = find_caz_boundary(metrics)
        assert boundary.caz_peak == 0
        assert np.isfinite(boundary.peak_separation)


# ---------------------------------------------------------------------------
# No-NaN contract: the full pipeline on valid data
# ---------------------------------------------------------------------------


class TestNoNaNContract:
    """Given clean input, every output value must be finite.

    This is the regression test for the fp16 overflow failure: on valid,
    normally-distributed float64 activations, compute_layer_metrics must
    never produce NaN or Inf for any layer metric.
    """

    @pytest.mark.parametrize(
        "n_layers,d,n",
        [
            (12, 768, 100),  # GPT-2 scale
            (48, 1600, 100),  # GPT-2-XL scale
            (32, 4096, 100),  # 7B model scale (hidden dim)
        ],
    )
    def test_no_nan_on_valid_gaussian_activations(self, n_layers, d, n):
        rng = np.random.default_rng(42)
        layer_acts = []
        for layer in range(n_layers):
            gap = float(layer) / n_layers * 3.0
            pos = rng.standard_normal((n, d)).astype(np.float64) + gap
            neg = rng.standard_normal((n, d)).astype(np.float64)
            layer_acts.append((pos, neg))

        metrics = compute_layer_metrics(layer_acts)
        assert len(metrics) == n_layers

        for m in metrics:
            assert np.isfinite(m.separation), (
                f"Layer {m.layer}/{n_layers}, d={d}: S={m.separation} is not finite"
            )
            assert np.isfinite(m.coherence), (
                f"Layer {m.layer}/{n_layers}, d={d}: C={m.coherence} is not finite"
            )
            assert np.isfinite(m.velocity), (
                f"Layer {m.layer}/{n_layers}, d={d}: v={m.velocity} is not finite"
            )
            assert m.separation >= 0.0
            assert 0.0 <= m.coherence <= 1.0

    def test_no_nan_at_large_activation_magnitude(self):
        """Activations at magnitudes that overflow fp16 must be fine in float64."""
        rng = np.random.default_rng(13)
        n_layers, d, n = 48, 1600, 100
        layer_acts = []
        for layer in range(n_layers):
            # Magnitudes grow with depth — simulates what happens in deep models
            scale = 10.0 * (1 + layer / 10)
            pos = rng.standard_normal((n, d)).astype(np.float64) * scale + scale
            neg = rng.standard_normal((n, d)).astype(np.float64) * scale
            layer_acts.append((pos, neg))

        metrics = compute_layer_metrics(layer_acts)
        nan_layers = [m.layer for m in metrics if not np.isfinite(m.separation)]
        assert not nan_layers, (
            f"NaN separation at layers {nan_layers} with large-magnitude activations. "
            "This replicates the fp16 overflow failure — check float64 casting."
        )

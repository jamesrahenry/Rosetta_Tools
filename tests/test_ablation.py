"""
test_ablation.py — Tests for rosetta_tools.ablation.

Tests concept direction computation, ablation math, and KL divergence.
Uses minimal torch tensors — no model loading, no GPU.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rosetta_tools.ablation import (
    DirectionalAblator,
    compute_dominant_direction,
    kl_divergence_from_logits,
)


# ---------------------------------------------------------------------------
# compute_dominant_direction
# ---------------------------------------------------------------------------


class TestComputeDominantDirection:

    def test_unit_vector(self):
        """Output should be a unit vector."""
        pos = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        neg = np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        d = compute_dominant_direction(pos, neg)
        assert d.dtype == np.float64
        assert np.linalg.norm(d) == pytest.approx(1.0)

    def test_direction_is_pos_minus_neg(self):
        """Direction should point from neg centroid to pos centroid."""
        pos = np.array([[4.0, 0.0], [6.0, 0.0]])  # centroid = [5, 0]
        neg = np.array([[0.0, 0.0], [0.0, 0.0]])   # centroid = [0, 0]
        d = compute_dominant_direction(pos, neg)
        np.testing.assert_allclose(d, [1.0, 0.0])

    def test_identical_distributions(self):
        """When pos == neg, direction should be zero (or near-zero)."""
        acts = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = compute_dominant_direction(acts, acts)
        assert np.linalg.norm(d) < 1e-10

    def test_single_sample_each(self):
        pos = np.array([[1.0, 0.0]])
        neg = np.array([[0.0, 1.0]])
        d = compute_dominant_direction(pos, neg)
        assert np.linalg.norm(d) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DirectionalAblator
# ---------------------------------------------------------------------------


class _SimpleLayer(nn.Module):
    """A no-op layer that returns input unchanged (as a tuple like HF layers)."""
    def forward(self, x):
        return (x,)


class TestDirectionalAblator:

    def test_removes_direction_component(self):
        """After ablation, projection onto direction should be zero."""
        layer = _SimpleLayer()
        direction = np.array([1.0, 0.0, 0.0])  # ablate x-axis
        hidden = torch.tensor([[[3.0, 4.0, 5.0]]])  # [1, 1, 3]

        with DirectionalAblator(layer, direction):
            output = layer(hidden)

        ablated = output[0]
        # x component should be removed
        assert ablated[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        # y, z should be unchanged
        assert ablated[0, 0, 1].item() == pytest.approx(4.0)
        assert ablated[0, 0, 2].item() == pytest.approx(5.0)

    def test_hook_removed_after_context(self):
        """Hook should be cleaned up after exiting context."""
        layer = _SimpleLayer()
        direction = np.array([1.0, 0.0])
        hidden = torch.tensor([[[3.0, 4.0]]])

        with DirectionalAblator(layer, direction):
            pass

        # After context exit, layer should behave normally
        output = layer(hidden)
        torch.testing.assert_close(output[0], hidden)

    def test_preserves_orthogonal_component(self):
        """Components orthogonal to direction should be perfectly preserved."""
        layer = _SimpleLayer()
        direction = np.array([0.0, 1.0, 0.0])  # ablate y-axis
        hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1, 2, 3]

        with DirectionalAblator(layer, direction):
            output = layer(hidden)

        ablated = output[0]
        # x and z unchanged
        assert ablated[0, 0, 0].item() == pytest.approx(1.0)
        assert ablated[0, 0, 2].item() == pytest.approx(3.0)
        assert ablated[0, 1, 0].item() == pytest.approx(4.0)
        assert ablated[0, 1, 2].item() == pytest.approx(6.0)
        # y zeroed
        assert ablated[0, 0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert ablated[0, 1, 1].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# kl_divergence_from_logits
# ---------------------------------------------------------------------------


class TestKLDivergence:

    def test_identical_logits(self):
        """KL divergence of identical distributions should be 0."""
        logits = torch.randn(100)
        kl = kl_divergence_from_logits(logits, logits)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self):
        """KL divergence should be positive for different distributions."""
        baseline = torch.randn(100)
        ablated = torch.randn(100)
        kl = kl_divergence_from_logits(baseline, ablated)
        assert kl > 0

    def test_more_different_means_higher_kl(self):
        """Larger perturbation → higher KL."""
        torch.manual_seed(42)
        baseline = torch.randn(50)
        small_noise = baseline + torch.randn(50) * 0.1
        large_noise = baseline + torch.randn(50) * 10.0

        kl_small = kl_divergence_from_logits(baseline, small_noise)
        kl_large = kl_divergence_from_logits(baseline, large_noise)
        assert kl_large > kl_small

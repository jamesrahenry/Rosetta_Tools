"""
test_feature_tracker.py — Tests for rosetta_tools.feature_tracker.

Tests cross-layer feature tracking with synthetic PC directions and eigenvalues.
No model loading, no GPU required.

Test philosophy:
- Known geometry: hand-craft PC directions with known cosine relationships
- Feature continuity: a direction that persists should be tracked as one feature
- Birth/death: new directions appear, old ones vanish
- Concept alignment: features aligned with known concepts should be detected
- Trajectory storage: per-layer alignment must be preserved, not just max
"""

import numpy as np
import pytest

from rosetta_tools.feature_tracker import Feature, FeatureMap, track_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_direction(dim: int, axis: int) -> np.ndarray:
    """Unit vector along a given axis."""
    v = np.zeros(dim)
    v[axis] = 1.0
    return v


def rotate_direction(v: np.ndarray, angle_deg: float, plane: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Rotate v by angle_deg in the plane defined by two axes."""
    theta = np.radians(angle_deg)
    result = v.copy()
    i, j = plane
    result[i] = v[i] * np.cos(theta) - v[j] * np.sin(theta)
    result[j] = v[i] * np.sin(theta) + v[j] * np.cos(theta)
    return result / (np.linalg.norm(result) + 1e-12)


# ---------------------------------------------------------------------------
# Basic tracking
# ---------------------------------------------------------------------------


class TestBasicTracking:
    """A single persistent direction across all layers."""

    def test_single_persistent_feature(self):
        dim = 16
        n_layers = 5
        direction = make_direction(dim, 0)

        layer_directions = [np.array([direction]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        assert fm.n_features == 1
        f = fm.features[0]
        assert f.birth_layer == 0
        assert f.death_layer == n_layers - 1
        assert f.lifespan == n_layers
        assert f.is_persistent

    def test_two_independent_features(self):
        dim = 16
        n_layers = 5
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)  # orthogonal

        layer_directions = [np.array([d1, d2]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0, 8.0] for _ in range(n_layers)]

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        assert fm.n_features == 2
        assert all(f.lifespan == n_layers for f in fm.features)

    def test_transient_feature(self):
        dim = 16
        n_layers = 6
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)  # appears only at layers 2-3

        layer_directions = []
        layer_eigenvalues = []
        for layer in range(n_layers):
            if 2 <= layer <= 3:
                layer_directions.append(np.array([d1, d2]))
                layer_eigenvalues.append([10.0, 8.0])
            else:
                layer_directions.append(np.array([d1]))
                layer_eigenvalues.append([10.0])

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        persistent = [f for f in fm.features if f.lifespan == n_layers]
        transient = [f for f in fm.features if f.lifespan == 2]
        assert len(persistent) == 1
        assert len(transient) == 1
        assert transient[0].is_transient


class TestBirthAndDeath:

    def test_feature_born_at_layer_3(self):
        dim = 16
        n_layers = 6
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)

        layer_directions = []
        layer_eigenvalues = []
        for layer in range(n_layers):
            if layer >= 3:
                layer_directions.append(np.array([d1, d2]))
                layer_eigenvalues.append([10.0, 8.0])
            else:
                layer_directions.append(np.array([d1]))
                layer_eigenvalues.append([10.0])

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        born_late = [f for f in fm.features if f.birth_layer == 3]
        assert len(born_late) == 1
        assert born_late[0].death_layer == 5

    def test_feature_dies_at_layer_2(self):
        dim = 16
        n_layers = 6
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)

        layer_directions = []
        layer_eigenvalues = []
        for layer in range(n_layers):
            if layer <= 2:
                layer_directions.append(np.array([d1, d2]))
                layer_eigenvalues.append([10.0, 8.0])
            else:
                layer_directions.append(np.array([d1]))
                layer_eigenvalues.append([10.0])

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        died_early = [f for f in fm.features if f.death_layer == 2]
        assert len(died_early) == 1
        assert died_early[0].birth_layer == 0


# ---------------------------------------------------------------------------
# Cosine threshold
# ---------------------------------------------------------------------------


class TestCosineThreshold:

    def test_gradual_rotation_tracked(self):
        """Small rotations stay above threshold — single feature."""
        dim = 16
        n_layers = 5
        direction = make_direction(dim, 0)

        layer_directions = []
        for layer in range(n_layers):
            rotated = rotate_direction(direction, layer * 10)  # 10° per layer
            layer_directions.append(np.array([rotated]))

        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        fm = track_features(layer_directions, layer_eigenvalues, n_layers, cos_threshold=0.5)
        assert fm.n_features == 1

    def test_abrupt_rotation_splits(self):
        """90° rotation breaks the chain — two features."""
        dim = 16
        n_layers = 4
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)  # orthogonal = cos=0

        layer_directions = [
            np.array([d1]),
            np.array([d1]),
            np.array([d2]),  # abrupt switch
            np.array([d2]),
        ]
        layer_eigenvalues = [[10.0]] * 4

        fm = track_features(layer_directions, layer_eigenvalues, n_layers, cos_threshold=0.5)
        assert fm.n_features == 2


# ---------------------------------------------------------------------------
# Concept alignment — flat mode (legacy)
# ---------------------------------------------------------------------------


class TestConceptAlignmentFlat:

    def test_aligned_feature_detected(self):
        dim = 16
        n_layers = 5
        direction = make_direction(dim, 0)
        concept = make_direction(dim, 0)  # identical

        layer_directions = [np.array([direction]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions={"test_concept": concept},
        )

        f = fm.features[0]
        assert f.concept_alignment["test_concept"] == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_feature_not_aligned(self):
        dim = 16
        n_layers = 5
        direction = make_direction(dim, 0)
        concept = make_direction(dim, 1)  # orthogonal

        layer_directions = [np.array([direction]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions={"test_concept": concept},
        )

        f = fm.features[0]
        assert f.concept_alignment["test_concept"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Concept alignment — per-layer mode with trajectory
# ---------------------------------------------------------------------------


class TestConceptAlignmentTrajectory:
    """The key new capability: per-layer concept alignment storage."""

    def test_trajectory_populated_per_layer(self):
        dim = 16
        n_layers = 5
        direction = make_direction(dim, 0)
        concept_dir = make_direction(dim, 0)

        layer_directions = [np.array([direction]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        # Per-layer concept directions
        concept_directions = {
            "test_concept": {layer: concept_dir for layer in range(n_layers)}
        }

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions=concept_directions,
        )

        f = fm.features[0]
        traj = f.concept_alignment_trajectory
        assert "test_concept" in traj
        assert len(traj["test_concept"]) == n_layers
        for layer in range(n_layers):
            assert layer in traj["test_concept"]
            assert traj["test_concept"][layer] == pytest.approx(1.0, abs=0.01)

    def test_trajectory_shows_rotation_through_concept_space(self):
        """Feature rotates from concept-aligned to orthogonal — trajectory captures it."""
        dim = 16
        n_layers = 5
        concept_dir = make_direction(dim, 0)

        # Feature gradually rotates away from concept direction
        layer_directions = []
        for layer in range(n_layers):
            angle = layer * 20  # 0°, 20°, 40°, 60°, 80°
            rotated = rotate_direction(make_direction(dim, 0), angle)
            layer_directions.append(np.array([rotated]))

        layer_eigenvalues = [[10.0] for _ in range(n_layers)]
        concept_directions = {
            "test_concept": {layer: concept_dir for layer in range(n_layers)}
        }

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions=concept_directions,
        )

        f = fm.features[0]
        traj = f.concept_alignment_trajectory["test_concept"]

        # Layer 0: perfectly aligned (cos²=1.0)
        assert traj[0] == pytest.approx(1.0, abs=0.01)
        # Layer 4: 80° rotation (cos²(80°) ≈ 0.03)
        assert traj[4] < 0.1
        # Trajectory should be monotonically decreasing
        values = [traj[l] for l in range(n_layers)]
        assert values == sorted(values, reverse=True)

    def test_trajectory_max_equals_concept_alignment(self):
        """The max of the trajectory should equal concept_alignment (backward compat)."""
        dim = 16
        n_layers = 5
        concept_dir = make_direction(dim, 0)

        layer_directions = []
        for layer in range(n_layers):
            angle = layer * 15
            rotated = rotate_direction(make_direction(dim, 0), angle)
            layer_directions.append(np.array([rotated]))

        layer_eigenvalues = [[10.0] for _ in range(n_layers)]
        concept_directions = {
            "test_concept": {layer: concept_dir for layer in range(n_layers)}
        }

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions=concept_directions,
        )

        f = fm.features[0]
        traj_max = max(f.concept_alignment_trajectory["test_concept"].values())
        assert f.concept_alignment["test_concept"] == pytest.approx(traj_max, abs=0.001)

    def test_concept_switch_at_midpoint(self):
        """Feature aligned with concept A at shallow, concept B at deep."""
        dim = 16
        n_layers = 6
        concept_a = make_direction(dim, 0)
        concept_b = make_direction(dim, 1)

        # Feature starts along axis 0, rotates to axis 1
        layer_directions = []
        for layer in range(n_layers):
            t = layer / (n_layers - 1)  # 0 to 1
            vec = (1 - t) * concept_a + t * concept_b
            vec = vec / np.linalg.norm(vec)
            layer_directions.append(np.array([vec]))

        layer_eigenvalues = [[10.0] for _ in range(n_layers)]
        concept_directions = {
            "concept_a": {layer: concept_a for layer in range(n_layers)},
            "concept_b": {layer: concept_b for layer in range(n_layers)},
        }

        fm = track_features(
            layer_directions, layer_eigenvalues, n_layers,
            concept_directions=concept_directions,
            cos_threshold=0.3,  # low threshold to keep it as one feature
        )

        f = fm.features[0]
        traj_a = f.concept_alignment_trajectory["concept_a"]
        traj_b = f.concept_alignment_trajectory["concept_b"]

        # At layer 0: mostly A
        assert traj_a[0] > traj_b[0]
        # At last layer: mostly B
        assert traj_b[n_layers - 1] > traj_a[n_layers - 1]

    def test_empty_trajectory_when_no_concepts(self):
        dim = 16
        n_layers = 3
        direction = make_direction(dim, 0)

        layer_directions = [np.array([direction]) for _ in range(n_layers)]
        layer_eigenvalues = [[10.0] for _ in range(n_layers)]

        fm = track_features(layer_directions, layer_eigenvalues, n_layers)

        f = fm.features[0]
        assert f.concept_alignment_trajectory == {}
        assert f.concept_alignment == {}


# ---------------------------------------------------------------------------
# FeatureMap helpers
# ---------------------------------------------------------------------------


class TestFeatureMap:

    def _make_map(self) -> FeatureMap:
        dim = 16
        n_layers = 10
        d1 = make_direction(dim, 0)
        d2 = make_direction(dim, 1)

        layer_directions = []
        layer_eigenvalues = []
        for layer in range(n_layers):
            if layer < 7:
                layer_directions.append(np.array([d1, d2]))
                layer_eigenvalues.append([10.0, 8.0])
            else:
                layer_directions.append(np.array([d1]))
                layer_eigenvalues.append([10.0])

        return track_features(layer_directions, layer_eigenvalues, n_layers)

    def test_features_at_layer(self):
        fm = self._make_map()
        at_3 = fm.features_at_layer(3)
        assert len(at_3) == 2

        at_8 = fm.features_at_layer(8)
        assert len(at_8) == 1

    def test_persistent_features(self):
        fm = self._make_map()
        persistent = fm.persistent_features()
        assert len(persistent) >= 1
        assert all(f.is_persistent for f in persistent)

    def test_max_concurrent(self):
        fm = self._make_map()
        assert fm.max_concurrent == 2

"""
test_gem.py — Tests for rosetta_tools.gem (GEM node/assembly construction).

All tests use synthetic caz_data dicts and hand-built dataclasses with known
geometric structure — no model loading, no GPU required, no dependency on
extraction pipeline output beyond the caz_data shape documented in
Rosetta_Full_Tutorial.ipynb cell 18.

Test philosophy:
- build_gem_node_k1: handoff_layer clamps correctly at model boundaries;
  cos_chain/entry_exit_cosine/max_rotation are sign-invariant (abs()) and
  match known geometry (constant direction → all-ones chain, orthogonal
  rotation → chain crosses zero); dimension mismatches raise, not corrupt
- build_concept_gem: k=1 happy path, the (currently unreachable in practice
  but still a real code path) zero-region branch, k>1 explicitly raises
  NotImplementedError rather than silently doing something wrong
- classify_node_dependencies: independence/upstream/downstream classification
  matches the documented "only forward passes count" asymmetry
- classify_caz_depth: every _CAZ_TYPE_BREAKPOINTS boundary lands on the
  correct side
- routing_held_ratio / routing_held_from_permutation / stage_detail:
  known inputs → known outputs, including the dict-or-object duck typing
  stage_detail relies on
- validate_gem_node: a node built by build_gem_node_k1 is clean (no
  warnings); each individual invariant violation is caught independently
- Serialization: gem_to_dict/gem_from_dict/save_gem/load_gem round-trip
  preserves every field
"""

from __future__ import annotations

import numpy as np
import pytest

from rosetta_tools.caz import CAZRegion
from rosetta_tools.gem import (
    ConceptGEM,
    EigThread,
    GEMNode,
    build_concept_gem,
    build_gem_node_k1,
    classify_caz_depth,
    classify_node_dependencies,
    gem_diagnostics,
    gem_from_dict,
    gem_to_dict,
    load_gem,
    routing_held_from_permutation,
    routing_held_ratio,
    save_gem,
    stage_detail,
    validate_gem_node,
)

from .conftest import make_caz_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _region(start: int, peak: int, end: int, caz_score: float = 1.0, **overrides) -> CAZRegion:
    """Build a CAZRegion with sane defaults, overridable per-test."""
    defaults = dict(
        peak_separation=1.0,
        peak_coherence=0.5,
        mean_separation=1.0,
        mean_coherence=0.5,
        prominence=0.5,
        depth_pct=0.0,
        width_pct=0.0,
        rise_span=max(peak - start, 1),
        fall_span=max(end - peak, 1),
        caz_score=caz_score,
        functional_caz_score=caz_score,
        is_functional_peak=False,
    )
    defaults.update(overrides)
    return CAZRegion(start=start, peak=peak, end=end, width=end - start + 1, **defaults)


def _make_node(
    settled_direction=(1.0, 0.0, 0.0),
    handoff_layer=5,
    handoff_cosine=1.0,
    concept_thread_id=0,
    cos_chain=(1.0, 1.0),
    entry_exit_cosine=1.0,
    caz_end=3,
    n_layers_total=10,
) -> GEMNode:
    """Build a minimal, deliberately-controllable GEMNode for validate_gem_node."""
    thread = EigThread(
        thread_id=0,
        layer_indices=[2, 3],
        directions=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        eigenvalues=[1.0, 1.0],
        concept_alignments=[1.0, 1.0],
        cos_chain=list(cos_chain),
    )
    return GEMNode(
        concept="c",
        model_id="m",
        caz_index=0,
        caz_start=2,
        caz_peak=3,
        caz_end=caz_end,
        caz_score=1.0,
        attention_paradigm="unknown",
        settled_direction=list(settled_direction),
        handoff_layer=handoff_layer,
        handoff_cosine=handoff_cosine,
        threads=[thread],
        concept_thread_id=concept_thread_id,
        entry_exit_cosine=entry_exit_cosine,
        max_rotation_per_layer=0.0,
        n_layers_total=n_layers_total,
        hidden_dim=3,
        k=1,
        phase=1,
    )


# ---------------------------------------------------------------------------
# build_gem_node_k1
# ---------------------------------------------------------------------------


class TestBuildGemNodeK1:
    def test_constant_direction_gives_unit_cos_chain(self):
        """Same direction at every layer in the window → cos_chain all 1.0,
        entry_exit_cosine 1.0, zero rotation."""
        n_layers = 10
        directions = [[1.0, 0.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[float(i) for i in range(n_layers)])
        region = _region(start=2, peak=4, end=5)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.caz_start == 2 and node.caz_peak == 4 and node.caz_end == 5
        assert np.linalg.norm(node.settled_direction) == pytest.approx(1.0)
        thread = node.concept_thread
        assert thread.cos_chain == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert node.entry_exit_cosine == pytest.approx(1.0)
        assert node.max_rotation_per_layer == pytest.approx(0.0)

    def test_orthogonal_rotation_crosses_zero(self):
        """Direction rotates 90 degrees across the window → cos_chain dips
        to 0, entry_exit_cosine ~0, max_rotation ~1."""
        n_layers = 6
        directions = [[0.0] * n_layers for _ in range(n_layers)]
        # region.start..end = 0..1: orthogonal unit vectors at those two layers
        directions[0] = [1.0, 0.0]
        directions[1] = [0.0, 1.0]
        for i in range(2, n_layers):
            directions[i] = [1.0, 0.0]
        caz_data = make_caz_data(directions, separations=[float(i) for i in range(n_layers)])
        region = _region(start=0, peak=0, end=1)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        thread = node.concept_thread
        assert thread.cos_chain[0] == pytest.approx(1.0)
        assert thread.cos_chain[1] == pytest.approx(0.0, abs=1e-9)
        assert node.entry_exit_cosine == pytest.approx(0.0, abs=1e-9)
        assert node.max_rotation_per_layer == pytest.approx(1.0, abs=1e-9)

    def test_handoff_layer_clamps_at_model_end(self):
        """region.end at the last layer → handoff_layer clamps to n_layers-1,
        not n_layers (out of bounds), and compares the settled direction
        against itself (handoff_cosine == 1.0)."""
        n_layers = 8
        directions = [[1.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=5, peak=6, end=n_layers - 1)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.handoff_layer == n_layers - 1
        assert node.handoff_cosine == pytest.approx(1.0)

    def test_handoff_layer_mid_model(self):
        n_layers = 12
        directions = [[1.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=2, peak=3, end=4)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.handoff_layer == 5  # region.end + 1

    def test_dimension_mismatch_raises(self):
        n_layers = 5
        directions = [[1.0, 0.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        # Corrupt one layer's dom_vector to a different dimension
        caz_data["layer_data"]["metrics"][2]["dom_vector"] = [1.0, 0.0]
        region = _region(start=1, peak=2, end=3)

        with pytest.raises(ValueError, match="corrupted extraction data"):
            build_gem_node_k1(caz_data, region, caz_index=0)

    def test_empty_window_falls_back_to_peak_layer(self):
        """An (invalid but constructible) region where start > end produces
        an empty window_layers range — falls back to [region.peak]."""
        n_layers = 8
        directions = [[float(i + 1), 0.0] for i in range(n_layers)]
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=5, peak=3, end=2)  # start > end: empty range(5, 3)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        thread = node.concept_thread
        assert thread.layer_indices == [3]
        assert node.entry_exit_cosine == pytest.approx(1.0)  # single direction vs itself

    def test_eigenvalues_are_separation_fisher_at_window_layers(self):
        n_layers = 6
        directions = [[1.0, 0.0]] * n_layers
        seps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        caz_data = make_caz_data(directions, separations=seps)
        region = _region(start=1, peak=2, end=3)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.concept_thread.eigenvalues == pytest.approx([20.0, 30.0, 40.0])
        assert node.concept_thread.concept_alignments == [1.0, 1.0, 1.0]
        assert node.concept_thread.mean_alignment == 1.0
        assert node.concept_thread.cumulative_score == pytest.approx(90.0)

    def test_k1_phase1_metadata(self):
        n_layers = 5
        directions = [[1.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=0, peak=1, end=2)

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.k == 1
        assert node.phase == 1
        assert node.concept_thread_id == 0
        assert node.concept_thread is node.threads[0]

    def test_depth_pct_property(self):
        n_layers = 11  # so n_layers_total - 1 = 10, clean percentages
        directions = [[1.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=2, peak=3, end=4)  # handoff_layer = 5

        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert node.handoff_layer == 5
        assert node.depth_pct == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# build_concept_gem
# ---------------------------------------------------------------------------


class TestBuildConceptGem:
    def _single_peak_caz_data(self, n_layers=16, peak_at=8, hidden_dim=4):
        rng = np.random.default_rng(0)
        seps = [float(max(0, 5 - abs(i - peak_at))) for i in range(n_layers)]
        directions = [rng.standard_normal(hidden_dim) for _ in range(n_layers)]
        return make_caz_data(directions, separations=seps, coherences=[0.6] * n_layers)

    def test_zero_regions_returns_empty_gem(self, monkeypatch):
        """profile.n_regions == 0 is not reachable via find_caz_regions_scored's
        current fallback-to-global-max behavior, but build_concept_gem's
        short-circuit for that branch is still real code — verify it directly
        by monkeypatching the detector."""
        import rosetta_tools.gem as gem_module
        from rosetta_tools.caz import CAZProfile

        empty_profile = CAZProfile(
            n_layers=4,
            regions=[],
            n_regions=0,
            global_peak_layer=0,
            global_peak_separation=0.0,
            global_mean_separation=0.0,
            is_multimodal=False,
        )
        monkeypatch.setattr(gem_module, "find_caz_regions_scored", lambda *a, **k: empty_profile)

        caz_data = self._single_peak_caz_data(n_layers=4)
        result = build_concept_gem(caz_data)

        assert result.n_nodes == 0
        assert result.nodes == []
        assert result.interaction_matrix is None
        assert result.node_types is None
        assert result.ablation_targets is None

    def test_single_peak_k1_happy_path(self):
        caz_data = self._single_peak_caz_data()
        result = build_concept_gem(caz_data)

        assert isinstance(result, ConceptGEM)
        assert result.n_nodes == 1
        assert result.node_types == ["independent"]
        assert result.ablation_targets == [0]
        assert result.nodes[0].k == 1

    def test_k_greater_than_1_requires_raw_activations(self):
        caz_data = self._single_peak_caz_data()
        with pytest.raises(ValueError, match="Phase 2"):
            build_concept_gem(caz_data, k=2)

    def test_multi_peak_produces_multiple_nodes_sorted_by_start(self):
        n_layers = 20
        rng = np.random.default_rng(1)
        seps = [1, 2, 4, 6, 8, 6, 5.8, 6, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        directions = [rng.standard_normal(4) for _ in range(n_layers)]
        caz_data = make_caz_data(directions, separations=seps, coherences=[0.6] * n_layers)

        result = build_concept_gem(caz_data)

        assert result.n_nodes == 2
        assert [n.caz_start for n in result.nodes] == sorted(n.caz_start for n in result.nodes)

    def test_multimodal_data_wires_dependency_classification(self):
        n_layers = 20
        rng = np.random.default_rng(2)
        seps = [1, 2, 4, 6, 8, 6, 5.8, 6, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        directions = [rng.standard_normal(4) for _ in range(n_layers)]
        caz_data = make_caz_data(directions, separations=seps, coherences=[0.6] * n_layers)

        # Two detected regions peak near layer 4 and layer 8 (see test above).
        multimodal_data = {
            "interaction_peaks": [4, 8],
            "interaction_matrix": [
                [100.0, 10.0],   # ablating node0 suppresses node1 (retained 10%)
                [100.0, 100.0],
            ],
        }
        result = build_concept_gem(caz_data, multimodal_data=multimodal_data)

        assert result.interaction_matrix is not None
        assert result.node_types == ["upstream", "downstream"]
        assert result.ablation_targets == [0]


# ---------------------------------------------------------------------------
# classify_node_dependencies
# ---------------------------------------------------------------------------


class TestClassifyNodeDependencies:
    def test_empty_matrix(self):
        assert classify_node_dependencies([]) == ([], [])

    def test_single_node(self):
        assert classify_node_dependencies([[100.0]]) == (["independent"], [0])

    def test_two_independent_nodes(self):
        matrix = [[100.0, 100.0], [100.0, 100.0]]
        types, targets = classify_node_dependencies(matrix)
        assert types == ["independent", "independent"]
        assert targets == [0, 1]

    def test_shallow_ablation_suppresses_deeper_node(self):
        """i=0 ablated suppresses j=1 (retained < threshold), i < j → upstream/downstream."""
        matrix = [[100.0, 20.0], [100.0, 100.0]]
        types, targets = classify_node_dependencies(matrix)
        assert types == ["upstream", "downstream"]
        assert targets == [0]

    def test_deep_ablation_suppressing_shallow_node_is_ignored(self):
        """i=1 (deeper) ablated 'suppresses' j=0 (shallower) — i > j, so the
        forward-pass-only rule means this does NOT count as a dependency."""
        matrix = [[100.0, 100.0], [10.0, 100.0]]
        types, targets = classify_node_dependencies(matrix)
        assert types == ["independent", "independent"]
        assert targets == [0, 1]

    def test_custom_independence_threshold(self):
        matrix = [[100.0, 85.0], [100.0, 100.0]]
        types_default, _ = classify_node_dependencies(matrix)  # 85 >= 80 default -> independent
        assert types_default == ["independent", "independent"]
        types_strict, targets_strict = classify_node_dependencies(matrix, independence_threshold=90.0)
        assert types_strict == ["upstream", "downstream"]
        assert targets_strict == [0]


# ---------------------------------------------------------------------------
# classify_caz_depth
# ---------------------------------------------------------------------------


class TestClassifyCazDepth:
    @pytest.mark.parametrize(
        "layer,expected",
        [
            (0, "embedding"),
            (9, "embedding"),
            (10, "active"),
            (34, "active"),
            (35, "deep"),
            (69, "deep"),
            (70, "functional"),
            (89, "functional"),
            (90, "artifact"),
            (100, "artifact"),
        ],
    )
    def test_breakpoints(self, layer, expected):
        # n_layers=101 so rel = layer/100 lands exactly on each breakpoint
        assert classify_caz_depth(layer, n_layers=101) == expected

    def test_single_layer_model(self):
        assert classify_caz_depth(0, n_layers=1) == "embedding"


# ---------------------------------------------------------------------------
# routing_held_ratio
# ---------------------------------------------------------------------------


class TestRoutingHeldRatio:
    def test_zero_peak_separation_returns_false(self):
        ratio, held = routing_held_ratio({5: 0.0, 20: 3.0}, peak_layer=5, final_layer=20)
        assert ratio == 0.0
        assert held is False

    def test_ratio_above_default_threshold_is_held(self):
        ratio, held = routing_held_ratio({5: 10.0, 20: 9.6}, peak_layer=5, final_layer=20)
        assert ratio == pytest.approx(0.96)
        assert held is True

    def test_ratio_below_default_threshold_is_not_held(self):
        ratio, held = routing_held_ratio({5: 10.0, 20: 5.0}, peak_layer=5, final_layer=20)
        assert ratio == pytest.approx(0.5)
        assert held is False

    def test_missing_final_layer_defaults_to_zero(self):
        ratio, held = routing_held_ratio({5: 10.0}, peak_layer=5, final_layer=99)
        assert ratio == 0.0
        assert held is False

    def test_custom_degradation_factor(self):
        ratio, held = routing_held_ratio(
            {5: 10.0, 20: 8.0}, peak_layer=5, final_layer=20, degradation_factor=0.7
        )
        assert ratio == pytest.approx(0.8)
        assert held is True


# ---------------------------------------------------------------------------
# routing_held_from_permutation
# ---------------------------------------------------------------------------


class TestRoutingHeldFromPermutation:
    def test_missing_key_returns_false(self):
        held = routing_held_from_permutation({}, node_idx=0, final_layer=10, baseline_final_sep=1.0)
        assert held is False

    def test_above_threshold_returns_true(self):
        subset_results = {"0": {"final_reduction": 0.10}}
        held = routing_held_from_permutation(
            subset_results, node_idx=0, final_layer=10, baseline_final_sep=1.0
        )
        assert held is True

    def test_below_threshold_returns_false(self):
        subset_results = {"0": {"final_reduction": 0.01}}
        held = routing_held_from_permutation(
            subset_results, node_idx=0, final_layer=10, baseline_final_sep=1.0
        )
        assert held is False

    def test_custom_contribution_threshold(self):
        subset_results = {"1": {"final_reduction": 0.10}}
        held = routing_held_from_permutation(
            subset_results, node_idx=1, final_layer=10, baseline_final_sep=1.0,
            contribution_threshold=0.2,
        )
        assert held is False


# ---------------------------------------------------------------------------
# stage_detail
# ---------------------------------------------------------------------------


class TestStageDetail:
    def test_no_nodes(self):
        assert stage_detail([], n_layers=12) == "No GEM nodes detected."

    def test_single_node_no_routing_info(self):
        node = {"caz_start": 4, "caz_end": 6, "caz_score": 0.42, "caz_peak": 5}
        result = stage_detail([node], n_layers=12)
        assert "L4–6" in result
        assert "0.42" in result
        assert "routing" not in result

    def test_single_node_routing_held(self):
        node = {"caz_start": 4, "caz_end": 6, "caz_score": 0.42, "caz_peak": 5}
        result = stage_detail([node], n_layers=12, routing_held_per_node={0: True})
        assert result.endswith("routing held.")

    def test_single_node_routing_lost(self):
        node = {"caz_start": 4, "caz_end": 6, "caz_score": 0.42, "caz_peak": 5}
        result = stage_detail([node], n_layers=12, routing_held_per_node={0: False})
        assert result.endswith("routing lost before final layer.")

    def test_multimodal_picks_dominant_by_caz_score(self):
        nodes = [
            {"caz_start": 0, "caz_end": 2, "caz_score": 0.10, "caz_peak": 1},
            {"caz_start": 8, "caz_end": 10, "caz_score": 0.90, "caz_peak": 9},
        ]
        result = stage_detail(nodes, n_layers=12)
        assert "Multimodal (2 regions)" in result
        assert "L8–10" in result  # dominant region (higher caz_score)
        assert "0.90" in result

    def test_multimodal_routing_string(self):
        nodes = [
            {"caz_start": 0, "caz_end": 2, "caz_score": 0.10, "caz_peak": 1},
            {"caz_start": 8, "caz_end": 10, "caz_score": 0.90, "caz_peak": 9},
        ]
        result = stage_detail(nodes, n_layers=12, routing_held_per_node={0: False, 1: True})
        assert "n0 lost" in result
        assert "n1 held" in result

    def test_accepts_gemnode_objects_not_just_dicts(self):
        node = _make_node()
        result = stage_detail([node], n_layers=10)
        assert "L2–3" in result  # caz_start=2, caz_end=3 per _make_node default


# ---------------------------------------------------------------------------
# validate_gem_node
# ---------------------------------------------------------------------------


class TestValidateGemNode:
    def test_node_from_build_gem_node_k1_is_clean(self):
        n_layers = 10
        directions = [[1.0, 0.0, 0.0]] * n_layers
        caz_data = make_caz_data(directions, separations=[1.0] * n_layers)
        region = _region(start=2, peak=3, end=4)
        node = build_gem_node_k1(caz_data, region, caz_index=0)

        assert validate_gem_node(node) == []

    def test_non_unit_settled_direction_warns(self):
        node = _make_node(settled_direction=(2.0, 0.0, 0.0))
        warnings = validate_gem_node(node)
        assert any("norm=" in w for w in warnings)

    def test_wrong_handoff_layer_warns(self):
        node = _make_node(handoff_layer=999)
        warnings = validate_gem_node(node)
        assert any("handoff_layer" in w for w in warnings)

    def test_out_of_range_concept_thread_id_warns(self):
        node = _make_node(concept_thread_id=5)
        warnings = validate_gem_node(node)
        assert any("concept_thread_id" in w for w in warnings)

    def test_cos_chain_out_of_range_warns(self):
        node = _make_node(cos_chain=(1.0, 1.5))
        warnings = validate_gem_node(node)
        assert any("cos_chain" in w for w in warnings)

    def test_entry_exit_cosine_out_of_range_warns(self):
        node = _make_node(entry_exit_cosine=1.5)
        warnings = validate_gem_node(node)
        assert any("entry_exit_cosine" in w for w in warnings)


# ---------------------------------------------------------------------------
# gem_diagnostics
# ---------------------------------------------------------------------------


class TestGemDiagnostics:
    def test_empty_gem(self):
        gem = ConceptGEM(concept="c", model_id="m", n_nodes=0, nodes=[])
        assert gem_diagnostics(gem) == {"n_nodes": 0, "status": "empty"}

    def test_populated_gem_aggregates(self):
        n_layers = 20
        rng = np.random.default_rng(3)
        seps = [1, 2, 4, 6, 8, 6, 5.8, 6, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        directions = [rng.standard_normal(4) for _ in range(n_layers)]
        caz_data = make_caz_data(directions, separations=seps, coherences=[0.6] * n_layers)
        gem = build_concept_gem(caz_data)

        diag = gem_diagnostics(gem)

        assert diag["n_nodes"] == gem.n_nodes
        assert diag["n_independent"] + diag["n_upstream"] + diag["n_downstream"] == gem.n_nodes
        assert diag["caz_score_max"] >= diag["caz_score_min"]
        assert 0.0 <= diag["handoff_cosine_mean"] <= 1.0 + 1e-6
        assert 0.0 <= diag["entry_exit_cosine_mean"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def _build_gem(self):
        n_layers = 16
        rng = np.random.default_rng(4)
        seps = [float(max(0, 5 - abs(i - 8))) for i in range(n_layers)]
        directions = [rng.standard_normal(4) for _ in range(n_layers)]
        caz_data = make_caz_data(directions, separations=seps, coherences=[0.6] * n_layers)
        return build_concept_gem(caz_data)

    def test_dict_round_trip_preserves_fields(self):
        gem = self._build_gem()
        data = gem_to_dict(gem)
        restored = gem_from_dict(data)

        assert restored.concept == gem.concept
        assert restored.model_id == gem.model_id
        assert restored.n_nodes == gem.n_nodes
        assert restored.node_types == gem.node_types
        assert restored.ablation_targets == gem.ablation_targets
        assert restored.nodes[0].settled_direction == pytest.approx(gem.nodes[0].settled_direction)
        assert restored.nodes[0].concept_thread.cos_chain == pytest.approx(
            gem.nodes[0].concept_thread.cos_chain
        )

    def test_save_load_round_trip(self, tmp_path):
        gem = self._build_gem()
        path = tmp_path / "gem_test.json"

        save_gem(gem, path)
        assert path.exists()
        restored = load_gem(path)

        assert restored.n_nodes == gem.n_nodes
        assert restored.nodes[0].handoff_layer == gem.nodes[0].handoff_layer
        assert restored.nodes[0].caz_score == pytest.approx(gem.nodes[0].caz_score)

    def test_encoder_handles_numpy_types(self):
        import json
        from rosetta_tools.gem import _GEMEncoder

        payload = {
            "a": np.float64(1.5),
            "b": np.int64(3),
            "c": np.array([1.0, 2.0, 3.0]),
        }
        encoded = json.dumps(payload, cls=_GEMEncoder)
        decoded = json.loads(encoded)
        assert decoded == {"a": 1.5, "b": 3, "c": [1.0, 2.0, 3.0]}

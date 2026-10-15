"""
test_gem_schema.py — Tests for GEM schema v2 (per-layer / per-sublayer
records, sidecar, back-compat loader) in rosetta_tools.gem and
rosetta_tools.gem_sublayer.

- Back-compat: synthetic v1 dicts always; real gem_*.json under
  ~/rosetta_data/models when present (skipped otherwise). Read-only.
- Statistics: formula fidelity against compute_separation and a direct
  implementation of the projected score; vectorized permutation null vs a
  slow loop; nulls separate signal from no-signal.
- Sublayer capture on tiny random-init models: see test_gem_sublayer_hooks.py.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosetta_tools.caz import compute_separation
from rosetta_tools.gem import (
    GEM_SCHEMA_VERSION,
    SettledSubspace,
    gem_from_dict,
    gem_to_dict,
    load_gem,
    load_gem_sidecar,
    save_gem,
)
from rosetta_tools.gem_sublayer import (
    _permutation_nulls,
    attach_layer_profile,
    attach_layer_settled_subspaces,
    build_layer_profile,
    decompose_settled_subspace,
    projected_separation,
)

MODELS = Path.home() / "rosetta_data" / "models"
QWEN = ["Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B", "Qwen_Qwen2.5_7B"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v1_dict(d=8):
    """A schema-v1 GEM dict as the pre-2026-10-15 code wrote it."""
    u = np.zeros(d); u[0] = 1.0
    node = dict(
        concept="c", model_id="m", caz_index=0, caz_start=1, caz_peak=2, caz_end=3,
        caz_score=0.5, attention_paradigm="gqa", settled_direction=u.tolist(),
        handoff_layer=4, handoff_cosine=0.9,
        threads=[dict(thread_id=0, layer_indices=[1, 2, 3], directions=[u.tolist()] * 3,
                      eigenvalues=[1.0, 2.0, 3.0], concept_alignments=[1.0] * 3,
                      cos_chain=[1.0, 1.0, 1.0], cumulative_score=6.0,
                      mean_eigenvalue=2.0, mean_alignment=1.0)],
        concept_thread_id=0, entry_exit_cosine=1.0, max_rotation_per_layer=0.0,
        n_layers_total=6, hidden_dim=d, k=1, phase=1,
    )
    return dict(concept="c", model_id="m", n_nodes=1, nodes=[node],
                interaction_matrix=None, node_types=["independent"],
                ablation_targets=[0], n_pairs=250)


def _synthetic_stream(L=4, n=60, d=16, signal=2.0, seed=0, sublayers=True):
    """Additive synthetic residual stream: x_l = x_{l-1} + a_l + m_l, with a
    concept offset written by attn at layer 1 and by mlp at layer 2."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls, sgn in (("pos", 1.0), ("neg", -1.0)):
        x = rng.normal(size=(n, d))
        emb = x.copy()
        streams, attns, mlps = [], [], []
        for l in range(L):
            a = 0.3 * rng.normal(size=(n, d))
            m = 0.3 * rng.normal(size=(n, d))
            if l == 1:
                a[:, 0] += sgn * signal
            if l == 2:
                m[:, 1] += sgn * signal
            x = x + a + m
            streams.append(x.copy()); attns.append(a); mlps.append(m)
        out[cls] = (emb, np.stack(streams), np.stack(attns), np.stack(mlps))
    return out


def _assert_preserved(raw: dict, d: dict, where: str = "") -> None:
    """Every key/value of the original v1 record survives a load -> dict
    round-trip. New v2 keys may appear, but only as None/defaults."""
    for k, v in raw.items():
        if k == "nodes":
            assert len(d["nodes"]) == len(v)
            for i, (rn, dn) in enumerate(zip(v, d["nodes"])):
                for nk, nv in rn.items():
                    assert dn[nk] == nv, f"{where} node {i} key {nk} changed"
                assert all(dn[nk] is None for nk in set(dn) - set(rn)), f"{where} node {i}"
        else:
            assert d[k] == v, f"{where} key {k} changed"
    assert all(d[k] is None for k in set(d) - set(raw) - {"schema_version"}), where


# ---------------------------------------------------------------------------
# Back-compat
# ---------------------------------------------------------------------------

class TestBackCompat:
    def test_v1_dict_loads_with_v2_fields_none(self):
        g = gem_from_dict(_v1_dict())
        assert g.schema_version == "1"
        assert g.layer_profile is None and g.sidecar is None
        assert g.nodes[0].settled_subspace is None
        assert g.extra == {"n_pairs": 250}
        assert g.nodes[0].concept_thread.cumulative_score == 6.0

    def test_v1_roundtrip_preserves_layout(self):
        raw = _v1_dict()
        d = gem_to_dict(gem_from_dict(raw))
        _assert_preserved(raw, d)
        assert d["n_pairs"] == 250 and "extra" not in d

    def test_does_not_mutate_input(self):
        raw = _v1_dict()
        before = json.dumps(raw, sort_keys=True)
        gem_from_dict(raw)
        assert json.dumps(raw, sort_keys=True) == before

    def test_unknown_node_key_kept_not_raised(self):
        raw = _v1_dict()
        raw["nodes"][0]["future_field"] = [1, 2]
        g = gem_from_dict(raw)
        assert g.nodes[0].extra == {"future_field": [1, 2]}
        assert gem_to_dict(g)["nodes"][0]["future_field"] == [1, 2]

    def test_save_writes_current_version(self, tmp_path):
        g = gem_from_dict(_v1_dict())
        written = save_gem(g, tmp_path / "gem_c.json")
        assert written.schema_version == GEM_SCHEMA_VERSION
        assert g.schema_version == "1"  # input not mutated
        assert load_gem(tmp_path / "gem_c.json").schema_version == GEM_SCHEMA_VERSION
        assert not (tmp_path / "gem_c.sidecar.npz").exists()

    @pytest.mark.parametrize("model", QWEN)
    def test_real_gem_roundtrip(self, model, tmp_path):
        p = MODELS / model / "gem_negation.json"
        if not p.exists():
            pytest.skip(f"{p} not present")
        raw = json.loads(p.read_text())
        g = load_gem(p)
        assert g.schema_version == "1"
        _assert_preserved(raw, gem_to_dict(g), model)
        out = tmp_path / "gem_negation.json"
        save_gem(g, out)
        g2 = load_gem(out)
        assert [n.settled_direction for n in g2.nodes] == [n.settled_direction for n in g.nodes]
        assert g2.extra == g.extra

    def test_all_real_qwen05_gems_load(self):
        d = MODELS / QWEN[0]
        files = sorted(d.glob("gem_*.json")) if d.exists() else []
        if not files:
            pytest.skip("no Qwen2.5-0.5B gem files")
        for p in files:
            g = load_gem(p)
            assert g.n_nodes == len(g.nodes)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_projected_separation_formula(self):
        rng = np.random.default_rng(1)
        pos = rng.normal(size=(50, 10)) + 1.0
        neg = rng.normal(size=(40, 10))
        s, raw, std, u = projected_separation(pos, neg)
        diff = pos.mean(0) - neg.mean(0)
        Sp, Sn = np.cov(pos.T, ddof=1), np.cov(neg.T, ddof=1)
        expected = np.linalg.norm(diff) / np.sqrt(0.5 * u @ (Sp + Sn) @ u)
        assert s == pytest.approx(expected, rel=1e-6)
        assert raw == pytest.approx(np.linalg.norm(diff))

    def test_projected_is_width_invariant_fisher_is_not(self):
        # Gauge effect: adding pure-noise dimensions shrinks the trace-based
        # Fisher score but leaves the projected score ~unchanged.
        rng = np.random.default_rng(2)
        base_p = rng.normal(size=(400, 4)); base_p[:, 0] += 2
        base_n = rng.normal(size=(400, 4))
        wide_p = np.hstack([base_p, rng.normal(size=(400, 200))])
        wide_n = np.hstack([base_n, rng.normal(size=(400, 200))])
        assert compute_separation(wide_p, wide_n) < 0.3 * compute_separation(base_p, base_n)
        s_base = projected_separation(base_p, base_n)[0]
        s_wide = projected_separation(wide_p, wide_n)[0]
        assert s_wide == pytest.approx(s_base, rel=0.35)

    def test_vectorized_null_matches_loop(self):
        rng = np.random.default_rng(3)
        pos, neg = rng.normal(size=(20, 6)), rng.normal(size=(15, 6))
        null_p, null_f = _permutation_nulls(pos, neg, 5, np.random.default_rng(7))
        X = np.concatenate([pos, neg])
        r = np.random.default_rng(7)
        for i in range(5):
            idx = r.permutation(len(X))
            m = np.zeros(len(X), bool); m[idx[:20]] = True
            assert null_f[i] == pytest.approx(compute_separation(X[m], X[~m]), rel=1e-6)
            assert null_p[i] == pytest.approx(projected_separation(X[m], X[~m])[0], rel=1e-6)

    def test_null_pvalues_signal_vs_none(self):
        s = _synthetic_stream(L=3, n=60, d=16, signal=2.0, sublayers=False)
        prof, _, _ = build_layer_profile(s["pos"][1], s["neg"][1], n_perm=99)
        assert prof.layers[0].null_projected.p_value > 0.01   # no concept yet at layer 0
        assert prof.layers[2].null_projected.p_value == pytest.approx(0.01)  # floor 1/(99+1)
        assert prof.layers[2].separation_fisher == pytest.approx(
            compute_separation(s["pos"][1][2], s["neg"][1][2]))


# ---------------------------------------------------------------------------
# Profile + sidecar
# ---------------------------------------------------------------------------

class TestProfile:
    def _profile(self, **kw):
        s = _synthetic_stream()
        (ep, sp, ap, mp), (en, sn, an, mn) = s["pos"], s["neg"]
        return build_layer_profile(sp, sn, ep, en, ap, an, mp, mn, n_perm=20, **kw)

    def test_every_layer_and_additivity(self):
        prof, arrays, _ = self._profile()
        assert len(prof.layers) == 4 and prof.embedding.layer == -1
        for rec in prof.layers:
            assert rec.recon_residual_max_rel_stream < 1e-10
            a, m = rec.sublayers
            assert (a.kind, m.kind) == ("attn", "mlp")
            assert a.share_of_concept_write_u_out + m.share_of_concept_write_u_out == pytest.approx(1.0)
            assert a.diff_proj_u_out + m.diff_proj_u_out == pytest.approx(rec.delta_diff_proj_u_out)
        # attn writes the concept at layer 1, mlp at layer 2
        assert prof.layers[1].sublayers[0].share_of_concept_write_u_out > 0.8
        assert prof.layers[2].sublayers[1].diff_proj_u_out > prof.layers[2].sublayers[0].diff_proj_u_out
        assert arrays["profile.proj_attn"].shape == (120, 4, 2)
        assert arrays["profile.null_projected"].shape == (5, 20)
        assert arrays["profile.attn_acts"].shape == (4, 120, 16)  # default on (D3)

    def test_stream_only_backfill(self):
        s = _synthetic_stream()
        prof, arrays, _ = build_layer_profile(s["pos"][1], s["neg"][1], n_perm=0)
        assert prof.embedding is None and prof.layers[0].delta_diff_proj_u_out is None
        assert prof.layers[1].delta_diff_proj_u_out is not None
        assert prof.layers[1].sublayers is None and prof.layers[1].null_projected is None
        assert "profile.proj_attn" not in arrays

    def test_save_load_sidecar(self, tmp_path):
        prof, arrays, docs = self._profile()
        gem = attach_layer_profile(gem_from_dict(_v1_dict(d=16)), prof)
        path = tmp_path / "gem_c.json"
        written = save_gem(gem, path, arrays=arrays, array_docs=docs)
        assert written.sidecar.path == "gem_c.sidecar.npz"
        assert written.sidecar.arrays["profile.directions"]["shape"] == [4, 16]
        assert "description" in written.sidecar.arrays["profile.directions"]
        raw = json.loads(path.read_text())
        assert "profile.directions" not in json.dumps(raw["layer_profile"])  # arrays not inline
        g2 = load_gem(path)
        assert g2.layer_profile.layers[2].sublayers[1].kind == "mlp"
        assert g2.layer_profile.layers[2].null_projected.p_value == prof.layers[2].null_projected.p_value
        assert g2.layer_profile.provenance.n_perm == 20
        assert gem_to_dict(g2) == gem_to_dict(written)
        with load_gem_sidecar(path) as sc:
            np.testing.assert_allclose(sc["profile.directions"], arrays["profile.directions"])
        # tamper -> sha mismatch
        with open(tmp_path / "gem_c.sidecar.npz", "ab") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="sha256"):
            load_gem_sidecar(path)

    def test_full_sublayer_acts_default_on_and_switchable(self):
        _, arrays, _ = self._profile()
        assert arrays["profile.attn_acts"].dtype == np.float16
        assert arrays["profile.mlp_acts"].shape == (4, 120, 16)
        _, arrays, _ = self._profile(store_full_sublayer_acts=False)
        assert "profile.attn_acts" not in arrays

    def test_final_norm_record(self):
        s = _synthetic_stream()
        rng = np.random.default_rng(5)
        fp = s["pos"][1][-1] / np.linalg.norm(s["pos"][1][-1], axis=1, keepdims=True)
        fn = s["neg"][1][-1] / np.linalg.norm(s["neg"][1][-1], axis=1, keepdims=True)
        prof, arrays, _ = self._profile(final_norm_pos=fp, final_norm_neg=fn)
        assert prof.final_norm.layer == 3 and prof.final_norm.sublayers is None
        assert prof.final_norm.separation_fisher == pytest.approx(compute_separation(fp, fn))
        assert prof.final_norm.null_projected.n_perm == 20
        assert arrays["profile.final_norm_direction"].shape == (16,)
        assert arrays["profile.final_norm_null_projected"].shape == (20,)
        g = gem_from_dict(json.loads(json.dumps(gem_to_dict(
            attach_layer_profile(gem_from_dict(_v1_dict(d=16)), prof)))))
        assert g.layer_profile.final_norm.separation_projected == prof.final_norm.separation_projected
        del rng

    def test_inline_directions_option(self, tmp_path):
        prof, arrays, docs = self._profile()
        gem = attach_layer_profile(gem_from_dict(_v1_dict(d=16)), prof)
        save_gem(gem, tmp_path / "a.json", arrays, docs)
        assert load_gem(tmp_path / "a.json").layer_profile.layers[0].direction is None
        save_gem(gem, tmp_path / "b.json", arrays, docs, inline_directions=True)
        g = load_gem(tmp_path / "b.json")
        np.testing.assert_allclose(g.layer_profile.layers[2].direction,
                                   arrays["profile.directions"][2], atol=1e-6)
        with pytest.raises(ValueError):
            save_gem(gem_from_dict(_v1_dict()), tmp_path / "c.json", inline_directions=True)

    def test_decompose_settled_subspace_identity(self):
        prof, arrays, _ = self._profile()
        rng = np.random.default_rng(0)
        W, _ = np.linalg.qr(rng.normal(size=(16, 3)))
        sub = SettledSubspace(layer=2, directions=W.T.tolist(), k_effective=3)
        sub2 = decompose_settled_subspace(sub, arrays)
        dec = sub2.sublayer_decomposition
        assert sub.sublayer_decomposition is None  # non-mutating
        assert np.max(np.abs(dec.reconstruction_residual)) < 1e-4  # float32 sidecar means
        assert len(dec.upstream_attn_diff_proj[0]) == 3
        assert dec.attn_diff_proj == [r[2] for r in dec.upstream_attn_diff_proj]
        # round-trips through a node
        node = replace(gem_from_dict(_v1_dict(d=16)).nodes[0], settled_subspace=sub2)
        g = replace(gem_from_dict(_v1_dict(d=16)), nodes=[node])
        g2 = gem_from_dict(json.loads(json.dumps(gem_to_dict(g))))
        assert g2.nodes[0].settled_subspace.sublayer_decomposition.layer == 2

    def test_attach_layer_settled_subspaces(self):
        prof, _, _ = self._profile()
        s = _synthetic_stream()
        hp = {2: s["pos"][1][2]}; hn = {2: s["neg"][1][2]}
        prof2 = attach_layer_settled_subspaces(prof, hp, hn, layers=[2], max_rounds=2, n_perm=5)
        assert prof.layers[2].settled_subspace is None
        sub = prof2.layers[2].settled_subspace
        assert sub.layer == 2 and sub.k_effective >= 1 and sub.provenance is not None


@pytest.mark.parametrize("model", ["EleutherAI_pythia_70m"])
def test_backfill_matches_existing_caz(model):
    """Stream-only profile from calibration_alllayer reproduces caz_*.json
    separation_fisher and dom_vector (same data, same formula)."""
    d = MODELS / model
    cal, caz = d / "calibration_alllayer_negation.npy", d / "caz_negation.json"
    if not (cal.exists() and caz.exists()):
        pytest.skip("pythia-70m negation calibration_alllayer not present")
    a = np.load(cal, mmap_mode="r")
    c = json.loads(caz.read_text())
    n = c["n_pairs"]
    prof, arrays, _ = build_layer_profile(a[:, :n], a[:, n:], n_perm=0, stream_source="hidden_states")
    for rec, m in zip(prof.layers, c["layer_data"]["metrics"]):
        assert rec.separation_fisher == pytest.approx(m["separation_fisher"], rel=1e-5)
        np.testing.assert_allclose(arrays["profile.directions"][rec.layer], m["dom_vector"], atol=1e-5)


# ---------------------------------------------------------------------------
# Config + parallel (projected) CAZ detection path
# ---------------------------------------------------------------------------

def _caz_and_profile(L=12, d=16, seed=0):
    """A caz_*.json-shaped dict and a stream-only profile for the same data,
    with one bump in separation at layer ~5."""
    from rosetta_tools.caz import compute_coherence, compute_velocity
    rng = np.random.default_rng(seed)
    n = 80
    amp = np.exp(-0.5 * ((np.arange(L) - 5) / 1.5) ** 2) * 3 + 0.2
    pos = rng.normal(size=(L, n, d)); neg = rng.normal(size=(L, n, d))
    pos[:, :, 0] += amp[:, None]
    seps = [compute_separation(pos[l], neg[l]) for l in range(L)]
    vel = compute_velocity(seps, window=3)
    metrics = []
    for l in range(L):
        dv = pos[l].mean(0) - neg[l].mean(0)
        metrics.append(dict(layer=l, separation_fisher=seps[l], coherence=compute_coherence(pos[l], neg[l]),
                            raw_distance=float(np.linalg.norm(dv)), dom_vector=(dv / np.linalg.norm(dv)).tolist(),
                            velocity=float(vel[l])))
    caz = dict(model_id="m", concept="c", n_pairs=n, layer_data=dict(n_layers=L, metrics=metrics))
    prof, _, _ = build_layer_profile(pos, neg, n_perm=0, stream_source="hidden_states")
    return caz, prof


class TestConfigAndDetection:
    def test_config_defaults_and_legacy(self):
        from rosetta_tools.gem import GEMExtractionConfig
        c = GEMExtractionConfig()
        assert c.layer_profile and c.sublayer_capture and c.store_full_sublayer_activations
        assert c.store_post_final_norm and not c.inline_directions
        assert c.n_permutations == 200 and c.detection_score == "fisher"
        lg = GEMExtractionConfig.legacy()
        assert not (lg.layer_profile or lg.sublayer_capture or lg.store_full_sublayer_activations
                    or lg.store_post_final_norm)

    def test_default_build_unchanged(self):
        from rosetta_tools.gem import build_concept_gem
        caz, prof = _caz_and_profile()
        a = gem_to_dict(build_concept_gem(caz))
        b = gem_to_dict(build_concept_gem(caz, layer_profile=prof))
        b.pop("layer_profile"); a.pop("layer_profile")
        assert a == b and a["caz_detection_score"] == "fisher"

    def test_projected_detection_path(self):
        from rosetta_tools.gem import build_concept_gem, find_caz_regions_projected, projected_layer_metrics
        caz, prof = _caz_and_profile()
        lm = projected_layer_metrics(caz, prof)
        assert [m.separation for m in lm] == [r.separation_projected for r in prof.layers]
        regions = find_caz_regions_projected(caz, prof)
        assert regions.n_regions >= 1
        assert any(r.start <= 5 <= r.end for r in regions.regions)
        g = build_concept_gem(caz, layer_profile=prof, detection_score="projected")
        assert g.caz_detection_score == "projected" and g.n_nodes == regions.n_regions
        with pytest.raises(ValueError):
            build_concept_gem(caz, detection_score="projected")
        with pytest.raises(ValueError):
            build_concept_gem(caz, detection_score="bogus")


# ---------------------------------------------------------------------------
# Pair-set provenance (2026-10-15)
# ---------------------------------------------------------------------------

class TestPairSetProvenance:
    def _pairs(self, n=4):
        from rosetta_tools.dataset import ConceptPair
        return [ConceptPair(pair_id=f"p{i}", domain="d", topic="t",
                            pos_text=f"pos {i}", neg_text=f"neg {i}") for i in range(n)]

    def test_hash_sensitive_to_membership_order_and_text(self):
        from rosetta_tools.dataset import pair_set_provenance
        ps = self._pairs()
        h = pair_set_provenance(ps, "c")["sha256"]
        assert pair_set_provenance(self._pairs(), "c")["sha256"] == h
        assert pair_set_provenance(ps[::-1], "c")["sha256"] != h
        assert pair_set_provenance(ps[:3], "c")["sha256"] != h
        alt = self._pairs(); alt[0].pos_text = "pos 0!"
        assert pair_set_provenance(alt, "c")["sha256"] != h
        d = pair_set_provenance(ps, "c", "train", 250)
        assert d["n_pairs"] == 4 and d["n_requested"] == 250 and d["split"] == "train"
        json.dumps(d)  # JSON-safe

    def test_provenance_roundtrip_and_backcompat(self):
        from rosetta_tools.dataset import pair_set_provenance
        from rosetta_tools.gem import Provenance
        s = _synthetic_stream()
        ps = pair_set_provenance(self._pairs(), "c")
        prof, _, _ = build_layer_profile(s["pos"][1], s["neg"][1], n_perm=0,
                                         provenance=Provenance(method="t", pair_set=ps))
        g = attach_layer_profile(gem_from_dict(_v1_dict(d=16)), prof)
        g2 = gem_from_dict(json.loads(json.dumps(gem_to_dict(g))))
        assert g2.layer_profile.provenance.pair_set == ps
        # a v2 provenance dict written before pair_set existed still loads
        d = gem_to_dict(g)
        del d["layer_profile"]["provenance"]["pair_set"]
        assert gem_from_dict(d).layer_profile.provenance.pair_set is None

    def test_real_dataset_records_commit(self):
        from rosetta_tools.dataset import load_concept_pairs, pair_set_provenance
        try:
            pairs = load_concept_pairs("negation", n=5, split="train")
        except FileNotFoundError:
            pytest.skip("Rosetta_Concept_Pairs not present")
        d = pair_set_provenance(pairs, "negation", "train", 5)
        assert d["pairs_file"].endswith("negation_consensus_pairs.jsonl")
        assert d["repo_commit"] is None or len(d["repo_commit"]) == 40

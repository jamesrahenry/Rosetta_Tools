"""
test_gem_phase2.py — Tests for the v2.1.0 Phase-2 GEM paths in rosetta_tools.gem:
extract_phase2_threads (k>1 eigen-threads) and the INLP settled subspace
(extract_settled_subspace / attach_settled_subspace).

Synthetic data only; no model loading. The INLP tests pin the *documented*
behaviour: pure noise -> k_effective 0; a planted rank-1 shift -> k_effective >= 1
with the first direction aligned to the planted axis. They deliberately do NOT
assert k_effective == 1 on rank-1 data, because the procedure is an upper
bound (see the extract_settled_subspace docstring).
"""

import numpy as np
import pytest

from rosetta_tools.gem import (
    EigThread,
    GEMNode,
    SettledSubspace,
    _topk_eigvecs,
    attach_settled_subspace,
    extract_phase2_threads,
    extract_settled_subspace,
)


def _planted(rng, n_per_class, d, shift, axis=0):
    Xp = rng.standard_normal((n_per_class, d))
    Xn = rng.standard_normal((n_per_class, d))
    Xp[:, axis] += shift
    return Xp, Xn


def _split(Xp, Xn, frac=0.3, rng=None):
    rng = rng or np.random.default_rng(0)
    n = Xp.shape[0]
    idx = rng.permutation(n)
    n_te = int(n * frac)
    te, tr = idx[:n_te], idx[n_te:]
    Xtr = np.concatenate([Xp[tr], Xn[tr]])
    ytr = np.concatenate([np.ones(len(tr)), np.zeros(len(tr))])
    Xte = np.concatenate([Xp[te], Xn[te]])
    yte = np.concatenate([np.ones(len(te)), np.zeros(len(te))])
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# _topk_eigvecs / extract_phase2_threads
# ---------------------------------------------------------------------------


class TestTopkEigvecs:
    def test_recovers_planted_axis(self):
        rng = np.random.default_rng(1)
        Xp, Xn = _planted(rng, 200, 32, shift=4.0, axis=3)
        vecs, vals = _topk_eigvecs(Xp, Xn, k=2)
        assert vecs.shape == (2, 32) and vals.shape == (2,)
        assert abs(vecs[0, 3]) > 0.9
        assert vals[0] > vals[1]

    def test_k_is_clamped_to_rank(self):
        rng = np.random.default_rng(2)
        Xp, Xn = _planted(rng, 4, 16, shift=2.0)  # only 4 difference rows
        vecs, vals = _topk_eigvecs(Xp, Xn, k=10)
        assert vecs.shape[0] == 4 and vals.shape == (4,)

    def test_unpaired_shapes_raise(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError):
            _topk_eigvecs(rng.standard_normal((5, 8)), rng.standard_normal((6, 8)), k=1)


class TestExtractPhase2Threads:
    @pytest.fixture
    def layers(self):
        rng = np.random.default_rng(4)
        layer_indices = [3, 4, 5, 6]
        hp, hn, doms = {}, {}, {}
        # A planted axis that rotates slowly from e0 toward e1 across layers.
        for i, L in enumerate(layer_indices):
            theta = 0.15 * i
            axis = np.zeros(24)
            axis[0], axis[1] = np.cos(theta), np.sin(theta)
            Xp = rng.standard_normal((150, 24))
            Xn = rng.standard_normal((150, 24))
            Xp += 4.0 * axis
            hp[L], hn[L], doms[L] = Xp, Xn, axis
        return layer_indices, hp, hn, doms

    def test_threads_are_well_formed(self, layers):
        layer_indices, hp, hn, doms = layers
        threads, best = extract_phase2_threads(hp, hn, layer_indices, k=3, dom_vectors_by_layer=doms)
        assert len(threads) == 3
        assert 0 <= best < 3
        for th in threads:
            assert isinstance(th, EigThread)
            assert th.layer_indices == layer_indices
            assert len(th.directions) == len(th.eigenvalues) == len(th.cos_chain) == len(layer_indices)
            assert len(th.concept_alignments) == len(layer_indices)
            assert th.cos_chain[0] == 1.0
            for d in th.directions:
                assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-6)

    def test_best_thread_tracks_the_planted_axis(self, layers):
        layer_indices, hp, hn, doms = layers
        threads, best = extract_phase2_threads(hp, hn, layer_indices, k=3, dom_vectors_by_layer=doms)
        th = threads[best]
        assert th.mean_alignment > 0.9
        assert min(th.cos_chain) > 0.9  # sign-aligned chain, slow rotation

    def test_no_dom_vectors_gives_unit_alignment(self, layers):
        layer_indices, hp, hn, _ = layers
        threads, _ = extract_phase2_threads(hp, hn, layer_indices, k=2)
        for th in threads:
            assert th.concept_alignments == [1.0] * len(layer_indices)

    def test_k_larger_than_n_does_not_crash(self):
        rng = np.random.default_rng(5)
        hp = {0: rng.standard_normal((3, 10)), 1: rng.standard_normal((3, 10))}
        hn = {0: rng.standard_normal((3, 10)), 1: rng.standard_normal((3, 10))}
        threads, best = extract_phase2_threads(hp, hn, [0, 1], k=8)
        assert len(threads) == 3
        assert 0 <= best < 3


# ---------------------------------------------------------------------------
# INLP settled subspace
# ---------------------------------------------------------------------------


class TestExtractSettledSubspace:
    def test_pure_noise_is_exhausted_immediately(self):
        rng = np.random.default_rng(6)
        Xp = rng.standard_normal((120, 40))
        Xn = rng.standard_normal((120, 40))
        Xtr, ytr, Xte, yte = _split(Xp, Xn)
        sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=7, max_rounds=4, n_perm=20)
        assert isinstance(sub, SettledSubspace)
        assert sub.layer == 7
        assert sub.k_effective == 0
        assert len(sub.held_out_acc_curve) >= 1

    def test_planted_rank1_signal_is_found_and_aligned(self):
        rng = np.random.default_rng(7)
        Xp, Xn = _planted(rng, 200, 40, shift=3.0, axis=5)
        Xtr, ytr, Xte, yte = _split(Xp, Xn)
        sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=3, max_rounds=5, n_perm=20)
        assert sub.k_effective >= 1
        w0 = np.asarray(sub.directions[0], dtype=float)
        assert abs(w0[5]) / np.linalg.norm(w0) > 0.5
        assert sub.held_out_acc_curve[0] > 0.8
        # upper-bound caveat: must not run away to max_rounds on a rank-1 signal
        assert sub.k_effective < 5

    def test_directions_are_orthonormal(self):
        rng = np.random.default_rng(8)
        Xp, Xn = _planted(rng, 200, 40, shift=3.0, axis=0)
        Xp[:, 1] += 3.0  # rank-2 planted
        Xtr, ytr, Xte, yte = _split(Xp, Xn)
        sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=0, max_rounds=5, n_perm=20)
        W = np.asarray(sub.directions, dtype=float)
        if W.shape[0] >= 2:
            G = W @ W.T
            assert np.allclose(G, np.eye(W.shape[0]), atol=1e-6)

    def test_rounds_bounded_by_max_rounds(self):
        rng = np.random.default_rng(9)
        Xp, Xn = _planted(rng, 200, 40, shift=5.0)
        Xp[:, 1:6] += 5.0
        Xtr, ytr, Xte, yte = _split(Xp, Xn)
        sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=0, max_rounds=2, n_perm=10)
        assert sub.k_effective <= 2
        assert len(sub.directions) <= 2


class TestAttachSettledSubspace:
    def _node(self):
        return GEMNode(
            concept="c", model_id="m", caz_index=0, caz_start=0, caz_peak=2, caz_end=4,
            caz_score=0.5, attention_paradigm="mha", settled_direction=[1.0, 0.0],
            handoff_layer=5, handoff_cosine=1.0, entry_exit_cosine=0.9,
            max_rotation_per_layer=0.1, hidden_dim=40, n_layers_total=12, k=1,
            concept_thread_id=0, threads=[],
        )

    def test_attach_is_functional_and_uses_handoff_layer(self):
        rng = np.random.default_rng(10)
        Xp, Xn = _planted(rng, 120, 40, shift=3.0)
        node = self._node()
        new = attach_settled_subspace(node, Xp, Xn, max_rounds=2, n_perm=10)
        assert node.settled_subspace is None
        assert new.settled_subspace is not None
        assert new.settled_subspace.layer == 5

    def test_unequal_class_sizes_are_accepted(self):
        """Regression: labels were built from h_pos.shape[0] for both classes
        and raised ValueError whenever n_pos != n_neg."""
        rng = np.random.default_rng(11)
        Xp = rng.standard_normal((90, 40)) + 3.0 * np.eye(40)[0]
        Xn = rng.standard_normal((130, 40))
        new = attach_settled_subspace(self._node(), Xp, Xn, max_rounds=2, n_perm=10)
        assert new.settled_subspace is not None
        assert new.settled_subspace.k_effective >= 1

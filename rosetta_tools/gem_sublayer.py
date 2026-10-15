"""
gem_sublayer.py — per-layer and per-sublayer (attention vs MLP) GEM research
records (schema v2). Design: Notes/gem_schema_redesign_2026-10-15.md.

Three layers of functionality, cheapest first:

  1. build_layer_profile (numpy only): per-layer records at EVERY layer from
     stream activations -- separation under the existing Fisher score and
     the gauge-corrected projected score, each with a label-permutation null.
     Works on existing calibration_alllayer_*.npy (stream-only backfill), or
     additionally on sublayer captures for the attention/MLP split.
  2. capture_sublayer_activations (torch, lazy import): forward hooks on each
     decoder block and its attention/MLP outputs, pooled per example.
     Checks Delta == attn + mlp numerically and RAISES when it fails (unless
     allow_inexact=True, in which case the residual is recorded and flagged).
  3. decompose_settled_subspace / attach_layer_settled_subspaces: optional
     extras on top of a profile, functional (dataclasses.replace) like
     attach_settled_subspace in gem.py.

Layer convention (same as gem.py / extract_layer_wise_metrics): GEM layer l
is the output of decoder block l. Block l reads GEM layer l-1 (embedding
output for l = 0) and writes GEM layer l. u_in / u_out are the unit
class-mean-difference directions of those two streams.

Additivity, per architecture family (hooks on what is actually ADDED):
  pre-LN (Llama/Qwen/Mistral/GPT-2/OPT pre-LN):   x_out = x_in + a + m exactly
  Gemma2/3 post-sublayer norms:  a, m hooked AFTER post_attention_layernorm /
                                 post_feedforward_layernorm -> still exact
  parallel blocks (GPT-J, NeoX use_parallel_residual, Phi): exact, but the
                                 MLP reads x_in, not x_in + a (flagged)
  post-LN (OPT-350m, BERT-style): NOT additive -> refused unless allow_inexact
Captures use decoder-block outputs, NOT output_hidden_states: in most HF
models hidden_states[-1] is post-final-norm, which would break additivity at
the last layer (and differs from existing CAZ data at that one layer).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rosetta_tools.caz import compute_separation
from rosetta_tools.gem import (
    ArchitectureFlags,
    ConceptGEM,
    GEMExtractionConfig,
    LayerProfile,
    LayerRecord,
    PermutationNull,
    Provenance,
    SettledSubspace,
    SublayerRecord,
    SubspaceSublayerDecomposition,
)

log = logging.getLogger(__name__)

_EPS = 1e-8

# Tolerance on max ||Delta - a - m|| / ||x_out|| by forward dtype. Relative to
# the STREAM norm, not the delta norm: in half precision each residual add
# rounds at ~eps * ||x||, and ||Delta|| << ||x|| deep in the network, so a
# delta-relative test would fail on rounding alone. Post-LN violations are
# O(1) and are caught at any of these thresholds.
ADDITIVITY_TOL = {
    "float64": 1e-8,
    "float32": 1e-4,
    "float16": 5e-3,
    "bfloat16": 2e-2,
}

# Sidecar index documentation (axes + description) for keys produced here.
# n = n_pos + n_neg, rows positive then negative (same layout as
# calibration_*.npy); L = n_layers; row l = GEM layer l.
SIDECAR_DOCS: dict[str, dict] = {
    "profile.labels": {"axes": "example", "description": "1 = positive, 0 = negative"},
    "profile.embed_mean_pos": {"axes": "hidden", "description": "embedding-output mean, positive class"},
    "profile.embed_mean_neg": {"axes": "hidden", "description": "embedding-output mean, negative class"},
    "profile.stream_mean_pos": {"axes": "layer, hidden", "description": "block-output (stream) mean, positive class"},
    "profile.stream_mean_neg": {"axes": "layer, hidden", "description": "block-output (stream) mean, negative class"},
    "profile.directions": {"axes": "layer, hidden", "description": "u_l = unit (stream_mean_pos - stream_mean_neg)"},
    "profile.attn_mean_pos": {"axes": "layer, hidden", "description": "attention write (as added to stream) mean, positive"},
    "profile.attn_mean_neg": {"axes": "layer, hidden", "description": "attention write mean, negative"},
    "profile.mlp_mean_pos": {"axes": "layer, hidden", "description": "MLP write (as added to stream) mean, positive"},
    "profile.mlp_mean_neg": {"axes": "layer, hidden", "description": "MLP write mean, negative"},
    "profile.proj_stream_u_out": {"axes": "example, layer", "description": "u_l . x_l per example"},
    "profile.proj_attn": {"axes": "example, layer, (u_in, u_out)", "description": "attention write projected onto u_in and u_out"},
    "profile.proj_mlp": {"axes": "example, layer, (u_in, u_out)", "description": "MLP write projected onto u_in and u_out"},
    "profile.proj_delta": {"axes": "example, layer, (u_in, u_out)", "description": "block write x_out - x_in projected onto u_in and u_out"},
    "profile.norm_attn": {"axes": "example, layer", "description": "||attention write||"},
    "profile.norm_mlp": {"axes": "example, layer", "description": "||MLP write||"},
    "profile.norm_delta": {"axes": "example, layer", "description": "||x_out - x_in||"},
    "profile.norm_stream_in": {"axes": "example, layer", "description": "||x_in|| (stream block reads)"},
    "profile.recon_residual_rel_stream": {"axes": "example, layer", "description": "||Delta - a - m|| / ||x_out||"},
    "profile.null_projected": {"axes": "layer+1 (row 0 = embedding), perm", "description": "permutation null samples, separation_projected"},
    "profile.null_fisher": {"axes": "layer+1 (row 0 = embedding), perm", "description": "permutation null samples, separation_fisher"},
    "profile.attn_acts": {"axes": "layer, example, hidden", "description": "raw pooled attention writes (float16; store_full_sublayer_activations)"},
    "profile.mlp_acts": {"axes": "layer, example, hidden", "description": "raw pooled MLP writes (float16; store_full_sublayer_activations)"},
    "profile.final_norm_mean_pos": {"axes": "hidden", "description": "post-final-norm stream (hidden_states[-1]) mean, positive"},
    "profile.final_norm_mean_neg": {"axes": "hidden", "description": "post-final-norm stream mean, negative"},
    "profile.final_norm_direction": {"axes": "hidden", "description": "unit mean-diff of the post-final-norm stream"},
    "profile.proj_final_norm": {"axes": "example", "description": "per-example projection of post-final-norm stream onto its direction"},
    "profile.final_norm_null_projected": {"axes": "perm", "description": "permutation null samples, separation_projected, post-final-norm"},
    "profile.final_norm_null_fisher": {"axes": "perm", "description": "permutation null samples, separation_fisher, post-final-norm"},
}


def code_version() -> str:
    """rosetta_tools version plus short git sha (and +dirty) when resolvable."""
    from rosetta_tools import __version__
    ver = f"rosetta_tools {__version__}"
    try:
        repo = Path(__file__).resolve().parent
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if sha:
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "."],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            ver += f" ({sha}{'+dirty' if dirty else ''})"
    except Exception:
        pass
    return ver


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# 1. Statistics (numpy only)
# ---------------------------------------------------------------------------

def projected_separation(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Gauge-corrected separation (Freenor, 2026-10-14):

        S_proj = ||mu+ - mu-|| / sqrt( 1/2 u^T (S+ + S-) u ),  u = (mu+ - mu-)/||mu+ - mu-||

    i.e. the centroid gap relative to the within-class spread ALONG the
    concept direction, instead of the total trace (compute_separation),
    which grows with width d regardless of the concept. ddof=1, matching
    compute_separation. Returns (S_proj, raw_distance, pooled_std, u)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    diff = pos.mean(0) - neg.mean(0)
    raw = float(np.linalg.norm(diff))
    u = diff / raw if raw > 0 else diff
    var = 0.5 * (np.var(pos @ u, ddof=1) + np.var(neg @ u, ddof=1))
    std = float(np.sqrt(var))
    return raw / (std + _EPS), raw, std, u


def _permutation_nulls(
    pos: np.ndarray, neg: np.ndarray, n_perm: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized label-permutation nulls for (projected, fisher) separation.
    Each permutation refits the mean-difference direction. Returns two
    (n_perm,) arrays."""
    X = np.concatenate([pos, neg], 0).astype(np.float64)
    n, n_pos = X.shape[0], pos.shape[0]
    n_neg = n - n_pos
    B = np.zeros((n_perm, n))
    for i in range(n_perm):
        B[i, rng.permutation(n)[:n_pos]] = 1.0

    total = X.sum(0)
    sum_pos = B @ X                                         # (P, d)
    mu_pos = sum_pos / n_pos
    mu_neg = (total - sum_pos) / n_neg
    diff = mu_pos - mu_neg
    raw = np.linalg.norm(diff, axis=1)                      # (P,)

    # Fisher: tr(S_c) = (sum ||x||^2 - n_c ||mu_c||^2) / (n_c - 1)
    sq = (X ** 2).sum(1)
    sq_pos = B @ sq
    tr_pos = (sq_pos - n_pos * (mu_pos ** 2).sum(1)) / (n_pos - 1)
    tr_neg = (sq.sum() - sq_pos - n_neg * (mu_neg ** 2).sum(1)) / (n_neg - 1)
    fisher = raw / (np.sqrt(np.clip(0.5 * (tr_pos + tr_neg), 0, None)) + _EPS)

    # Projected: within-class variance of X @ u_perm
    U = diff / np.where(raw > 0, raw, 1.0)[:, None]
    proj = X @ U.T                                          # (n, P)
    Bt = B.T
    s_pos = (proj * Bt).sum(0)
    ss_pos = (proj ** 2 * Bt).sum(0)
    s_all, ss_all = proj.sum(0), (proj ** 2).sum(0)
    var_pos = (ss_pos - s_pos ** 2 / n_pos) / (n_pos - 1)
    var_neg = ((ss_all - ss_pos) - (s_all - s_pos) ** 2 / n_neg) / (n_neg - 1)
    projected = raw / (np.sqrt(np.clip(0.5 * (var_pos + var_neg), 0, None)) + _EPS)
    return projected, fisher


def _null_summary(statistic: str, samples: np.ndarray, observed: float, seed: int) -> PermutationNull:
    return PermutationNull(
        statistic=statistic,
        n_perm=int(len(samples)),
        seed=int(seed),
        mean=float(np.mean(samples)),
        std=float(np.std(samples)),
        q95=float(np.quantile(samples, 0.95)),
        max=float(np.max(samples)),
        p_value=float((np.sum(samples >= observed) + 1) / (len(samples) + 1)),
    )


def _check_finite(name: str, *arrs: np.ndarray | None) -> None:
    for a in arrs:
        if a is not None and not np.all(np.isfinite(a)):
            raise ValueError(
                f"build_layer_profile: non-finite values in {name} (fp16 overflow?); "
                "refusing to compute statistics on them"
            )


def _stream_record(
    layer: int, pos: np.ndarray, neg: np.ndarray, n_perm: int, seed: int,
) -> tuple[LayerRecord, np.ndarray, np.ndarray | None, np.ndarray | None]:
    sep_proj, raw, std, u = projected_separation(pos, neg)
    rec = LayerRecord(
        layer=layer,
        separation_fisher=float(compute_separation(pos, neg)),
        separation_projected=float(sep_proj),
        raw_distance=raw,
        projected_pooled_std=std,
        stream_norm_mean=float(np.mean(np.linalg.norm(np.concatenate([pos, neg], 0), axis=1))),
        sidecar_row=layer if layer >= 0 else None,
    )
    null_p = null_f = None
    if n_perm > 0:
        # Per-layer seed so a layer's null is reproducible on its own.
        rng = np.random.default_rng([seed, layer + 1])
        null_p, null_f = _permutation_nulls(pos, neg, n_perm, rng)
        rec.null_projected = _null_summary("separation_projected", null_p, sep_proj, seed)
        rec.null_fisher = _null_summary("separation_fisher", null_f, rec.separation_fisher, seed)
    return rec, u, null_p, null_f


def _sublayer_record(
    kind: str, hook: str, sp: np.ndarray, sn: np.ndarray,
    u_in: np.ndarray | None, u_out: np.ndarray, delta_diff_u_out: float | None,
    norm_in: np.ndarray | None, norm_delta: np.ndarray | None,
) -> tuple[SublayerRecord, np.ndarray, np.ndarray]:
    """Returns (record, per-example proj [n,2] (u_in,u_out), per-example norm [n])."""
    S = np.concatenate([sp, sn], 0)
    n_pos = sp.shape[0]
    dmu = sp.mean(0) - sn.mean(0)
    p_out = S @ u_out
    p_in = S @ u_in if u_in is not None else np.full(S.shape[0], np.nan)
    norms = np.linalg.norm(S, axis=1)
    var_out = 0.5 * (np.var(p_out[:n_pos], ddof=1) + np.var(p_out[n_pos:], ddof=1))
    diff_out = float(dmu @ u_out)
    dmu_norm = float(np.linalg.norm(dmu))
    rec = SublayerRecord(
        kind=kind,
        hook_point=hook,
        diff_proj_u_in=float(dmu @ u_in) if u_in is not None else None,
        diff_proj_u_out=diff_out,
        pos_mean_proj_u_in=float(np.mean(p_in[:n_pos])) if u_in is not None else None,
        neg_mean_proj_u_in=float(np.mean(p_in[n_pos:])) if u_in is not None else None,
        pos_mean_proj_u_out=float(np.mean(p_out[:n_pos])),
        neg_mean_proj_u_out=float(np.mean(p_out[n_pos:])),
        var_proj_u_out=float(var_out),
        separation_projected_u_out=float(diff_out / (np.sqrt(var_out) + _EPS)),
        share_of_concept_write_u_out=(
            float(diff_out / delta_diff_u_out)
            if delta_diff_u_out is not None and abs(delta_diff_u_out) > _EPS else None
        ),
        mean_diff_norm=dmu_norm,
        cos_mean_diff_u_out=float(diff_out / dmu_norm) if dmu_norm > 0 else None,
        norm_mean=float(np.mean(norms)),
        norm_frac_stream_mean=float(np.mean(norms / np.maximum(norm_in, _EPS))) if norm_in is not None else None,
        norm_frac_delta_mean=float(np.mean(norms / np.maximum(norm_delta, _EPS))) if norm_delta is not None else None,
    )
    return rec, np.stack([p_in, p_out], 1), norms


def build_layer_profile(
    stream_pos: np.ndarray,
    stream_neg: np.ndarray,
    embed_pos: np.ndarray | None = None,
    embed_neg: np.ndarray | None = None,
    attn_pos: np.ndarray | None = None,
    attn_neg: np.ndarray | None = None,
    mlp_pos: np.ndarray | None = None,
    mlp_neg: np.ndarray | None = None,
    n_perm: int = 200,
    seed: int = 0,
    architecture: ArchitectureFlags | None = None,
    provenance: Provenance | None = None,
    stream_source: str = "decoder_layer_output",
    store_full_sublayer_acts: bool = True,
    final_norm_pos: np.ndarray | None = None,
    final_norm_neg: np.ndarray | None = None,
) -> tuple[LayerProfile, dict[str, np.ndarray], dict[str, dict]]:
    """Per-layer records at every layer (numpy only, no model needed).

    stream_pos/neg : (L, n_pos|n_neg, d) GEM-layer activations (row l =
        output of block l). E.g. split calibration_alllayer_{concept}.npy
        (stream_source="hidden_states" in that case -- see the final-layer
        caveat in the module docstring).
    embed_pos/neg : (n, d) embedding output (block 0's input). Optional; if
        absent, block 0 has no u_in and no Delta.
    attn_*/mlp_* : (L, n, d) sublayer writes as added to the stream (from
        capture_sublayer_activations). Optional; both or neither.
    n_perm : label permutations per layer (0 disables the nulls).
    store_full_sublayer_acts : put the raw attn/mlp writes in the sidecar
        too (float16). Default on (James, 2026-10-15); only applies when
        sublayers are given.
    final_norm_pos/neg : (n, d) post-final-norm stream at the last layer
        (hidden_states[-1]); recorded as profile.final_norm alongside the raw
        block output in layers[-1].

    Returns (profile, arrays, array_docs); pass arrays/array_docs to
    save_gem(..., arrays=, array_docs=) so they land in the sidecar.
    """
    if (attn_pos is None) != (mlp_pos is None):
        raise ValueError("pass both attn_* and mlp_* or neither")
    stream_pos = np.asarray(stream_pos, dtype=np.float64)
    stream_neg = np.asarray(stream_neg, dtype=np.float64)
    L, n_pos, d = stream_pos.shape
    n_neg = stream_neg.shape[1]
    have_embed = embed_pos is not None and embed_neg is not None
    have_sub = attn_pos is not None
    _check_finite("stream", stream_pos, stream_neg)
    _check_finite("embed", embed_pos, embed_neg)
    _check_finite("sublayers", attn_pos, attn_neg, mlp_pos, mlp_neg)
    arch = architecture or ArchitectureFlags()
    n = n_pos + n_neg

    arrays: dict[str, np.ndarray] = {
        "profile.labels": np.concatenate([np.ones(n_pos, np.int8), np.zeros(n_neg, np.int8)]),
        "profile.stream_mean_pos": stream_pos.mean(1).astype(np.float32),
        "profile.stream_mean_neg": stream_neg.mean(1).astype(np.float32),
    }
    directions = np.zeros((L, d))
    proj_stream = np.zeros((n, L))
    null_p_rows, null_f_rows = [], []

    embedding_rec = None
    u_emb = None
    if have_embed:
        ep = np.asarray(embed_pos, dtype=np.float64)
        en = np.asarray(embed_neg, dtype=np.float64)
        embedding_rec, u_emb, np_, nf_ = _stream_record(-1, ep, en, n_perm, seed)
        arrays["profile.embed_mean_pos"] = ep.mean(0).astype(np.float32)
        arrays["profile.embed_mean_neg"] = en.mean(0).astype(np.float32)
    if n_perm > 0:
        null_p_rows.append(np_ if have_embed else np.full(n_perm, np.nan))
        null_f_rows.append(nf_ if have_embed else np.full(n_perm, np.nan))

    if have_sub:
        proj_attn = np.full((n, L, 2), np.nan)
        proj_mlp = np.full((n, L, 2), np.nan)
        norm_attn = np.zeros((n, L))
        norm_mlp = np.zeros((n, L))
        recon_rel = np.full((n, L), np.nan)
        for k, a in (("attn_mean_pos", attn_pos), ("attn_mean_neg", attn_neg),
                     ("mlp_mean_pos", mlp_pos), ("mlp_mean_neg", mlp_neg)):
            arrays[f"profile.{k}"] = np.asarray(a, dtype=np.float64).mean(1).astype(np.float32)
        if store_full_sublayer_acts:
            arrays["profile.attn_acts"] = np.concatenate([attn_pos, attn_neg], 1).astype(np.float16)
            arrays["profile.mlp_acts"] = np.concatenate([mlp_pos, mlp_neg], 1).astype(np.float16)
    proj_delta = np.full((n, L, 2), np.nan)
    norm_delta = np.full((n, L), np.nan)
    norm_in = np.full((n, L), np.nan)

    records: list[LayerRecord] = []
    u_prev = u_emb
    for l in range(L):
        rec, u, np_, nf_ = _stream_record(l, stream_pos[l], stream_neg[l], n_perm, seed)
        directions[l] = u
        X_out = np.concatenate([stream_pos[l], stream_neg[l]], 0)
        proj_stream[:, l] = X_out @ u
        if n_perm > 0:
            null_p_rows.append(np_)
            null_f_rows.append(nf_)
        if u_prev is not None:
            rec.cos_prev_direction = float(u @ u_prev)

        # Block l's input stream: previous GEM layer, or embedding for l = 0.
        if l > 0:
            X_in = np.concatenate([stream_pos[l - 1], stream_neg[l - 1]], 0)
        elif have_embed:
            X_in = np.concatenate([np.asarray(embed_pos, np.float64), np.asarray(embed_neg, np.float64)], 0)
        else:
            X_in = None

        delta = None
        delta_diff_out = None
        if X_in is not None:
            delta = X_out - X_in
            dmu = delta[:n_pos].mean(0) - delta[n_pos:].mean(0)
            delta_diff_out = float(dmu @ u)
            rec.delta_diff_proj_u_out = delta_diff_out
            rec.delta_diff_proj_u_in = float(dmu @ u_prev) if u_prev is not None else None
            nd = np.linalg.norm(delta, axis=1)
            ni = np.linalg.norm(X_in, axis=1)
            rec.delta_norm_mean = float(np.mean(nd))
            norm_delta[:, l], norm_in[:, l] = nd, ni
            proj_delta[:, l, 1] = delta @ u
            if u_prev is not None:
                proj_delta[:, l, 0] = delta @ u_prev

        if have_sub:
            A = (np.asarray(attn_pos[l], np.float64), np.asarray(attn_neg[l], np.float64))
            M = (np.asarray(mlp_pos[l], np.float64), np.asarray(mlp_neg[l], np.float64))
            ra, pa, na = _sublayer_record(
                "attn", arch.attn_hook or "attn", A[0], A[1], u_prev, u, delta_diff_out,
                norm_in[:, l] if X_in is not None else None,
                norm_delta[:, l] if X_in is not None else None,
            )
            rm, pm, nm = _sublayer_record(
                "mlp", arch.mlp_hook or "mlp", M[0], M[1], u_prev, u, delta_diff_out,
                norm_in[:, l] if X_in is not None else None,
                norm_delta[:, l] if X_in is not None else None,
            )
            rec.sublayers = [ra, rm]
            proj_attn[:, l], proj_mlp[:, l] = pa, pm
            norm_attn[:, l], norm_mlp[:, l] = na, nm
            if delta is not None:
                res = delta - np.concatenate(A, 0) - np.concatenate(M, 0)
                res_norm = np.linalg.norm(res, axis=1)
                rel_stream = res_norm / np.maximum(np.linalg.norm(X_out, axis=1), _EPS)
                recon_rel[:, l] = rel_stream
                rec.recon_residual_max_abs = float(np.max(np.abs(res)))
                rec.recon_residual_max_rel_delta = float(np.max(res_norm / np.maximum(norm_delta[:, l], _EPS)))
                rec.recon_residual_max_rel_stream = float(np.max(rel_stream))

        records.append(rec)
        u_prev = u

    arrays["profile.directions"] = directions.astype(np.float32)
    arrays["profile.proj_stream_u_out"] = proj_stream.astype(np.float32)
    arrays["profile.proj_delta"] = proj_delta.astype(np.float32)
    arrays["profile.norm_delta"] = norm_delta.astype(np.float32)
    arrays["profile.norm_stream_in"] = norm_in.astype(np.float32)
    if have_sub:
        arrays["profile.proj_attn"] = proj_attn.astype(np.float32)
        arrays["profile.proj_mlp"] = proj_mlp.astype(np.float32)
        arrays["profile.norm_attn"] = norm_attn.astype(np.float32)
        arrays["profile.norm_mlp"] = norm_mlp.astype(np.float32)
        arrays["profile.recon_residual_rel_stream"] = recon_rel.astype(np.float32)
    if n_perm > 0:
        arrays["profile.null_projected"] = np.stack(null_p_rows).astype(np.float32)
        arrays["profile.null_fisher"] = np.stack(null_f_rows).astype(np.float32)

    final_rec = None
    if final_norm_pos is not None and final_norm_neg is not None:
        fp = np.asarray(final_norm_pos, dtype=np.float64)
        fn = np.asarray(final_norm_neg, dtype=np.float64)
        _check_finite("final_norm", fp, fn)
        final_rec, u_f, fnp, fnf = _stream_record(L - 1, fp, fn, n_perm, seed)
        final_rec.sidecar_row = None
        final_rec.cos_prev_direction = float(u_f @ directions[L - 1])  # vs raw block output
        arrays["profile.final_norm_mean_pos"] = fp.mean(0).astype(np.float32)
        arrays["profile.final_norm_mean_neg"] = fn.mean(0).astype(np.float32)
        arrays["profile.final_norm_direction"] = u_f.astype(np.float32)
        arrays["profile.proj_final_norm"] = (np.concatenate([fp, fn], 0) @ u_f).astype(np.float32)
        if n_perm > 0:
            arrays["profile.final_norm_null_projected"] = fnp.astype(np.float32)
            arrays["profile.final_norm_null_fisher"] = fnf.astype(np.float32)

    prov = provenance or Provenance(method="build_layer_profile")
    prov = replace(
        prov,
        code_version=prov.code_version or code_version(),
        created_utc=prov.created_utc or _utc_now(),
        stats_dtype="float64",
        n_pos=n_pos, n_neg=n_neg, seed=seed, n_perm=n_perm,
    )
    profile = LayerProfile(
        n_layers=L, hidden_dim=d, layers=records, embedding=embedding_rec,
        final_norm=final_rec, stream_source=stream_source, architecture=arch, provenance=prov,
    )
    docs = {k: SIDECAR_DOCS[k] for k in arrays if k in SIDECAR_DOCS}
    return profile, arrays, docs


def attach_layer_profile(gem: ConceptGEM, profile: LayerProfile) -> ConceptGEM:
    """Return a new ConceptGEM with `profile` attached (non-mutating)."""
    return replace(gem, layer_profile=profile)


# ---------------------------------------------------------------------------
# 2. Settled-subspace extras
# ---------------------------------------------------------------------------

def decompose_settled_subspace(
    sub: SettledSubspace,
    arrays: Any,
    provenance: Provenance | None = None,
) -> SettledSubspace:
    """Attach a SubspaceSublayerDecomposition to `sub` (non-mutating).

    arrays : mapping with the profile.* class-mean keys (the dict returned by
        build_layer_profile, or load_gem_sidecar(...)). Needs embedding and
        sublayer means, i.e. a profile built from capture_sublayer_activations.

    For each direction w_j at layer L = sub.layer, splits w_j . dmu(x_L) into
    the embedding term plus every block's attention and MLP write up to L.
    reconstruction_residual is what the sum misses (~0 for exact-additive
    architectures). Note: if `sub` was fit on hidden_states activations at
    the LAST layer, those are post-final-norm while the profile stream is
    not; the residual will show it (flagged in provenance notes).
    """
    for k in ("profile.embed_mean_pos", "profile.attn_mean_pos", "profile.mlp_mean_pos"):
        if k not in arrays:
            raise KeyError(f"decompose_settled_subspace needs {k} (profile built with sublayer captures)")
    W = np.asarray(sub.directions, dtype=np.float64).reshape(-1, np.asarray(arrays["profile.embed_mean_pos"]).shape[0])
    L = sub.layer
    f = lambda k: np.asarray(arrays[k], dtype=np.float64)
    d_emb = f("profile.embed_mean_pos") - f("profile.embed_mean_neg")
    d_attn = f("profile.attn_mean_pos")[: L + 1] - f("profile.attn_mean_neg")[: L + 1]
    d_mlp = f("profile.mlp_mean_pos")[: L + 1] - f("profile.mlp_mean_neg")[: L + 1]
    d_stream = f("profile.stream_mean_pos")[L] - f("profile.stream_mean_neg")[L]

    up_attn = W @ d_attn.T          # (k, L+1)
    up_mlp = W @ d_mlp.T
    emb = W @ d_emb
    stream = W @ d_stream
    resid = stream - (emb + up_attn.sum(1) + up_mlp.sum(1))

    n_layers = f("profile.stream_mean_pos").shape[0]
    prov = provenance or Provenance(method="decompose_settled_subspace (class-mean projections from sidecar)")
    notes = list(prov.notes)
    if L == n_layers - 1:
        notes.append("layer is final: profile stream is pre-final-norm; a subspace fit on "
                     "hidden_states[-1] lives in post-norm space")
    prov = replace(prov, code_version=prov.code_version or code_version(),
                   created_utc=prov.created_utc or _utc_now(), stats_dtype="float64", notes=notes)

    dec = SubspaceSublayerDecomposition(
        layer=L,
        attn_diff_proj=up_attn[:, L].tolist() if W.size else [],
        mlp_diff_proj=up_mlp[:, L].tolist() if W.size else [],
        stream_diff_proj=stream.tolist(),
        embed_diff_proj=emb.tolist(),
        upstream_attn_diff_proj=up_attn.tolist(),
        upstream_mlp_diff_proj=up_mlp.tolist(),
        reconstruction_residual=resid.tolist(),
        provenance=prov,
    )
    return replace(sub, sublayer_decomposition=dec)


def attach_layer_settled_subspaces(
    profile: LayerProfile,
    h_pos_by_layer: dict[int, np.ndarray],
    h_neg_by_layer: dict[int, np.ndarray],
    layers: list[int],
    test_size: float = 0.3,
    random_state: int = 0,
    **extract_kwargs: Any,
) -> LayerProfile:
    """Run INLP (extract_settled_subspace) at each of `layers` and attach the
    result to that LayerRecord. Non-mutating. EXPENSIVE -- see the caution
    on extract_settled_subspace (30-90+ min per layer on a 70M model when
    the permutation fallback fires); pass explicit layers, never all."""
    from sklearn.model_selection import train_test_split
    from rosetta_tools.gem import extract_settled_subspace

    new_layers = list(profile.layers)
    for l in layers:
        hp, hn = h_pos_by_layer[l], h_neg_by_layer[l]
        X = np.concatenate([hp, hn], 0).astype(np.float64)
        y = np.concatenate([np.ones(len(hp)), np.zeros(len(hn))])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                              random_state=random_state, stratify=y)
        sub = extract_settled_subspace(Xtr, ytr, Xte, yte, layer=l, **extract_kwargs)
        sub = replace(sub, provenance=Provenance(
            method="extract_settled_subspace (INLP) via attach_layer_settled_subspaces",
            code_version=code_version(), created_utc=_utc_now(),
            n_pos=len(hp), n_neg=len(hn), seed=random_state, stats_dtype="float64",
            notes=[f"test_size={test_size}", f"kwargs={extract_kwargs}"],
        ))
        new_layers[l] = replace(new_layers[l], settled_subspace=sub)
    return replace(profile, layers=new_layers)


# ---------------------------------------------------------------------------
# 3. Hook-based sublayer capture (torch)
# ---------------------------------------------------------------------------

_LAYER_PATHS = (
    "model.layers", "layers",                    # Llama/Qwen/Mistral/Gemma (CausalLM / base)
    "transformer.h", "h",                        # GPT-2, GPT-J, CodeGen, Falcon
    "gpt_neox.layers",                           # GPT-NeoX / Pythia (CausalLM)
    "model.decoder.layers", "decoder.layers",    # OPT
    "transformer.layers",
)


def _get_attr_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_decoder_layers(model) -> tuple[str, Any]:
    """Return (dotted path, ModuleList) of the decoder blocks."""
    for p in _LAYER_PATHS:
        mods = _get_attr_path(model, p)
        if mods is not None and hasattr(mods, "__len__") and len(mods) > 0:
            return p, mods
    raise NotImplementedError(
        f"capture_sublayer_activations: cannot locate decoder layers on {type(model).__name__}; "
        f"tried {_LAYER_PATHS}"
    )


def resolve_architecture(model) -> ArchitectureFlags:
    """Decide which modules to hook and whether Delta == attn + mlp should
    hold exactly. Raises NotImplementedError for unknown block layouts
    rather than guessing."""
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", "unknown")
    path, layers = find_decoder_layers(model)
    l0 = layers[0]
    flags = ArchitectureFlags(model_type=mt, decoder_class=type(l0).__name__, layer_path=path)

    for bad in ("altup", "laurel", "crossattention", "per_layer_input_gate"):
        if hasattr(l0, bad):
            raise NotImplementedError(
                f"{flags.decoder_class} has '{bad}': extra stream writes beyond attn+mlp "
                "are not modelled by this extractor"
            )

    if hasattr(l0, "pre_feedforward_layernorm") and hasattr(l0, "post_feedforward_layernorm"):
        # Gemma2/3 "sandwich": what is added is the NORMED sublayer output.
        # (Careful: in Llama, post_attention_layernorm is the PRE-MLP norm.)
        flags.attn_hook, flags.mlp_hook = "post_attention_layernorm", "post_feedforward_layernorm"
        flags.sublayer_post_norm = True
        flags.notes.append("Gemma-style post-sublayer norms: hooks are after the norm, on the added tensor")
    elif mt == "opt":
        flags.attn_hook, flags.mlp_hook = "self_attn", "fc2"
        if not getattr(cfg, "do_layer_norm_before", True):
            flags.post_ln = True
    elif hasattr(l0, "self_attn") and hasattr(l0, "block_sparse_moe"):
        flags.attn_hook, flags.mlp_hook = "self_attn", "block_sparse_moe"  # Mixtral
    elif hasattr(l0, "self_attn") and hasattr(l0, "mlp"):
        flags.attn_hook, flags.mlp_hook = "self_attn", "mlp"
    elif hasattr(l0, "attn") and hasattr(l0, "mlp"):
        flags.attn_hook, flags.mlp_hook = "attn", "mlp"
    elif hasattr(l0, "attention") and hasattr(l0, "mlp"):
        flags.attn_hook, flags.mlp_hook = "attention", "mlp"
    elif hasattr(l0, "self_attention") and hasattr(l0, "mlp"):
        flags.attn_hook, flags.mlp_hook = "self_attention", "mlp"
    else:
        raise NotImplementedError(
            f"unknown decoder block layout {flags.decoder_class} (model_type={mt}); "
            f"children={[n for n, _ in l0.named_children()]}"
        )

    mlp_mod = getattr(l0, flags.mlp_hook)
    if hasattr(mlp_mod, "experts") or hasattr(mlp_mod, "gate") and "moe" in type(mlp_mod).__name__.lower():
        flags.moe = True
        flags.notes.append("MoE MLP: hooked write = router-weighted expert sum (+ shared expert if any)")

    # Final norm: the module producing hidden_states[-1]. Sibling of the
    # layer list. OPT with project_out applies a projection AFTER the norm,
    # so the norm output is not hidden_states[-1] there -> not recorded.
    parent = model if "." not in path else _get_attr_path(model, path.rsplit(".", 1)[0])
    prefix = "" if "." not in path else path.rsplit(".", 1)[0] + "."
    for name in ("norm", "ln_f", "final_layer_norm", "final_norm"):
        mod = getattr(parent, name, None)
        if mod is not None and hasattr(mod, "register_forward_hook"):
            if getattr(parent, "project_out", None) is not None:
                flags.notes.append(f"{prefix}{name} is followed by project_out; post-final-norm stream not recorded")
            else:
                flags.final_norm_hook = prefix + name
            break
    else:
        flags.notes.append("no final norm module found; post-final-norm stream not recorded")

    if mt in ("gptj", "codegen", "phi"):
        flags.parallel_block = True
    elif mt == "gpt_neox":
        flags.parallel_block = bool(getattr(cfg, "use_parallel_residual", True))
    elif mt == "falcon":
        flags.parallel_block = bool(getattr(cfg, "parallel_attn", False)
                                    or getattr(cfg, "new_decoder_architecture", False))
    if flags.parallel_block:
        flags.notes.append("parallel block: attn and mlp both read x_in; no stream point between them")
    if flags.post_ln:
        flags.additivity_expected_exact = False
        flags.notes.append("post-LN: x_out = LN(x_in + sublayer); Delta != attn + mlp")
    return flags


@dataclass
class SublayerCapture:
    """Pooled per-example captures. Arrays are float32, rows in text order.
    stream[l] = output of block l; stream_in[l] = its input."""

    embed: np.ndarray            # (n, d) = stream_in[0]
    stream: np.ndarray           # (L, n, d)
    attn: np.ndarray             # (L, n, d)
    mlp: np.ndarray              # (L, n, d)
    architecture: ArchitectureFlags
    model_dtype: str
    max_recon_rel_stream: float  # max ||Delta - a - m|| / ||x_out|| over examples, layers
    max_continuity_gap: float    # max |stream_in[l] - stream[l-1]|, should be 0
    max_recon_rel_stream_per_layer: list | None = None  # per-layer max of the above residual
    final_norm: np.ndarray | None = None  # (n, d) post-final-norm stream (hidden_states[-1])


def _first_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def capture_sublayer_activations(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 8,
    pool: str = "last",
    max_length: int = 512,
    allow_inexact: bool = False,
    tol: float | None = None,
    capture_final_norm: bool = True,
) -> SublayerCapture:
    """Run `texts` through `model` with forward hooks on every decoder block
    (input and output) and on its attention/MLP writes; pool each capture
    with the same _pool as extract_layer_activations.

    Raises:
      - RuntimeError if model.training (dropout breaks additivity),
      - ValueError for pool="last" with left padding (_pool would pick a pad),
      - NotImplementedError for unknown/unsupported block layouts,
      - ValueError for post-LN models, or if max ||Delta - a - m||/||x_out||
        exceeds `tol` (default ADDITIVITY_TOL[dtype]), or if a block's input
        is not the previous block's output -- unless allow_inexact=True, in
        which case flags.additivity_expected_exact is set False and the
        residuals are left in the data for build_layer_profile to record.
    """
    import torch
    from rosetta_tools.extraction import _pool

    if model.training:
        raise RuntimeError("capture_sublayer_activations: call model.eval() first (dropout breaks additivity)")
    if pool == "last" and getattr(tokenizer, "padding_side", "right") == "left":
        raise ValueError("pool='last' assumes right padding; tokenizer.padding_side is 'left'")
    arch = resolve_architecture(model)
    if arch.post_ln and not allow_inexact:
        raise ValueError(f"{arch.decoder_class}: post-LN block, Delta != attn + mlp; "
                         "pass allow_inexact=True to record anyway")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, layers = find_decoder_layers(model)
    L = len(layers)
    state: dict[str, Any] = {}
    buf = {k: [[] for _ in range(L)] for k in ("in", "out", "attn", "mlp")}
    final_buf: list = []

    def put(kind, i, t):
        t = _first_tensor(t)
        if t.dim() == 2:
            # OPT runs fc1/fc2 on a flattened (batch*seq, d) view
            t = t.reshape(*state["mask"].shape, -1)
        pooled = _pool(t, state["mask"], strategy=pool)
        buf[kind][i].append(pooled.detach().float().cpu().numpy())

    handles = []
    for i, layer in enumerate(layers):
        def pre(mod, args, kwargs, i=i):
            h = args[0] if args else kwargs["hidden_states"]
            put("in", i, h)
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(lambda m, a, o, i=i: put("out", i, o)))
        handles.append(getattr(layer, arch.attn_hook).register_forward_hook(
            lambda m, a, o, i=i: put("attn", i, o)))
        handles.append(getattr(layer, arch.mlp_hook).register_forward_hook(
            lambda m, a, o, i=i: put("mlp", i, o)))
    if capture_final_norm and arch.final_norm_hook:
        def fin(mod, a, o):
            t = _first_tensor(o)
            final_buf.append(_pool(t, state["mask"], strategy=pool).detach().float().cpu().numpy())
        handles.append(_get_attr_path(model, arch.final_norm_hook).register_forward_hook(fin))

    model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    try:
        for b in range(0, len(texts), batch_size):
            enc = tokenizer(texts[b:b + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            state["mask"] = enc["attention_mask"].to(device)
            with torch.no_grad():
                model(input_ids=enc["input_ids"].to(device), attention_mask=state["mask"], use_cache=False)
    finally:
        for h in handles:
            h.remove()

    cat = {k: np.stack([np.concatenate(v[i], 0) for i in range(L)]) for k, v in buf.items()}
    if any(len(cat[k]) != L for k in cat) or cat["in"].shape != cat["attn"].shape:
        raise RuntimeError("capture_sublayer_activations: hooks fired an unexpected number of times")

    delta = cat["out"] - cat["in"]
    res = np.linalg.norm(delta - cat["attn"] - cat["mlp"], axis=-1)
    rel = res / np.maximum(np.linalg.norm(cat["out"], axis=-1), _EPS)
    max_rel = float(rel.max())
    gap = float(np.max(np.abs(cat["in"][1:] - cat["out"][:-1]))) if L > 1 else 0.0
    tol = ADDITIVITY_TOL.get(model_dtype, 1e-4) if tol is None else tol
    problems = []
    if max_rel > tol:
        problems.append(f"max ||Delta - attn - mlp||/||x_out|| = {max_rel:.3g} > tol {tol:.3g}")
    if gap > tol * max(1.0, float(np.abs(cat["out"]).max())):
        problems.append(f"block input != previous block output (max gap {gap:.3g})")
    if problems:
        msg = f"{arch.decoder_class} ({arch.model_type}): " + "; ".join(problems)
        if not allow_inexact:
            raise ValueError("sublayer additivity check failed: " + msg)
        log.warning("additivity not exact (allow_inexact=True): %s", msg)
        arch.additivity_expected_exact = False
        arch.notes.append("additivity check failed: " + "; ".join(problems))

    final = None
    if final_buf:
        final = np.concatenate(final_buf, 0)
        if final.shape != cat["out"][-1].shape:
            # norm module fired more/less than once per forward (reused?)
            arch.notes.append(f"final norm hook shape {final.shape} != stream; dropped")
            final = None

    return SublayerCapture(
        embed=cat["in"][0], stream=cat["out"], attn=cat["attn"], mlp=cat["mlp"],
        architecture=arch, model_dtype=model_dtype,
        max_recon_rel_stream=max_rel, max_continuity_gap=gap,
        max_recon_rel_stream_per_layer=rel.max(axis=1).tolist(),
        final_norm=final,
    )


def extract_concept_layer_profile(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    model_id: str = "",
    device: str = "cpu",
    config: GEMExtractionConfig | None = None,
    dataset: str | None = None,
    pair_set: dict | None = None,
) -> tuple[LayerProfile, dict[str, np.ndarray], dict[str, dict]]:
    """Capture + build_layer_profile for one concept's contrastive texts,
    driven by a GEMExtractionConfig (default: GEMExtractionConfig(), i.e.
    everything on). Returns (profile, arrays, array_docs) for
    save_gem(..., arrays=, array_docs=, inline_directions=config.inline_directions)."""
    cfg = config or GEMExtractionConfig()
    kw = dict(device=device, batch_size=cfg.batch_size, pool=cfg.pooling,
              max_length=cfg.max_length, allow_inexact=cfg.allow_inexact,
              tol=cfg.additivity_tol, capture_final_norm=cfg.store_post_final_norm)
    cap_p = capture_sublayer_activations(model, tokenizer, pos_texts, **kw)
    cap_n = capture_sublayer_activations(model, tokenizer, neg_texts, **kw)
    arch = cap_p.architecture
    if not cap_n.architecture.additivity_expected_exact:
        arch = replace(arch, additivity_expected_exact=False,
                       notes=arch.notes + [n for n in cap_n.architecture.notes if n not in arch.notes])
    lp = arch.layer_path
    hooks = {
        "stream_in": f"{lp}.{{l}} (forward pre-hook, hidden_states arg)",
        "stream_out": f"{lp}.{{l}} (forward hook, output[0])",
        "attn": f"{lp}.{{l}}.{arch.attn_hook} (forward hook, output[0])",
        "mlp": f"{lp}.{{l}}.{arch.mlp_hook} (forward hook, output[0])",
    }
    if cap_p.final_norm is not None:
        hooks["final_norm"] = f"{arch.final_norm_hook} (forward hook, output)"
    per_layer = np.maximum(cap_p.max_recon_rel_stream_per_layer, cap_n.max_recon_rel_stream_per_layer)
    prov = Provenance(
        method="extract_concept_layer_profile: forward hooks on decoder blocks + sublayers, "
               "pooled per example, class statistics in float64",
        model_id=model_id or getattr(getattr(model, "config", None), "_name_or_path", ""),
        model_dtype=cap_p.model_dtype,
        capture_dtype="float32",
        hook_points=hooks,
        pooling=cfg.pooling, max_length=cfg.max_length, batch_size=cfg.batch_size,
        device=str(device), dataset=dataset, pair_set=pair_set,
        notes=[f"additivity tol: {cfg.additivity_tol if cfg.additivity_tol is not None else ADDITIVITY_TOL.get(cap_p.model_dtype)}",
               "max recon residual rel stream per layer: "
               + ", ".join(f"{x:.3g}" for x in per_layer)],
    )
    return build_layer_profile(
        cap_p.stream, cap_n.stream, cap_p.embed, cap_n.embed,
        cap_p.attn, cap_n.attn, cap_p.mlp, cap_n.mlp,
        n_perm=cfg.n_permutations, seed=cfg.permutation_seed, architecture=arch,
        provenance=prov, stream_source="decoder_layer_output",
        store_full_sublayer_acts=cfg.store_full_sublayer_activations,
        final_norm_pos=cap_p.final_norm if cfg.store_post_final_norm else None,
        final_norm_neg=cap_n.final_norm if cfg.store_post_final_norm else None,
    )

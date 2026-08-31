"""
extraction.py — Model-agnostic activation extraction for CAZ experiments.

Built on raw HuggingFace transformers — no TransformerLens dependency.
Works with any model that supports output_hidden_states=True, which covers
the full HF ecosystem including GPT-2, GPT-Neo, OPT, Pythia, Llama,
Mistral, Qwen, etc.

Typical usage
-------------
    import torch
    from transformers import AutoModel, AutoTokenizer
    from rosetta_tools.extraction import extract_layer_activations
    from rosetta_tools.gpu_utils import get_device, get_dtype

    device = get_device()
    dtype  = get_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    model     = AutoModel.from_pretrained("gpt2-xl", dtype=dtype).to(device)
    model.eval()

    texts  = ["The study conclusively shows...", "The evidence is mixed..."]
    acts   = extract_layer_activations(model, tokenizer, texts, device=device)
    # acts: list of [n_texts, hidden_dim] arrays, one per layer (including embedding)
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Literal

import numpy as np
try:
    import torch
except ImportError as _e:
    raise ImportError(
        "rosetta_tools.extraction requires torch. Install it with: pip install torch"
    ) from _e
from numpy.typing import NDArray

from .caz import compute_coherence, compute_separation, compute_velocity, concept_quality_report

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

PoolStrategy = Literal["mean", "last", "first", "cls"]


def _pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: PoolStrategy = "last",
) -> torch.Tensor:
    """Pool a batch of hidden states to one vector per sequence.

    Parameters
    ----------
    hidden_states:
        Shape ``[batch, seq_len, hidden_dim]``.
    attention_mask:
        Shape ``[batch, seq_len]``.  1 for real tokens, 0 for padding.
    strategy:
        ``"last"``  — last non-padding token (default; correct for causal/decoder
                      models — GPT-2, Llama, Mistral, Qwen, etc.  The last token
                      position is where the model has attended to full context and
                      committed to a representation, matching the CAZ framework's
                      residual stream tracking methodology).
        ``"mean"``  — mean over non-padding tokens.
        ``"first"`` — first token (embedding / CLS position).
        ``"cls"``   — alias for ``"first"``.

    Returns
    -------
    torch.Tensor
        Shape ``[batch, hidden_dim]``.
    """
    if strategy in ("first", "cls"):
        return hidden_states[:, 0, :]

    if strategy == "last":
        # Index of the last non-padding token per sequence.
        # With device_map="auto" (multi-GPU pipeline), hidden_states at deep layers
        # may live on a different device than attention_mask (which stays on cuda:0).
        # Move lengths to match hidden_states before indexing.
        lengths = attention_mask.sum(dim=1) - 1  # [batch] — on attention_mask's device
        lengths = lengths.to(hidden_states.device)
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, lengths, :]

    # mean — mask out padding, average over real tokens
    mask_expanded = attention_mask.to(hidden_states.device).unsqueeze(-1).float()  # [batch, seq, 1]
    summed = (hidden_states.float() * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1e-8)
    return summed / count


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_layer_activations(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 8,
    pool: PoolStrategy = "last",
    max_length: int = 512,
) -> list[NDArray[np.float32]]:
    """Extract residual-stream activations at every layer for a list of texts.

    Uses output_hidden_states=True — compatible with any HuggingFace model
    that exposes hidden states (AutoModel, GPT2Model, OPTModel, etc.).

    Parameters
    ----------
    model:
        HuggingFace model (already loaded, already on device).
        Must support ``output_hidden_states=True``.
    tokenizer:
        Corresponding tokenizer.
    texts:
        List of raw text strings to encode and run through the model.
    device:
        Device string (``"cuda"`` or ``"cpu"``).
    batch_size:
        Number of texts per forward pass.  Reduce if OOM.
    pool:
        Pooling strategy for collapsing the sequence dimension.
    max_length:
        Maximum tokenized length.  Texts longer than this are truncated.

    Returns
    -------
    list[NDArray[np.float32]]
        One array per layer (embedding + all transformer blocks).
        Each array has shape ``[n_texts, hidden_dim]``.
        Activations are cast to float32 regardless of model dtype —
        important for numerical stability in Fisher normalization.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = None
    all_layer_acts: list[list[NDArray]] = []  # [layer][batch_chunk]

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (n_layers + 1) tensors
        # each [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states

        if n_layers is None:
            n_layers = len(hidden_states)
            all_layer_acts = [[] for _ in range(n_layers)]

        for layer_idx, hs in enumerate(hidden_states):
            pooled = _pool(hs, attention_mask, strategy=pool)
            # Cast to float32 — critical for Fisher normalization accuracy
            all_layer_acts[layer_idx].append(pooled.detach().cpu().float().numpy())

    # Concatenate batch chunks
    return [np.concatenate(chunks, axis=0) for chunks in all_layer_acts]


def extract_contrastive_activations(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str = "cpu",
    batch_size: int = 8,
    pool: PoolStrategy = "last",
    max_length: int = 512,
) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """Extract activations for positive and negative class texts.

    Convenience wrapper over ``extract_layer_activations`` for the
    standard CAZ contrastive pair workflow.

    Parameters
    ----------
    pos_texts:
        Texts for the positive class (label 1).
    neg_texts:
        Texts for the negative class (label 0).
    (other parameters same as extract_layer_activations)

    Returns
    -------
    list[tuple[NDArray, NDArray]]
        One ``(pos_acts, neg_acts)`` tuple per layer.
        Each array is ``[n_texts, hidden_dim]``, float32.
    """
    pos_acts = extract_layer_activations(
        model,
        tokenizer,
        pos_texts,
        device=device,
        batch_size=batch_size,
        pool=pool,
        max_length=max_length,
    )
    neg_acts = extract_layer_activations(
        model,
        tokenizer,
        neg_texts,
        device=device,
        batch_size=batch_size,
        pool=pool,
        max_length=max_length,
    )
    return list(zip(pos_acts, neg_acts))


# ---------------------------------------------------------------------------
# CAZ pipeline — layer-wise metrics + record assembly
# ---------------------------------------------------------------------------


def extract_layer_wise_metrics(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int,
    compute_quality: bool = True,
) -> tuple[dict, NDArray[np.float32], NDArray[np.float32]]:
    """Run the full CAZ per-layer pipeline for one concept's contrastive texts.

    Extracts activations at every layer, computes separation/coherence/velocity/
    DOM per layer, locates the peak layer, and (optionally) embeds an inline
    QA/stability report (see ``rosetta_tools.caz.concept_quality_report``).

    Returns
    -------
    tuple[dict, NDArray, NDArray]
        ``(metrics_dict, cal_acts, all_layer_cal)`` where ``metrics_dict`` has
        keys ``n_layers``, ``metrics`` (per-layer S/C/velocity/DOM), ``peak_layer``,
        ``peak_separation``, ``peak_depth_pct``, and (if ``compute_quality``)
        ``quality``. ``cal_acts`` is the peak-layer calibration activations
        (pos then neg stacked); ``all_layer_cal`` is the same but for every layer.
    """
    pos_by_layer = extract_layer_activations(
        model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
    )
    neg_by_layer = extract_layer_activations(
        model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
    )
    pos_by_layer = pos_by_layer[1:]
    neg_by_layer = neg_by_layer[1:]

    # Some architectures (e.g. OPT-350m: word_embed_proj_dim=512, hidden_size=1024)
    # have layers with mismatched hidden dims that survive the [1:] skip.
    # Keep only layers at the modal dimension so np.stack doesn't crash.
    if pos_by_layer:
        dims = [p.shape[1] for p in pos_by_layer]
        modal_dim = Counter(dims).most_common(1)[0][0]
        if len(set(dims)) > 1:
            log.warning(
                "Heterogeneous hidden dims across layers %s — keeping %d-dim layers only",
                dict(Counter(dims)), modal_dim,
            )
            keep = [p.shape[1] == modal_dim for p in pos_by_layer]
            pos_by_layer = [p for p, ok in zip(pos_by_layer, keep) if ok]
            neg_by_layer = [n for n, ok in zip(neg_by_layer, keep) if ok]

    n_layers = len(pos_by_layer)
    separations, coherences, dom_vectors, raw_distances = [], [], [], []

    for pos, neg in zip(pos_by_layer, neg_by_layer):
        S = compute_separation(pos, neg)
        C = compute_coherence(pos, neg)
        pos64, neg64 = pos.astype(np.float64), neg.astype(np.float64)
        diff = pos64.mean(axis=0) - neg64.mean(axis=0)
        raw_dist = float(np.linalg.norm(diff))
        norm = np.linalg.norm(diff)
        dom = (diff / norm).tolist() if norm > 0 else diff.tolist()
        separations.append(S)
        coherences.append(C)
        dom_vectors.append(dom)
        raw_distances.append(raw_dist)

    vel_array = compute_velocity(separations, window=3)
    peak_layer = int(np.argmax(separations))
    peak_depth_pct = 100.0 * peak_layer / n_layers if n_layers > 0 else 0.0

    if peak_depth_pct < 5.0:
        log.warning(
            "  ⚠ Peak at L%d (%.1f%% depth) — likely embedding leakage, "
            "not genuine concept assembly. Interpret with caution.",
            peak_layer, peak_depth_pct,
        )

    # Calibration activations at peak layer — pos and neg combined.
    # Saved alongside the metrics JSON for Procrustes alignment.
    cal_acts = np.concatenate(
        [pos_by_layer[peak_layer].astype(np.float32),
         neg_by_layer[peak_layer].astype(np.float32)],
        axis=0,
    )

    # All-layer calibration activations for depth-matched alignment.
    # Shape: [n_layers, n_pos+n_neg, hidden_dim].
    all_layer_cal = np.stack(
        [np.concatenate([pos_by_layer[i].astype(np.float32),
                         neg_by_layer[i].astype(np.float32)], axis=0)
         for i in range(n_layers)],
        axis=0,
    )

    metrics_dict = {
        "n_layers": n_layers,
        "metrics": [
            {
                "layer": i,
                "separation_fisher": separations[i],
                "coherence": coherences[i],
                "raw_distance": raw_distances[i],
                "dom_vector": dom_vectors[i],
                "velocity": float(vel_array[i]),
            }
            for i in range(n_layers)
        ],
        "peak_layer": peak_layer,
        "peak_separation": separations[peak_layer],
        "peak_depth_pct": round(100.0 * peak_layer / n_layers, 1),
    }

    # Extraction-time QA / stability signals, embedded so every extraction
    # carries its own quality report (split-half DOM reproducibility, peak
    # separation, class balance, non-finite fraction). The one check it cannot
    # do from a single extraction (cross-model direction consistency, which
    # catches a clean systematic label inversion) is a separate corpus-level
    # pass — see rosetta_tools.caz.concept_quality_report.
    if compute_quality:
        metrics_dict["quality"] = concept_quality_report(
            list(zip(pos_by_layer, neg_by_layer)), peak_layer,
        )
    return metrics_dict, cal_acts, all_layer_cal


def build_caz_record(
    model_id: str,
    concept: str,
    n_pairs: int,
    split: str,
    hidden_dim: int,
    n_layers: int,
    extraction_seconds: float,
    layer_data: dict,
) -> dict:
    """Assemble the canonical caz record dict written to ``caz_<concept>.json``.

    Pure schema assembly — no I/O. Callers own writing the file (and any
    upload), so the on-disk layout stays whatever the caller wants.
    """
    return {
        "model_id": model_id,
        "concept": concept,
        "n_pairs": n_pairs,
        "split": split,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "token_pos": -1,
        "extraction_seconds": round(extraction_seconds, 1),
        "layer_data": layer_data,
    }

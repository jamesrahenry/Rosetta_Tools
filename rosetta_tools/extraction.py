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

import warnings
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

warnings.filterwarnings("ignore")


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
        # Index of the last non-padding token per sequence
        lengths = attention_mask.sum(dim=1) - 1  # [batch]
        batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch, lengths, :]

    # mean — mask out padding, average over real tokens
    mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
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

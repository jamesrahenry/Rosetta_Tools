"""
ablation.py — Directional ablation for CAZ mid-stream hypothesis testing.

Implements orthogonal projection ablation against HuggingFace AutoModel —
no TransformerLens dependency.

The Mid-Stream Ablation Hypothesis (CAZ Prediction 1) states that removing a
concept direction at the CAZ peak produces maximum behavioral suppression with
minimum collateral capability damage, compared to ablation at pre-CAZ or
post-CAZ layers.

How ablation works
------------------
For a concept direction v̂ (unit vector), orthogonal projection removes the
component of each token's hidden state that lies along v̂:

    h' = h − (h · v̂) v̂

This is applied as a forward hook on the target transformer layer. Downstream
layers process the ablated representation, so the causal effect propagates
through the rest of the network.

Typical usage
-------------
    from rosetta_tools.ablation import DirectionalAblator, compute_dominant_direction
    from rosetta_tools.extraction import extract_layer_activations

    # 1. Get the concept direction at the peak layer
    pos_acts = extract_layer_activations(model, tokenizer, pos_texts)[peak_layer]
    neg_acts = extract_layer_activations(model, tokenizer, neg_texts)[peak_layer]
    direction = compute_dominant_direction(pos_acts, neg_acts)

    # 2. Run ablated forward passes
    layers = get_transformer_layers(model)
    with DirectionalAblator(layers[peak_layer], direction):
        ablated_acts = extract_layer_activations(model, tokenizer, pos_texts)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------


def get_transformer_layers(model) -> list:
    """Return the list of transformer block modules for a HuggingFace model.

    Supports GPT-2, Llama, Mistral, Pythia (GPT-NeoX), GPT-Neo, OPT, Gemma,
    and most other HF decoder-only architectures.

    Parameters
    ----------
    model:
        A loaded HuggingFace model (AutoModel or CausalLM).

    Returns
    -------
    list
        Ordered list of transformer layer modules.

    Raises
    ------
    RuntimeError
        If the layer structure cannot be detected automatically.
    """
    # Common attribute paths for decoder-only transformer layers.
    # Paths are tried in order.  "layers" must come after "model.layers"
    # because some wrappers (CausalLM) have both, and "model.layers" is
    # the correct one for those.  "layers" catches AutoModel-loaded
    # Qwen/Llama/Gemma where there's no .model wrapper.
    candidates = [
        "transformer.h",         # GPT-2
        "model.layers",          # Llama, Mistral, Gemma, Qwen (CausalLM)
        "gpt_neox.layers",       # Pythia, GPT-NeoX
        "transformer.blocks",    # GPT-Neo (some variants)
        "model.decoder.layers",  # OPT
        "model.transformer.h",   # Some GPT-2 wrappers
        "layers",                # Qwen, Llama, Gemma via AutoModel (no .model wrapper)
        "decoder.layers",        # OPT via AutoModel
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    raise RuntimeError(
        f"Cannot auto-detect transformer layers for {type(model).__name__}. "
        "Pass the layer list directly to DirectionalAblator."
    )


# ---------------------------------------------------------------------------
# Concept direction
# ---------------------------------------------------------------------------


def compute_dominant_direction(
    pos_acts: NDArray,
    neg_acts: NDArray,
) -> NDArray:
    """Compute the unit concept direction from positive/negative activations.

    The direction is the normalized difference of class centroids — the same
    vector used for dom_vector in the extraction pipeline.

    Parameters
    ----------
    pos_acts:
        Activations for positive class — shape [n_pos, hidden_dim], float32/64.
    neg_acts:
        Activations for negative class — shape [n_neg, hidden_dim], float32/64.

    Returns
    -------
    NDArray
        Unit vector of shape [hidden_dim], float64.
    """
    pos = np.asarray(pos_acts, dtype=np.float64)
    neg = np.asarray(neg_acts, dtype=np.float64)
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return diff
    return diff / norm


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------


class DirectionalAblator:
    """Context manager that hooks a transformer layer to ablate a concept direction.

    Applies orthogonal projection ablation to every forward pass through the
    hooked layer while active. Compatible with any HF AutoModel layer module.

    Parameters
    ----------
    layer_module:
        The transformer block module to hook (e.g. ``layers[peak_layer]``).
        Get this from ``get_transformer_layers(model)[layer_idx]``.
    direction:
        Concept direction to ablate — shape [hidden_dim]. Will be L2-normalised.
    dtype:
        Torch dtype for the direction tensor. Defaults to float32.

    Example
    -------
        layers = get_transformer_layers(model)
        with DirectionalAblator(layers[peak_layer], direction):
            outputs = model(**inputs, output_hidden_states=True)
    """

    def __init__(
        self,
        layer_module,
        direction: NDArray,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._layer = layer_module
        self._handle = None

        d = torch.tensor(direction, dtype=dtype)
        self._direction = d / d.norm()  # unit vector

    def _hook(self, module, input, output):
        # HF layer outputs are tuples: (hidden_state, ...) or just a tensor.
        # The hidden state is the element whose last dimension matches the
        # concept direction.  For most architectures it's output[0], but some
        # (e.g. OPT) may include attention outputs with different shapes.
        if isinstance(output, tuple):
            hidden = None
            hidden_idx = 0
            target_dim = self._direction.shape[0]
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and o.dim() == 3 and o.shape[-1] == target_dim:
                    hidden = o
                    hidden_idx = i
                    break
            if hidden is None:
                # Fallback: use first tensor element
                hidden = output[0]
                hidden_idx = 0
        else:
            hidden = output
            hidden_idx = None

        dev = hidden.device
        dt = hidden.dtype
        v = self._direction.to(device=dev, dtype=dt)

        # Dimension mismatch guard — skip ablation rather than crash
        if hidden.shape[-1] != v.shape[0]:
            return output

        # Orthogonal projection: h' = h - (h·v)v
        # hidden shape: [batch, seq_len, hidden_dim]
        proj_scalar = torch.einsum("bsh,h->bs", hidden, v)           # [batch, seq]
        proj_vec    = torch.einsum("bs,h->bsh", proj_scalar, v)       # [batch, seq, hidden]
        ablated = hidden - proj_vec

        if isinstance(output, tuple):
            return output[:hidden_idx] + (ablated,) + output[hidden_idx + 1:]
        return ablated

    def __enter__(self):
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def kl_divergence_from_logits(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
) -> float:
    """KL divergence KL(baseline ‖ ablated) from last-token logits.

    Parameters
    ----------
    baseline_logits:
        Logits without ablation — shape [vocab_size].
    ablated_logits:
        Logits with ablation active — shape [vocab_size].

    Returns
    -------
    float
        KL divergence. Higher means more capability disruption.
    """
    p = F.log_softmax(baseline_logits.float(), dim=-1)
    q = F.log_softmax(ablated_logits.float(), dim=-1)
    return float(F.kl_div(q, p, reduction="sum", log_target=True).item())


def compute_baseline_logits(
    model,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Get last-token logits for a list of prompts (no ablation).

    Parameters
    ----------
    model:
        HuggingFace CausalLM (must have a ``lm_head`` / logits output).
    tokenizer:
        Corresponding tokenizer.
    prompts:
        General capability prompts (not concept-specific).
    device:
        Device string.

    Returns
    -------
    list[torch.Tensor]
        One [vocab_size] tensor per prompt.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1, :].cpu()
        results.append(logits)
    return results

"""
gpu_utils.py — Shared GPU/device utilities for the Rosetta Program.

Centralises device selection, dtype resolution, VRAM reporting, and
model teardown so that individual experiments don't reinvent this
boilerplate.  Designed to be library-agnostic: works with both
HuggingFace ``transformers`` and TransformerLens ``HookedTransformer``.

Typical usage
-------------
    from shared.gpu_utils import get_device, get_dtype, log_vram, release_model

    # -- Setup ---------------------------------------------------------
    device = get_device()            # "cuda" or "cpu"
    dtype  = get_dtype(device)       # torch.float16 or torch.float32

    # -- HuggingFace ---------------------------------------------------
    model = SomeModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    log_vram("after model load")

    # -- TransformerLens -----------------------------------------------
    model = HookedTransformer.from_pretrained(model_id, device=device, dtype=dtype)
    log_vram("after model load")

    # -- Teardown (important when loading multiple large models) --------
    release_model(model)
    log_vram("after release")

Public API
----------
get_device(prefer: str = "auto") -> str
    Returns "cuda" or "cpu".

get_dtype(device: str) -> torch.dtype
    Returns torch.float16 for CUDA, torch.float32 for CPU.

log_vram(label: str = "", device_index: int = 0) -> None
    Prints current allocated and total VRAM for the given CUDA device.
    No-ops silently on CPU.

vram_stats(device_index: int = 0) -> dict | None
    Returns a dict with VRAM numbers, or None on CPU.

release_model(model, *, clear_cache: bool = True) -> None
    Deletes the model reference and optionally calls torch.cuda.empty_cache().
    Safe to call on CPU (empty_cache is a no-op there).
"""

from __future__ import annotations

import gc
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(prefer: str = "auto") -> str:
    """Return the device string to use for model loading and tensor ops.

    Parameters
    ----------
    prefer:
        ``"auto"``  — use CUDA if available, otherwise CPU (default).
        ``"cuda"``  — require CUDA; raises ``RuntimeError`` if unavailable.
        ``"cpu"``   — always use CPU, regardless of hardware.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was requested but CUDA is not available on this system."
            )
        return "cuda"
    # "auto"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# dtype resolution
# ---------------------------------------------------------------------------


def get_dtype(device: str) -> torch.dtype:
    """Return the appropriate floating-point dtype for the given device.

    Uses fp16 on CUDA to reduce VRAM usage (e.g. GPT-2-XL fits in ~3 GB
    instead of ~6 GB).  Falls back to fp32 on CPU where fp16 arithmetic
    is typically slower and numerically less stable.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    torch.dtype
        ``torch.float16`` for CUDA, ``torch.float32`` for CPU.
    """
    return torch.float16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# VRAM reporting
# ---------------------------------------------------------------------------


def vram_stats(device_index: int = 0) -> Optional[dict]:
    """Return a dict with VRAM usage figures, or None if CUDA is unavailable.

    All values are in gibibytes (GiB, 1024³ bytes).

    Parameters
    ----------
    device_index:
        CUDA device ordinal (0 for the first GPU).

    Returns
    -------
    dict or None
        Keys: ``"allocated_gib"``, ``"reserved_gib"``, ``"total_gib"``,
        ``"free_gib"``, ``"device_name"``.
        Returns ``None`` on CPU-only systems.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(device_index)
    total = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    free = total - reserved  # conservative free estimate

    return {
        "device_name": props.name,
        "allocated_gib": allocated,
        "reserved_gib": reserved,
        "total_gib": total,
        "free_gib": free,
    }


def log_vram(label: str = "", device_index: int = 0) -> None:
    """Print a one-line VRAM summary to stdout.

    No-ops silently when CUDA is unavailable, so it is safe to call
    unconditionally in scripts that support both CPU and GPU paths.

    Parameters
    ----------
    label:
        Optional context string printed alongside the numbers
        (e.g. ``"after model load"``).
    device_index:
        CUDA device ordinal.
    """
    stats = vram_stats(device_index)
    if stats is None:
        return

    context = f" [{label}]" if label else ""
    print(
        f"VRAM{context}: {stats['allocated_gib']:.2f} GiB allocated / "
        f"{stats['reserved_gib']:.2f} GiB reserved / "
        f"{stats['total_gib']:.1f} GiB total  "
        f"({stats['device_name']})"
    )


# ---------------------------------------------------------------------------
# Model teardown
# ---------------------------------------------------------------------------


def release_model(model, *, clear_cache: bool = True) -> None:
    """Delete a model and free GPU memory.

    When loading multiple large models sequentially (e.g. source and
    target models in an alignment experiment) it is important to
    explicitly release each model before loading the next one.
    Python's reference-counting GC is not always enough because
    CUDA tensors can linger in the memory allocator's pool.

    Parameters
    ----------
    model:
        Any PyTorch model (HuggingFace, TransformerLens, etc.).
        The reference is deleted; do not use the variable afterward.
    clear_cache:
        If ``True`` (default), call ``torch.cuda.empty_cache()`` after
        deletion, returning pooled memory to the OS/CUDA allocator.
        Set to ``False`` only if you are loading a replacement model
        immediately and want to avoid the cache-flush overhead.
    """
    del model
    gc.collect()
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Convenience: print a startup banner
# ---------------------------------------------------------------------------


def log_device_info(device: str, dtype: torch.dtype) -> None:
    """Print a short startup summary of the selected device and dtype.

    Useful at the top of experiment scripts to confirm configuration
    before the (potentially slow) model download/load begins.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    dtype:
        The torch dtype that will be used for model parameters.
    """
    dtype_name = "fp16" if dtype == torch.float16 else "fp32"
    if device == "cuda":
        stats = vram_stats()
        if stats:
            print(
                f"Device: {device} ({stats['device_name']})  |  "
                f"dtype: {dtype_name}  |  "
                f"VRAM available: {stats['free_gib']:.1f} / {stats['total_gib']:.1f} GiB"
            )
        else:
            print(f"Device: {device}  |  dtype: {dtype_name}")
    else:
        print(f"Device: {device}  |  dtype: {dtype_name}  |  (no GPU)")

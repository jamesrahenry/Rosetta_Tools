"""
gpu_utils.py — Device selection, VRAM reporting, and model teardown.

Environment-aware: dtype defaults differ between laptop (consumer GPU or CPU)
and datacenter hardware (H100/A100 class).

dtype policy
------------
H100 / A100 (Ampere+, VRAM ≥ 40 GB):
    Default: bfloat16.  Same exponent range as float32 — cannot overflow at
    deep-layer activations.  The fp16 overflow that corrupted the original
    GPT-2-XL credibility results (L32+ Fisher normalization → 0) cannot
    occur with bfloat16.  On 80 GB VRAM, memory is not the bottleneck.

Consumer / laptop GPU (VRAM < 40 GB, or pre-Ampere):
    Default: float32.  fp16 saves memory but introduces overflow risk for
    deep models.  On a 4 GB laptop GPU running GPT-2-XL for a PoC, float32
    is preferred; the model still fits and results are trustworthy.

CPU:
    Always float32.  bfloat16 CPU support is inconsistent.

The threshold (40 GB) distinguishes H100 (80 GB) and A100 (40/80 GB) from
consumer cards (RTX 500 Ada = 4 GB, RTX 4090 = 24 GB, etc.).

Metric computation (Fisher normalization, PCA, covariance) is always done
in float64 regardless of model dtype — see rosetta_tools.caz and
rosetta_tools.extraction.  This module controls only forward-pass dtype.

Public API
----------
get_device(prefer="auto") -> str
    Returns "cuda" or "cpu".

get_dtype(device, prefer="auto") -> torch.dtype
    Returns the environment-appropriate dtype.
    prefer="auto"     — bfloat16 on datacenter GPU, float32 elsewhere.
    prefer="bfloat16" — bfloat16 if supported, else float32.
    prefer="float32"  — always float32.

vram_stats(device_index=0) -> dict | None
    Current VRAM usage in GiB.  None on CPU.

log_vram(label="", device_index=0) -> None
    Print a one-line VRAM summary.  No-ops silently on CPU.

log_device_info(device, dtype) -> None
    Print a startup banner confirming device, dtype, and VRAM.

release_model(model, *, clear_cache=True) -> None
    Delete model and free GPU memory.
"""

from __future__ import annotations

import gc
from typing import Literal, Optional

import torch

DtypePreference = Literal["auto", "bfloat16", "float32"]

# VRAM threshold above which hardware is treated as datacenter-class.
# H100 = 80 GiB, A100 = 40/80 GiB.  Consumer cards top out at ~24 GiB.
_DATACENTER_VRAM_GIB = 40.0


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(prefer: str = "auto") -> str:
    """Return the device string for model loading and tensor ops.

    Parameters
    ----------
    prefer:
        ``"auto"``  — CUDA if available, otherwise CPU (default).
        ``"cuda"``  — require CUDA; raises RuntimeError if unavailable.
        ``"cpu"``   — always CPU.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device 'cuda' was requested but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# dtype selection
# ---------------------------------------------------------------------------


def _is_datacenter_gpu(device_index: int = 0) -> bool:
    """Return True if the current GPU has datacenter-class VRAM (≥ 40 GiB)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device_index)
    total_gib = props.total_memory / 1024**3
    return total_gib >= _DATACENTER_VRAM_GIB


def get_dtype(
    device: str,
    prefer: DtypePreference = "auto",
) -> torch.dtype:
    """Return the appropriate dtype for model weights and forward passes.

    Environment-aware: dtype is chosen based on detected hardware when
    ``prefer="auto"`` (the default).

    Datacenter GPU (H100 / A100, VRAM ≥ 40 GiB):
        bfloat16.  Same exponent range as float32 — cannot overflow at
        deep-layer activations.  80 GB VRAM means memory is not the
        constraint.  This is the dtype for H100 DO droplet runs.

    Consumer / laptop GPU (VRAM < 40 GiB):
        float32.  fp16 saves memory but introduces silent overflow risk
        for deep models (the Fisher normalization failure in the original
        GPT-2-XL credibility results was caused by fp16 on a 4 GB card).
        For PoC runs on a laptop, correctness matters more than throughput.

    CPU:
        float32 always.  bfloat16 CPU support is inconsistent.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    prefer:
        ``"auto"``     — environment-aware selection (default).
        ``"bfloat16"`` — bfloat16 if GPU supports it, else float32.
        ``"float32"``  — always float32.

    Returns
    -------
    torch.dtype
    """
    if device == "cpu":
        return torch.float32

    if prefer == "float32":
        return torch.float32

    if prefer == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Pre-Ampere GPU — fall back to float32, not float16
        return torch.float32

    # "auto" — environment-aware
    if _is_datacenter_gpu():
        # Datacenter: bfloat16 if supported (it will be on any H100/A100)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    else:
        # Consumer / laptop: float32 for numerical safety
        return torch.float32


# ---------------------------------------------------------------------------
# VRAM reporting
# ---------------------------------------------------------------------------


def vram_stats(device_index: int = 0) -> Optional[dict]:
    """Return current VRAM usage figures, or None on CPU.

    All values are in gibibytes (GiB).

    Returns
    -------
    dict with keys:
        ``"device_name"``, ``"allocated_gib"``, ``"reserved_gib"``,
        ``"total_gib"``, ``"free_gib"``.
    None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(device_index)
    total = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3

    return {
        "device_name": props.name,
        "allocated_gib": allocated,
        "reserved_gib": reserved,
        "total_gib": total,
        "free_gib": total - reserved,  # conservative: reserved - allocated stays pooled
    }


def log_vram(label: str = "", device_index: int = 0) -> None:
    """Print a one-line VRAM summary.  No-ops silently on CPU."""
    stats = vram_stats(device_index)
    if stats is None:
        return
    context = f" [{label}]" if label else ""
    print(
        f"VRAM{context}: "
        f"{stats['allocated_gib']:.2f} GiB allocated / "
        f"{stats['reserved_gib']:.2f} GiB reserved / "
        f"{stats['total_gib']:.1f} GiB total  "
        f"({stats['device_name']})"
    )


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def log_device_info(device: str, dtype: torch.dtype) -> None:
    """Print a startup summary before a long model run.

    Confirms device, dtype, and available VRAM so that misconfiguration
    is visible before the (potentially slow) model download begins.
    """
    dtype_name = {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
    }.get(dtype, str(dtype))

    if device == "cuda":
        stats = vram_stats()
        if stats:
            print(
                f"Device: {device} ({stats['device_name']})  |  "
                f"dtype: {dtype_name}  |  "
                f"VRAM free: {stats['free_gib']:.1f} / {stats['total_gib']:.1f} GiB"
            )
            return
    print(f"Device: {device}  |  dtype: {dtype_name}")


# ---------------------------------------------------------------------------
# Model teardown
# ---------------------------------------------------------------------------


def release_model(model, *, clear_cache: bool = True) -> None:
    """Delete a model and free GPU memory.

    When loading multiple large models sequentially, explicit release is
    important — Python's reference-counting GC is not sufficient because
    CUDA tensors can linger in the memory allocator's pool.

    Parameters
    ----------
    model:
        Any PyTorch model.  Do not use the variable after calling this.
    clear_cache:
        If True (default), call torch.cuda.empty_cache() after deletion,
        returning pooled memory to the CUDA allocator.  Set to False only
        if loading a replacement model immediately.
    """
    del model
    gc.collect()
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

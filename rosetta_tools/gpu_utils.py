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
import json
import os
import shutil
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# HuggingFace download filtering  (2026-04-10)
# ---------------------------------------------------------------------------
# huggingface_hub >=1.5.0 enables the Xet download backend by default.
# Xet pulls entire model repos (tflite, onnx, flax, rust weights) instead
# of just safetensors + config + tokenizer, causing multi-GB stalls.
# Disable it at import time so notebooks and scripts that import
# rosetta_tools get the fix automatically — no per-user env setup needed.

if not os.environ.get("HF_HUB_DISABLE_XET"):
    os.environ["HF_HUB_DISABLE_XET"] = "1"

_HF_IGNORE_PATTERNS = [
    "*.tflite",
    "*.onnx",
    "*.ot",           # rust_model.ot
    "*.msgpack",      # flax_model.msgpack
    "*.h5",           # tf_model.h5
    "onnx/*",
]

try:
    import huggingface_hub.constants as _hf_constants
    if hasattr(_hf_constants, "DEFAULT_IGNORE_PATTERNS"):
        _hf_constants.DEFAULT_IGNORE_PATTERNS = list(
            set(getattr(_hf_constants, "DEFAULT_IGNORE_PATTERNS", []))
            | set(_HF_IGNORE_PATTERNS)
        )
except ImportError:
    pass

DtypePreference = Literal["auto", "bfloat16", "float32"]

# VRAM threshold above which hardware is treated as datacenter-class.
# H100 = 80 GiB, A100 = 40/80 GiB.  Consumer cards top out at ~24 GiB.
_DATACENTER_VRAM_GIB = 40.0

# GPU architectures known to have robust bf16 support (Ampere+).
# Used to safely enable bf16 even on cards below the datacenter VRAM threshold.
_BF16_CAPABLE_ARCHS = {"Ada", "Ampere", "Hopper", "Blackwell"}

# Partial name matches for GPUs with native bf16 (for fallback when arch isn't reported)
_BF16_CAPABLE_NAMES = {"L4", "L40", "A10", "A16", "RTX 40", "RTX 50", "RTX 30"}


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


def _is_bf16_capable(device_index: int = 0) -> bool:
    """Return True if the GPU natively supports bfloat16.

    Checks compute capability (>= 8.0 is Ampere+) and falls back to
    name-based matching for known bf16-capable cards like L4, A10, etc.
    This allows bf16 on multi-GPU cloud VMs (e.g. 2×L4) where individual
    GPUs are below the datacenter VRAM threshold but fully support bf16.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device_index)
    # Compute capability 8.0+ = Ampere or newer = native bf16
    if props.major >= 8:
        return True
    # Fallback: name-based check
    name = props.name
    return any(n in name for n in _BF16_CAPABLE_NAMES)


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
    if _is_datacenter_gpu() or _is_bf16_capable():
        # Datacenter or known bf16-capable GPU (L4, A10, RTX 30/40/50 series)
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
# Adaptive batch size
# ---------------------------------------------------------------------------


def safe_batch_size(requested: int, device: str = "cuda", reserve_gib: float = 2.0) -> int:
    """Scale down batch size if VRAM is tight after model loading.

    Heuristic: if free VRAM (minus a safety reserve) is less than 25% of
    total, halve the batch size.  Repeat until batch_size=1 or there's room.

    Returns the (possibly reduced) batch size.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return requested

    stats = vram_stats()
    if stats is None:
        return requested

    free = stats["free_gib"] - reserve_gib
    total = stats["total_gib"]
    batch = requested

    while batch > 1 and free < 0.25 * total:
        batch = max(1, batch // 2)
        # After halving, the headroom estimate improves proportionally
        # (rough: half the batch → half the activation memory)
        free *= 2

    if batch < requested:
        print(
            f"WARNING: Reduced batch_size {requested} → {batch} "
            f"(only {stats['free_gib']:.1f} GiB free of {total:.1f} GiB)"
        )

    return batch


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
    # Move all parameters to CPU before deletion to free CUDA tensors
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reset memory stats to avoid stale peak tracking
        torch.cuda.reset_peak_memory_stats()
        # Second gc pass — catches reference cycles broken by first pass
        gc.collect()
        torch.cuda.empty_cache()


def purge_hf_cache(model_id: str) -> None:
    """Delete a HuggingFace model from the local cache.

    Useful when running many large models sequentially on disk-constrained
    machines (e.g. cloud VMs). Call after extraction is complete and results
    are saved — model weights are no longer needed.

    Parameters
    ----------
    model_id:
        HuggingFace model ID (e.g. ``"EleutherAI/pythia-6.9b"``).
        Converted to the cache directory format (``models--org--name``).
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    # HF cache uses models--<org>--<name> directory format
    cache_name = "models--" + model_id.replace("/", "--")
    cache_dir = cache_root / cache_name

    if cache_dir.exists():
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1024**3
        shutil.rmtree(cache_dir)
        print(f"Purged HF cache: {model_id} ({size_gb:.1f} GB freed)")
    else:
        # Try HF_HOME env var
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = Path(hf_home) / "hub" / cache_name
            if cache_dir.exists():
                size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1024**3
                shutil.rmtree(cache_dir)
                print(f"Purged HF cache: {model_id} ({size_gb:.1f} GB freed)")


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays.

    numpy int64/int32 and float32/float64 scalars are not JSON-serializable
    by default.  This encoder converts them to their Python equivalents so
    that json.dump/json.dumps works on dicts that contain numpy values — e.g.
    feature maps produced by feature_tracker.py where np.unravel_index returns
    int64 indices that flow into the saved results.

    Usage::

        import json
        from rosetta_tools.gpu_utils import NumpyJSONEncoder
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2)
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Resilient model loading (retry on network drops)
# ---------------------------------------------------------------------------

def load_model_with_retry(
    model_cls,
    model_id: str,
    *,
    dtype,
    device: str,
    max_retries: int = 15,
    retry_delay: float = 10.0,
):
    """Load a HuggingFace model, retrying on network errors.

    Standard HTTP downloads can drop mid-shard on flaky networks
    (IncompleteRead / ChunkedEncodingError).  The HF cache stores partial
    downloads, so retrying resumes from where it left off.

    Parameters
    ----------
    model_cls:
        AutoModel class to use, e.g. ``AutoModelForCausalLM`` or ``AutoModel``.
    model_id:
        HuggingFace model ID.
    dtype:
        torch.dtype to load with.
    device:
        Device string (``"cuda"``, ``"cpu"``, etc.).
    max_retries:
        Number of download attempts before raising.
    retry_delay:
        Seconds to wait between retries.
    """
    import logging
    log = logging.getLogger(__name__)

    # Check disk pressure before downloading.  Models range from ~300 MB to
    # ~14 GB; if /home is near full the download will corrupt mid-shard.
    # Purge the HF cache if we're below 20 GiB so the download has headroom.
    _DISK_FLOOR_GIB = 20
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        stat = os.statvfs(hf_home)
        free_gib = stat.f_bavail * stat.f_frsize / 1024**3
        if free_gib < _DISK_FLOOR_GIB:
            log.warning(
                "Low disk before loading %s: %.1f GiB free — purging HF cache",
                model_id, free_gib,
            )
            purge_hf_cache(model_id)  # only purges this model's cache if present
            # Also clear other cached models to free headroom
            hub_dir = hf_home / "hub"
            if hub_dir.exists():
                import shutil as _shutil
                cleared = 0
                for entry in sorted(hub_dir.iterdir()):
                    if entry.name.startswith("models--") and entry.name != f"models--{model_id.replace('/', '--')}":
                        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                        _shutil.rmtree(entry, ignore_errors=True)
                        cleared += size
                        if cleared > 10 * 1024**3:  # stop after freeing 10 GB
                            break
                log.info("Purged %.1f GB of cached models to free disk", cleared / 1024**3)
    except OSError:
        pass  # statvfs can fail on some mounts; just continue

    # Phase 1: download weights sequentially (max_workers=1) with retries.
    #
    # from_pretrained on sharded models calls snapshot_download internally
    # with a ThreadPoolExecutor — multiple parallel shard downloads.  On a
    # flaky connection every parallel attempt fails.  Pre-downloading with
    # max_workers=1 serialises the shards so one bad shard doesn't kill
    # the whole batch, and lets us retry just the failed shard.
    # After a successful snapshot_download, Phase 2 loads from local cache
    # only (local_files_only=True) so no more network access.
    from huggingface_hub import snapshot_download

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info(
                "Downloading %s (attempt %d/%d, sequential shards)…",
                model_id, attempt, max_retries,
            )
            snapshot_download(
                model_id,
                max_workers=1,           # one shard at a time — survives flaky links
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "rust_model*",
                                  "tf_model*", "*.ot"],
            )
            break  # download succeeded
        except OSError as exc:
            last_exc = exc
            if attempt < max_retries:
                log.warning(
                    "Download error for %s (attempt %d/%d): %s — retrying in %.0fs",
                    model_id, attempt, max_retries, exc, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                log.error("All %d download attempts failed for %s", max_retries, model_id)
                raise

    # Phase 2: load from local cache — no network access.
    try:
        return model_cls.from_pretrained(
            model_id, dtype=dtype, device_map=device, local_files_only=True,
        )
    except (ValueError, ImportError):
        return model_cls.from_pretrained(
            model_id, dtype=dtype, local_files_only=True,
        ).to(device)

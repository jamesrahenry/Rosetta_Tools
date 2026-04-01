"""
test_gpu_utils.py — Tests for rosetta_tools.gpu_utils.

Tests device selection, dtype logic, and model teardown.
Runs on CPU — GPU tests are skipped if CUDA is unavailable.
"""

import gc
from unittest.mock import MagicMock, patch

import pytest
import torch

from rosetta_tools.gpu_utils import (
    get_device,
    get_dtype,
    release_model,
    vram_stats,
    _is_bf16_capable,
)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


class TestGetDevice:

    def test_cpu_always_works(self):
        assert get_device("cpu") == "cpu"

    def test_auto_returns_string(self):
        result = get_device("auto")
        assert result in ("cuda", "cpu")

    def test_cuda_raises_if_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test unavailable path")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            get_device("cuda")


# ---------------------------------------------------------------------------
# get_dtype
# ---------------------------------------------------------------------------


class TestGetDtype:

    def test_cpu_always_float32(self):
        assert get_dtype("cpu") == torch.float32

    def test_cpu_ignores_prefer(self):
        assert get_dtype("cpu", prefer="bfloat16") == torch.float32

    def test_float32_override(self):
        assert get_dtype("cpu", prefer="float32") == torch.float32
        # Even on CUDA, float32 should be honored
        if torch.cuda.is_available():
            assert get_dtype("cuda", prefer="float32") == torch.float32


# ---------------------------------------------------------------------------
# vram_stats
# ---------------------------------------------------------------------------


class TestVramStats:

    def test_returns_none_on_cpu(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA available")
        assert vram_stats() is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_returns_dict_on_gpu(self):
        stats = vram_stats()
        assert isinstance(stats, dict)
        assert "total_gib" in stats
        assert "device_name" in stats
        assert stats["total_gib"] > 0


# ---------------------------------------------------------------------------
# release_model
# ---------------------------------------------------------------------------


class TestReleaseModel:

    def test_deletes_model(self):
        """release_model should not raise even on a simple object."""
        model = torch.nn.Linear(4, 4)
        release_model(model)
        # model is deleted — just verify no exception

    def test_clears_cache_flag(self):
        model = torch.nn.Linear(4, 4)
        # Should not raise with clear_cache=False either
        release_model(model, clear_cache=False)


# ---------------------------------------------------------------------------
# _is_bf16_capable
# ---------------------------------------------------------------------------


class TestBf16Capable:

    def test_no_cuda_returns_false(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA available")
        assert _is_bf16_capable() is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_returns_bool_on_gpu(self):
        result = _is_bf16_capable()
        assert isinstance(result, bool)

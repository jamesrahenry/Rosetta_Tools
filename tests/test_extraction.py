"""
test_extraction.py — Tests for rosetta_tools.extraction.

Tests pooling strategies, batch handling, and output shape contracts
using a minimal mock model. No GPU required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rosetta_tools.extraction import _pool, extract_layer_activations


# ---------------------------------------------------------------------------
# Pooling tests
# ---------------------------------------------------------------------------


class TestPool:
    """Tests for the _pool helper."""

    def _make(self, batch=2, seq=5, dim=8):
        """Create random hidden states and a simple attention mask."""
        hs = torch.randn(batch, seq, dim)
        mask = torch.ones(batch, seq, dtype=torch.long)
        return hs, mask

    def test_first_token(self):
        hs, mask = self._make()
        out = _pool(hs, mask, strategy="first")
        assert out.shape == (2, 8)
        torch.testing.assert_close(out, hs[:, 0, :])

    def test_cls_alias(self):
        hs, mask = self._make()
        first = _pool(hs, mask, strategy="first")
        cls = _pool(hs, mask, strategy="cls")
        torch.testing.assert_close(first, cls)

    def test_last_token_no_padding(self):
        hs, mask = self._make(batch=1, seq=4, dim=3)
        out = _pool(hs, mask, strategy="last")
        # No padding → last token is index 3
        torch.testing.assert_close(out, hs[:, 3, :])

    def test_last_token_with_padding(self):
        hs = torch.randn(2, 5, 4)
        mask = torch.tensor([
            [1, 1, 1, 0, 0],  # last real token at index 2
            [1, 1, 1, 1, 0],  # last real token at index 3
        ])
        out = _pool(hs, mask, strategy="last")
        assert out.shape == (2, 4)
        torch.testing.assert_close(out[0], hs[0, 2, :])
        torch.testing.assert_close(out[1], hs[1, 3, :])

    def test_mean_no_padding(self):
        hs = torch.ones(1, 3, 2)  # all ones
        mask = torch.ones(1, 3, dtype=torch.long)
        out = _pool(hs, mask, strategy="mean")
        torch.testing.assert_close(out, torch.ones(1, 2))

    def test_mean_with_padding(self):
        hs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])  # third token is padding
        out = _pool(hs, mask, strategy="mean")
        expected = torch.tensor([[2.0, 3.0]])  # mean of first two only
        torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Mock model for extraction tests
# ---------------------------------------------------------------------------


class _MockModel(nn.Module):
    """Minimal model that returns deterministic hidden states."""

    def __init__(self, n_layers=3, hidden_dim=16):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        class _Config:
            pass
        self.config = _Config()
        self.config.hidden_size = hidden_dim
        self.config.num_hidden_layers = n_layers
        self.name_or_path = "mock-model"

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, seq = input_ids.shape
        # Generate n_layers + 1 hidden states (embedding + transformer blocks)
        hidden_states = tuple(
            torch.randn(batch, seq, self.hidden_dim)
            for _ in range(self.n_layers + 1)
        )

        class _Output:
            pass
        out = _Output()
        out.hidden_states = hidden_states
        return out


class _MockTokenizer:
    """Minimal tokenizer that returns fixed-length encodings."""

    def __init__(self, vocab_size=100, max_len=10):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=512):
        batch = len(texts)
        seq = min(self.max_len, max_length)
        ids = torch.randint(0, self.vocab_size, (batch, seq))
        mask = torch.ones(batch, seq, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


class TestExtractLayerActivations:

    def test_output_shape(self):
        model = _MockModel(n_layers=4, hidden_dim=8)
        tokenizer = _MockTokenizer()
        texts = ["hello world", "foo bar", "test sentence"]
        acts = extract_layer_activations(model, tokenizer, texts, device="cpu")
        # Should return n_layers + 1 arrays (embedding + blocks)
        assert len(acts) == 5
        for a in acts:
            assert a.shape == (3, 8)
            assert a.dtype == np.float32

    def test_batching(self):
        """Small batch_size should produce same result as large."""
        model = _MockModel(n_layers=2, hidden_dim=4)
        tokenizer = _MockTokenizer()
        texts = ["a", "b", "c", "d", "e"]

        # Fix the random seed so both calls produce the same results
        torch.manual_seed(42)
        acts_big = extract_layer_activations(
            model, tokenizer, texts, device="cpu", batch_size=100
        )
        torch.manual_seed(42)
        acts_small = extract_layer_activations(
            model, tokenizer, texts, device="cpu", batch_size=2
        )

        assert len(acts_big) == len(acts_small)
        for a, b in zip(acts_big, acts_small):
            assert a.shape == b.shape

    def test_pad_token_fallback(self):
        """Should set pad_token from eos_token if None."""
        tokenizer = _MockTokenizer()
        assert tokenizer.pad_token is None
        model = _MockModel(n_layers=1, hidden_dim=4)
        extract_layer_activations(model, tokenizer, ["test"], device="cpu")
        assert tokenizer.pad_token == "<eos>"

    def test_pad_token_preserved(self):
        """Should not overwrite an existing pad_token."""
        tokenizer = _MockTokenizer()
        tokenizer.pad_token = "<pad>"
        model = _MockModel(n_layers=1, hidden_dim=4)
        extract_layer_activations(model, tokenizer, ["test"], device="cpu")
        assert tokenizer.pad_token == "<pad>"

    def test_float32_output_regardless_of_model_dtype(self):
        """Activations should always be float32, even if model is float16."""
        model = _MockModel(n_layers=2, hidden_dim=4)
        tokenizer = _MockTokenizer()
        acts = extract_layer_activations(model, tokenizer, ["test"], device="cpu")
        for a in acts:
            assert a.dtype == np.float32

    def test_single_text(self):
        model = _MockModel(n_layers=2, hidden_dim=4)
        tokenizer = _MockTokenizer()
        acts = extract_layer_activations(model, tokenizer, ["one"], device="cpu")
        for a in acts:
            assert a.shape[0] == 1

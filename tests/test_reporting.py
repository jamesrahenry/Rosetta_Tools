"""
test_reporting.py — Tests for rosetta_tools.reporting.

Tests JSON result loading, concept inference, and DataFrame construction
using temporary files. No GPU, no model loading.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rosetta_tools.reporting import (
    _infer_concept,
    load_result_df,
    load_results_dir,
    load_run_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_caz_json(
    model_id="test-model",
    concept="credibility",
    n_pairs=50,
    n_layers=4,
    peak_layer=2,
):
    """Create a minimal frontier-format CAZ JSON structure."""
    metrics = []
    for i in range(n_layers):
        sep = 0.5 + (0.3 if i == peak_layer else 0.0)
        metrics.append({
            "layer": i,
            "separation_fisher": sep,
            "coherence": 0.9,
            "raw_distance": 1.0,
            "dom_vector": [0.0] * 8,
            "velocity": 0.1,
        })
    return {
        "model_id": model_id,
        "concept": concept,
        "n_pairs": n_pairs,
        "hidden_dim": 8,
        "n_layers": n_layers,
        "token_pos": -1,
        "extraction_seconds": 1.0,
        "layer_data": {
            "n_layers": n_layers,
            "metrics": metrics,
            "peak_layer": peak_layer,
            "peak_separation": metrics[peak_layer]["separation_fisher"],
            "peak_depth_pct": round(100.0 * peak_layer / n_layers, 1),
        },
    }


def _make_run_summary(model_id="test-model", concepts=None):
    """Create a minimal run_summary.json structure."""
    if concepts is None:
        concepts = ["credibility", "negation"]
    return {
        "model_id": model_id,
        "concepts": concepts,
        "n_pairs_cap": 100,
        "device": "cuda",
        "dtype": "torch.bfloat16",
        "total_seconds": 10.0,
        "timestamp": "20260401_120000",
        "results": [
            {
                "concept": c,
                "n_pairs": 100,
                "peak_layer": 5,
                "peak_separation": 0.8,
                "peak_depth_pct": 50.0,
                "extraction_seconds": 5.0,
                "output_file": f"caz_{c}.json",
            }
            for c in concepts
        ],
    }


# ---------------------------------------------------------------------------
# _infer_concept
# ---------------------------------------------------------------------------


class TestInferConcept:

    def test_frontier_filename(self):
        assert _infer_concept(Path("results/run1/caz_credibility.json")) == "credibility"
        assert _infer_concept(Path("results/run1/caz_negation.json")) == "negation"
        assert _infer_concept(Path("results/run1/caz_moral_valence.json")) == "moral_valence"

    def test_legacy_directory(self):
        assert _infer_concept(Path("expanded_credibility_gpt2_20260315/caz_extraction.json")) == "credibility"

    def test_unknown_returns_empty(self):
        assert _infer_concept(Path("results/random_dir/unknown.json")) == ""


# ---------------------------------------------------------------------------
# load_result_df
# ---------------------------------------------------------------------------


class TestLoadResultDf:

    def test_basic_load(self, tmp_path):
        data = _make_caz_json(n_layers=4, peak_layer=2)
        path = tmp_path / "caz_credibility.json"
        path.write_text(json.dumps(data))

        df = load_result_df(path)
        assert len(df) == 4
        assert set(df.columns) >= {"model_id", "concept", "layer", "separation", "is_peak", "depth_pct"}
        assert df["concept"].iloc[0] == "credibility"
        assert df["model_id"].iloc[0] == "test-model"
        # Peak should be at layer 2
        peak_rows = df[df["is_peak"]]
        assert len(peak_rows) == 1
        assert peak_rows.iloc[0]["layer"] == 2

    def test_depth_pct_calculation(self, tmp_path):
        data = _make_caz_json(n_layers=10, peak_layer=5)
        path = tmp_path / "caz_certainty.json"
        path.write_text(json.dumps(data))

        df = load_result_df(path)
        row_5 = df[df["layer"] == 5].iloc[0]
        assert row_5["depth_pct"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# load_results_dir
# ---------------------------------------------------------------------------


class TestLoadResultsDir:

    def test_loads_multiple_concepts(self, tmp_path):
        for concept in ["credibility", "negation"]:
            data = _make_caz_json(concept=concept, n_layers=4, peak_layer=2)
            path = tmp_path / f"caz_{concept}.json"
            path.write_text(json.dumps(data))

        df = load_results_dir(tmp_path)
        assert len(df) == 8  # 2 concepts × 4 layers
        assert set(df["concept"].unique()) == {"credibility", "negation"}

    def test_multiple_directories(self, tmp_path):
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()

        data_a = _make_caz_json(model_id="model-a", concept="credibility", n_layers=3)
        data_b = _make_caz_json(model_id="model-b", concept="credibility", n_layers=3)
        (dir_a / "caz_credibility.json").write_text(json.dumps(data_a))
        (dir_b / "caz_credibility.json").write_text(json.dumps(data_b))

        df = load_results_dir([dir_a, dir_b])
        assert df["model_id"].nunique() == 2

    def test_empty_directory(self, tmp_path):
        df = load_results_dir(tmp_path)
        assert df.empty

    def test_skips_run_summary(self, tmp_path):
        # run_summary.json should not be loaded as a CAZ result
        summary = _make_run_summary()
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))
        data = _make_caz_json()
        (tmp_path / "caz_credibility.json").write_text(json.dumps(data))

        df = load_results_dir(tmp_path)
        assert len(df) == 4  # only the caz_credibility.json, not run_summary


# ---------------------------------------------------------------------------
# load_run_summary
# ---------------------------------------------------------------------------


class TestLoadRunSummary:

    def test_basic(self, tmp_path):
        summary = _make_run_summary(concepts=["credibility", "negation", "sentiment"])
        path = tmp_path / "run_summary.json"
        path.write_text(json.dumps(summary))

        df = load_run_summary(path)
        assert len(df) == 3
        assert set(df.columns) >= {"model_id", "concept", "peak_layer", "peak_separation"}

    def test_from_directory(self, tmp_path):
        summary = _make_run_summary()
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))

        df = load_run_summary(tmp_path)  # pass directory, not file
        assert len(df) == 2

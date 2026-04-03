"""
reporting.py — Pandas-based result loading for CAZ experiments.

Converts JSON checkpoint files produced by the extraction pipeline into
tidy long-form DataFrames suitable for analysis and visualization.

Two result formats are supported:

  Frontier format (extract_caz_frontier.py):
    caz_<concept>.json with top-level 'concept', 'n_pairs', and pre-computed
    peak fields in layer_data.

  Legacy format (extract_vectors_caz.py):
    caz_extraction.json — concept not stored in file; inferred from the
    parent directory name (e.g. expanded_credibility_gpt2_<timestamp>).

Both produce the same output schema.

Typical usage
-------------
    from rosetta_tools.reporting import load_result_df, load_results_dir

    # Single checkpoint file
    df = load_result_df("results/frontier_llama/caz_credibility.json")

    # All checkpoints in a run directory → combined DataFrame
    df = load_results_dir("results/frontier_llama_20260318_120000/")

    # Multiple run dirs (cross-model comparison)
    df = load_results_dir(["results/dir_a/", "results/dir_b/"])

Schema
------
One row per (model_id, concept, layer).

    model_id          str    HuggingFace model identifier
    concept           str    concept name (credibility, negation, …)
    n_pairs           int    number of contrastive pairs used (−1 if unknown)
    n_layers          int    total transformer layers in this model
    layer             int    layer index (0-based, embedding excluded)
    depth_pct         float  layer / n_layers * 100
    separation        float  Fisher separation score
    coherence         float  coherence score
    raw_distance      float  L2 distance between class centroids
    velocity          float  rate of change in separation (windowed)
    is_peak           bool   True for the peak separation layer
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# All known concept names — used to extract concept from legacy directory names
_KNOWN_CONCEPTS = {
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order", "plurality",
}


def _infer_concept(path: Path) -> str:
    """Best-effort concept extraction from a file or directory path.

    Checks the filename stem first (frontier: caz_credibility.json → credibility),
    then walks parent directory name tokens against the known concept set.
    Returns an empty string if no concept can be determined.
    """
    # Frontier filename: caz_<concept>.json
    stem = path.stem  # e.g. "caz_credibility"
    if stem.startswith("caz_"):
        candidate = stem[4:]
        if candidate in _KNOWN_CONCEPTS:
            return candidate

    # Legacy directory: expanded_credibility_gpt2_20260315_131312
    for part in path.parts:
        for token in part.split("_"):
            if token in _KNOWN_CONCEPTS:
                return token
        # multi-word concepts joined by underscore (moral_valence, temporal_order)
        for concept in _KNOWN_CONCEPTS:
            if concept in part:
                return concept

    return ""


def load_result_df(path: str | Path) -> pd.DataFrame:
    """Load a single CAZ checkpoint JSON into a tidy long-form DataFrame.

    Handles both the frontier format (caz_<concept>.json) and the legacy
    format (caz_extraction.json). See module docstring for schema.

    Parameters
    ----------
    path:
        Path to a CAZ checkpoint JSON file.

    Returns
    -------
    pd.DataFrame
        Tidy long-form with one row per layer.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    model_id = data["model_id"]
    layer_data = data["layer_data"]
    n_layers = layer_data["n_layers"]
    metrics = layer_data["metrics"]

    # Determine concept and n_pairs — present in frontier format, absent in legacy
    concept = data.get("concept") or _infer_concept(path)
    n_pairs = data.get("n_pairs", -1)

    # Peak layer — pre-computed in frontier format, derived here for legacy
    if "peak_layer" in layer_data:
        peak_layer = layer_data["peak_layer"]
    else:
        seps = [m["separation_fisher"] for m in metrics]
        peak_layer = int(np.argmax(seps))

    rows = []
    for m in metrics:
        layer = m["layer"]
        rows.append(
            {
                "model_id": model_id,
                "concept": concept,
                "n_pairs": n_pairs,
                "n_layers": n_layers,
                "layer": layer,
                "depth_pct": round(100.0 * layer / n_layers, 2),
                "separation": m["separation_fisher"],
                "coherence": m["coherence"],
                "raw_distance": m["raw_distance"],
                "velocity": m["velocity"],
                "is_peak": layer == peak_layer,
            }
        )

    return pd.DataFrame(rows)


def load_results_dir(
    path: str | Path | list[str | Path],
    glob: str = "caz_*.json",
    include_legacy: bool = True,
) -> pd.DataFrame:
    """Load all CAZ checkpoint files from one or more result directories.

    Parameters
    ----------
    path:
        A single directory path, or a list of directory paths.
    glob:
        Glob pattern for frontier-format checkpoint files.
        Default matches ``caz_<concept>.json``.
    include_legacy:
        If True (default), also load ``caz_extraction.json`` files found
        in subdirectories (legacy format from extract_vectors_caz.py).

    Returns
    -------
    pd.DataFrame
        Combined tidy long-form DataFrame, sorted by model_id, concept, layer.
        Returns an empty DataFrame if no files are found.
    """
    if isinstance(path, (str, Path)):
        dirs = [Path(path)]
    else:
        dirs = [Path(p) for p in path]

    frames: list[pd.DataFrame] = []

    # Files handled by the legacy path — skip in the frontier glob
    _legacy_names = {"caz_extraction.json", "run_summary.json"}

    for d in dirs:
        # Frontier-format files matching the glob (skip legacy and analysis files)
        for f in sorted(d.glob(glob)):
            if f.name in _legacy_names or f.name.startswith("caz_analysis"):
                continue
            try:
                frames.append(load_result_df(f))
            except (KeyError, json.JSONDecodeError) as e:
                warnings.warn(f"Skipping {f}: {e}", stacklevel=2)

        # Legacy extraction files (recursive search)
        if include_legacy:
            for f in sorted(d.rglob("caz_extraction.json")):
                try:
                    frames.append(load_result_df(f))
                except (KeyError, json.JSONDecodeError) as e:
                    warnings.warn(f"Skipping {f}: {e}", stacklevel=2)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["model_id", "concept", "layer"]).reset_index(drop=True)


def load_region_df(
    layer_df: pd.DataFrame,
    min_prominence_frac: float = 0.10,
) -> pd.DataFrame:
    """Compute per-region structural summary from a layer-wise DataFrame.

    Takes the output of ``load_results_dir`` (one row per layer) and returns
    one row per assembly region per (model_id, concept) pair, using the
    multi-modal ``find_caz_regions`` detector.

    Parameters
    ----------
    layer_df:
        Tidy DataFrame from ``load_results_dir`` or ``load_result_df``.
    min_prominence_frac:
        Passed to ``find_caz_regions``.  Default 0.10.

    Returns
    -------
    pd.DataFrame
        One row per region.  Columns:

        model_id, concept, n_layers, is_multimodal, n_regions,
        region_idx (0 = shallowest), is_dominant (bool),
        start, peak, end, width, width_pct, depth_pct,
        peak_separation, peak_coherence, mean_separation, mean_coherence,
        prominence, rise_span, fall_span
    """
    from rosetta_tools.caz import find_caz_regions, LayerMetrics

    rows = []
    for (model_id, concept), sub in layer_df.groupby(["model_id", "concept"]):
        sub = sub.sort_values("layer")
        metrics = [
            LayerMetrics(
                layer=int(r["layer"]),
                separation=r["separation"],
                coherence=r["coherence"],
                velocity=r["velocity"],
            )
            for _, r in sub.iterrows()
        ]
        try:
            profile = find_caz_regions(metrics, min_prominence_frac=min_prominence_frac)
        except Exception:
            continue

        dominant_peak = profile.dominant.peak
        for i, region in enumerate(profile.regions):
            rows.append({
                "model_id": model_id,
                "concept": concept,
                "n_layers": profile.n_layers,
                "is_multimodal": profile.is_multimodal,
                "n_regions": profile.n_regions,
                "region_idx": i,
                "is_dominant": region.peak == dominant_peak,
                "start": region.start,
                "peak": region.peak,
                "end": region.end,
                "width": region.width,
                "width_pct": region.width_pct,
                "depth_pct": region.depth_pct,
                "peak_separation": region.peak_separation,
                "peak_coherence": region.peak_coherence,
                "mean_separation": region.mean_separation,
                "mean_coherence": region.mean_coherence,
                "prominence": region.prominence,
                "rise_span": region.rise_span,
                "fall_span": region.fall_span,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["model_id", "concept", "region_idx"]
    ).reset_index(drop=True)


def load_scored_region_df(
    layer_df: pd.DataFrame,
    min_prominence_frac: float = 0.005,
) -> pd.DataFrame:
    """Compute per-region structural summary using the scored CAZ detector.

    Like ``load_region_df`` but uses ``find_caz_regions_scored`` with a low
    prominence floor (default 0.5%) to catch gentle CAZes.  Includes
    ``caz_score`` in output for downstream filtering/ranking.

    Parameters
    ----------
    layer_df:
        Tidy DataFrame from ``load_results_dir`` or ``load_result_df``.
    min_prominence_frac:
        Passed to ``find_caz_regions_scored``.  Default 0.005 (0.5%).

    Returns
    -------
    pd.DataFrame
        One row per region.  Columns include all of ``load_region_df`` plus
        ``caz_score`` (composite strength metric).
    """
    from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics

    rows = []
    for (model_id, concept), sub in layer_df.groupby(["model_id", "concept"]):
        sub = sub.sort_values("layer")
        metrics = [
            LayerMetrics(
                layer=int(r["layer"]),
                separation=r["separation"],
                coherence=r["coherence"],
                velocity=r["velocity"],
            )
            for _, r in sub.iterrows()
        ]
        try:
            profile = find_caz_regions_scored(
                metrics, min_prominence_frac=min_prominence_frac,
            )
        except Exception:
            continue

        dominant_peak = profile.dominant.peak
        for i, region in enumerate(profile.regions):
            rows.append({
                "model_id": model_id,
                "concept": concept,
                "n_layers": profile.n_layers,
                "is_multimodal": profile.is_multimodal,
                "n_regions": profile.n_regions,
                "region_idx": i,
                "is_dominant": region.peak == dominant_peak,
                "start": region.start,
                "peak": region.peak,
                "end": region.end,
                "width": region.width,
                "width_pct": region.width_pct,
                "depth_pct": region.depth_pct,
                "peak_separation": region.peak_separation,
                "peak_coherence": region.peak_coherence,
                "mean_separation": region.mean_separation,
                "mean_coherence": region.mean_coherence,
                "prominence": region.prominence,
                "rise_span": region.rise_span,
                "fall_span": region.fall_span,
                "caz_score": region.caz_score,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["model_id", "concept", "region_idx"]
    ).reset_index(drop=True)


def load_run_summary(path: str | Path) -> pd.DataFrame:
    """Load a run_summary.json index as a wide-form summary DataFrame.

    One row per concept. Useful for quick cross-concept peak comparisons
    without loading full layer-wise data.

    Parameters
    ----------
    path:
        Path to a ``run_summary.json`` file, or its parent directory.

    Returns
    -------
    pd.DataFrame
        Columns: model_id, concept, n_pairs, peak_layer, peak_separation,
        peak_depth_pct, extraction_seconds.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "run_summary.json"

    with path.open() as f:
        summary = json.load(f)

    model_id = summary["model_id"]
    rows = [
        {
            "model_id": model_id,
            "concept": r["concept"],
            "n_pairs": r["n_pairs"],
            "peak_layer": r["peak_layer"],
            "peak_separation": r["peak_separation"],
            "peak_depth_pct": r["peak_depth_pct"],
            "extraction_seconds": r["extraction_seconds"],
        }
        for r in summary["results"]
    ]
    return pd.DataFrame(rows)

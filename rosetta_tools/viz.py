"""
viz.py — Matplotlib helpers for CAZ and alignment visualizations.

All plotting functions accept pandas DataFrames produced by
``rosetta_tools.reporting`` and write PNG files to disk.

Concept metadata
----------------
CONCEPT_META maps concept names to display type and color:
    type: epistemic | affective | relational | syntactic
    color: hex string for consistent cross-plot coloring

Typical usage
-------------
    from rosetta_tools.viz import plot_caz_profile, plot_peak_heatmap
    from rosetta_tools.reporting import load_results_dir

    df = load_results_dir("results/")
    plot_caz_profile(df, concept="credibility", model_id="EleutherAI/pythia-410m",
                     out_path="viz/credibility_pythia410m.png")
    plot_peak_heatmap(df, out_path="viz/peak_heatmap.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Concept metadata
# ---------------------------------------------------------------------------

CONCEPT_META: dict[str, dict] = {
    "credibility":    {"type": "epistemic",  "color": "#6A1B9A"},
    "certainty":      {"type": "epistemic",  "color": "#AB47BC"},
    "sentiment":      {"type": "affective",  "color": "#2E7D32"},
    "moral_valence":  {"type": "affective",  "color": "#66BB6A"},
    "causation":      {"type": "relational", "color": "#E65100"},
    "temporal_order": {"type": "relational", "color": "#FFA726"},
    "negation":       {"type": "syntactic",  "color": "#1565C0"},
    "plurality":      {"type": "syntactic",  "color": "#42A5F5"},
}

TYPE_COLORS = {
    "epistemic":  "#6A1B9A",
    "affective":  "#2E7D32",
    "relational": "#E65100",
    "syntactic":  "#1565C0",
}

TYPE_BG = {
    "epistemic":  "#EDE7F6",
    "affective":  "#E8F5E9",
    "relational": "#FFF3E0",
    "syntactic":  "#E3F2FD",
}

CONCEPT_ORDER = [
    "temporal_order", "causation",    # relational
    "negation", "plurality",           # syntactic
    "sentiment", "moral_valence",      # affective
    "certainty", "credibility",        # epistemic
]


# ---------------------------------------------------------------------------
# Single concept × model profile
# ---------------------------------------------------------------------------


def plot_caz_profile(
    df: pd.DataFrame,
    concept: str,
    model_id: str,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Three-panel CAZ profile (separation, coherence, velocity) for one run.

    Parameters
    ----------
    df:
        Tidy DataFrame from ``load_results_dir`` or ``load_result_df``.
    concept:
        Concept name to plot.
    model_id:
        Model identifier to plot.
    out_path:
        Path to write the PNG file.
    title:
        Optional plot title override.
    """
    sub = df[(df["concept"] == concept) & (df["model_id"] == model_id)].copy()
    if sub.empty:
        raise ValueError(f"No data for concept={concept!r} model={model_id!r}")

    sub = sub.sort_values("layer")
    peak_row = sub[sub["is_peak"]]
    peak_layer = int(peak_row["layer"].iloc[0]) if not peak_row.empty else int(sub["separation"].idxmax())

    color = CONCEPT_META.get(concept, {}).get("color", "#333333")
    model_short = model_id.split("/")[-1]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for ax, col, ylabel in zip(
        axes,
        ["separation", "coherence", "velocity"],
        ["S(l) — Separation", "C(l) — Coherence", "V(l) — Velocity"],
    ):
        ax.plot(sub["depth_pct"], sub[col], "o-", color=color, linewidth=1.8, markersize=3)
        ax.axvline(sub.loc[sub["layer"] == peak_layer, "depth_pct"].iloc[0],
                   color="red", linestyle="--", alpha=0.6, label=f"Peak L{peak_layer}")
        if col == "velocity":
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Relative depth (% of model layers)")
    axes[0].set_title(
        title or f"CAZ Profile — {concept}  ·  {model_short}",
        fontweight="bold",
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-concept overlay
# ---------------------------------------------------------------------------


def plot_concept_comparison(
    df: pd.DataFrame,
    out_path: str | Path,
    model_ids: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Multi-concept overlay (S, C, V) across one or two model scales.

    x-axis normalised to relative depth for cross-scale comparability.
    Each concept gets its canonical color; different models get different
    line styles.

    Parameters
    ----------
    df:
        Tidy DataFrame from ``load_results_dir``.
    out_path:
        Path to write the PNG file.
    model_ids:
        Models to include. Defaults to all models in df (up to 4 line styles).
    title:
        Optional plot title override.
    """
    model_ids = model_ids or sorted(df["model_id"].unique().tolist())
    linestyles = ["-", "--", ":", "-."]
    concepts = [c for c in CONCEPT_ORDER if c in df["concept"].unique()]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for ax, col, ylabel in zip(
        axes,
        ["separation", "coherence", "velocity"],
        ["S(l) — Separation (Fisher-normalised)",
         "C(l) — Coherence (explained variance)",
         "V(l) — Velocity (dS/dLayer)"],
    ):
        for concept in concepts:
            color = CONCEPT_META.get(concept, {}).get("color", "#888")
            for model_id, ls in zip(model_ids, linestyles):
                sub = df[(df["concept"] == concept) & (df["model_id"] == model_id)]
                if sub.empty:
                    continue
                sub = sub.sort_values("depth_pct")
                ax.plot(
                    sub["depth_pct"], sub[col],
                    color=color, linestyle=ls,
                    linewidth=1.4, alpha=0.75,
                )
        if col == "velocity":
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("Relative depth (% of model layers)")

    concept_handles = [
        Patch(color=CONCEPT_META.get(c, {}).get("color", "#888"),
              label=f"{c}  ({CONCEPT_META.get(c, {}).get('type', '?')})")
        for c in concepts
    ]
    model_handles = [
        Line2D([0], [0], color="black", linestyle=ls, linewidth=1.4, label=mid.split("/")[-1])
        for mid, ls in zip(model_ids, linestyles)
    ]
    axes[0].legend(handles=concept_handles + model_handles, loc="upper left",
                   fontsize=7, ncol=2)
    axes[0].set_title(
        title or "CAZ metrics — all concepts",
        fontweight="bold",
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Peak depth heatmap
# ---------------------------------------------------------------------------


def plot_peak_heatmap(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Heatmap of peak depth percentages — concepts × models.

    Parameters
    ----------
    df:
        Tidy DataFrame from ``load_results_dir``.
    out_path:
        Path to write the PNG file.
    title:
        Optional title override.
    """
    peaks = df[df["is_peak"]].copy()
    pivot = peaks.pivot_table(
        index="model_id", columns="concept", values="depth_pct", aggfunc="first"
    )
    # Reorder columns by CONCEPT_ORDER where available
    col_order = [c for c in CONCEPT_ORDER if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.3),
                                    max(4, len(pivot) * 0.7)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.split("/")[-1] for m in pivot.index], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="black" if 20 < val < 80 else "white")

    plt.colorbar(im, ax=ax, label="Peak depth (%)", fraction=0.03)
    ax.set_title(title or "Peak CAZ depth by concept × model", fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

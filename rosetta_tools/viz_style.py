"""
viz_style.py — Style constants and helpers for all Rosetta visualizations.

No matplotlib backend is set here — safe to import in Jupyter notebooks,
interactive sessions, and headless scripts alike.

Canonical reference: Rosetta_Program/VIZ_STYLE_GUIDE.md

Usage
-----
    from rosetta_tools.viz_style import (
        concept_color,
        CONCEPT_COLORS, CONCEPT_TYPE, CONCEPTS,
        CONCEPT_COLORS_ACCESSIBLE,
        FAMILY_COLORS, FAMILY_MAP, FAMILY_ORDER,
        CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat,
        THEME, apply_theme, layer_ticks, model_label, sort_models,
        add_outside_callouts,
        # backwards-compat
        CONCEPT_META, TYPE_COLORS, TYPE_BG, CONCEPT_ORDER,
    )

Design decisions:
  - White background, print-ready (PDF-safe)
  - 18 named concept colors, deep/saturated (L≈0.34, S≈0.76),
    max-separation placement across the hue wheel
  - Unlimited concept support via concept_color() hash fallback (SAE-scale)
  - Paul Tol accessible palette available as CONCEPT_COLORS_ACCESSIBLE
  - Architecture family colors are distinct and legible at small sizes
  - X-axis: layer count (L0, L7...) with % in parentheses
  - Callouts go OUTSIDE the plot area with straight vertical lines
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Concept palette
# ---------------------------------------------------------------------------
# 18 named concepts, deep/saturated, greedy max-separation hue placement.
# Core 7 pinned; extended 11 algorithm-placed.
# For any other name, call concept_color(name) for a hash-derived color
# in the same aesthetic — stable across runs, SAE-scale safe.

CONCEPT_COLORS: dict[str, str] = {
    # ── Core 7 (pinned) ───────────────────────────────────────────
    "credibility":    "#7B1FA2",   # deep purple       H≈282°
    "certainty":      "#AD1457",   # deep pink         H≈334°
    "negation":       "#C62828",   # deep red          H≈0°
    "causation":      "#E65100",   # deep orange       H≈21°
    "temporal_order": "#827717",   # olive             H≈54°
    "sentiment":      "#2E7D32",   # forest green      H≈123°
    "moral_valence":  "#00695C",   # teal              H≈173°
    # ── Extended 11 (algorithm-placed) ────────────────────────────
    "sarcasm":        "#986714",   # bronze-brown      H≈38°
    "plurality":      "#809814",   # yellow-olive      H≈71°
    "exfiltration":   "#599814",   # lime-green        H≈89°
    "agency":         "#339814",   # medium green      H≈106°
    "threat_severity":"#149852",   # teal-green        H≈148°
    "formality":      "#148A98",   # cyan              H≈187°
    "obfuscation":    "#146C98",   # steel blue        H≈200°
    "specificity":    "#144F98",   # medium blue       H≈214°
    "authorization":  "#143098",   # strong blue       H≈228°
    "deception":      "#351498",   # deep indigo       H≈255°
    "urgency":        "#981487",   # deep magenta      H≈308°
}

CONCEPT_TYPE: dict[str, str] = {
    "credibility":    "epistemic",
    "certainty":      "epistemic",
    "deception":      "epistemic",
    "sarcasm":        "epistemic",
    "specificity":    "epistemic",
    "formality":      "epistemic",
    "negation":       "syntactic",
    "plurality":      "syntactic",
    "causation":      "relational",
    "temporal_order": "relational",
    "agency":         "relational",
    "sentiment":      "affective",
    "moral_valence":  "affective",
    "urgency":        "affective",
    "threat_severity":"affective",
    "authorization":  "security",
    "exfiltration":   "security",
    "obfuscation":    "security",
}

CONCEPTS: list[str] = list(CONCEPT_COLORS.keys())

# Accessible palette (Paul Tol — colorblind-safe, ≤ 8 concepts per panel)
CONCEPT_COLORS_ACCESSIBLE: dict[str, str] = {
    "credibility":    "#AA3377",
    "certainty":      "#882255",
    "negation":       "#CC6677",
    "causation":      "#EE7733",
    "temporal_order": "#CCBB44",
    "sentiment":      "#228833",
    "moral_valence":  "#44AA99",
    "sarcasm":        "#EE6677",
    "plurality":      "#999933",
    "exfiltration":   "#117733",
    "agency":         "#66CCEE",
    "threat_severity":"#CC3311",
    "formality":      "#009988",
    "obfuscation":    "#0077BB",
    "specificity":    "#4477AA",
    "authorization":  "#332288",
    "deception":      "#AA4499",
    "urgency":        "#DDCC77",
}


def concept_color(name: str, *, accessible: bool = False) -> str:
    """Return the canonical color for any concept name.

    Named concepts return their fixed palette entry. Any other name
    (SAE features, novel concepts, etc.) gets a stable deterministic
    color via MD5 hash in the same deep/saturated aesthetic (L=0.34, S=0.76).
    """
    palette = CONCEPT_COLORS_ACCESSIBLE if accessible else CONCEPT_COLORS
    if name in palette:
        return palette[name]
    h_int = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h_int % 3600) / 3600.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.34, 0.76)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


# ---------------------------------------------------------------------------
# Architecture family palette
# ---------------------------------------------------------------------------

FAMILY_ORDER = ["Pythia", "OPT", "GPT-2", "Qwen", "Llama", "Mistral", "Gemma", "Phi", "Other"]

FAMILY_COLORS: dict[str, str] = {
    "Pythia":  "#1565C0",
    "OPT":     "#6A1B9A",
    "GPT-2":   "#558B2F",
    "Qwen":    "#E65100",
    "Llama":   "#AD1457",
    "Mistral": "#0277BD",
    "Gemma":   "#00695C",
    "Phi":     "#4E342E",
    "Other":   "#546E7A",
}

FAMILY_MAP: dict[str, tuple[str, int]] = {
    "EleutherAI/pythia-70m":           ("Pythia",  70),
    "EleutherAI/pythia-160m":          ("Pythia",  160),
    "EleutherAI/pythia-410m":          ("Pythia",  410),
    "EleutherAI/pythia-1b":            ("Pythia",  1000),
    "EleutherAI/pythia-1.4b":          ("Pythia",  1400),
    "EleutherAI/pythia-2.8b":          ("Pythia",  2800),
    "EleutherAI/pythia-6.9b":          ("Pythia",  6900),
    "EleutherAI/pythia-12b":           ("Pythia",  12000),
    "facebook/opt-125m":               ("OPT",     125),
    "facebook/opt-350m":               ("OPT",     350),
    "facebook/opt-1.3b":               ("OPT",     1300),
    "facebook/opt-2.7b":               ("OPT",     2700),
    "facebook/opt-6.7b":               ("OPT",     6700),
    "openai-community/gpt2":           ("GPT-2",   117),
    "openai-community/gpt2-medium":    ("GPT-2",   345),
    "openai-community/gpt2-large":     ("GPT-2",   800),
    "openai-community/gpt2-xl":        ("GPT-2",   1500),
    "Qwen/Qwen2.5-0.5B":              ("Qwen",    500),
    "Qwen/Qwen2.5-1.5B":              ("Qwen",    1500),
    "Qwen/Qwen2.5-3B":                ("Qwen",    3000),
    "Qwen/Qwen2.5-7B":                ("Qwen",    7000),
    "Qwen/Qwen2.5-14B":               ("Qwen",    14000),
    "meta-llama/Llama-3.2-1B":         ("Llama",   1000),
    "meta-llama/Llama-3.2-3B":         ("Llama",   3000),
    "mistralai/Mistral-7B-v0.3":       ("Mistral", 7000),
    "google/gemma-2-2b":               ("Gemma",   2000),
    "google/gemma-2-9b":               ("Gemma",   9000),
    "microsoft/phi-2":                 ("Phi",     2700),
}

# ---------------------------------------------------------------------------
# CAZ score category palette
# ---------------------------------------------------------------------------

CAZ_CAT_COLORS: dict[str, str] = {
    "black_hole": "#C62828",
    "strong":     "#E65100",
    "moderate":   "#F9A825",
    "gentle":     "#1565C0",
}

CAZ_CAT_FILL: dict[str, str] = {
    "black_hole": "#FFCDD2",
    "strong":     "#FFE0B2",
    "moderate":   "#FFF9C4",
    "gentle":     "#BBDEFB",
}

CAZ_CAT_LABELS: dict[str, str] = {
    "black_hole": "Black hole",
    "strong":     "Strong",
    "moderate":   "Moderate",
    "gentle":     "Gentle",
}


def caz_score_cat(score: float) -> str:
    """Return the CAZ strength category for a score value."""
    if score > 0.5:  return "black_hole"
    if score > 0.2:  return "strong"
    if score > 0.05: return "moderate"
    return "gentle"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME: dict[str, str] = {
    "bg":       "white",
    "panel_bg": "white",
    "grid":     "#e8e8e8",
    "spine":    "#cccccc",
    "text":     "#111111",
    "dim":      "#555555",
    "cka_line": "#1565C0",
    "coh_line": "#546E7A",
    "annot":    "#222222",
}


def apply_theme(ax, ax_twin=None, *, grid: bool = True) -> None:
    """Apply the standard Rosetta white-background theme to an Axes."""
    ax.set_facecolor(THEME["panel_bg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["dim"], labelsize=7.5, length=3, width=0.7)
    if grid:
        ax.grid(True, color=THEME["grid"], linewidth=0.6, alpha=1.0, zorder=0)
    if ax_twin is not None:
        ax_twin.set_facecolor(THEME["panel_bg"])
        for spine in ax_twin.spines.values():
            spine.set_edgecolor(THEME["spine"])
            spine.set_linewidth(0.8)
        ax_twin.tick_params(colors=THEME["cka_line"], labelsize=7, length=3, width=0.7)


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def layer_ticks(n_layers: int, pcts: tuple[int, ...] = (0, 25, 50, 75, 100)):
    """Return (positions, labels) for an x-axis showing layer count and depth %."""
    positions = [int(p / 100 * (n_layers - 1)) for p in pcts]
    labels = [f"L{l}\n({p}%)" for l, p in zip(positions, pcts)]
    return positions, labels


def model_label(model_id: str) -> str:
    """Short human-readable label, e.g. 'Pythia-1.4B', 'Qwen-3B'."""
    family, _ = FAMILY_MAP.get(model_id, ("", 0))
    short = model_id.split("/")[-1]
    for prefix in ("pythia-", "opt-", "gpt2-", "Qwen2.5-", "Llama-3.2-",
                   "Mistral-", "gemma-2-", "phi-"):
        if short.lower().startswith(prefix.lower()):
            size = short[len(prefix):]
            return f"{family}-{size.upper()}"
    return short


def sort_models(model_ids: list[str]) -> list[str]:
    """Sort model IDs by family order, then by parameter count."""
    def key(mid):
        fam, params = FAMILY_MAP.get(mid, ("Other", 0))
        fam_idx = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        return (fam_idx, params)
    return sorted(model_ids, key=key)


# ---------------------------------------------------------------------------
# Outside-axes callout system
# ---------------------------------------------------------------------------

def _assign_callout_slots(callouts: list[dict], n_layers: int) -> list[tuple[str, float]]:
    if not callouts:
        return []
    char_w = 7.0 * n_layers / 787.0
    SLOTS = [("top", 1.09), ("bottom", -0.10), ("top", 1.22), ("bottom", -0.24)]
    sorted_idx = sorted(range(len(callouts)), key=lambda i: callouts[i]["x"])
    slot_right = [-999.0] * len(SLOTS)
    result: list[Any] = [None] * len(callouts)
    for orig_i in sorted_idx:
        c = callouts[orig_i]
        x = c["x"]
        max_line = max(len(ln) for ln in c["label"].split("\n"))
        half_w = max_line * char_w / 2.0 + 1.0
        placed = False
        for s_i, (side, y_frac) in enumerate(SLOTS):
            if (x - half_w) > slot_right[s_i]:
                result[orig_i] = (side, y_frac)
                slot_right[s_i] = x + half_w
                placed = True
                break
        if not placed:
            best = min(range(len(SLOTS)), key=lambda i: slot_right[i])
            result[orig_i] = SLOTS[best]
            slot_right[best] = x + half_w
    return result


def add_outside_callouts(ax, callouts: list[dict], n_layers: int) -> None:
    """Draw outside-the-axes callouts with vertical connector lines."""
    slots = _assign_callout_slots(callouts, n_layers)
    for pt, (side, y_label) in zip(callouts, slots):
        va = "bottom" if side == "top" else "top"
        ax.annotate(
            pt["label"],
            xy=(pt["x"], pt["y"]),
            xycoords="data",
            xytext=(pt["x"], y_label),
            textcoords=("data", "axes fraction"),
            ha="center", va=va,
            color=pt["color"],
            fontsize=7.5,
            fontweight="bold" if pt.get("bold", True) else "normal",
            clip_on=False,
            arrowprops=dict(arrowstyle="-", color=pt["color"], lw=0.9,
                            shrinkA=4, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=pt["color"],
                      lw=0.8, alpha=0.97),
            zorder=20,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------
# Scripts that predate viz_style.py used CONCEPT_META (nested dict),
# TYPE_COLORS (type → color), TYPE_BG, and CONCEPT_ORDER.
# These are derived from the canonical flat dicts above.

CONCEPT_META: dict[str, dict] = {
    name: {"color": color, "type": CONCEPT_TYPE.get(name, "other")}
    for name, color in CONCEPT_COLORS.items()
}

TYPE_COLORS: dict[str, str] = {
    "epistemic":  "#7B1FA2",
    "affective":  "#2E7D32",
    "relational": "#E65100",
    "syntactic":  "#C62828",
    "security":   "#143098",
    "other":      "#546E7A",
}

TYPE_BG: dict[str, str] = {
    "epistemic":  "#EDE7F6",
    "affective":  "#E8F5E9",
    "relational": "#FFF3E0",
    "syntactic":  "#E3F2FD",
    "security":   "#E8EAF6",
    "other":      "#ECEFF1",
}

CONCEPT_ORDER: list[str] = [
    "temporal_order", "causation", "agency",
    "negation", "plurality",
    "sentiment", "moral_valence", "urgency", "threat_severity",
    "certainty", "credibility", "deception", "sarcasm",
    "specificity", "formality",
    "authorization", "exfiltration",
]

"""
paths.py — Canonical path constants for the Rosetta data layout.

All scripts should import from here rather than constructing paths locally.
"""

from pathlib import Path

ROSETTA_DATA      = Path.home() / "rosetta_data"
ROSETTA_DATA_ROOT = ROSETTA_DATA          # alias used by some scripts
ROSETTA_MODELS    = ROSETTA_DATA / "models"
ROSETTA_PAPER_N250 = ROSETTA_DATA / "paper_n250"  # frozen HF snapshot (preferred for paper figures)
ROSETTA_MODELS_SNAPSHOTS = ROSETTA_DATA / "model_snapshots"  # isolated per-paper extraction archives
ROSETTA_RESULTS   = ROSETTA_DATA / "results"


def iter_paper_model_dirs(
    *,
    exclude_instruct: bool = True,
    exclude_patterns: tuple[str, ...] = ("deepdive", "dark", "manifold"),
) -> "list[Path]":
    """Return one directory per model, sourced from the frozen snapshot when available.

    Priority per model slug: paper_n250/ > models/ > model_snapshots/
    When paper_n250/ has a model, that copy is used; otherwise models/ is used.
    This ensures paper figures always prefer the frozen HF snapshot but fall back
    to the live models/ directory for models not yet in the snapshot.

    Args:
        exclude_instruct: skip instruction-tuned variants (Instruct/_it_/_it suffix)
        exclude_patterns: additional slug prefixes/substrings to skip
    """
    seen: dict[str, Path] = {}  # slug → preferred path

    def _is_instruct(name: str) -> bool:
        lo = name.lower()
        return "instruct" in lo or "_it_" in lo or lo.endswith("_it")

    candidates = [
        p for p in (ROSETTA_MODELS, ROSETTA_MODELS_SNAPSHOTS)
        if p.exists()
    ]
    # paper_n250 first so its entries win the dedup
    ordered = ([ROSETTA_PAPER_N250] if ROSETTA_PAPER_N250.exists() else []) + candidates

    for root in ordered:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if any(pat in name for pat in exclude_patterns):
                continue
            if exclude_instruct and _is_instruct(name):
                continue
            if name not in seen:          # first root wins (paper_n250 is first)
                seen[name] = d

    return sorted(seen.values(), key=lambda p: p.name)

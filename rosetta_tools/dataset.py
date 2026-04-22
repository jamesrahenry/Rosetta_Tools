"""
dataset.py — Dataset loading and validation utilities for CAZ experiments.

Handles the JSONL contrastive pair format used across all Rosetta datasets.
Each file contains pairs of records (label 1 = positive class, label 0 =
negative class) with fields: pair_id, label, domain, model_name, text, topic.

Typical usage
-------------
    from rosetta_tools.dataset import load_pairs, load_concept_pairs

    # High-level: canonical data root, split-aware, sampled
    pairs = load_concept_pairs("causation", n=200, split="train")

    # Low-level: load a specific file directly
    pairs = load_pairs("data/credibility_pairs.jsonl")
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Canonical concept registry
# ---------------------------------------------------------------------------

# Concepts valid for PRH/CAZ analysis — obfuscation excluded because its
# positive class is tokenisation-encoded text (leet/base64/homoglyphs),
# not a semantic contrast.  It lives in the CIA pipeline only.
CAZ_PRH_CONCEPTS: list[str] = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# All 18 concepts including CIA-only ones
ALL_CONCEPTS: list[str] = CAZ_PRH_CONCEPTS + ["obfuscation"]

_SPLIT_FILE_NAME = "v1_validation_split.json"


def _concepts_root() -> Path:
    """Resolve canonical data root from env or repo-relative fallback."""
    env = os.environ.get("ROSETTA_CONCEPTS_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    # Walk up from this file to find Rosetta_Program root
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate Rosetta_Concept_Pairs/pairs/raw/v1/. "
        "Set ROSETTA_CONCEPTS_ROOT to the directory containing *_consensus_pairs.jsonl files."
    )


def _load_split() -> dict[str, dict[str, list[str]]]:
    """Load the fixed validation split from metadata."""
    root = _concepts_root()
    split_path = root.parent.parent.parent / "metadata" / _SPLIT_FILE_NAME
    if not split_path.exists():
        raise FileNotFoundError(
            f"Validation split not found: {split_path}. "
            "Run Rosetta_Concept_Pairs/generate_split.py to create it."
        )
    with open(split_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def load_concept_pairs(
    concept: str,
    *,
    split: Literal["train", "validation", "all"] = "train",
    n: int = 200,
    seed: int | None = None,
) -> list["ConceptPair"]:
    """Load up to *n* pairs for *concept* from the canonical dataset.

    Parameters
    ----------
    concept:
        One of the entries in ``CAZ_PRH_CONCEPTS`` or ``ALL_CONCEPTS``.
    split:
        ``"train"`` (default) — pair_ids in the fixed 80% training split.
        ``"validation"`` — held-out 20% split, never used in extraction.
        ``"all"`` — all pair_ids (use for CIA probe building only).
    n:
        Maximum pairs to return (250 pos + 250 neg up to *n*).
        Silently clamped to however many are available — no error.
    seed:
        Random seed for sampling.  ``None`` uses a deterministic seed
        derived from ``(concept, split)`` for reproducibility without
        requiring callers to manage seeds.
    """
    root = _concepts_root()
    path = root / f"{concept}_consensus_pairs.jsonl"
    all_pairs = load_pairs(path)

    if split != "all":
        split_map = _load_split()
        if concept not in split_map:
            raise KeyError(f"Concept '{concept}' not found in validation split.")
        allowed = set(split_map[concept][split])
        # pair_id in ConceptPair is composite "base_id__model_name"; extract base
        all_pairs = [
            p for p in all_pairs
            if p.pair_id.split("__")[0] in allowed
        ]

    if len(all_pairs) <= n:
        return all_pairs

    rng = random.Random(seed if seed is not None else hash((concept, split)) & 0xFFFFFFFF)
    return rng.sample(all_pairs, n)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConceptPair:
    """One contrastive pair — positive (label 1) and negative (label 0) text."""

    pair_id: str
    domain: str
    topic: str
    pos_text: str  # label 1
    neg_text: str  # label 0
    concept: str = ""
    model_name: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_pairs(path: str | Path) -> list[ConceptPair]:
    """Load a JSONL contrastive pair file into a list of ConceptPair objects.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.

    Returns
    -------
    list[ConceptPair]
        One ``ConceptPair`` per pair_id.  Pairs are returned in the order
        they appear in the file (sorted by pair_id within each domain).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains incomplete pairs (a pair_id with only one record).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Group records by (pair_id, model_name) so multi-generator consensus
    # datasets produce one ConceptPair per generator rather than collapsing
    # all versions into one.  For single-generator files model_name is
    # constant so the key reduces to pair_id — fully backward-compatible.
    by_pair: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e
            pair_id = record["pair_id"]
            model_name = record.get("model_name", "")
            key = f"{pair_id}__{model_name}" if model_name else pair_id
            label = record["label"]
            by_pair[key][label] = record

    pairs = []
    for pair_id, records in sorted(by_pair.items()):
        if 1 not in records or 0 not in records:
            missing = 1 if 1 not in records else 0
            raise ValueError(f"Incomplete pair '{pair_id}': missing label {missing}")
        pos = records[1]
        neg = records[0]
        pairs.append(
            ConceptPair(
                pair_id=pair_id,
                domain=pos.get("domain", ""),
                topic=pos.get("topic", ""),
                pos_text=pos["text"],
                neg_text=neg["text"],
                concept=pos.get("concept", ""),
                model_name=pos.get("model_name", ""),
                metadata={
                    k: v
                    for k, v in pos.items()
                    if k
                    not in (
                        "pair_id",
                        "label",
                        "domain",
                        "topic",
                        "text",
                        "concept",
                        "model_name",
                    )
                },
            )
        )

    return pairs


def iter_texts(
    pairs: list[ConceptPair],
) -> Iterator[tuple[str, str, ConceptPair]]:
    """Iterate over (pos_text, neg_text, pair) tuples.

    Convenience wrapper for the common pattern of iterating over paired texts.
    """
    for pair in pairs:
        yield pair.pos_text, pair.neg_text, pair


def texts_by_label(
    pairs: list[ConceptPair],
) -> tuple[list[str], list[str]]:
    """Return all positive and negative texts as two flat lists.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(pos_texts, neg_texts)`` — same order, same length.
    """
    pos_texts = [p.pos_text for p in pairs]
    neg_texts = [p.neg_text for p in pairs]
    return pos_texts, neg_texts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(path: str | Path) -> list[str]:
    """Validate a dataset file and return a list of issues (empty = clean).

    Checks performed:
    - File exists and is valid JSONL
    - All required fields present on every record
    - All pair_ids have both label 0 and label 1
    - Domain counts are consistent (warns if any domain has fewer than
      the modal count)
    - No duplicate (pair_id, label) combinations

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.

    Returns
    -------
    list[str]
        Human-readable issue descriptions.  Empty list means clean.
    """
    path = Path(path)
    issues = []

    if not path.exists():
        return [f"File not found: {path}"]

    required_fields = {"pair_id", "label", "text"}
    # Key: (pair_id, model_name) — same composite as load_pairs so multi-generator
    # consensus files don't produce false duplicate warnings.
    by_pair: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    domain_counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, int]] = set()

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {lineno}: invalid JSON — {e}")
                continue

            # Required fields
            missing = required_fields - set(record.keys())
            if missing:
                issues.append(f"Line {lineno}: missing fields {missing}")
                continue

            pair_id = record["pair_id"]
            model_name = record.get("model_name", "")
            label = record["label"]
            pair_key = (pair_id, model_name)

            # Duplicate check — true duplicate only if same pair_id, model_name, AND label
            seen_key = (pair_id, model_name, label)
            if seen_key in seen:
                qualifier = f" model={model_name!r}" if model_name else ""
                issues.append(
                    f"Duplicate record: pair_id={pair_id!r}{qualifier} label={label}"
                )
            seen.add(seen_key)

            by_pair[pair_key][label] = by_pair[pair_key].get(label, 0) + 1

            if "domain" in record:
                domain_counts[record["domain"]] += 1

    # Completeness
    for (pair_id, model_name), labels in by_pair.items():
        qualifier = f" model={model_name!r}" if model_name else ""
        if 0 not in labels:
            issues.append(f"Pair {pair_id!r}{qualifier}: missing label 0 (negative)")
        if 1 not in labels:
            issues.append(f"Pair {pair_id!r}{qualifier}: missing label 1 (positive)")

    # Domain balance
    if domain_counts:
        modal = max(domain_counts.values())
        for domain, count in domain_counts.items():
            if count < modal * 0.8:
                issues.append(
                    f"Domain {domain!r} has {count} records vs modal {modal} "
                    f"(expected ~{modal})"
                )

    return issues


def load_pairs_df(path: str | Path) -> "pd.DataFrame":
    """Load a JSONL contrastive pair file as a flat pandas DataFrame.

    Each row is one pair. Columns: pair_id, domain, topic, concept,
    model_name, pos_text, neg_text.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.

    Returns
    -------
    pd.DataFrame
        One row per pair, indexed by pair_id.
    """
    import pandas as pd

    pairs = load_pairs(path)
    rows = [
        {
            "pair_id": p.pair_id,
            "domain": p.domain,
            "topic": p.topic,
            "concept": p.concept,
            "model_name": p.model_name,
            "pos_text": p.pos_text,
            "neg_text": p.neg_text,
        }
        for p in pairs
    ]
    return pd.DataFrame(rows).set_index("pair_id")


def dataset_summary(path: str | Path) -> dict:
    """Return a summary dict for a dataset file.

    Suitable for printing or logging before a long run.
    """
    path = Path(path)
    pairs = load_pairs(path)

    domains = defaultdict(int)
    concepts = defaultdict(int)
    for p in pairs:
        domains[p.domain] += 1
        if p.concept:
            concepts[p.concept] += 1

    return {
        "path": str(path),
        "n_pairs": len(pairs),
        "domains": dict(domains),
        "concepts": dict(concepts) or None,
        "model_names": list({p.model_name for p in pairs if p.model_name}),
    }

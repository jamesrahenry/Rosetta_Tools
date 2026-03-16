"""
dataset.py — Dataset loading and validation utilities for CAZ experiments.

Handles the JSONL contrastive pair format used across all Rosetta datasets.
Each file contains pairs of records (label 1 = positive class, label 0 =
negative class) with fields: pair_id, label, domain, model_name, text, topic.

Typical usage
-------------
    from rosetta_tools.dataset import load_pairs, iter_texts, validate_dataset

    pairs = load_pairs("data/credibility_pairs.jsonl")
    print(f"Loaded {len(pairs)} pairs")

    # Iterate positive and negative texts together
    for pos_text, neg_text, meta in iter_texts(pairs):
        ...

    # Validate before a long run
    issues = validate_dataset("data/negation_pairs.jsonl")
    if issues:
        raise ValueError(f"Dataset issues: {issues}")
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


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

    # Group records by pair_id
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
            label = record["label"]
            by_pair[pair_id][label] = record

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
    by_pair: dict[str, dict[int, int]] = defaultdict(dict)  # pair_id -> {label: count}
    domain_counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, int]] = set()

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
            label = record["label"]

            # Duplicate check
            key = (pair_id, label)
            if key in seen:
                issues.append(f"Duplicate record: pair_id={pair_id!r} label={label}")
            seen.add(key)

            by_pair[pair_id][label] = by_pair[pair_id].get(label, 0) + 1

            if "domain" in record:
                domain_counts[record["domain"]] += 1

    # Completeness
    for pair_id, labels in by_pair.items():
        if 0 not in labels:
            issues.append(f"Pair {pair_id!r}: missing label 0 (negative)")
        if 1 not in labels:
            issues.append(f"Pair {pair_id!r}: missing label 1 (positive)")

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

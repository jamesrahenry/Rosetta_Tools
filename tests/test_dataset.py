"""
test_dataset.py — Tests for rosetta_tools.dataset.

Tests JSONL loading, validation, and edge cases using temporary files.
No external dependencies beyond pytest and the standard library.
"""

import json
import tempfile
from pathlib import Path

import pytest

from rosetta_tools.dataset import (
    ConceptPair,
    load_pairs,
    texts_by_label,
    iter_texts,
    validate_dataset,
    dataset_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_pair(pair_id="p1", pos_text="good", neg_text="bad", domain="test", topic="t"):
    return [
        {"pair_id": pair_id, "label": 1, "text": pos_text, "domain": domain, "topic": topic},
        {"pair_id": pair_id, "label": 0, "text": neg_text, "domain": domain, "topic": topic},
    ]


# ---------------------------------------------------------------------------
# load_pairs
# ---------------------------------------------------------------------------


class TestLoadPairs:

    def test_basic_load(self, tmp_path):
        records = _make_pair("p1", "positive", "negative")
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        pairs = load_pairs(path)
        assert len(pairs) == 1
        assert pairs[0].pair_id == "p1"
        assert pairs[0].pos_text == "positive"
        assert pairs[0].neg_text == "negative"
        assert pairs[0].domain == "test"

    def test_multiple_pairs(self, tmp_path):
        records = _make_pair("p1") + _make_pair("p2", "yes", "no")
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        pairs = load_pairs(path)
        assert len(pairs) == 2

    def test_sorted_by_pair_id(self, tmp_path):
        records = _make_pair("z_last") + _make_pair("a_first")
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        pairs = load_pairs(path)
        assert pairs[0].pair_id == "a_first"
        assert pairs[1].pair_id == "z_last"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_pairs("/nonexistent/path.jsonl")

    def test_incomplete_pair_missing_neg(self, tmp_path):
        records = [{"pair_id": "p1", "label": 1, "text": "only positive", "domain": "d", "topic": "t"}]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        with pytest.raises(ValueError, match="missing label 0"):
            load_pairs(path)

    def test_incomplete_pair_missing_pos(self, tmp_path):
        records = [{"pair_id": "p1", "label": 0, "text": "only negative", "domain": "d", "topic": "t"}]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        with pytest.raises(ValueError, match="missing label 1"):
            load_pairs(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text("not valid json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_pairs(path)

    def test_empty_lines_ignored(self, tmp_path):
        records = _make_pair("p1")
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write("\n")
            f.write(json.dumps(records[0]) + "\n")
            f.write("\n")
            f.write(json.dumps(records[1]) + "\n")
            f.write("\n")

        pairs = load_pairs(path)
        assert len(pairs) == 1

    def test_extra_fields_in_metadata(self, tmp_path):
        records = [
            {"pair_id": "p1", "label": 1, "text": "pos", "domain": "d", "topic": "t", "custom_field": "val"},
            {"pair_id": "p1", "label": 0, "text": "neg", "domain": "d", "topic": "t"},
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        pairs = load_pairs(path)
        assert pairs[0].metadata.get("custom_field") == "val"


# ---------------------------------------------------------------------------
# texts_by_label
# ---------------------------------------------------------------------------


class TestTextsByLabel:

    def test_basic(self):
        pairs = [
            ConceptPair("p1", "d", "t", "pos1", "neg1"),
            ConceptPair("p2", "d", "t", "pos2", "neg2"),
        ]
        pos, neg = texts_by_label(pairs)
        assert pos == ["pos1", "pos2"]
        assert neg == ["neg1", "neg2"]

    def test_empty(self):
        pos, neg = texts_by_label([])
        assert pos == []
        assert neg == []


# ---------------------------------------------------------------------------
# iter_texts
# ---------------------------------------------------------------------------


class TestIterTexts:

    def test_yields_tuples(self):
        pairs = [ConceptPair("p1", "d", "t", "pos", "neg")]
        result = list(iter_texts(pairs))
        assert len(result) == 1
        pos_text, neg_text, pair = result[0]
        assert pos_text == "pos"
        assert neg_text == "neg"
        assert pair.pair_id == "p1"


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:

    def test_clean_dataset(self, tmp_path):
        records = _make_pair("p1") + _make_pair("p2")
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        issues = validate_dataset(path)
        assert issues == []

    def test_missing_file(self):
        issues = validate_dataset("/no/such/file.jsonl")
        assert len(issues) == 1
        assert "not found" in issues[0].lower()

    def test_missing_required_field(self, tmp_path):
        records = [{"pair_id": "p1", "label": 1}]  # missing 'text'
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        issues = validate_dataset(path)
        assert any("missing fields" in i for i in issues)

    def test_duplicate_record(self, tmp_path):
        records = _make_pair("p1") + [
            {"pair_id": "p1", "label": 1, "text": "dupe", "domain": "d", "topic": "t"}
        ]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        issues = validate_dataset(path)
        assert any("Duplicate" in i for i in issues)

    def test_incomplete_pair_detected(self, tmp_path):
        records = [{"pair_id": "p1", "label": 1, "text": "only pos", "domain": "d", "topic": "t"}]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        issues = validate_dataset(path)
        assert any("missing label 0" in i for i in issues)


# ---------------------------------------------------------------------------
# dataset_summary
# ---------------------------------------------------------------------------


class TestDatasetSummary:

    def test_basic(self, tmp_path):
        records = _make_pair("p1") + _make_pair("p2")
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, records)

        summary = dataset_summary(path)
        assert summary["n_pairs"] == 2
        assert "test" in summary["domains"]

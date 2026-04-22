"""
models.py — Rosetta model and concept registry loader.

Data lives in rosetta_tools/data/models.yaml and concepts.yaml.
This module loads them at import time and exposes query functions.

    from rosetta_tools.models import (
        all_models, models_by_cluster, models_by_family,
        get_model, vram_gb, attention_paradigm_of,
        instruct_pairs, concepts_by_pipeline,
    )
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    model_id: str
    family: str
    hidden_dim: int
    n_layers: int
    vram_bf16: float
    cluster: str | None          # A/B/C/D/E/F/G/scale/null
    encoding: str                # redundant | sparse | unknown
    attention: str               # mha | gqa | alternating | unknown
    quantization: str | None     # null | 4bit | 8bit
    enabled: bool
    gated: bool
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ConceptEntry:
    name: str
    category: str                # epistemic | syntactic | relational | affective | security
    pipelines: list[str]         # caz | cia
    assembly_depth_pct: float | None
    assembly_depth_std: float | None
    notes: str = ""


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> Any:
    try:
        ref = importlib.resources.files("rosetta_tools.data").joinpath(filename)
        return yaml.safe_load(ref.read_text())
    except Exception:
        # Fallback: path relative to this file
        from pathlib import Path
        p = Path(__file__).parent / "data" / filename
        with open(p) as f:
            return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _model_registry() -> list[ModelEntry]:
    data = _load_yaml("models.yaml")
    entries = []
    for m in data["models"]:
        entries.append(ModelEntry(
            model_id=m["model_id"],
            family=m["family"],
            hidden_dim=m["hidden_dim"],
            n_layers=m["n_layers"],
            vram_bf16=m["vram_bf16"],
            cluster=m.get("cluster"),
            encoding=m.get("encoding", "unknown"),
            attention=m.get("attention", "unknown"),
            quantization=m.get("quantization"),
            enabled=m.get("enabled", True),
            gated=m.get("gated", False),
            tags=m.get("tags") or [],
            notes=m.get("notes") or "",
        ))
    return entries


@lru_cache(maxsize=1)
def _concept_registry() -> list[ConceptEntry]:
    data = _load_yaml("concepts.yaml")
    entries = []
    for c in data["concepts"]:
        entries.append(ConceptEntry(
            name=c["name"],
            category=c["category"],
            pipelines=c.get("pipelines") or [],
            assembly_depth_pct=c.get("assembly_depth_pct"),
            assembly_depth_std=c.get("assembly_depth_std"),
            notes=c.get("notes") or "",
        ))
    return entries


# ---------------------------------------------------------------------------
# Model queries
# ---------------------------------------------------------------------------

def get_model(model_id: str) -> ModelEntry | None:
    for m in _model_registry():
        if m.model_id == model_id:
            return m
    return None


def all_models(include_disabled: bool = False) -> list[str]:
    return [m.model_id for m in _model_registry()
            if include_disabled or m.enabled]


def models_by_cluster(cluster: str, include_disabled: bool = False) -> list[str]:
    """Return model IDs in a named PRH cluster (A/B/C/D/E/F/G/scale)."""
    return [m.model_id for m in _model_registry()
            if m.cluster == cluster and (include_disabled or m.enabled)]


def models_by_family(family: str, include_disabled: bool = False) -> list[str]:
    return [m.model_id for m in _model_registry()
            if m.family == family and (include_disabled or m.enabled)]


def families(include_disabled: bool = False) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for m in _model_registry():
        if include_disabled or m.enabled:
            result.setdefault(m.family, []).append(m.model_id)
    return result


def models_by_tag(tag: str, include_disabled: bool = False) -> list[str]:
    return [m.model_id for m in _model_registry()
            if tag in m.tags and (include_disabled or m.enabled)]


def models_by_encoding(strategy: str, include_disabled: bool = False) -> list[str]:
    return [m.model_id for m in _model_registry()
            if m.encoding == strategy and (include_disabled or m.enabled)]


def models_by_hidden_dim(dim: int, include_disabled: bool = False) -> list[str]:
    return [m.model_id for m in _model_registry()
            if m.hidden_dim == dim and (include_disabled or m.enabled)]


def zero_pca_clusters(include_disabled: bool = False) -> dict[int, list[str]]:
    """Return {hidden_dim: [model_ids]} for dims with 2+ models (Procrustes pairs)."""
    from collections import defaultdict
    by_dim: dict[int, list[str]] = defaultdict(list)
    for m in _model_registry():
        if m.hidden_dim > 0 and (include_disabled or m.enabled):
            by_dim[m.hidden_dim].append(m.model_id)
    return {dim: ids for dim, ids in sorted(by_dim.items()) if len(ids) >= 2}


def vram_gb(model_id: str) -> float:
    m = get_model(model_id)
    return m.vram_bf16 if m is not None else 0.0


def family_of(model_id: str) -> str:
    m = get_model(model_id)
    return m.family if m is not None else "unknown"


def hidden_dim_of(model_id: str) -> int:
    m = get_model(model_id)
    return m.hidden_dim if m is not None else 0


def attention_paradigm_of(model_id: str) -> str:
    """Return attention paradigm: 'mha' | 'gqa' | 'alternating' | 'unknown'."""
    m = get_model(model_id)
    return m.attention if m is not None else "unknown"


def requires_quantization(model_id: str) -> str | None:
    """Return required quantization ('4bit' | '8bit') or None."""
    m = get_model(model_id)
    return m.quantization if m is not None else None


def instruct_pairs(include_disabled: bool = False) -> list[tuple[str, str]]:
    """Return (base_model_id, instruct_model_id) pairs matched by family+size."""
    base_by_key: dict[tuple[str, float], ModelEntry] = {}
    inst_by_key: dict[tuple[str, float], ModelEntry] = {}
    for m in _model_registry():
        if not include_disabled and not m.enabled:
            continue
        if "rlhf-pair" not in m.tags:
            continue
        key = (m.family.removesuffix("-instruct"), m.vram_bf16)
        if "instruct" in m.tags:
            inst_by_key[key] = m
        else:
            base_by_key[key] = m

    pairs = []
    for key, base in base_by_key.items():
        if key in inst_by_key:
            pairs.append((base.model_id, inst_by_key[key].model_id))
    return pairs


# ---------------------------------------------------------------------------
# Concept queries
# ---------------------------------------------------------------------------

def get_concept(name: str) -> ConceptEntry | None:
    for c in _concept_registry():
        if c.name == name:
            return c
    return None


def all_concepts() -> list[str]:
    return [c.name for c in _concept_registry()]


def concepts_by_pipeline(pipeline: str) -> list[str]:
    """Return concept names for a pipeline ('caz' or 'cia')."""
    return [c.name for c in _concept_registry() if pipeline in c.pipelines]


def concepts_by_category(category: str) -> list[str]:
    return [c.name for c in _concept_registry() if c.category == category]


def concept_assembly_depths() -> dict[str, float]:
    """Return {concept: mean_assembly_depth_pct} for measured concepts only."""
    return {c.name: c.assembly_depth_pct
            for c in _concept_registry()
            if c.assembly_depth_pct is not None}

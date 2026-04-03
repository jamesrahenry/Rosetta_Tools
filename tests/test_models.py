"""
test_models.py — Tests for rosetta_tools.models registry and query functions.

Tests the central model/concept registry that all scripts import from.
No model loading, no GPU required. Pure data structure and query tests.

Test philosophy:
- Registry integrity: no duplicate IDs, all required fields populated
- Query correctness: filters return the right subsets
- Invariants: instruct models never in all_models(), enabled/disabled logic
- Pairing: instruct_pairs matches by architecture, not by name heuristics
- Concepts: ordered by assembly depth, categories are valid
"""

import pytest

from rosetta_tools.models import (
    CONCEPTS,
    REGISTRY,
    ModelEntry,
    ConceptEntry,
    all_models,
    concept_datasets,
    concept_names,
    concepts_by_category,
    families,
    family_of,
    get_model,
    instruct_pairs,
    models_by_family,
    models_by_tag,
    vram_gb,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    """The registry itself should be well-formed."""

    def test_no_duplicate_model_ids(self):
        ids = [m.model_id for m in REGISTRY]
        assert len(ids) == len(set(ids)), f"Duplicate model IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_models_have_family(self):
        for m in REGISTRY:
            assert m.family, f"{m.model_id} has empty family"

    def test_all_models_have_positive_vram(self):
        for m in REGISTRY:
            assert m.vram_gb > 0, f"{m.model_id} has non-positive vram_gb={m.vram_gb}"

    def test_no_duplicate_concept_names(self):
        names = [c.name for c in CONCEPTS]
        assert len(names) == len(set(names)), f"Duplicate concept names"

    def test_concepts_have_valid_categories(self):
        valid = {"epistemic", "syntactic", "relational", "affective"}
        for c in CONCEPTS:
            assert c.category in valid, f"{c.name} has invalid category '{c.category}'"

    def test_concepts_ordered_by_assembly_depth(self):
        depths = [c.mean_assembly_depth_pct for c in CONCEPTS]
        assert depths == sorted(depths), "CONCEPTS not ordered by mean_assembly_depth_pct"

    def test_concept_dataset_files_have_jsonl_extension(self):
        for c in CONCEPTS:
            assert c.dataset_file.endswith(".jsonl"), f"{c.name} dataset_file doesn't end with .jsonl"

    def test_instruct_models_tagged(self):
        """Every model in a *-instruct family should have the 'instruct' tag."""
        for m in REGISTRY:
            if m.family.endswith("-instruct"):
                assert "instruct" in m.tags, f"{m.model_id} in instruct family but missing 'instruct' tag"

    def test_instruct_models_disabled_by_default(self):
        """Instruct models should be disabled (not in --all runs)."""
        for m in REGISTRY:
            if "instruct" in m.tags:
                assert not m.enabled, f"{m.model_id} is tagged instruct but enabled=True"


# ---------------------------------------------------------------------------
# all_models()
# ---------------------------------------------------------------------------


class TestAllModels:

    def test_returns_list_of_strings(self):
        result = all_models()
        assert isinstance(result, list)
        assert all(isinstance(m, str) for m in result)

    def test_excludes_disabled(self):
        disabled = {m.model_id for m in REGISTRY if not m.enabled}
        enabled = all_models()
        assert disabled.isdisjoint(set(enabled))

    def test_include_disabled_flag(self):
        all_enabled = all_models(include_disabled=False)
        all_inc = all_models(include_disabled=True)
        assert len(all_inc) >= len(all_enabled)
        assert set(all_enabled).issubset(set(all_inc))

    def test_no_instruct_in_default(self):
        """Instruct models must not appear in default all_models()."""
        result = all_models()
        for model_id in result:
            entry = get_model(model_id)
            assert "instruct" not in entry.tags, f"{model_id} is instruct but in all_models()"


# ---------------------------------------------------------------------------
# models_by_family()
# ---------------------------------------------------------------------------


class TestModelsByFamily:

    def test_known_family_returns_models(self):
        # Pythia should always have models
        pythia = models_by_family("pythia")
        assert len(pythia) > 0
        assert all("pythia" in m.lower() for m in pythia)

    def test_unknown_family_returns_empty(self):
        assert models_by_family("nonexistent_family_xyz") == []

    def test_excludes_disabled(self):
        for fam_name, model_ids in families().items():
            for mid in model_ids:
                entry = get_model(mid)
                assert entry.enabled


# ---------------------------------------------------------------------------
# models_by_tag()
# ---------------------------------------------------------------------------


class TestModelsByTag:

    def test_instruct_tag(self):
        result = models_by_tag("instruct", include_disabled=True)
        assert len(result) > 0
        for mid in result:
            entry = get_model(mid)
            assert "instruct" in entry.tags

    def test_unknown_tag_returns_empty(self):
        assert models_by_tag("nonexistent_tag_xyz") == []

    def test_respects_enabled_flag(self):
        """models_by_tag excludes disabled models by default."""
        all_instruct = models_by_tag("instruct", include_disabled=True)
        enabled_only = models_by_tag("instruct")
        for mid in enabled_only:
            entry = get_model(mid)
            assert entry.enabled
        assert len(all_instruct) >= len(enabled_only)


# ---------------------------------------------------------------------------
# families()
# ---------------------------------------------------------------------------


class TestFamilies:

    def test_returns_dict(self):
        result = families()
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_all_values_are_enabled(self):
        for fam, model_ids in families().items():
            for mid in model_ids:
                entry = get_model(mid)
                assert entry.enabled, f"{mid} in families() but not enabled"

    def test_no_instruct_families_in_default(self):
        """Instruct families shouldn't appear since their models are disabled."""
        fams = families()
        for fam_name in fams:
            assert not fam_name.endswith("-instruct"), f"instruct family '{fam_name}' in families()"


# ---------------------------------------------------------------------------
# Scalar lookups
# ---------------------------------------------------------------------------


class TestScalarLookups:

    def test_vram_gb_known_model(self):
        result = vram_gb("EleutherAI/pythia-70m")
        assert result > 0

    def test_vram_gb_unknown_model(self):
        assert vram_gb("nonexistent/model") == 0.0

    def test_family_of_known_model(self):
        assert family_of("EleutherAI/pythia-70m") == "pythia"

    def test_family_of_unknown_model(self):
        assert family_of("nonexistent/model") == "unknown"

    def test_get_model_known(self):
        entry = get_model("EleutherAI/pythia-70m")
        assert entry is not None
        assert entry.model_id == "EleutherAI/pythia-70m"
        assert isinstance(entry, ModelEntry)

    def test_get_model_unknown(self):
        assert get_model("nonexistent/model") is None


# ---------------------------------------------------------------------------
# instruct_pairs()
# ---------------------------------------------------------------------------


class TestInstructPairs:

    def test_returns_list_of_tuples(self):
        pairs = instruct_pairs(include_disabled=True)
        assert isinstance(pairs, list)
        for item in pairs:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_base_and_instruct_are_different_models(self):
        for base, inst in instruct_pairs(include_disabled=True):
            assert base != inst

    def test_base_and_instruct_same_vram(self):
        """Paired models should be the same size."""
        for base, inst in instruct_pairs(include_disabled=True):
            assert vram_gb(base) == vram_gb(inst), f"{base} ({vram_gb(base)}) != {inst} ({vram_gb(inst)})"

    def test_base_is_not_instruct_family(self):
        for base, inst in instruct_pairs(include_disabled=True):
            base_entry = get_model(base)
            assert not base_entry.family.endswith("-instruct")

    def test_instruct_is_instruct_family(self):
        for base, inst in instruct_pairs(include_disabled=True):
            inst_entry = get_model(inst)
            assert inst_entry.family.endswith("-instruct")

    def test_family_roots_match(self):
        """Base family should be the root of instruct family (e.g. qwen2 / qwen2-instruct)."""
        for base, inst in instruct_pairs(include_disabled=True):
            base_fam = get_model(base).family
            inst_fam = get_model(inst).family
            assert inst_fam == f"{base_fam}-instruct", f"{base_fam} -> {inst_fam}"

    def test_default_respects_enabled(self):
        """Without include_disabled, only pairs where BOTH are enabled appear."""
        pairs = instruct_pairs(include_disabled=False)
        for base, inst in pairs:
            assert get_model(base).enabled
            assert get_model(inst).enabled


# ---------------------------------------------------------------------------
# Concept accessors
# ---------------------------------------------------------------------------


class TestConceptAccessors:

    def test_concept_names_returns_all(self):
        names = concept_names()
        assert len(names) == len(CONCEPTS)
        assert names == [c.name for c in CONCEPTS]

    def test_concept_datasets_returns_dict(self):
        ds = concept_datasets()
        assert isinstance(ds, dict)
        assert len(ds) == len(CONCEPTS)
        for name, filename in ds.items():
            assert filename.endswith(".jsonl")

    def test_concepts_by_category_epistemic(self):
        result = concepts_by_category("epistemic")
        assert "credibility" in result
        assert "certainty" in result

    def test_concepts_by_category_relational(self):
        result = concepts_by_category("relational")
        assert "causation" in result
        assert "temporal_order" in result

    def test_concepts_by_category_unknown(self):
        assert concepts_by_category("nonexistent") == []

    def test_seven_concepts(self):
        """We should have exactly 7 concept probes."""
        assert len(CONCEPTS) == 7

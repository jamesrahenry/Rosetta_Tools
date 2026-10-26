"""
test_models.py — Tests for the rosetta_tools.models registry (YAML-backed).

The registry data lives in rosetta_tools/data/models.yaml and concepts.yaml;
this module tests the loader and the query functions built on it.  No model
loading, no GPU.

Test philosophy:
- Registry integrity: no duplicate IDs, required fields populated, tags/flags
  consistent with each other.
- Query correctness: every filter returns a subset of the registry, and the
  ``include_disabled`` switch behaves the same way everywhere.
- Invariants: disabled models never leak into default queries; instruct pairs
  are matched by family + size, not by name heuristics.
- Concepts: categories and pipelines come from a closed vocabulary; assembly
  depths are percentages and only present where measured.

The suite deliberately avoids asserting exact registry sizes — the YAML grows
as models are added — and pins only a handful of entries that the papers
depend on (gpt2, pythia-70m, Qwen2.5-72B quantization).
"""

import pytest

from rosetta_tools.models import (
    ConceptEntry,
    ModelEntry,
    _concept_registry,
    _model_registry,
    all_concepts,
    all_models,
    attention_paradigm_of,
    concept_assembly_depths,
    concepts_by_category,
    concepts_by_pipeline,
    families,
    family_of,
    get_concept,
    get_model,
    hidden_dim_of,
    instruct_pairs,
    models_by_cluster,
    models_by_encoding,
    models_by_family,
    models_by_hidden_dim,
    models_by_tag,
    requires_quantization,
    vram_gb,
    zero_pca_clusters,
)

VALID_CATEGORIES = {"epistemic", "syntactic", "relational", "affective", "security"}
VALID_PIPELINES = {"caz", "cia"}
VALID_ENCODINGS = {"redundant", "sparse", "unknown"}
VALID_ATTENTION = {"mha", "gqa", "alternating", "unknown"}
VALID_QUANT = {None, "4bit", "8bit"}


@pytest.fixture(scope="module")
def registry() -> list[ModelEntry]:
    return _model_registry()


@pytest.fixture(scope="module")
def concepts() -> list[ConceptEntry]:
    return _concept_registry()


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    def test_registry_nonempty(self, registry):
        assert len(registry) > 0

    def test_no_duplicate_model_ids(self, registry):
        ids = [m.model_id for m in registry]
        dups = sorted({x for x in ids if ids.count(x) > 1})
        assert not dups, f"Duplicate model IDs: {dups}"

    def test_required_fields_populated(self, registry):
        for m in registry:
            assert m.model_id, "empty model_id"
            assert m.family, f"{m.model_id}: empty family"
            assert m.hidden_dim > 0, f"{m.model_id}: hidden_dim={m.hidden_dim}"
            assert m.n_layers > 0, f"{m.model_id}: n_layers={m.n_layers}"
            assert m.vram_bf16 > 0, f"{m.model_id}: vram_bf16={m.vram_bf16}"

    def test_closed_vocabularies(self, registry):
        for m in registry:
            assert m.encoding in VALID_ENCODINGS, f"{m.model_id}: encoding={m.encoding!r}"
            assert m.attention in VALID_ATTENTION, f"{m.model_id}: attention={m.attention!r}"
            assert m.quantization in VALID_QUANT, f"{m.model_id}: quantization={m.quantization!r}"

    def test_instruct_family_models_carry_instruct_tag(self, registry):
        for m in registry:
            if m.family.endswith("-instruct"):
                assert "instruct" in m.tags, f"{m.model_id}: instruct family without 'instruct' tag"

    def test_rlhf_pair_models_have_a_partner(self, registry):
        """Every model tagged rlhf-pair has a counterpart (same family stem, same size)."""
        tagged = [m for m in registry if "rlhf-pair" in m.tags]
        for m in tagged:
            key = (m.family.removesuffix("-instruct"), m.vram_bf16)
            partners = [
                o for o in tagged
                if o is not m
                and (o.family.removesuffix("-instruct"), o.vram_bf16) == key
                and (("instruct" in o.tags) != ("instruct" in m.tags))
            ]
            assert partners, f"{m.model_id}: tagged rlhf-pair but no base/instruct partner"


# ---------------------------------------------------------------------------
# Model queries
# ---------------------------------------------------------------------------


class TestAllModels:
    def test_default_excludes_disabled(self, registry):
        enabled = {m.model_id for m in registry if m.enabled}
        assert set(all_models()) == enabled

    def test_include_disabled_is_whole_registry(self, registry):
        assert set(all_models(include_disabled=True)) == {m.model_id for m in registry}

    def test_include_disabled_is_superset(self):
        assert set(all_models()) <= set(all_models(include_disabled=True))

    def test_preserves_yaml_order(self, registry):
        assert all_models(include_disabled=True) == [m.model_id for m in registry]

    def test_paper_anchor_models_present(self):
        ids = set(all_models(include_disabled=True))
        assert "openai-community/gpt2" in ids
        assert "EleutherAI/pythia-70m" in ids


class TestFilters:
    @pytest.mark.parametrize(
        "fn,arg",
        [
            (models_by_family, "pythia"),
            (models_by_tag, "instruct"),
            (models_by_cluster, "A"),
            (models_by_encoding, "redundant"),
            (models_by_hidden_dim, 768),
        ],
    )
    def test_filter_respects_enabled_flag(self, fn, arg):
        default = set(fn(arg))
        everything = set(fn(arg, include_disabled=True))
        assert default <= everything
        assert default <= set(all_models())
        assert everything <= set(all_models(include_disabled=True))

    def test_models_by_family_matches_entries(self, registry):
        for fam in {m.family for m in registry}:
            for mid in models_by_family(fam, include_disabled=True):
                assert get_model(mid).family == fam

    def test_models_by_family_unknown_is_empty(self):
        assert models_by_family("no-such-family") == []

    def test_models_by_tag_matches_entries(self, registry):
        for tag in {t for m in registry for t in m.tags}:
            for mid in models_by_tag(tag, include_disabled=True):
                assert tag in get_model(mid).tags

    def test_models_by_tag_unknown_is_empty(self):
        assert models_by_tag("no-such-tag", include_disabled=True) == []

    def test_models_by_cluster_contains_gpt2(self):
        assert "openai-community/gpt2" in models_by_cluster("A", include_disabled=True)

    def test_models_by_hidden_dim_matches_entries(self):
        for mid in models_by_hidden_dim(768, include_disabled=True):
            assert hidden_dim_of(mid) == 768


class TestFamilies:
    def test_keys_are_enabled_families(self, registry):
        assert set(families()) == {m.family for m in registry if m.enabled}

    def test_values_partition_all_models(self):
        flat = [mid for ids in families().values() for mid in ids]
        assert sorted(flat) == sorted(all_models())

    def test_include_disabled_covers_registry(self, registry):
        flat = [mid for ids in families(include_disabled=True).values() for mid in ids]
        assert sorted(flat) == sorted(m.model_id for m in registry)

    def test_every_member_belongs_to_its_family(self):
        for fam, ids in families(include_disabled=True).items():
            for mid in ids:
                assert family_of(mid) == fam


class TestZeroPCAClusters:
    def test_clusters_have_at_least_two_models(self):
        for dim, ids in zero_pca_clusters(include_disabled=True).items():
            assert len(ids) >= 2, f"dim {dim} has {len(ids)} model(s)"

    def test_cluster_members_share_hidden_dim(self):
        for dim, ids in zero_pca_clusters(include_disabled=True).items():
            for mid in ids:
                assert hidden_dim_of(mid) == dim

    def test_768_cluster_exists(self):
        assert 768 in zero_pca_clusters(include_disabled=True)


class TestScalarLookups:
    def test_get_model_known(self):
        m = get_model("openai-community/gpt2")
        assert isinstance(m, ModelEntry)
        assert m.family == "gpt2"
        assert m.hidden_dim == 768
        assert m.n_layers == 12

    def test_get_model_unknown_is_none(self):
        assert get_model("nobody/no-such-model") is None

    def test_vram_gb(self):
        assert vram_gb("openai-community/gpt2") == pytest.approx(0.5)
        assert vram_gb("nobody/no-such-model") == 0.0

    def test_family_of(self):
        assert family_of("EleutherAI/pythia-70m") == "pythia"
        assert family_of("nobody/no-such-model") == "unknown"

    def test_hidden_dim_of(self):
        assert hidden_dim_of("openai-community/gpt2") == 768
        assert hidden_dim_of("nobody/no-such-model") == 0

    def test_attention_paradigm_of(self):
        assert attention_paradigm_of("openai-community/gpt2") == "mha"
        assert attention_paradigm_of("nobody/no-such-model") == "unknown"

    def test_requires_quantization(self):
        assert requires_quantization("openai-community/gpt2") is None
        assert requires_quantization("Qwen/Qwen2.5-72B") == "8bit"
        assert requires_quantization("nobody/no-such-model") is None


class TestInstructPairs:
    def test_pairs_are_well_formed(self):
        for base, inst in instruct_pairs(include_disabled=True):
            b, i = get_model(base), get_model(inst)
            assert b is not None and i is not None
            assert "instruct" in i.tags and "instruct" not in b.tags
            assert "rlhf-pair" in b.tags and "rlhf-pair" in i.tags
            assert b.family.removesuffix("-instruct") == i.family.removesuffix("-instruct")
            assert b.vram_bf16 == pytest.approx(i.vram_bf16)

    def test_default_pairs_only_use_enabled_models(self):
        enabled = set(all_models())
        for base, inst in instruct_pairs():
            assert base in enabled and inst in enabled

    def test_include_disabled_is_superset(self):
        assert set(instruct_pairs()) <= set(instruct_pairs(include_disabled=True))

    def test_no_duplicate_bases(self):
        bases = [b for b, _ in instruct_pairs(include_disabled=True)]
        assert len(bases) == len(set(bases))

    def test_every_rlhf_pair_instruct_model_is_paired(self, registry):
        """Regression: before 2026-10-26 no enabled base model carried the
        rlhf-pair tag, so instruct_pairs() returned [] for the whole enabled
        registry while the API reference promised matched pairs."""
        paired_instruct = {i for _, i in instruct_pairs(include_disabled=True)}
        for m in registry:
            if "rlhf-pair" in m.tags and "instruct" in m.tags:
                assert m.model_id in paired_instruct, f"{m.model_id}: no base partner"
        assert len(instruct_pairs()) >= 8


# ---------------------------------------------------------------------------
# Concept registry
# ---------------------------------------------------------------------------


class TestConcepts:
    def test_no_duplicate_concept_names(self, concepts):
        names = [c.name for c in concepts]
        assert len(names) == len(set(names))

    def test_all_concepts_matches_registry(self, concepts):
        assert all_concepts() == [c.name for c in concepts]

    def test_categories_are_valid(self, concepts):
        for c in concepts:
            assert c.category in VALID_CATEGORIES, f"{c.name}: category={c.category!r}"

    def test_pipelines_are_valid_and_nonempty(self, concepts):
        for c in concepts:
            assert c.pipelines, f"{c.name}: no pipelines"
            assert set(c.pipelines) <= VALID_PIPELINES, f"{c.name}: pipelines={c.pipelines}"

    def test_get_concept(self):
        c = get_concept("negation")
        assert isinstance(c, ConceptEntry)
        assert c.name == "negation"
        assert get_concept("no-such-concept") is None

    def test_concepts_by_pipeline_matches_entries(self):
        for pipeline in VALID_PIPELINES:
            names = concepts_by_pipeline(pipeline)
            assert set(names) <= set(all_concepts())
            for n in names:
                assert pipeline in get_concept(n).pipelines

    def test_caz_pipeline_has_17_concepts(self):
        """The CAZ/PRH corpus is fixed at 17 concepts across all four papers."""
        assert len(concepts_by_pipeline("caz")) == 17

    def test_concepts_by_category_matches_entries(self):
        for cat in VALID_CATEGORIES:
            for n in concepts_by_category(cat):
                assert get_concept(n).category == cat

    def test_assembly_depths_are_percentages_for_measured_only(self, concepts):
        depths = concept_assembly_depths()
        measured = {c.name for c in concepts if c.assembly_depth_pct is not None}
        assert set(depths) == measured
        for name, d in depths.items():
            assert 0.0 <= d <= 100.0, f"{name}: assembly_depth_pct={d}"

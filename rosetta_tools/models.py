"""Central model & concept registry for Rosetta Program experiments.

All scripts should import from here rather than maintaining their own lists.

    from rosetta_tools.models import all_models, families, CONCEPTS, get_model
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """One model in the registry."""
    model_id: str
    family: str
    vram_gb: float  # approximate bf16 VRAM usage
    enabled: bool = True  # False = excluded from --all runs (e.g. OOM)
    gated: bool = False  # requires HF license acceptance
    tags: list[str] = field(default_factory=list)  # e.g. ["instruct", "rlhf"]
    quirks: list[str] = field(default_factory=list)  # known issues/notes


REGISTRY: list[ModelEntry] = [
    # Pythia — EleutherAI, GPT-NeoX architecture, learned position embeddings
    ModelEntry("EleutherAI/pythia-70m",   "pythia",  0.2),
    ModelEntry("EleutherAI/pythia-160m",  "pythia",  0.4),
    ModelEntry("EleutherAI/pythia-410m",  "pythia",  1.0),
    ModelEntry("EleutherAI/pythia-1b",    "pythia",  2.1),
    ModelEntry("EleutherAI/pythia-1.4b",  "pythia",  2.9),
    ModelEntry("EleutherAI/pythia-2.8b",  "pythia",  5.7),
    ModelEntry("EleutherAI/pythia-6.9b",  "pythia",  14.0),

    # GPT-2 — OpenAI, learned position embeddings
    # AutoModel loads GPT2Model (not LMHead) — layers at .h not .transformer.h
    ModelEntry("openai-community/gpt2",         "gpt2", 0.5,
               quirks=["AutoModel: layers at .h (not .transformer.h)"]),
    ModelEntry("openai-community/gpt2-medium",  "gpt2", 1.4,
               quirks=["AutoModel: layers at .h"]),
    ModelEntry("openai-community/gpt2-large",   "gpt2", 3.1,
               quirks=["AutoModel: layers at .h"]),
    ModelEntry("openai-community/gpt2-xl",      "gpt2", 6.3,
               quirks=["AutoModel: layers at .h"]),

    # OPT — Meta, learned position embeddings + ALiBi variants
    # OPT-350m: word_embed_proj_dim=512 != hidden_size=1024 — embedding layer
    # outputs different dim than transformer layers. Feature tracker handles this.
    ModelEntry("facebook/opt-125m",  "opt", 0.3),
    ModelEntry("facebook/opt-350m",  "opt", 0.7,
               quirks=["embed_proj_dim=512 != hidden_size=1024 — dim mismatch at layer 0"]),
    ModelEntry("facebook/opt-1.3b",  "opt", 2.6),
    ModelEntry("facebook/opt-2.7b",  "opt", 5.4),
    ModelEntry("facebook/opt-6.7b",  "opt", 13.4),

    # Qwen 2.5 — Alibaba, RoPE, GQA
    ModelEntry("Qwen/Qwen2.5-0.5B",  "qwen2", 1.0),
    ModelEntry("Qwen/Qwen2.5-1.5B",  "qwen2", 3.1),
    ModelEntry("Qwen/Qwen2.5-3B",    "qwen2", 6.2),
    ModelEntry("Qwen/Qwen2.5-7B",    "qwen2", 14.5),

    # Gemma 2 — Google, RoPE, sliding window + global attention alternating
    ModelEntry("google/gemma-2-2b",  "gemma2", 5.1),
    ModelEntry("google/gemma-2-9b",  "gemma2", 18.5, enabled=False,
               quirks=["OOM on L4 (22 GiB) — needs H200 or 8-bit"]),

    # Llama 3.2 — Meta, RoPE, GQA
    ModelEntry("meta-llama/Llama-3.2-1B", "llama3", 2.4, gated=True, enabled=False,
               quirks=["Gated: pending Meta license approval"]),
    ModelEntry("meta-llama/Llama-3.2-3B", "llama3", 6.4, gated=True, enabled=False,
               quirks=["Gated: pending Meta license approval"]),

    # Mistral — Mistral AI, RoPE, sliding window attention, GQA
    ModelEntry("mistralai/Ministral-8B-Instruct-2410", "mistral", 16.0, enabled=False,
               quirks=["Tight on L4 — 16GB bf16, instruct-only (no base)"]),
    ModelEntry("mistralai/Mistral-7B-v0.3", "mistral", 14.5),
    ModelEntry("mistralai/Mistral-Small-3.1-24B-Base-2503", "mistral", 48.0, enabled=False,
               quirks=["Way too large for L4"]),

    # Phi — Microsoft, "textbook" training data, MHA
    ModelEntry("microsoft/phi-2", "phi", 5.6,
               quirks=["Unusual training mix (synthetic textbooks) — may affect geometry"]),

    # ── Instruct variants (RLHF / alignment-tuned) ──
    # Same weights + architecture as base, with RLHF/DPO fine-tuning.
    # For RLHF feature destruction audit: diff feature maps base vs instruct.

    # ── Instruct variants (RLHF / alignment-tuned) ──
    # Disabled by default — not included in --all runs.
    # Use models_by_tag("instruct") or instruct_pairs() to query.

    # Qwen 2.5 Instruct
    ModelEntry("Qwen/Qwen2.5-0.5B-Instruct", "qwen2-instruct", 1.0,
               enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-1.5B-Instruct", "qwen2-instruct", 3.1,
               enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-3B-Instruct",   "qwen2-instruct", 6.2,
               enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-7B-Instruct",   "qwen2-instruct", 14.5,
               enabled=False, tags=["instruct"]),

    # Gemma 2 IT (instruction-tuned)
    ModelEntry("google/gemma-2-2b-it", "gemma2-instruct", 5.1,
               enabled=False, tags=["instruct"]),
    ModelEntry("google/gemma-2-9b-it", "gemma2-instruct", 18.5,
               enabled=False, tags=["instruct"],
               quirks=["OOM on L4 — same as base gemma-2-9b"]),

    # Llama 3.2 Instruct
    ModelEntry("meta-llama/Llama-3.2-1B-Instruct", "llama3-instruct", 2.4,
               enabled=False, gated=True, tags=["instruct"],
               quirks=["Gated: requires Meta license acceptance on HuggingFace"]),
    ModelEntry("meta-llama/Llama-3.2-3B-Instruct", "llama3-instruct", 6.4,
               enabled=False, gated=True, tags=["instruct"],
               quirks=["Gated: requires Meta license acceptance on HuggingFace"]),

    # Mistral Instruct
    ModelEntry("mistralai/Mistral-7B-Instruct-v0.3", "mistral-instruct", 14.5,
               enabled=False, tags=["instruct"]),
]


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------

@dataclass
class ConceptEntry:
    """One concept probe in the registry."""
    name: str
    category: str  # epistemic, syntactic, relational, affective
    dataset_file: str  # filename in consensus_pairs/
    mean_assembly_depth_pct: float  # from 22-model scaling study
    assembly_depth_std: float  # cross-model standard deviation


CONCEPTS: list[ConceptEntry] = [
    ConceptEntry("credibility",    "epistemic",   "credibility_consensus_pairs.jsonl",    39.6, 29.7),
    ConceptEntry("negation",       "syntactic",   "negation_consensus_pairs.jsonl",       49.8, 17.3),
    ConceptEntry("causation",      "relational",  "causation_consensus_pairs.jsonl",      53.6, 18.2),
    ConceptEntry("temporal_order", "relational",  "temporal_order_consensus_pairs.jsonl",  56.1, 16.8),
    ConceptEntry("sentiment",      "affective",   "sentiment_consensus_pairs.jsonl",      61.8, 13.6),
    ConceptEntry("certainty",      "epistemic",   "certainty_consensus_pairs.jsonl",      64.2, 14.2),
    ConceptEntry("moral_valence",  "affective",   "moral_valence_consensus_pairs.jsonl",  68.1, 14.1),
]


# ---------------------------------------------------------------------------
# Convenience accessors — Models
# ---------------------------------------------------------------------------

def all_models(include_disabled: bool = False) -> list[str]:
    """Return model IDs for all enabled models (or all if include_disabled)."""
    return [m.model_id for m in REGISTRY if include_disabled or m.enabled]


def models_by_family(family: str) -> list[str]:
    """Return enabled model IDs for a given family."""
    return [m.model_id for m in REGISTRY if m.family == family and m.enabled]


def families() -> dict[str, list[str]]:
    """Return {family_name: [model_ids]} for all enabled models."""
    result: dict[str, list[str]] = {}
    for m in REGISTRY:
        if m.enabled:
            result.setdefault(m.family, []).append(m.model_id)
    return result


def vram_gb(model_id: str) -> float:
    """Return approximate bf16 VRAM usage in GB."""
    for m in REGISTRY:
        if m.model_id == model_id:
            return m.vram_gb
    return 0.0


def family_of(model_id: str) -> str:
    """Return family name for a model ID."""
    for m in REGISTRY:
        if m.model_id == model_id:
            return m.family
    return "unknown"


def models_by_tag(tag: str, include_disabled: bool = False) -> list[str]:
    """Return model IDs that have a given tag."""
    return [m.model_id for m in REGISTRY
            if tag in m.tags and (include_disabled or m.enabled)]


def get_model(model_id: str) -> ModelEntry | None:
    """Look up a model entry by ID."""
    for m in REGISTRY:
        if m.model_id == model_id:
            return m
    return None


def instruct_pairs(include_disabled: bool = False) -> list[tuple[str, str]]:
    """Return (base_model_id, instruct_model_id) pairs.

    Matches families by name: 'foo' pairs with 'foo-instruct'.
    Only returns pairs where both base and instruct are enabled
    (or all if include_disabled).
    """
    base_by_family: dict[str, list[ModelEntry]] = {}
    inst_by_family: dict[str, list[ModelEntry]] = {}
    for m in REGISTRY:
        if not include_disabled and not m.enabled:
            continue
        if m.family.endswith("-instruct"):
            inst_by_family.setdefault(m.family.removesuffix("-instruct"), []).append(m)
        else:
            base_by_family.setdefault(m.family, []).append(m)

    pairs = []
    for fam, bases in base_by_family.items():
        instructs = inst_by_family.get(fam, [])
        # Match by VRAM (same size = same model, different tuning)
        inst_by_vram = {m.vram_gb: m for m in instructs}
        for b in bases:
            if b.vram_gb in inst_by_vram:
                pairs.append((b.model_id, inst_by_vram[b.vram_gb].model_id))
    return pairs


# ---------------------------------------------------------------------------
# Convenience accessors — Concepts
# ---------------------------------------------------------------------------

def concept_names() -> list[str]:
    """Return all concept names in assembly order (shallow to deep)."""
    return [c.name for c in CONCEPTS]


def concept_datasets() -> dict[str, str]:
    """Return {concept_name: dataset_filename}."""
    return {c.name: c.dataset_file for c in CONCEPTS}


def concepts_by_category(category: str) -> list[str]:
    """Return concept names for a given category."""
    return [c.name for c in CONCEPTS if c.category == category]

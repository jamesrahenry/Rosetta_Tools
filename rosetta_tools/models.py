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
    # How the model encodes information across layers:
    #   "redundant" — older MHA architectures (GPT-2, Pythia, OPT, Phi): features
    #                 persist across many layers; high cross-layer cosine similarity.
    #                 Deep dive finds many persistent labeled features.
    #   "sparse"    — newer GQA+SwiGLU architectures (Qwen, Gemma-2, Llama-3.x,
    #                 Mistral): representations change more rapidly layer-to-layer;
    #                 features are less redundant. Deep dive finds fewer persistent
    #                 features; concept directions don't align as strongly with top PCs.
    #   "unknown"   — not yet classified empirically.
    encoding_strategy: str = "unknown"  # "redundant" | "sparse" | "unknown"


REGISTRY: list[ModelEntry] = [
    # Pythia — EleutherAI, GPT-NeoX architecture, MHA, learned position embeddings
    ModelEntry("EleutherAI/pythia-70m",   "pythia",  0.2,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-160m",  "pythia",  0.4,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-410m",  "pythia",  1.0,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-1b",    "pythia",  2.1,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-1.4b",  "pythia",  2.9,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-2.8b",  "pythia",  5.7,  encoding_strategy="redundant"),
    ModelEntry("EleutherAI/pythia-6.9b",  "pythia",  14.0, encoding_strategy="redundant"),

    # GPT-2 — OpenAI, MHA, learned position embeddings
    # AutoModel loads GPT2Model (not LMHead) — layers at .h not .transformer.h
    ModelEntry("openai-community/gpt2",         "gpt2", 0.5, encoding_strategy="redundant",
               quirks=["AutoModel: layers at .h (not .transformer.h)"]),
    ModelEntry("openai-community/gpt2-medium",  "gpt2", 1.4, encoding_strategy="redundant",
               quirks=["AutoModel: layers at .h"]),
    ModelEntry("openai-community/gpt2-large",   "gpt2", 3.1, encoding_strategy="redundant",
               quirks=["AutoModel: layers at .h"]),
    ModelEntry("openai-community/gpt2-xl",      "gpt2", 6.3, encoding_strategy="redundant",
               quirks=["AutoModel: layers at .h"]),

    # OPT — Meta, MHA, learned position embeddings + ALiBi variants
    # OPT-350m: word_embed_proj_dim=512 != hidden_size=1024 — embedding layer
    # outputs different dim than transformer layers. Feature tracker handles this.
    ModelEntry("facebook/opt-125m",  "opt", 0.3, encoding_strategy="redundant"),
    ModelEntry("facebook/opt-350m",  "opt", 0.7, encoding_strategy="redundant",
               quirks=["embed_proj_dim=512 != hidden_size=1024 — dim mismatch at layer 0"]),
    ModelEntry("facebook/opt-1.3b",  "opt", 2.6, encoding_strategy="redundant"),
    ModelEntry("facebook/opt-2.7b",  "opt", 5.4, encoding_strategy="redundant"),
    ModelEntry("facebook/opt-6.7b",  "opt", 13.4, encoding_strategy="redundant"),

    # Qwen 2.5 — Alibaba, RoPE, GQA, SwiGLU
    ModelEntry("Qwen/Qwen2.5-0.5B",  "qwen2", 1.0,  encoding_strategy="sparse"),
    ModelEntry("Qwen/Qwen2.5-1.5B",  "qwen2", 3.1,  encoding_strategy="sparse"),
    ModelEntry("Qwen/Qwen2.5-3B",    "qwen2", 6.2,  encoding_strategy="sparse"),
    ModelEntry("Qwen/Qwen2.5-7B",    "qwen2", 14.5, encoding_strategy="sparse"),

    # Gemma 2 — Google, RoPE, GQA, sliding window + global attention alternating
    ModelEntry("google/gemma-2-2b",  "gemma2", 5.1,  encoding_strategy="sparse",
               quirks=["Extreme sparse encoder — alternating local/global attn produces layer-local features; cos-threshold has no effect on feature count"]),
    ModelEntry("google/gemma-2-9b",  "gemma2", 18.5, encoding_strategy="sparse", enabled=False,
               quirks=["OOM in bfloat16 on L4 — use --load-in-8bit (~11 GiB)",
                       "Extreme sparse encoder — cos-threshold has no effect on feature count"]),

    # Llama 3.2 — Meta, RoPE, GQA, SwiGLU
    ModelEntry("meta-llama/Llama-3.2-1B", "llama3", 2.4, encoding_strategy="sparse", gated=True),
    ModelEntry("meta-llama/Llama-3.2-3B", "llama3", 6.4, encoding_strategy="sparse", gated=True),

    # Mistral — Mistral AI, RoPE, GQA, sliding window attention, SwiGLU
    ModelEntry("mistralai/Mistral-7B-v0.3", "mistral", 14.5, encoding_strategy="sparse"),
    ModelEntry("mistralai/Mistral-Small-3.1-24B-Base-2503", "mistral", 48.0, encoding_strategy="sparse",
               enabled=False, quirks=["Way too large for L4"]),

    # Phi — Microsoft, MHA, "textbook" training data
    ModelEntry("microsoft/phi-2", "phi", 5.6, encoding_strategy="redundant",
               quirks=["Unusual training mix (synthetic textbooks) — may affect geometry"]),

    # ── Instruct variants (RLHF / alignment-tuned) ──
    # Same weights + architecture as base, with RLHF/DPO fine-tuning.
    # For RLHF feature destruction audit: diff feature maps base vs instruct.

    # ── Instruct variants (RLHF / alignment-tuned) ──
    # Disabled by default — not included in --all runs.
    # Use models_by_tag("instruct") or instruct_pairs() to query.

    # Qwen 2.5 Instruct
    ModelEntry("Qwen/Qwen2.5-0.5B-Instruct", "qwen2-instruct", 1.0,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-1.5B-Instruct", "qwen2-instruct", 3.1,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-3B-Instruct",   "qwen2-instruct", 6.2,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),
    ModelEntry("Qwen/Qwen2.5-7B-Instruct",   "qwen2-instruct", 14.5,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),

    # Gemma 2 IT (instruction-tuned)
    ModelEntry("google/gemma-2-2b-it", "gemma2-instruct", 5.1,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),
    ModelEntry("google/gemma-2-9b-it", "gemma2-instruct", 18.5,
               encoding_strategy="sparse", enabled=False, tags=["instruct"],
               quirks=["OOM in bfloat16 on L4 — use --load-in-8bit (~11 GiB)"]),

    # Llama 3.2 Instruct
    ModelEntry("meta-llama/Llama-3.2-1B-Instruct", "llama3-instruct", 2.4,
               encoding_strategy="sparse", enabled=False, gated=True, tags=["instruct"]),
    ModelEntry("meta-llama/Llama-3.2-3B-Instruct", "llama3-instruct", 6.4,
               encoding_strategy="sparse", enabled=False, gated=True, tags=["instruct"]),

    # Mistral Instruct
    ModelEntry("mistralai/Mistral-7B-Instruct-v0.3", "mistral-instruct", 14.5,
               encoding_strategy="sparse", enabled=False, tags=["instruct"]),
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


def encoding_strategy_of(model_id: str) -> str:
    """Return encoding strategy ('redundant' | 'sparse' | 'unknown') for a model."""
    m = get_model(model_id)
    return m.encoding_strategy if m is not None else "unknown"


def models_by_encoding(strategy: str, include_disabled: bool = False) -> list[str]:
    """Return model IDs with the given encoding strategy."""
    return [m.model_id for m in REGISTRY
            if m.encoding_strategy == strategy and (include_disabled or m.enabled)]


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

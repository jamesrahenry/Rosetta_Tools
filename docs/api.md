# rosetta_tools API Reference

*Updated: 2026-10-26 UTC — covers v2.1.0.*

Quick-reference index for every public module. All public functions have full docstrings in
source; where this page and a docstring disagree, the docstring wins. Signatures are abridged
(`...` = further keyword arguments; see source).

Modules fall into three groups:

| Group | Modules | Needs torch / transformers? |
|---|---|---|
| Model-facing | `gpu_utils`, `extraction`, `ablation` | yes (imported lazily; informative `ImportError` otherwise) |
| Pure numpy / pandas | `caz`, `probes`, `gem`, `alignment`, `dataset`, `reporting`, `compound_monitor` | no |
| Registry, style, paths | `models`, `viz`, `viz_style`, `paths` | no (`viz` imports matplotlib with the Agg backend) |

---

## gpu_utils — Device Management

Dtype policy: bfloat16 on datacenter GPUs (Ampere+, ≥ 40 GiB), float32 on consumer GPUs and CPU.
MPS (Apple Silicon) is supported.

```python
from rosetta_tools.gpu_utils import get_device, get_dtype, log_device_info, release_model
```

| Function | Signature | Purpose |
|---|---|---|
| `get_device` | `(prefer="auto") -> str` | Best available device: `cuda`, `mps`, `cpu` |
| `get_dtype` | `(device, prefer="auto") -> torch.dtype` | Environment-aware dtype |
| `log_device_info` | `(device, dtype) -> None` | Startup banner: device, dtype, VRAM |
| `vram_stats` | `(device_index=0) -> dict \| None` | `total_gib`, `used_gib`, `free_gib`, `pct_used` |
| `log_vram` | `(label="", device_index=0) -> None` | One-line VRAM summary; no-op on CPU |
| `disk_free_gib` | `(path=None) -> float` | Free disk at `path` (default: HF cache root) |
| `safe_batch_size` | `(requested, device="cuda", reserve_gib=2.0) -> int` | Scale batch size down when VRAM is tight |
| `load_causal_lm` | `(model_id, device, dtype, *, device_map=None, load_in_8bit=False, trust_remote_code=False) -> (model, tokenizer)` | Load a causal LM with the project's defaults |
| `load_model_with_retry` | `(model_cls, model_id, *, dtype, device, device_map=None, load_in_8bit=False, quantization_config=None, max_memory=None, ...)` | Load with OOM fallback (balanced device map, capped `max_memory`, quantization) |
| `requires_quantization` | `(model_id) -> str \| None` | `"4bit"` / `"8bit"` from the registry, else `None` |
| `release_model` | `(model, *, clear_cache=True) -> None` | Delete model and free GPU memory (also resets CUDA peak stats) |
| `purge_hf_cache` | `(model_id, min_free_gb=15.0) -> None` | Remove a model from the local HF cache when disk is low |
| `NumpyJSONEncoder` | `json.JSONEncoder` subclass | Serialises numpy scalars/arrays in `json.dump` |

---

## extraction — Activation Extraction and the CAZ Record

Model-agnostic extraction via native HuggingFace (`output_hidden_states=True`). No TransformerLens.
Layer `l` of every result is `hidden_states[l+1]`, the output of decoder block `l`; the embedding
entry is dropped. **The last layer is post-final-norm** in most HF models (Qwen2/Llama `norm`,
GPT-2 `ln_f`).

```python
from rosetta_tools.extraction import (
    extract_layer_activations, extract_contrastive_activations,
    extract_layer_wise_metrics, build_caz_record,
)
```

| Function | Signature | Purpose |
|---|---|---|
| `extract_layer_activations` | `(model, tokenizer, texts, device="cpu", batch_size=8, pool="last", max_length=512) -> list[NDArray]` | `[n_layers][n_texts, hidden_dim]` pooled residual-stream activations |
| `extract_contrastive_activations` | `(model, tokenizer, pos_texts, neg_texts, device="cpu", batch_size=8, pool="last", max_length=512) -> list[tuple[NDArray, NDArray]]` | Paired `(pos, neg)` activations per layer — the input every CAZ/probe function takes |
| `extract_layer_wise_metrics` | `(model, tokenizer, pos_texts, neg_texts, device, batch_size, compute_quality=True) -> (layer_data: dict, all_layer_acts: NDArray)` | Per-layer S / C / velocity / DOM vector + peak detection + inline QA report (`caz.concept_quality_report`). Migrated from Rosetta_Analysis in v2.0.0 |
| `build_caz_record` | `(model_id, concept, n_pairs, split, hidden_dim, n_layers, extraction_seconds, layer_data) -> dict` | Assemble the `caz_<concept>.json` schema from `extract_layer_wise_metrics` output |

**Pooling:** `"last"` (last token; default for causal LMs), `"mean"`.
**Metrics are always float64** regardless of forward-pass dtype — fp16 overflows in variance at
deep layers of large models.

---

## caz — Concept Assembly Zone Metrics

Pure numpy. Takes the `list[(pos, neg)]` structure from `extraction` and returns NamedTuples /
dataclasses. A **CAZ** is a scored saddle-to-saddle segment of the separation curve
(`papers/shared/CAZ_DEFINITION_OF_RECORD.md`); the detector tiles depth and `caz_score` grades
salience — it is not a membership threshold.

```python
from rosetta_tools.caz import (
    compute_separation, compute_coherence, compute_velocity, compute_layer_metrics,
    find_caz_regions_scored, concept_quality_report,
)
```

**Metrics**

| Function | Signature | Purpose |
|---|---|---|
| `compute_separation` | `(pos, neg, eps=1e-8) -> float` | Fisher-normalised class separation `S` |
| `compute_coherence` | `(pos, neg, n_components=1) -> float` | PCA coherence `C` of the pooled cloud |
| `compute_velocity` | `(separations, window=None) -> NDArray` | Smoothed layer-to-layer change in `S` |
| `compute_layer_metrics` | `(layer_activations, velocity_window=3) -> list[LayerMetrics]` | `S`, `C`, `v` for every layer |
| `LayerMetrics` | `(layer, separation, coherence, velocity)` | NamedTuple |
| `split_half_dom_reproducibility` | `(pos, neg, *, seed=0, n_splits=5) -> dict` | Split-half cosine of the mean-difference direction (P3 §6.9 stability check) |
| `concept_quality_report` | `(layer_activations, peak_layer, *, seed=0, n_splits=5, thresholds=None) -> dict` | Extraction-time QA: split-half DOM cosine, participation ratio, flags. `QUALITY_THRESHOLDS` holds the defaults |
| `final_global_attention_layer` | `(n_layers) -> int` | Architecture-aware "functional peak" layer used by the region detector |

**Boundaries and regions**

| Function | Signature | Purpose |
|---|---|---|
| `find_caz_boundary` | `(metrics, threshold_factor=0.5, min_sustained=2) -> CAZBoundary` | Single-zone boundary (legacy, P1 §4) |
| `compute_caz_statistics` | `(metrics, boundary) -> dict` | Summary stats for a `CAZBoundary` |
| `find_caz_regions` | `(metrics, min_prominence_frac=0.1, min_peak_distance=2, attention_paradigm="unknown", functional_peak_layer=None) -> CAZProfile` | Multi-region detector, strict prominence |
| `find_caz_regions_scored` | `(metrics, min_prominence_frac=0.005, min_peak_distance=2, min_valley_depth_frac=0.03, attention_paradigm="unknown", ...) -> CAZProfile` | **The corpus detector.** Gentle prominence + per-region `caz_score`; global-peak fallback guarantees ≥ 1 region |
| `CAZRegion` | `(start, peak, end, width, peak_separation, peak_coherence, mean_separation, mean_coherence, prominence, caz_score, ...)` | One segment |
| `CAZProfile` | `(n_layers, regions, n_regions, global_peak_layer, global_peak_separation, global_mean_separation, is_multimodal, ...)` | All segments for one concept × model; `.dominant` is the highest-scoring region |

---

## probes — Probe Extraction from Contrastive Activations

Pure numpy / scikit-learn. This is the module Monitor and CIA build their probe libraries on.

```python
from rosetta_tools.probes import extract_probe, extract_gem_probe, score_direction, ProbeResult
```

| Function | Signature | Purpose |
|---|---|---|
| `score_direction` | `(acts, direction) -> NDArray[float64]` | Cosine of each row of `acts` against a unit `direction` |
| `extract_probe` | `(layer_activations, method="raw", threshold_strategy="midpoint", target_tpr=..., eval_frac=0.2, seed=42, concept="", ...) -> ProbeResult` | Peak-layer probe. `method`: `"raw"` (max raw separation), `"fisher"`, `"auroc"`. Falls back gracefully when a layer is degenerate |
| `extract_gem_probe` | `(layer_activations, eval_frac=0.2, seed=42, concept="", attention_paradigm="unknown", direction_method="fisher_weighted", calibration_percentile=5.0) -> ProbeResult` | CAZ-aware probe: runs `find_caz_regions_scored`, takes the dominant region, builds the direction (`"fisher_weighted"` over the region's layers, or `"endpoint"` DOM at `dom.end`), scores at the **handoff layer** `min(dom.end + 1, N-1)`, and sets a per-region assembly threshold (`gem_region_thresholds`) at the `calibration_percentile` of training-positive max cosines. Falls back to `extract_probe(method="raw")` (`probe_type="raw_fallback"`) when no region is found |
| `ProbeResult` | dataclass | `concept, layer, direction (float32), threshold, sep_curve, pos_mean, neg_mean, auroc, n_pos, n_neg, probe_type, caz_regions, gem_region_thresholds, ...`; `.to_dict()` for JSON |

**Train/eval split (v2.0.0+).** `_split_indices` is pair-aware: when `n_pos == n_neg` one
permutation over pair indices drives both classes, so a pair never straddles train and eval.
Pass `paired=False` for genuinely unpaired data. With independent per-class permutations (v1.x)
each pair had ≈ 32 % probability of straddling at `eval_frac=0.2`, inflating fitted estimators'
held-out AUROC.

**Thresholds.** `ProbeResult.threshold` is the midpoint of positive and negative mean scores on
the *held-out contrastive negatives*. That is not a benign-traffic operating point; deployments
must recalibrate the alert threshold on their own benign distribution.

---

## gem — Geometric Evolution Maps

Pure numpy. A **GEM** is the complete set of assembly events for one concept in one model — one
`GEMNode` per CAZ segment, carrying the segment, its per-layer trajectory, the handoff layer and
the settled direction (`papers/shared/GEM_DEFINITION.md`).

```python
from rosetta_tools.gem import (
    build_concept_gem, save_gem, load_gem, gem_diagnostics,
    extract_phase2_threads, extract_settled_subspace, attach_settled_subspace,
)
```

**Data types**

| Type | Key fields | Notes |
|---|---|---|
| `EigThread` | `thread_id, layer_indices, directions, eigenvalues, concept_alignments, cos_chain, ...` | One eigenvector tracked across the CAZ window |
| `GEMNode` | `concept, model_id, caz_index, caz_start, caz_peak, caz_end, caz_score, attention_paradigm, settled_direction, handoff_layer, handoff_cosine, entry_exit_cosine, max_rotation_per_layer, k, threads, phase, settled_subspace, ...` | One assembly event |
| `ConceptGEM` | `concept, model_id, n_nodes, nodes, interaction_matrix, node_types, ablation_targets` | The GEM (atlas) |
| `SettledSubspace` | `layer, extraction_method="inlp", directions, held_out_acc_curve, held_out_logloss_curve, n_rounds, stop_reason, ...` | Rank-k settled structure at one layer |

**Construction**

| Function | Signature | Purpose |
|---|---|---|
| `build_gem_node_k1` | `(caz_data, region, caz_index, attention_paradigm="unknown") -> GEMNode` | k = 1 node from a `caz_<concept>.json` dict |
| `build_gem_node_phase2` | `(caz_data, region, caz_index, h_pos_by_layer, h_neg_by_layer, k=3, attention_paradigm="unknown", ...) -> GEMNode` | k > 1 node with eigen-threads (v2.1.0) |
| `build_concept_gem` | `(caz_data, multimodal_data=None, attention_paradigm="unknown", k=1, h_pos_by_layer=None, h_neg_by_layer=None, compute_settled_subspace=False, ...) -> ConceptGEM` | Full GEM from a CAZ record (+ optional ablation/multimodal data for node typing) |
| `classify_node_dependencies` | `(interaction_matrix, independence_threshold=80.0) -> (node_types, ablation_targets)` | Independent / dependent node typing from pairwise ablation interactions |
| `extract_phase2_threads` | `(h_pos_by_layer, h_neg_by_layer, layer_indices, k=3, dom_vectors_by_layer=None, ...) -> list[EigThread]` | Greedy bipartite matching of top-k eigenvectors across layers (v2.1.0) |
| `extract_settled_subspace` | `(X_train, y_train, X_test, y_test, layer, max_rounds=10, stop_alpha=0.05, ...) -> SettledSubspace` | **Experimental.** INLP: iteratively fit a linear probe, project its direction out, stop when held-out accuracy is no longer above the majority-class rate (one-sided binomial triage; permutation test when ambiguous). `k_effective` is an **upper bound** on settled rank: with n ≪ d the fitted direction is noisy and the residual re-detects (synthetic rank-1 data returns 2–3). Compare against a matched rank-1 control before claiming rank > 1 (v2.1.0) |
| `attach_settled_subspace` | `(node, h_pos, h_neg, test_size=0.3, random_state=0, **extract_kwargs) -> GEMNode` | Functional: returns a copy of `node` with `settled_subspace` filled at its handoff layer (v2.1.0) |

**Serialisation and diagnostics**

| Function | Signature | Purpose |
|---|---|---|
| `gem_to_dict` / `gem_from_dict` | `(gem) -> dict` / `(data) -> ConceptGEM` | Round-trip. ≥ 2.1.0 emits a `settled_subspace` key per node; v2.0.0's loader rejects unknown node keys, so read new files with ≥ 2.1.0 |
| `save_gem` / `load_gem` | `(gem, path)` / `(path) -> ConceptGEM` | JSON on disk (`gem_<concept>.json`) |
| `validate_gem_node` | `(node) -> list[str]` | Schema / range problems, empty list when clean |
| `gem_diagnostics` | `(gem) -> dict` | Node count, depth classes, rotation summary |
| `classify_caz_depth` | `(layer, n_layers) -> str` | `"shallow"` / `"mid"` / `"deep"` by relative depth |
| `routing_held_ratio` | `(baseline_per_layer, peak_layer, final_layer, degradation_factor=0.95) -> (ratio, held)` | Does separation survive from peak to final layer? |
| `routing_held_from_permutation` | `(subset_results, node_idx, final_layer, baseline_final_sep, contribution_threshold=0.05) -> bool` | Same question from a permutation-ablation result set |
| `stage_detail` | `(nodes, n_layers, routing_held_per_node=None) -> str` | Human-readable stage summary |

**Discovery helpers** (read the local `ROSETTA_MODELS` tree): `find_extraction_dir(model_id)`,
`discover_concepts(extraction_dir, source="gem")`, `discover_all_models()`, `discover_base_models()`.

---

## dataset — Contrastive Pair Loading

```python
from rosetta_tools.dataset import load_concept_pairs, load_pairs, texts_by_label, ConceptPair
```

| Function | Signature | Purpose |
|---|---|---|
| `load_concept_pairs` | `(concept, *, split="train", n=200, seed=None) -> list[ConceptPair]` | **Primary entry point.** Resolves the canonical Rosetta_Concept_Pairs root (`ROSETTA_CONCEPTS_ROOT` or repo-relative), applies the fixed train/validation split, samples up to `n` pairs. Default seed is SHA256-derived from the concept name, so the sample reproduces across processes (v2.0.0; v1.x used process-randomised `hash()`) |
| `load_pairs` | `(path) -> list[ConceptPair]` | Load one JSONL file. Drops blank-text pairs instead of misaligning rows (v2.0.0) |
| `texts_by_label` | `(pairs) -> (pos_texts, neg_texts)` | Split into the two text lists `extraction` wants |
| `iter_texts` | `(pairs) -> Iterator[(text, label, pair)]` | Flat iteration |
| `validate_dataset` | `(path) -> list[str]` | Schema problems, empty when clean |
| `load_pairs_df` | `(path) -> pd.DataFrame` | Tidy frame |
| `dataset_summary` | `(path) -> dict` | Counts by domain/topic |
| `ConceptPair` | `(pair_id, domain, topic, pos_text, neg_text, concept="", model_name="", metadata={})` | dataclass |
| `ALL_CONCEPTS`, `CAZ_PRH_CONCEPTS` | constants | 18 registered concepts; the 17 used by the CAZ/PRH corpus |

---

## alignment — Procrustes Rotation

```python
from rosetta_tools.alignment import compute_procrustes_rotation, apply_rotation, align_and_score
```

| Function | Signature | Purpose |
|---|---|---|
| `compute_procrustes_rotation` | `(source_acts, target_acts, n_components=None) -> R` | Orthogonal Procrustes on paired activation clouds (optionally after PCA) |
| `apply_rotation` | `(vector, R) -> NDArray` | Rotate a concept vector into the target basis |
| `cosine_similarity` | `(v1, v2) -> float` | |
| `align_and_score` | `(source_vec, target_vec, source_acts, target_acts, n_components=None) -> dict` | Rotation + aligned cosine + diagnostics in one call |
| `pairwise_alignment_df` | `(vectors, activations) -> pd.DataFrame` | All ordered model pairs |

---

## ablation — Directional Ablation

Forward hooks on HF `AutoModel` decoder blocks. Requires torch.

```python
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers, kl_divergence_from_logits
```

| Symbol | Signature | Purpose |
|---|---|---|
| `get_transformer_layers` | `(model) -> list[nn.Module]` | Decoder blocks for any supported architecture (GPT-2 `.h`, Llama/Qwen `.layers`, NeoX, OPT, Gemma, ...) |
| `compute_dominant_direction` | `(pos_acts, neg_acts) -> NDArray` | Unit mean-difference direction |
| `DirectionalAblator` | `(layer_module, direction, dtype=torch.float32)` | Context manager: project `direction` out of the block output |
| `DirectionalShifter` | `(layer_module, shift_vector, coefficient=1.0, dtype=torch.float32)` | Context manager: add `coefficient * shift_vector` to the block output (v1.4.0) |
| `compute_baseline_logits` | `(model, tokenizer, prompts, device="cuda") -> list[Tensor]` | |
| `kl_divergence_from_logits` | `(baseline_logits, ablated_logits) -> float` | Behavioural effect size |

---

## reporting — Result Loading

```python
from rosetta_tools.reporting import load_results_dir, load_region_df, load_scored_region_df
```

| Function | Signature | Purpose |
|---|---|---|
| `load_result_df` | `(path) -> pd.DataFrame` | One `caz_<concept>.json` → long layer frame |
| `load_results_dir` | `(path \| [paths], glob="caz_*.json", include_legacy=True) -> pd.DataFrame` | Every checkpoint under one or more directories |
| `load_run_summary` | `(path) -> pd.DataFrame` | `run_summary.json` → wide form |
| `load_region_df` | `(layer_df, min_prominence_frac=0.1) -> pd.DataFrame` | Per-region summary (strict detector) |
| `load_scored_region_df` | `(layer_df, min_prominence_frac=0.005) -> pd.DataFrame` | Per-region summary (scored / gentle detector — the corpus default) |

---

## models — Model and Concept Registry

YAML-backed (`rosetta_tools/data/models.yaml`, `concepts.yaml`), loaded once at import. All
scripts import from here rather than keeping their own model lists.

```python
from rosetta_tools.models import (
    all_models, families, models_by_family, models_by_tag, models_by_cluster,
    get_model, vram_gb, attention_paradigm_of, requires_quantization, instruct_pairs,
    all_concepts, get_concept, concepts_by_pipeline,
)
```

| Function | Signature | Purpose |
|---|---|---|
| `get_model` | `(model_id) -> ModelEntry \| None` | `model_id, family, hidden_dim, n_layers, vram_bf16, cluster, encoding, attention, quantization, enabled, gated, tags, notes` |
| `all_models` | `(include_disabled=False) -> list[str]` | Enabled model IDs in YAML order |
| `models_by_family` / `models_by_tag` / `models_by_cluster` / `models_by_encoding` / `models_by_hidden_dim` | `(key, include_disabled=False) -> list[str]` | Filters; all respect the `enabled` flag by default |
| `families` | `(include_disabled=False) -> dict[str, list[str]]` | `{family: [model_ids]}` |
| `zero_pca_clusters` | `(include_disabled=False) -> dict[int, list[str]]` | `{hidden_dim: ids}` for dims with ≥ 2 models (Procrustes pairs without PCA) |
| `family_of` / `hidden_dim_of` / `vram_gb` / `attention_paradigm_of` / `requires_quantization` | `(model_id) -> scalar` | Lookups with safe defaults (`"unknown"`, `0`, `0.0`, `None`) |
| `instruct_pairs` | `(include_disabled=False) -> list[(base, instruct)]` | Models tagged `rlhf-pair`, matched by family stem + `vram_bf16` |
| `all_concepts` / `get_concept` | `() -> list[str]` / `(name) -> ConceptEntry \| None` | `name, category, pipelines, assembly_depth_pct, assembly_depth_std, notes` |
| `concepts_by_pipeline` / `concepts_by_category` | `(key) -> list[str]` | `pipeline ∈ {"caz", "cia"}`; `category ∈ {epistemic, syntactic, relational, affective, security}` |
| `concept_assembly_depths` | `() -> dict[str, float]` | Mean assembly depth (%) for measured concepts only |

Tags in use: `scale`, `frontier`, `instruct`, `rlhf-pair`, `moe`, `abliterated`, `framework-fig`,
`validation-fig`. Disabled entries (instruct variants, MoE, very large models) are excluded from
every default query; pass `include_disabled=True` to see them.

---

## viz and viz_style — Figures

`viz_style` holds constants and helpers and imports no matplotlib backend, so it is safe to import
anywhere (including Rosetta_Analysis figure scripts that pick their own backend). `viz` selects
Agg and provides the file-writing plot functions.

```python
from rosetta_tools.viz_style import THEME, CONCEPT_META, CONCEPT_ORDER, concept_color, apply_theme
from rosetta_tools.viz import plot_caz_profile, plot_concept_comparison, plot_peak_heatmap
```

| Symbol | Purpose |
|---|---|
| `THEME`, `apply_theme(ax, ax_twin=None, *, grid=True)` | House style for axes |
| `CONCEPT_META`, `CONCEPT_ORDER`, `CONCEPT_TYPE`, `CONCEPT_COLORS`, `CONCEPT_COLORS_ACCESSIBLE`, `concept_color(name, *, accessible=False)` | Per-concept display name, type, colour |
| `TYPE_COLORS`, `TYPE_BG` | Concept-type palette |
| `FAMILY_ORDER`, `FAMILY_COLORS`, `FAMILY_MAP`, `model_label(model_id)`, `sort_models(ids)` | Model-family palette and ordering |
| `CAZ_CAT_COLORS`, `CAZ_CAT_FILL`, `CAZ_CAT_LABELS`, `caz_score_cat(score)` | Major → Gentle CAZ score categories |
| `layer_ticks(n_layers, pcts=(0,25,50,75,100))`, `add_outside_callouts(ax, callouts, n_layers)` | Depth axes helpers |
| `plot_caz_profile(df, concept, model_id, out_path, title=None)` | Three-panel S / C / v plot |
| `plot_concept_comparison(df, out_path, model_ids=None, title=None)` | Multi-concept overlay |
| `plot_peak_heatmap(df, out_path, title=None)` | Peak depth % heatmap, concepts × models |

---

## compound_monitor — Vigilance Thresholds

Pure Python. A small, generic helper for monitors that combine several evidence modules: tighten
an allocation threshold in proportion to anomaly evidence from other modules. It carries **no
verdict rules** — those belong to the consuming monitor.

```python
from rosetta_tools.compound_monitor import ModuleEvidence, vigilance_threshold, apply_vigilance
```

| Symbol | Signature | Purpose |
|---|---|---|
| `ModuleEvidence` | `(name, confidence ∈ [0,1], detail={})` | One module's anomaly confidence |
| `vigilance_threshold` | `(base_threshold, evidence, weights, floor_factor=0.5) -> float` | `base × (1 − Σ w·conf)`, floored at `base × floor_factor` |
| `apply_vigilance` | `(allocations, base_threshold, evidence, weights, floor_factor=0.5) -> dict[str, bool]` | Per-concept allocated-under-vigilance verdicts |

---

## paths — Canonical Data Paths

Resolves to `Path.home() / "rosetta_data" / ...` so the same code runs on the dev machine and on
GPU hosts (whose source layout differs). Never hard-code `~/Source/...` in scripts that run on
hosts.

| Constant / function | Value |
|---|---|
| `ROSETTA_DATA_ROOT`, `ROSETTA_DATA` | `~/rosetta_data` |
| `ROSETTA_MODELS` | `~/rosetta_data/models` — per-model extraction dirs |
| `ROSETTA_RESULTS` | `~/rosetta_data/results` |
| `ROSETTA_PAPER_N250` | `~/rosetta_data/paper_n250` — frozen N = 250 paper snapshot (HF `paper-n250` tag) |
| `ROSETTA_MODELS_SNAPSHOTS` | legacy snapshots dir |
| `iter_paper_model_dirs(*, exclude_instruct=True, exclude_patterns=("deepdive","dark","manifold"))` | Model dirs that count toward paper tables |

---

## Migration Notes

### `shared.gpu_utils` → `rosetta_tools.gpu_utils` (April 2026)

```python
# Before
from shared.gpu_utils import get_device, get_dtype, log_vram
# After
from rosetta_tools.gpu_utils import get_device, get_dtype, log_vram
```

### Model registry centralisation (April 2026)

Model lists previously hard-coded in scripts live in `rosetta_tools.models` (YAML since v1.1.0).
`CONCEPTS` / `REGISTRY` module-level lists no longer exist; use `all_concepts()` / `all_models()`.

### Analysis modules moved out (May 2026, v1.3.x → v2.0.0)

`feature_tracker`, `manifold_detector`, `tracking` → [Rosetta_Analysis](https://github.com/jamesrahenry/Rosetta_Analysis);
`feature_atlas`, `caz_registry`, `dark_feature_scanner`, `consensus_generator` → Rosetta_Feature_Library.
`rosetta_tools` is the reusable CAZ/GEM pipeline only. The only module that moved *in* is the CAZ
record assembler (`extraction.extract_layer_wise_metrics`, `extraction.build_caz_record`) in v2.0.0.

### v1.1.0 → v2.x for probe consumers (Monitor, CIA)

`extract_probe`, `extract_gem_probe` and `score_direction` have identical signatures across
v1.1.0, v2.0.0 and v2.1.0. What changed underneath: the pair-aware train/eval split (affects
held-out AUROC and the midpoint threshold when `eval_frac > 0`), the reproducible
`load_concept_pairs` seed, and blank-pair dropping in `load_pairs`. Consumers that split texts
themselves before calling `extract_*` are not protected by the pair-aware split and must shuffle
pair indices, not the two text lists independently.

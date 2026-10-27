# rosetta_tools API Reference

*Updated: 2026-10-26 UTC — covers v2.2.0.*

Quick-reference index for every public module. All public functions have full docstrings in
source; where this page and a docstring disagree, the docstring wins. Signatures are abridged
(`...` = further keyword arguments; see source).

Modules fall into three groups:

| Group | Modules | Needs torch / transformers? |
|---|---|---|
| Model-facing | `gpu_utils`, `extraction`, `ablation`, `gem_sublayer` (capture path) | yes (imported lazily; informative `ImportError` otherwise) |
| Pure numpy / pandas | `caz`, `probes`, `gem`, `gem_sublayer` (numpy path), `alignment`, `dataset`, `reporting`, `compound_monitor` | no |
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
| `extract_probe` | `(layer_activations, method="raw", threshold_strategy="midpoint", target_tpr=..., eval_frac=0.2, seed=42, concept="", direction_estimator="dom", ...) -> ProbeResult` | Peak-layer probe. `method`: `"raw"` (max raw separation), `"fisher"`, `"auroc"`. `direction_estimator` (v2.2.0): `"dom"` (default, difference-of-means, identical to every earlier release), `"logreg"` or `"lda"`; orthogonal to `method` (which picks the layer), and with `method="auroc"` the layer scan itself uses the chosen estimator. Falls back gracefully when a layer is degenerate |
| `extract_gem_probe` | `(layer_activations, eval_frac=0.2, seed=42, concept="", attention_paradigm="unknown", direction_method="fisher_weighted", calibration_percentile=5.0, direction_estimator="dom") -> ProbeResult` | CAZ-aware probe: runs `find_caz_regions_scored`, takes the dominant region, builds the direction (`"fisher_weighted"` over the region's layers, or `"endpoint"` at `dom.end`; both use `direction_estimator`, default `"dom"`), scores at the **handoff layer** `min(dom.end + 1, N-1)`, and sets a per-region assembly threshold (`gem_region_thresholds`) at the `calibration_percentile` of training-positive max cosines. Falls back to `extract_probe(method="raw")` (`probe_type="raw_fallback"`) when no region is found |
| `ProbeResult` | dataclass | `concept, layer, direction (float32), threshold, sep_curve, pos_mean, neg_mean, auroc, n_pos, n_neg, probe_type, caz_regions, gem_region_thresholds, direction_estimator, ...`; `.to_dict()` for JSON |

**Train/eval split (v2.0.0+).** `_split_indices` is pair-aware: when `n_pos == n_neg` one
permutation over pair indices drives both classes, so a pair never straddles train and eval.
Pass `paired=False` for genuinely unpaired data. With independent per-class permutations (v1.x)
each pair had ≈ 32 % probability of straddling at `eval_frac=0.2`, inflating fitted estimators'
held-out AUROC.

**Thresholds.** `ProbeResult.threshold` is the midpoint of positive and negative mean scores on
the *held-out contrastive negatives*. That is not a benign-traffic operating point; deployments
must recalibrate the alert threshold on their own benign distribution.

**Direction estimators (v2.2.0).** `"logreg"` and `"lda"` fit on the training split and return the unit coefficient vector, signed positive-class-ward like `"dom"`. They need a pair-aware split to be compared fairly against `"dom"`: a fitted estimator can score the held-out mate of a training item, a leak a centroid contrast cannot exploit. Use `eval_frac > 0` with paired data (the default split is pair-aware) and expect fitted estimators to sit near AUROC 1.0 when `n_train` is small relative to the hidden dimension, so compare magnitudes, not win rates. scikit-learn is imported lazily, only when a non-`"dom"` estimator is requested.

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

**Schema v2 (v2.2.0).** `GEM_SCHEMA_VERSION = "2"`. A `ConceptGEM` now optionally carries
`layer_profile` (a `LayerProfile`), `sidecar` (a `SidecarRef`), `caz_detection_score` and a
`schema_version` string, and each `SettledSubspace` an optional `sublayer_decomposition`. Everything is
optional and defaults to `None`: v1 files (no `schema_version`) load with `schema_version="1"` and
every v2 field `None`, and unknown top-level or node keys are kept rather than rejected.
`save_gem` always writes the current version. **Reader compatibility:** v2.1.0 and earlier loaders
reject the new node keys, so GEM JSON written by ≥ 2.2.0 with v2 fields must be read with ≥ 2.2.0.
Files written without any v2 data (`GEMExtractionConfig.legacy()`) round-trip as before.

| Type | Key fields | Notes |
|---|---|---|
| `GEMExtractionConfig` | `layer_profile=True, sublayer_capture=True, store_full_sublayer_activations=True, store_post_final_norm=True, inline_directions=False, n_permutations=200, permutation_seed=0, additivity_tol=None, allow_inexact=False, pooling="last", batch_size=16, max_length=512, detection_score="fisher"` | One switchboard for what a GEM build records. **Defaults are everything on**, so net-new generation records the v2 data; `GEMExtractionConfig.legacy()` reproduces pre-v2 output (`caz_*.json` only). `sublayer_capture=False` gives a stream-only profile from `calibration_alllayer_<concept>.npy` with no model. `store_full_sublayer_activations` is large (about 43 MB per concept at 0.5B, 147 MB at 3B). `additivity_tol=None` uses `gem_sublayer.ADDITIVITY_TOL[model dtype]` |
| `LayerProfile` | `n_layers, hidden_dim, layers, embedding, final_norm, stream_source, direction_convention, architecture, provenance` | Per-layer records at **every** layer, not only the CAZ window. `.record(layer)` returns `embedding` for `-1`. `final_norm` is the post-final-norm stream at the last layer (= `hidden_states[-1]`, what existing CAZ data holds there) alongside the raw block output in `layers[-1]` |
| `LayerRecord` | `layer, separation_fisher, separation_projected, raw_distance, projected_pooled_std, null_fisher, null_projected, cos_prev_direction, stream_norm_mean, delta_* , recon_residual_*, sublayers, settled_subspace, sidecar_row, direction` | GEM layer `l` is the output of decoder block `l` (`-1` = embedding output). `separation_projected` is the gauge-corrected score `‖μ₊−μ₋‖ / √(½uᵀ(S₊+S₋)u)`. `direction` (`u_l`) is `None` unless saved with `inline_directions=True`; the sidecar always holds it |
| `SublayerRecord` | `diff_proj_u_in/out, pos/neg_mean_proj_u_in/out, var_proj_u_out, separation_projected_u_out, share_of_concept_write_u_out, mean_diff_norm, cos_mean_diff_u_out, ...` | Attention or MLP write of one block, projected on the unit mean-difference directions of the stream the block reads (`u_in`) and writes (`u_out`) |
| `PermutationNull` | `statistic, n_perm, seed, mean, std, q95, max, p_value` | Label-permutation null for one separation statistic at one layer; the direction is refit on every permutation. `p_value = (#{null ≥ observed} + 1) / (n_perm + 1)` |
| `ArchitectureFlags` | `model_type, decoder_class, layer_path, attn_hook, mlp_hook, parallel_block, sublayer_post_norm, post_ln, moe, final_norm_hook, additivity_expected_exact, notes` | What the sublayer hooks mean for this model and whether `Δ = attn + mlp` should hold exactly |
| `Provenance` | `code_version, created_utc, model_id, model_dtype, capture_dtype, stats_dtype, hook_points, pooling, n_pos, n_neg, seed, n_perm, max_length, batch_size, device, dataset, pair_set, notes` | How a record was produced. `pair_set` holds `dataset.pair_set_provenance(...)`; `None` on older records |
| `SubspaceSublayerDecomposition` | `layer, attn_diff_proj, mlp_diff_proj, stream_diff_proj, embed_diff_proj, upstream_attn_diff_proj, upstream_mlp_diff_proj, reconstruction_residual` | Splits each `SettledSubspace` direction's class-mean separation at layer `L` into the writes that built it; exact in pre-LN models |
| `SidecarRef` | `path, format="npz", sha256, arrays` | Index of the sidecar arrays: relative path, checksum, per-key shape, dtype, axes and description |

| Function | Signature | Purpose |
|---|---|---|
| `save_gem` | `(gem, path, arrays=None, array_docs=None, inline_directions=False) -> ConceptGEM` | As before, plus: given `arrays` (from `build_layer_profile`) it writes a `gem_<concept>.sidecar.npz` next to the JSON and records its path, sha256 and per-key index in `gem.sidecar`. Returns the `ConceptGEM` as written; the input is not mutated. `inline_directions=True` also writes each `u_l` inline (needs a `layer_profile` and `arrays["profile.directions"]`) |
| `sidecar_path_for` | `(json_path) -> Path` | `gem_<concept>.json` → `gem_<concept>.sidecar.npz`, same directory |
| `load_gem_sidecar` | `(json_path, gem=None, verify=True)` | Lazy `np.load` handle onto the sidecar (use as a mapping, close when done). `FileNotFoundError` if the GEM has none. `ValueError` on a sha256 mismatch, **and when the stored sha256 is empty**, unless `verify=False` |
| `find_caz_regions_projected` | `(caz_data, layer_profile, attention_paradigm="unknown", **detector_kwargs) -> CAZProfile` | Opt-in research path: the scored detector run over the projected separation instead of the Fisher score. The default detector is untouched. The detector's thresholds were tuned on the Fisher scale, so differences mix "better score" with "untuned thresholds" |

Builders for these records live in [`gem_sublayer`](#gem_sublayer--per-layer-and-per-sublayer-gem-records); `rosetta_tools.gem` also re-exports them lazily (so `gem.build_layer_profile` works without importing torch).

---

## gem_sublayer — Per-Layer and Per-Sublayer GEM Records

Builds the schema-v2 records described under [gem](#gem--geometric-evolution-maps). Three tiers,
cheapest first; only tier 2 needs torch (imported lazily).

```python
from rosetta_tools.gem_sublayer import (
    build_layer_profile, attach_layer_profile,
    capture_sublayer_activations, extract_concept_layer_profile,
    decompose_settled_subspace, attach_layer_settled_subspaces,
)
```

**Layer convention.** GEM layer `l` is the output of decoder block `l`; block `l` reads GEM layer
`l-1` (the embedding output for `l = 0`). `u_in` and `u_out` are the unit class-mean-difference
directions of those two streams.

| Tier | Function | Signature | Purpose |
|---|---|---|---|
| 1 (numpy) | `build_layer_profile` | `(stream_pos, stream_neg, embed_pos=None, embed_neg=None, attn_pos=None, attn_neg=None, mlp_pos=None, mlp_neg=None, n_perm=200, seed=0, architecture=None, provenance=None, stream_source="decoder_layer_output", store_full_sublayer_acts=True, final_norm_pos=None, final_norm_neg=None) -> (LayerProfile, arrays, array_docs)` | Per-layer records at every layer from `(L, n, d)` stream activations: Fisher and gauge-corrected projected separation, each with a label-permutation null. Works on existing `calibration_alllayer_<concept>.npy` (stream-only backfill, `stream_source="hidden_states"`), or additionally on sublayer captures for the attention/MLP split. Pass `arrays` and `array_docs` straight to `save_gem` |
| 1 | `attach_layer_profile` | `(gem, profile) -> ConceptGEM` | Functional: copy of `gem` with `layer_profile` set |
| 2 (torch) | `capture_sublayer_activations` | `(model, tokenizer, texts, device="cpu", batch_size=8, pool="last", max_length=512, allow_inexact=False, tol=None, capture_final_norm=True) -> SublayerCapture` | Forward hooks on each decoder block and its attention and MLP outputs, pooled per example. Checks `Δ = attn + mlp` numerically and **raises** when it fails, unless `allow_inexact=True` (the residual is then recorded and flagged). Hooks are removed even if registration fails partway |
| 2 | `extract_concept_layer_profile` | `(model, tokenizer, pos_texts, neg_texts, model_id="", device="cpu", config=None, dataset=None, pair_set=None) -> (LayerProfile, arrays, array_docs)` | Capture plus `build_layer_profile` for one concept, driven by a `GEMExtractionConfig` (default: everything on). Pass `pair_set=pair_set_provenance(pairs, ...)` to record which pairs were used |
| 2 | `resolve_architecture` | `(model) -> ArchitectureFlags` | Decide which modules to hook and whether `Δ = attn + mlp` should be exact. Raises `NotImplementedError` for unknown block layouts rather than guessing |
| 2 | `find_decoder_layers` | `(model) -> (path, layers)` | Locate the decoder block list |
| 3 | `decompose_settled_subspace` | `(sub, arrays, provenance=None) -> SettledSubspace` | Attach a `SubspaceSublayerDecomposition` to a `SettledSubspace` |
| 3 | `attach_layer_settled_subspaces` | `(profile, h_pos_by_layer, h_neg_by_layer, layers, test_size=0.3, random_state=0, **extract_kwargs) -> LayerProfile` | Run INLP at the listed layers and attach the results; expensive, so not computed by default. Carries the same caution as `extract_settled_subspace` |
| — | `code_version` | `() -> str` | `rosetta_tools <version> (<sha>[+dirty])`, stored in `Provenance.code_version` |
| — | `projected_separation` | `(pos, neg) -> (separation, raw_distance, pooled_std, direction)` | The gauge-corrected statistic used throughout |
| — | `ADDITIVITY_TOL`, `SIDECAR_DOCS` | constants | Per-dtype tolerance on `max‖Δ−a−m‖/‖x_out‖` (relative to the stream norm); index documentation for every sidecar key this module writes |

**Which architectures are exact.** Hooks sit on what is actually added to the stream, so
`x_out = x_in + attn + mlp` holds exactly for pre-LN models (Llama, Qwen, Mistral, GPT-2,
OPT pre-LN), Gemma 2/3 (hooked after the post-sublayer norms), and parallel blocks (GPT-J,
GPT-NeoX with `use_parallel_residual`, Phi), where the MLP reads `x_in` rather than
`x_in + attn` (flagged in `ArchitectureFlags.parallel_block`). Mixture-of-experts MLPs (Mixtral,
Qwen2/3-MoE) are exact too; the `mlp` write is then the router-weighted expert sum
(`ArchitectureFlags.moe`). Post-LN models (OPT-350m, BERT-style) are not additive and are refused
unless `allow_inexact=True`. Captures use decoder-block outputs, not `output_hidden_states`,
because `hidden_states[-1]` is post-final-norm in most HF models; the post-final-norm stream is
recorded separately as `LayerProfile.final_norm`.

**Tolerances.** The additivity check is relative to the stream norm, not the delta norm: in half
precision each residual add rounds at about `eps · ‖x‖`, and `‖Δ‖ ≪ ‖x‖` deep in the network.
Defaults: float64 `1e-8`, float32 `1e-4`, float16 `5e-3`, bfloat16 `2e-2`.

**Typical flow.**

```python
pairs = load_concept_pairs("agency", n=250)
pos, neg = texts_by_label(pairs)
profile, arrays, docs = extract_concept_layer_profile(
    model, tok, pos, neg, model_id=model_id, device="cuda",
    pair_set=pair_set_provenance(pairs, "agency", "train", 250),
)
gem = attach_layer_profile(build_concept_gem(caz_data), profile)
save_gem(gem, "gem_agency.json", arrays=arrays, array_docs=docs)
with load_gem_sidecar("gem_agency.json") as sc:
    directions = sc["profile.directions"]          # (L, d), u_l per layer
```

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
| `pair_set_provenance` | `(pairs, concept="", split=None, n_requested=None) -> dict` | Identify exactly which pairs a run used (v2.2.0). Returns `sha256` over the ordered `(pair_id, pos_text, neg_text)` triples (any change in membership, order or text changes it), `n_pairs`, `concept`, `split`, `n_requested`, `pairs_file` (relative to the Rosetta_Concept_Pairs repo when resolvable), `repo_commit` (that repo's git HEAD, or `None`) and `file_dirty`. Motivation: dataset fixes changed which 250 pairs `load_concept_pairs()` returns for the same call, silently; this makes that detectable. JSON-safe; store it in `Provenance.pair_set` |
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

### v2.1.0 → v2.2.0

Additive. `direction_estimator` is a new keyword on `extract_probe` and `extract_gem_probe` with
default `"dom"`, so existing call sites and stored probes are unchanged. GEM schema v2 is opt-in on
the read side (v1 files load unchanged) but **GEM JSON written by ≥ 2.2.0 with v2 fields cannot be
read by ≤ 2.1.0**, which rejects the new keys. Net-new generation through
`extract_concept_layer_profile` records v2 data by default; pass
`GEMExtractionConfig.legacy()` for pre-v2 output. `load_gem_sidecar` now treats an empty stored
sha256 as an error rather than skipping verification; pass `verify=False` to load unverified.

### v1.1.0 → v2.x for probe consumers (Monitor, CIA)

`extract_probe`, `extract_gem_probe` and `score_direction` have identical signatures across
v1.1.0, v2.0.0 and v2.1.0; v2.2.0 adds the optional `direction_estimator` keyword (default
`"dom"`) to the first two. What changed underneath: the pair-aware train/eval split (affects
held-out AUROC and the midpoint threshold when `eval_frac > 0`), the reproducible
`load_concept_pairs` seed, and blank-pair dropping in `load_pairs`. Consumers that split texts
themselves before calling `extract_*` are not protected by the pair-aware split and must shuffle
pair indices, not the two text lists independently.

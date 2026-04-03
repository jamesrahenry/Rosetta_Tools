# rosetta_tools API Reference

Production-grade Python library for the Rosetta Program interpretability research.

All public functions have full docstrings in source. This document provides a quick-reference index.

---

## gpu_utils — Device Management

Environment-aware GPU utilities. Dtype policy: bfloat16 on datacenter GPUs (Ampere+, >=40 GiB), float32 on consumer GPUs and CPU.

```python
from rosetta_tools.gpu_utils import get_device, get_dtype, vram_stats, log_vram
```

| Function | Signature | Purpose |
|---|---|---|
| `get_device` | `(prefer="auto") -> str` | Detect best available device (`cuda`, `cpu`) |
| `get_dtype` | `(device, prefer="auto") -> torch.dtype` | Environment-aware dtype selection |
| `vram_stats` | `(device_index=0) -> dict \| None` | Current VRAM: `total_gib`, `used_gib`, `free_gib`, `pct_used` |
| `log_vram` | `(label="", device_index=0) -> None` | One-line VRAM summary to logger. No-ops on CPU |
| `log_device_info` | `(device, dtype) -> None` | Startup banner: device, dtype, VRAM |
| `safe_batch_size` | `(requested, device="cuda", reserve_gib=2.0) -> int` | Scale down batch size if VRAM is tight |
| `release_model` | `(model, *, clear_cache=True) -> None` | Delete model and free GPU memory |
| `purge_hf_cache` | `(model_id) -> None` | Delete a model from HuggingFace local cache |

---

## extraction — Activation Extraction

Model-agnostic activation extraction via native HuggingFace (`output_hidden_states=True`). No TransformerLens dependency.

```python
from rosetta_tools.extraction import extract_layer_activations, extract_contrastive_activations
```

| Function | Signature | Purpose |
|---|---|---|
| `extract_layer_activations` | `(model, tokenizer, texts, device="cpu", batch_size=8, pool="last", max_length=512) -> list[NDArray]` | Residual-stream activations at every layer. Returns `[n_layers][n_texts, hidden_dim]` |
| `extract_contrastive_activations` | `(model, tokenizer, pos_texts, neg_texts, ...) -> list[tuple[NDArray, NDArray]]` | Paired (positive, negative) activations per layer for CAZ metrics |

**Pooling strategies:** `"last"` (last token, default for causal LMs), `"mean"` (mean pool all tokens).

**Metrics are always computed in float64** regardless of forward-pass dtype, to prevent fp16 overflow in Fisher normalization.

---

## caz — Concept Assembly Zone Metrics

Core CAZ math: separation, coherence, velocity, boundary detection. Library-agnostic — operates on numpy arrays, not model objects.

```python
from rosetta_tools.caz import (
    compute_separation, compute_coherence, compute_velocity,
    compute_layer_metrics, find_caz_boundary, find_caz_regions_scored,
)
```

**Metric functions:**

| Function | Signature | Purpose |
|---|---|---|
| `compute_separation` | `(pos, neg, eps=1e-8) -> float` | Fisher-normalized centroid distance: `S(l) = \|mu_A - mu_B\| / sqrt(0.5 * (tr(Sigma_A) + tr(Sigma_B)))` |
| `compute_coherence` | `(pos, neg, n_components=1) -> float` | Explained variance ratio of primary PCA component |
| `compute_velocity` | `(separations, window=3) -> NDArray` | Smoothed first derivative of S(l) |
| `compute_layer_metrics` | `(layer_activations, velocity_window=3) -> list[LayerMetrics]` | Compute S, C, v for every layer |

**Boundary/region detection:**

| Function | Signature | Purpose |
|---|---|---|
| `find_caz_boundary` | `(metrics, threshold_factor=0.5, min_sustained=2) -> CAZBoundary` | Single-peak CAZ boundary detection |
| `find_caz_regions` | `(metrics, min_prominence_frac=0.1, ...) -> CAZProfile` | Multi-peak region detection (10% prominence floor) |
| `find_caz_regions_scored` | `(metrics, min_prominence_frac=0.005, ...) -> CAZProfile` | Scored detection (0.5% floor) — finds gentle CAZes |
| `compute_caz_statistics` | `(metrics, boundary) -> dict` | Per-region summary statistics |

**Data classes:** `LayerMetrics`, `CAZBoundary`, `CAZRegion`, `CAZProfile`

---

## dataset — Contrastive Pair Loading

JSONL format: `{"positive": "text", "negative": "text"}` per line.

```python
from rosetta_tools.dataset import load_pairs, texts_by_label, validate_dataset
```

| Function | Signature | Purpose |
|---|---|---|
| `load_pairs` | `(path) -> list[ConceptPair]` | Load JSONL file into ConceptPair objects |
| `load_pairs_df` | `(path) -> pd.DataFrame` | Load as flat DataFrame with `pos_text`, `neg_text` columns |
| `texts_by_label` | `(pairs) -> tuple[list[str], list[str]]` | Split into positive and negative text lists |
| `iter_texts` | `(pairs) -> Iterator[tuple[str, str, ConceptPair]]` | Iterate (pos, neg, pair) tuples |
| `validate_dataset` | `(path) -> list[str]` | Check for issues. Empty list = clean |
| `dataset_summary` | `(path) -> dict` | Count pairs, check balance, report stats |

---

## alignment — Procrustes Rotation

Cross-architecture concept vector alignment. Handles different hidden dimensions via shared PCA projection.

```python
from rosetta_tools.alignment import (
    cosine_similarity, compute_procrustes_rotation,
    apply_rotation, align_and_score, pairwise_alignment_df,
)
```

| Function | Signature | Purpose |
|---|---|---|
| `cosine_similarity` | `(v1, v2) -> float` | Cosine similarity between two vectors |
| `compute_procrustes_rotation` | `(source_acts, target_acts, n_components=None) -> NDArray` | Orthogonal R mapping target → source space |
| `apply_rotation` | `(vector, R) -> NDArray` | Apply R to a concept vector |
| `align_and_score` | `(source_vec, target_vec, source_acts, target_acts, n_components=None) -> dict` | End-to-end: fit rotation, apply, report `raw_cosine`, `aligned_cosine`, `alignment_gain` |
| `pairwise_alignment_df` | `(vectors, activations) -> pd.DataFrame` | All-pairs alignment matrix as DataFrame |

**Cross-dimension handling:** When source and target have different hidden dimensions, both are projected to a shared PCA subspace before fitting the rotation. Raw cosine is NaN in this case.

---

## ablation — Directional Ablation

Mid-stream concept removal via forward hooks. No weight modification.

```python
from rosetta_tools.ablation import (
    get_transformer_layers, compute_dominant_direction,
    DirectionalAblator, kl_divergence_from_logits,
)
```

| Function | Signature | Purpose |
|---|---|---|
| `get_transformer_layers` | `(model) -> list` | Auto-detect transformer blocks across architectures |
| `compute_dominant_direction` | `(pos_acts, neg_acts) -> NDArray` | Unit concept direction from contrastive activations |
| `DirectionalAblator` | context manager | Hook a layer to project out a direction during forward pass |
| `compute_baseline_logits` | `(model, tokenizer, prompts, device) -> list[Tensor]` | Last-token logits without ablation |
| `kl_divergence_from_logits` | `(baseline_logits, ablated_logits) -> float` | KL(baseline \|\| ablated) for measuring behavioral impact |

**Supported architectures:** GPT-2 (`.h`), Pythia/GPT-NeoX (`.gpt_neox.layers`), OPT (`.model.decoder.layers`), Llama/Mistral/Gemma/Qwen (`.model.layers` or `.layers`). The layer detector tries 9 attribute paths in priority order.

**Dimension mismatch guard:** If hidden state dimension doesn't match the concept direction (e.g., OPT-350m embedding layer), the hook skips ablation silently rather than crashing.

---

## reporting — Result Loading

Load CAZ checkpoint JSONs into tidy DataFrames for analysis.

```python
from rosetta_tools.reporting import load_results_dir, load_scored_region_df
```

| Function | Signature | Purpose |
|---|---|---|
| `load_result_df` | `(path) -> pd.DataFrame` | Single `caz_*.json` → long-form DataFrame |
| `load_results_dir` | `(path, glob="caz_*.json", include_legacy=True) -> pd.DataFrame` | All checkpoints from one or more directories |
| `load_run_summary` | `(path) -> pd.DataFrame` | `run_summary.json` → wide-form summary |
| `load_region_df` | `(layer_df, min_prominence_frac=0.1) -> pd.DataFrame` | Per-region structural summary from layer data |
| `load_scored_region_df` | `(layer_df, min_prominence_frac=0.005) -> pd.DataFrame` | Scored regions (gentle CAZ detection) |

---

## feature_tracker — Cross-Layer Feature Tracking

Tracks principal component directions across layers via greedy cosine matching. Discovers model-native features (dark matter).

```python
from rosetta_tools.feature_tracker import track_features, Feature, FeatureMap
```

| Function | Signature | Purpose |
|---|---|---|
| `track_features` | `(layer_directions, layer_eigenvalues, n_layers_total, cos_threshold=0.5, concept_directions=None, model_id="") -> FeatureMap` | Track all features from birth to death across layers |

**`Feature` fields:** `feature_id`, `birth_layer`, `death_layer`, `lifespan`, `eigenvalues` (per-layer), `cos_chain` (continuity), `peak_layer`, `peak_eigenvalue`, `concept_alignment` (max cos² per concept), `concept_alignment_trajectory` (per-layer cos² per concept).

**`FeatureMap` methods:** `features_at_layer(layer)`, `persistent_features()`, `unlabeled_features()`

**Concept direction formats:** Flat (`dict[str, NDArray]` — checks at peak layer only) or per-layer (`dict[str, dict[int, NDArray]]` — checks at every layer, stores trajectory).

---

## manifold_detector — Unsupervised Manifold Census

Eigenvalue analysis with Marchenko-Pastur noise floor. Quantifies dark matter.

```python
from rosetta_tools.manifold_detector import layer_manifold_census, LayerManifoldResult
```

| Function | Signature | Purpose |
|---|---|---|
| `layer_manifold_census` | `(layer_activations, concept_directions=None, n_top_eigenvalues=50, store_directions=False) -> ManifoldCensus` | Per-layer eigenvalue census: effective dimensionality, significant dimensions, concept coverage, residual structure |

**`LayerManifoldResult` fields:** `layer_idx`, `effective_dim` (participation ratio), `significant_dims` (above MP threshold), `concept_coverage` (fraction explained by known probes), `top_eigenvalues`, `top_directions` (if `store_directions=True`).

---

## models — Model & Concept Registry

Central registry for all models and concepts. All scripts import from here rather than maintaining their own lists.

```python
from rosetta_tools.models import (
    all_models, families, models_by_family, models_by_tag,
    instruct_pairs, get_model, concept_names, concept_datasets,
)
```

**Model queries:**

| Function | Signature | Purpose |
|---|---|---|
| `all_models` | `(include_disabled=False) -> list[str]` | All enabled model IDs (base models only by default) |
| `models_by_family` | `(family) -> list[str]` | Enabled models for one family (e.g., `"pythia"`) |
| `models_by_tag` | `(tag) -> list[str]` | Models with a given tag, ignores enabled flag |
| `families` | `() -> dict[str, list[str]]` | `{family: [model_ids]}` for enabled models |
| `family_of` | `(model_id) -> str` | Family name for a model (`"unknown"` if not found) |
| `vram_gb` | `(model_id) -> float` | Approximate bf16 VRAM in GB (`0.0` if not found) |
| `get_model` | `(model_id) -> ModelEntry \| None` | Full entry with `enabled`, `gated`, `tags`, `quirks` |
| `instruct_pairs` | `(include_disabled=False) -> list[tuple[str, str]]` | Matched `(base, instruct)` model pairs for RLHF comparison |

**Concept queries:**

| Function | Signature | Purpose |
|---|---|---|
| `concept_names` | `() -> list[str]` | All 7 concepts in assembly order (shallow → deep) |
| `concept_datasets` | `() -> dict[str, str]` | `{concept: dataset_filename}` |
| `concepts_by_category` | `(category) -> list[str]` | Concepts for a category: `"epistemic"`, `"syntactic"`, `"relational"`, `"affective"` |

**Model families:** pythia, gpt2, opt, qwen2, gemma2, llama3, mistral, phi, plus `-instruct` variants.

**Instruct models** are tagged `"instruct"` and disabled by default — they don't appear in `all_models()` or `families()`. Query them with `models_by_tag("instruct")` or `instruct_pairs()`.

---

## tracking — MLflow Integration

Optional experiment tracking. Fails gracefully (no-ops) if MLflow is unavailable.

```python
from rosetta_tools.tracking import ensure_server, start_run, log_concept, end_run
```

| Function | Signature | Purpose |
|---|---|---|
| `configure` | `(tracking_uri=None) -> str \| None` | Set MLflow tracking URI |
| `ensure_server` | `(store=None, port=5111) -> str` | Start MLflow server if not running |
| `start_run` | `(experiment, model_id, params, run_name=None) -> Any \| None` | Begin tracked run. Returns run object or None |
| `log_concept` | `(concept, summary) -> None` | Log per-concept metrics to active run |
| `end_run` | `(run, out_dir=None) -> None` | Log artifacts and close run |

---

## viz — Visualization

Standard CAZ profile plots and heatmaps.

```python
from rosetta_tools.viz import plot_caz_profile, plot_peak_heatmap, CONCEPT_META
```

| Function | Signature | Purpose |
|---|---|---|
| `plot_caz_profile` | `(df, concept, model_id, out_path, title=None)` | Three-panel S/C/v plot for one concept × model |
| `plot_concept_comparison` | `(df, out_path, model_ids=None, title=None)` | Multi-concept overlay across models |
| `plot_peak_heatmap` | `(df, out_path, title=None)` | Peak depth % heatmap: concepts × models |

`CONCEPT_META` and `CONCEPT_ORDER` provide display names, colors, and ordering.

---

## Migration Notes

### `shared.gpu_utils` → `rosetta_tools.gpu_utils` (April 2026)

The original `shared/` directory was removed. All GPU utilities now live in `rosetta_tools.gpu_utils`. Update imports:

```python
# Before
from shared.gpu_utils import get_device, get_dtype, log_vram

# After
from rosetta_tools.gpu_utils import get_device, get_dtype, log_vram
```

All function signatures are unchanged. `release_model` and `purge_hf_cache` moved with the same API.

### Model registry centralization (April 2026)

Model lists previously hardcoded in individual scripts are now in `rosetta_tools.models`. Scripts should import rather than maintain their own:

```python
# Before (in each script)
MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", ...]

# After
from rosetta_tools.models import all_models, families
```

### Feature tracker trajectory storage (April 2026)

`track_features()` now stores per-layer concept alignment trajectories when given per-layer concept directions. The `concept_alignment` field (max cos²) is still populated for backward compatibility. New field:

```python
feature.concept_alignment_trajectory  # dict[str, dict[int, float]]
# concept_name → {layer_index: cos²}
```

Pass per-layer directions to enable:
```python
concept_directions = {"credibility": {0: vec0, 1: vec1, ...}}  # per-layer
# vs
concept_directions = {"credibility": vec_peak}  # flat (legacy, peak-only)
```

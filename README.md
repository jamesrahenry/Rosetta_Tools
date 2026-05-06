# rosetta_tools

Shared tooling for the Rosetta interpretability research program.

---

## Install

Pin to a release tag — unpinned `main` can change between installs:

```bash
pip install git+https://github.com/jamesrahenry/Rosetta_Tools.git@v1.1.0
```

Or for local development (editable):

```bash
git clone git@github.com:jamesrahenry/Rosetta_Tools.git
cd Rosetta_Tools
pip install -e .
```

---

## Modules

Full API reference with signatures and examples: [`docs/api.md`](docs/api.md)

| Module | Purpose |
|---|---|
| `rosetta_tools.gpu_utils` | Device selection, dtype policy, VRAM reporting, model teardown |
| `rosetta_tools.extraction` | Contrastive activation extraction — raw HuggingFace, no TransformerLens |
| `rosetta_tools.caz` | CAZ metric computation (S/C/v), boundary and region detection |
| `rosetta_tools.dataset` | Load and validate JSONL contrastive pair datasets |
| `rosetta_tools.alignment` | Procrustes rotation, cross-architecture concept alignment |
| `rosetta_tools.ablation` | Directional ablation via forward hooks, KL divergence measurement |
| `rosetta_tools.gem` | GEM (Geometric Evolution Map) node construction and diagnostics |
| `rosetta_tools.models` | Central model and concept registry — all scripts import from here |
| `rosetta_tools.probes` | Linear probe training and evaluation for concept directions |
| `rosetta_tools.feature_tracker` | Cross-layer feature tracking via greedy cosine matching |
| `rosetta_tools.manifold_detector` | Eigenvalue census with Marchenko-Pastur noise floor |
| `rosetta_tools.reporting` | Load CAZ checkpoint JSONs into tidy DataFrames |
| `rosetta_tools.tracking` | Optional MLflow experiment tracking (fails gracefully if unavailable) |
| `rosetta_tools.viz` | Standard CAZ profile plots and peak heatmaps |
| `rosetta_tools.dataset` | JSONL pair loading, validation, and summary |
| `rosetta_tools.paths` | Canonical data path resolution (`ROSETTA_CONCEPTS_ROOT`, repo-relative fallback) |

### Quick example

```python
from transformers import AutoModel, AutoTokenizer
from rosetta_tools.gpu_utils import get_device, get_dtype, release_model
from rosetta_tools.extraction import extract_contrastive_activations
from rosetta_tools.caz import compute_layer_metrics, find_caz_regions_scored
from rosetta_tools.dataset import load_pairs, texts_by_label

device    = get_device()
dtype     = get_dtype(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
model     = AutoModel.from_pretrained("EleutherAI/pythia-1.4b", torch_dtype=dtype).to(device)
model.eval()

pairs    = load_pairs("credibility_pairs.jsonl")
pos, neg = texts_by_label(pairs)

layer_acts = extract_contrastive_activations(model, tokenizer, pos, neg, device=device)
metrics    = compute_layer_metrics(layer_acts)
profile    = find_caz_regions_scored(metrics)

for region in profile.regions:
    print(f"CAZ peak layer {region.peak_layer}: S={region.peak_separation:.3f}, score={region.score:.3f}")

release_model(model)
```

---

## Design Notes

- **No TransformerLens dependency.** All activation extraction uses raw HuggingFace
  `transformers`. TransformerLens has persistent compatibility issues with
  transformers 5.x that make it fragile for new model families.

- **fp64 metrics always.** Activations may be extracted in fp16/bf16 for GPU efficiency,
  but all metric computation (Fisher normalization, PCA) uses float64 internally.
  This is critical — fp16 overflows in variance computation at deep layers of
  large models, silently producing wrong results.

- **Library-agnostic CAZ math.** The `caz` module takes numpy arrays and returns
  numpy/NamedTuple results. No torch, no HF, no TransformerLens required.
  The extraction step handles the model-specific part; the metric step is pure math.

---

## Related

- [Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs) — contrastive pair dataset (18 concepts, 38k records)
- [Rosetta_Analysis](https://github.com/jamesrahenry/Rosetta_Analysis) — analysis scripts for CAZ, GEM, and PRH papers

*jamesrahenry@henrynet.ca*

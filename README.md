# rosetta_tools

Shared tooling for the Rosetta interpretability research program.

---

## Install

Pin to a release tag — unpinned `main` can change between installs:

```bash
pip install git+https://github.com/jamesrahenry/Rosetta_Tools.git@v2.1.0
```

`torch` and `transformers` are not declared dependencies — install the versions that suit your
hardware. They are imported only by the model-facing modules (`extraction`, `ablation`,
`gpu_utils`); the CAZ/GEM metric code needs only numpy/scipy/scikit-learn.

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
| `rosetta_tools.gem` | GEM (Geometric Evolution Map) construction, diagnostics, Phase-2 eigen-threads, INLP settled subspace |
| `rosetta_tools.models` | Central model and concept registry — all scripts import from here |
| `rosetta_tools.probes` | Probe extraction from contrastive activations: `extract_probe` (raw/fisher/auroc), `extract_gem_probe` (CAZ handoff + Fisher-weighted direction), `score_direction` |
| `rosetta_tools.reporting` | Load CAZ checkpoint JSONs into tidy DataFrames |
| `rosetta_tools.viz` | Standard CAZ profile plots and peak heatmaps |
| `rosetta_tools.viz_style` | Shared style constants for Rosetta figures |
| `rosetta_tools.compound_monitor` | Vigilance thresholds: tighten an allocation threshold from cross-module evidence |
| `rosetta_tools.paths` | Canonical data path resolution (`ROSETTA_CONCEPTS_ROOT`, repo-relative fallback) |

`feature_tracker`, `manifold_detector` and `tracking` moved to
[Rosetta_Analysis](https://github.com/jamesrahenry/Rosetta_Analysis) in May 2026; this package is
now the reusable CAZ/GEM pipeline only. The shell scripts in `bin/` (GPU job queue, host
provisioning) are retired legacy tooling kept for reproducibility — see `bin/README.md`.

### Which version to pin

| Consumer | Pin | Why |
|---|---|---|
| New deployments and analysis (Monitor, CIA, Rosetta_Analysis) | **`v2.1.0`** | Current supported surface. Pair-aware train/eval split, reproducible pair sampling, blank-pair handling, CAZ record assembler, INLP settled subspace. |
| Reproducing P1 (CAZ framework), P2 (GEM), P3 (CAZ validation), P4 (PRH) | `v1.3.1` | The version the papers cite (Zenodo `10.5281/zenodo.20361433`). Frozen; do not move. |
| Reproducing the round-3 Gemma-2 rebuild | `v1.4.0` | As recorded in `papers/shared/round3_gpu/GEMMA2_REBUILD_NOTES.md`. |

`v1.5.0` and `v1.6.0` appear as in-tree version strings in some notes and commit messages but were
never tagged; the code they refer to shipped in `v2.0.0`.

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
    print(f"CAZ peak layer {region.peak}: S={region.peak_separation:.3f}, score={region.caz_score:.3f}")

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

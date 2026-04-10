# rosetta_tools

Shared tooling for the [Rosetta interpretability research program](https://github.com/jamesrahenry/Rosetta_Manifold).

---

## Install

```bash
pip install git+https://github.com/jamesrahenry/Rosetta_Tools.git
```

Or for local development (editable):

```bash
git clone git@github.com:jamesrahenry/Rosetta_Tools.git
cd Rosetta_Tools
pip install -e .
```

---

## Modules

### `rosetta_tools.gpu_utils`

Device selection, dtype resolution, VRAM reporting, and model teardown.
Designed to be library-agnostic — works with HuggingFace and any other framework.

```python
from rosetta_tools.gpu_utils import get_device, get_dtype, log_vram, release_model

device = get_device()        # "cuda" or "cpu"
dtype  = get_dtype(device)   # torch.float16 or torch.float32

model = SomeModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
log_vram("after model load")

# Between models — important on 4GB GPUs
release_model(model)
```

### `rosetta_tools.caz`

Concept Allocation Zone metric computation — no model library dependency.
Feed in activation arrays, get S/C/v metrics and CAZ boundaries back.

```python
from rosetta_tools.caz import compute_layer_metrics, find_caz_boundary
from rosetta_tools.extraction import extract_contrastive_activations

layer_acts = extract_contrastive_activations(model, tokenizer, pos_texts, neg_texts)
metrics    = compute_layer_metrics(layer_acts)
boundary   = find_caz_boundary(metrics)

print(f"CAZ peak: layer {boundary.caz_peak}, S={boundary.peak_separation:.3f}")
```

### `rosetta_tools.extraction`

Model-agnostic activation extraction using raw HuggingFace `transformers`.
No TransformerLens dependency. Works with any model supporting
`output_hidden_states=True`.

```python
from transformers import AutoModel, AutoTokenizer
from rosetta_tools.extraction import extract_contrastive_activations
from rosetta_tools.gpu_utils import get_device, get_dtype

device    = get_device()
dtype     = get_dtype(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model     = AutoModel.from_pretrained("gpt2-xl", torch_dtype=dtype).to(device)
model.eval()

layer_acts = extract_contrastive_activations(
    model, tokenizer,
    pos_texts=["The evidence clearly shows...", ...],
    neg_texts=["The evidence is mixed...", ...],
    device=device,
)
# layer_acts: list of (pos_acts, neg_acts) tuples, one per layer
```

### `rosetta_tools.dataset`

Load and validate JSONL contrastive pair datasets.

```python
from rosetta_tools.dataset import load_pairs, texts_by_label, validate_dataset

# Validate before a long run
issues = validate_dataset("data/credibility_pairs.jsonl")
assert not issues, issues

pairs     = load_pairs("data/credibility_pairs.jsonl")
pos, neg  = texts_by_label(pairs)
```

---

## Design Notes

- **No TransformerLens dependency.** All activation extraction uses raw HuggingFace
  `transformers`. TransformerLens has persistent compatibility issues with
  transformers 5.x that make it fragile for new model families.

- **fp32 metrics always.** Activations may be extracted in fp16 for GPU efficiency,
  but all metric computation (Fisher normalization, PCA) uses float64 internally.
  This is critical — fp16 overflows in variance computation at deep layers of
  large models, silently producing wrong results.

- **Library-agnostic CAZ math.** The `caz` module takes numpy arrays and returns
  numpy/NamedTuple results. No torch, no HF, no TransformerLens required.
  The extraction step handles the model-specific part; the metric step is pure math.

---

## Related

- [Rosetta Manifold](https://github.com/jamesrahenry/Rosetta_Manifold) — empirical CAZ pipeline
- [Concept Allocation Zone](https://github.com/jamesrahenry/Concept_Allocation_Zone) — theoretical framework
- [Pop Goes the Easel](https://github.com/jamesrahenry/pop_goes_the_easel) — companion study

*jamesrahenry@henrynet.ca*

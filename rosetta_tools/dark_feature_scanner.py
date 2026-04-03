"""Scan texts to find max-activating examples for dark (unlabeled) features.

For each dark feature, projects activations at its peak layer onto the feature
direction and ranks texts by activation magnitude.  The top-activating texts
often reveal what the feature is encoding.

Usage:
    python -m rosetta_tools.dark_feature_scanner \
        --model EleutherAI/pythia-6.9b \
        --library feature_library/ \
        --texts datasets/consensus_pairs/ \
        --top-k 10 \
        --output dark_feature_labels.json
"""

from __future__ import annotations

import json
import logging
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class FeatureActivation:
    """One feature's activation on one text."""
    feature_id: int
    text_idx: int
    activation: float
    text_snippet: str
    source_file: str
    label: int | None  # original contrastive label if available


@dataclass
class FeatureProfile:
    """Activation profile for one dark feature across all texts."""
    feature_id: int
    peak_layer: int
    peak_pc: int
    lifespan: int
    uf_id: str | None
    handoff_label: str
    top_positive: list[FeatureActivation]  # highest activation
    top_negative: list[FeatureActivation]  # most negative activation
    mean_activation: float
    std_activation: float
    suggested_label: str | None  # auto-suggested concept


def load_texts(text_dir: Path, max_per_file: int = 500) -> list[dict]:
    """Load texts from JSONL files. Returns [{text, source, label}, ...]."""
    texts = []
    for jsonl in sorted(text_dir.glob("*.jsonl")):
        count = 0
        for line in open(jsonl):
            try:
                data = json.loads(line)
                texts.append({
                    "text": data["text"],
                    "source": jsonl.stem,
                    "label": data.get("label"),
                    "pair_id": data.get("pair_id", ""),
                })
                count += 1
                if count >= max_per_file:
                    break
            except (json.JSONDecodeError, KeyError):
                continue
    log.info("Loaded %d texts from %s", len(texts), text_dir)
    return texts


def _pool_last_token(hidden_state, attention_mask):
    """Extract last non-padding token's hidden state."""
    lengths = attention_mask.sum(dim=1) - 1  # [batch]
    batch_idx = torch.arange(hidden_state.size(0), device=hidden_state.device)
    return hidden_state[batch_idx, lengths]


def extract_activations_at_layer(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
) -> NDArray:
    """Extract activations at a single layer for efficiency.

    Returns array of shape [n_texts, hidden_dim].
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_acts = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        encoding = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hs = outputs.hidden_states[layer]  # [batch, seq, hidden]
        pooled = _pool_last_token(hs, attention_mask)
        all_acts.append(pooled.cpu().float().numpy())

    return np.concatenate(all_acts, axis=0)


def scan_dark_features(
    model_id: str,
    library_dir: Path,
    text_dir: Path,
    top_k: int = 10,
    device: str = "auto",
    batch_size: int = 16,
    only_dark: bool = True,
    min_lifespan: int = 5,
) -> list[FeatureProfile]:
    """Scan texts to profile dark features.

    Parameters
    ----------
    model_id : HuggingFace model ID
    library_dir : path to feature_library/
    text_dir : path to directory with .jsonl text files
    top_k : number of top-activating texts per feature
    device : 'cuda', 'cpu', or 'auto'
    only_dark : if True, only scan features with no concept alignment
    min_lifespan : minimum feature lifespan to scan
    """
    from transformers import AutoModel, AutoTokenizer
    from .gpu_utils import get_device, get_dtype, release_model, log_vram

    # Load model features from library
    slug = model_id.split("/")[-1]
    features_path = library_dir / "models" / slug / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"No library entry for {model_id} at {features_path}")

    model_data = json.loads(features_path.read_text())
    features = model_data["features"]
    n_layers = model_data["n_layers"]

    # Filter to dark persistent features
    dark_features = []
    for f in features:
        if f["lifespan"] < min_lifespan:
            continue
        if only_dark and f.get("handoff_label", "") != "unlabeled":
            continue
        dark_features.append(f)

    if not dark_features:
        log.info("No dark features to scan for %s", model_id)
        return []

    log.info("Scanning %d dark features for %s", len(dark_features), model_id)

    # Load directions
    directions_dir = library_dir / "models" / slug / "directions"

    # Group features by peak layer to minimize forward passes
    layer_features: dict[int, list[dict]] = {}
    for f in dark_features:
        # Find peak layer (max eigenvalue)
        eigs = f["eigenvalues"]
        layers = f["layer_indices"]
        if not eigs or not layers:
            continue
        peak_idx = int(np.argmax(eigs))
        peak_layer = layers[peak_idx]

        # Load direction at peak layer
        npy_path = directions_dir / f"directions_L{peak_layer:03d}.npy"
        if not npy_path.exists():
            continue

        # Get PC index at peak layer
        pc_indices = f.get("pc_indices", [])
        if peak_idx >= len(pc_indices):
            continue
        peak_pc = pc_indices[peak_idx]

        f["_peak_layer"] = peak_layer
        f["_peak_pc"] = peak_pc
        layer_features.setdefault(peak_layer, []).append(f)

    unique_layers = sorted(layer_features.keys())
    log.info("Need activations at %d unique layers", len(unique_layers))

    # Load texts
    texts_data = load_texts(text_dir)
    text_strings = [t["text"] for t in texts_data]

    # Load model
    if device == "auto":
        device = get_device("auto")
    dtype = get_dtype(device)

    log.info("Loading %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    except (ValueError, ImportError):
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    log_vram("after model load")

    # Extract and project
    profiles = []
    for layer_idx in unique_layers:
        layer_feats = layer_features[layer_idx]
        log.info("  Layer %d: extracting activations for %d features...", layer_idx, len(layer_feats))

        # Extract activations at this layer
        acts = extract_activations_at_layer(
            model, tokenizer, text_strings, layer_idx,
            device=device, batch_size=batch_size,
        )  # [n_texts, hidden_dim]

        # Load directions for this layer
        npy_path = directions_dir / f"directions_L{layer_idx:03d}.npy"
        all_dirs = np.load(npy_path)  # [n_pcs, hidden_dim]

        for f in layer_feats:
            pc_idx = f["_peak_pc"]
            if pc_idx >= len(all_dirs):
                continue

            direction = all_dirs[pc_idx].astype(np.float64)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm

            # Project activations onto feature direction
            projections = acts.astype(np.float64) @ direction  # [n_texts]

            mean_act = float(projections.mean())
            std_act = float(projections.std())

            # Top positive activations
            top_pos_idx = np.argsort(projections)[-top_k:][::-1]
            top_pos = []
            for idx in top_pos_idx:
                top_pos.append(FeatureActivation(
                    feature_id=f["feature_id"],
                    text_idx=int(idx),
                    activation=float(projections[idx]),
                    text_snippet=text_strings[idx][:200],
                    source_file=texts_data[idx]["source"],
                    label=texts_data[idx].get("label"),
                ))

            # Top negative activations
            top_neg_idx = np.argsort(projections)[:top_k]
            top_neg = []
            for idx in top_neg_idx:
                top_neg.append(FeatureActivation(
                    feature_id=f["feature_id"],
                    text_idx=int(idx),
                    activation=float(projections[idx]),
                    text_snippet=text_strings[idx][:200],
                    source_file=texts_data[idx]["source"],
                    label=texts_data[idx].get("label"),
                ))

            # Auto-suggest label: which source file dominates the top activations?
            source_counts = {}
            for a in top_pos[:5]:
                src = a.source_file.replace("_consensus_pairs", "")
                source_counts[src] = source_counts.get(src, 0) + 1
            suggested = None
            if source_counts:
                top_source, top_count = max(source_counts.items(), key=lambda x: x[1])
                if top_count >= 3:  # majority of top-5
                    suggested = top_source

            profiles.append(FeatureProfile(
                feature_id=f["feature_id"],
                peak_layer=layer_idx,
                peak_pc=f["_peak_pc"],
                lifespan=f["lifespan"],
                uf_id=f.get("uf_id"),
                handoff_label=f.get("handoff_label", "unlabeled"),
                top_positive=top_pos,
                top_negative=top_neg,
                mean_activation=mean_act,
                std_activation=std_act,
                suggested_label=suggested,
            ))

    release_model(model)
    log.info("Scanned %d features", len(profiles))
    return profiles


def save_profiles(profiles: list[FeatureProfile], output_path: Path) -> None:
    """Save scan results to JSON."""
    results = []
    for p in profiles:
        results.append({
            "feature_id": p.feature_id,
            "peak_layer": p.peak_layer,
            "peak_pc": p.peak_pc,
            "lifespan": p.lifespan,
            "uf_id": p.uf_id,
            "handoff_label": p.handoff_label,
            "mean_activation": round(p.mean_activation, 4),
            "std_activation": round(p.std_activation, 4),
            "suggested_label": p.suggested_label,
            "top_positive": [
                {
                    "text_snippet": a.text_snippet,
                    "activation": round(a.activation, 4),
                    "source": a.source_file,
                    "label": a.label,
                }
                for a in p.top_positive
            ],
            "top_negative": [
                {
                    "text_snippet": a.text_snippet,
                    "activation": round(a.activation, 4),
                    "source": a.source_file,
                    "label": a.label,
                }
                for a in p.top_negative
            ],
        })

    output_path.write_text(json.dumps({"profiles": results}, indent=2))
    log.info("Saved %d profiles to %s", len(results), output_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Scan texts for dark feature activations")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--library", type=Path, required=True, help="Path to feature_library/")
    parser.add_argument("--texts", type=Path, required=True, help="Path to directory with .jsonl files")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K texts per feature")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--all-features", action="store_true", help="Scan all features, not just dark ones")
    args = parser.parse_args()

    output = args.output or Path(f"dark_scan_{args.model.split('/')[-1]}.json")

    profiles = scan_dark_features(
        model_id=args.model,
        library_dir=args.library,
        text_dir=args.texts,
        top_k=args.top_k,
        device=args.device,
        batch_size=args.batch_size,
        only_dark=not args.all_features,
    )

    save_profiles(profiles, output)

    # Summary
    labeled = [p for p in profiles if p.suggested_label]
    print(f"\n{len(profiles)} dark features scanned")
    print(f"{len(labeled)} auto-labeled by max-activation source dominance:")
    for p in labeled:
        print(f"  F{p.feature_id:03d} (L{p.peak_layer}, {p.lifespan}L): suggested={p.suggested_label}")

    unlabeled = [p for p in profiles if not p.suggested_label]
    if unlabeled:
        print(f"\n{len(unlabeled)} remain unlabeled — candidates for gradient-ascent optimization:")
        for p in unlabeled[:10]:
            sources = set(a.source_file.replace("_consensus_pairs", "") for a in p.top_positive[:5])
            print(f"  F{p.feature_id:03d} (L{p.peak_layer}, {p.lifespan}L): top sources={sources}")

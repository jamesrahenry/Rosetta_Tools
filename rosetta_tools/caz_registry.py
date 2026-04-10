"""CAZ Registry — tracks Concept Assembly Zones in the feature library.

Stores detected CAZ regions per model × concept, cross-linked to Universal
Features (UF###) from the feature atlas. Answers:
  - Which UF features overlap with credibility's CAZ in pythia-6.9b?
  - Which models share a CAZ for certainty at ~60% depth?
  - Which CAZes does UF001 span?

Storage layout (inside feature_library/):
  cazs/
    _caz_index.json              # all CAZes: {model, concept, peak_depth, score, uf_ids}
    credibility/
      pythia-6.9b.json           # CAZ regions for this model×concept
    certainty/
      gpt2-xl.json
      ...
  models/pythia-6.9b/
    cazs.json                    # all CAZes for this model, with feature crosslinks
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _jsonify(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence",
            "causation", "temporal_order", "negation"]


def classify_caz(
    peak_layer: int,
    n_layers: int,
    caz_score: float,
    functional_caz_score: float | None = None,
    attention_paradigm: str = "unknown",
) -> str:
    """Classify a CAZ region by depth and functional quality.

    Now ablation-calibrated (tc17bb65).  Uses functional_caz_score when
    available to distinguish genuine allocation zones from geometric
    artifacts ("black holes").

    Returns
    -------
    str — one of:
        'embedding'  — peak in first ~8% of depth.  Passive tokenizer-driven
                       separation; the model hasn't computed anything yet.
        'active'     — peak between 8% and 80% depth AND functional score
                       indicates real ablation impact.  The main zone of
                       interest for downstream use.
        'deep'       — peak above 80% depth.  Late-layer integration /
                       unembedding pressure zone.
        'functional' — architecture-determined functional peak (e.g. final
                       global attention layer in alternating architectures).
                       Verified by patching recovery regardless of Fisher
                       separation strength.
        'artifact'   — geometric separation exists but functional score is
                       near-zero.  Common in GQA/alternating architectures
                       where Fisher peaks don't predict ablation impact.
    """
    depth_pct = peak_layer / max(n_layers - 1, 1) * 100

    # Depth-based classification first
    if depth_pct < 8.0:
        return "embedding"

    # If we have functional scoring, use it for quality assessment
    if functional_caz_score is not None and attention_paradigm != "unknown":
        # Functional peak (e.g. Gemma final global attention layer)
        # gets its own classification — these are verified by patching
        # regardless of their Fisher separation profile.
        if attention_paradigm == "alternating":
            from rosetta_tools.caz import final_global_attention_layer
            func_layer = final_global_attention_layer(n_layers)
            # Check if this peak IS the functional peak (within 1 layer)
            if abs(peak_layer - func_layer) <= 1:
                return "functional"

        # Artifact detection: high geometric score but near-zero functional score
        if caz_score > 0.15 and functional_caz_score < 0.05:
            return "artifact"

    # Standard depth classification
    if depth_pct > 80.0:
        return "deep"
    return "active"


@dataclass
class CAZRegion:
    """One detected CAZ for one model × concept."""
    model_id: str
    concept: str
    # Layer positions
    start_layer: int
    peak_layer: int
    end_layer: int
    n_layers_total: int
    # Depth percentages
    peak_depth_pct: float
    start_depth_pct: float
    end_depth_pct: float
    width_depth_pct: float
    # Scores
    caz_score: float
    functional_caz_score: float   # ablation-calibrated (see caz.ABLATION_PRIORS)
    peak_separation: float
    # Classification (ablation-calibrated — see classify_caz())
    caz_type: str  # "embedding" | "active" | "deep" | "functional" | "artifact"
    # Cross-links
    overlapping_ufs: list[str]   # UF### ids whose feature trajectories overlap this CAZ
    overlapping_feature_ids: list[int]  # model-specific feature IDs


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _find_extraction_dir(results_root: Path, model_id: str) -> Path | None:
    """Find the latest extraction result directory for a model."""
    slug = _short(model_id)
    # Try all naming conventions
    candidates = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        # Check if it contains caz_*.json files (extraction results)
        if any(d.glob("caz_*.json")):
            sf = d / "run_summary.json"
            if sf.exists():
                try:
                    s = json.loads(sf.read_text())
                    if s.get("model_id") == model_id:
                        candidates.append(d)
                except (json.JSONDecodeError, KeyError):
                    pass
    return sorted(candidates)[-1] if candidates else None


def build_caz_registry(
    library_dir: Path,
    results_root: Path,
) -> dict[str, list[CAZRegion]]:
    """Build CAZ registry from extraction results, crosslinked to feature atlas.

    Parameters
    ----------
    library_dir : path to feature_library/
    results_root : path to results/ directory with extraction results

    Returns
    -------
    dict mapping model_id -> list of CAZRegion
    """
    from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
    from rosetta_tools.models import attention_paradigm_of

    # Load atlas for UF assignments
    atlas_path = library_dir / "atlas.json"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found at {atlas_path}")

    # Build feature trajectory lookup: {model_id: [{feature_id, layer_indices, uf_id, concept_alignment_trajectory}]}
    feature_lookup: dict[str, list[dict]] = {}
    models_dir = library_dir / "models"
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        feat_path = model_dir / "features.json"
        if not feat_path.exists():
            continue
        feat_data = json.loads(feat_path.read_text())
        model_id = feat_data["model_id"]
        feature_lookup[model_id] = feat_data.get("features", [])

    all_cazs: dict[str, list[CAZRegion]] = {}

    for model_id, features in feature_lookup.items():
        slug = _short(model_id)
        ext_dir = _find_extraction_dir(results_root, model_id)
        if ext_dir is None:
            log.warning("No extraction results for %s", slug)
            continue

        log.info("Processing %s...", slug)
        model_cazs = []
        paradigm = attention_paradigm_of(model_id)

        for concept in CONCEPTS:
            caz_file = ext_dir / f"caz_{concept}.json"
            if not caz_file.exists():
                continue

            data = json.loads(caz_file.read_text())
            n_layers = data["n_layers"]
            metrics = data["layer_data"]["metrics"]

            lm = [LayerMetrics(
                m["layer"],
                m.get("separation_fisher", 0),
                m.get("coherence", 0),
                m.get("velocity", 0),
            ) for m in metrics]

            profile = find_caz_regions_scored(lm, attention_paradigm=paradigm)

            for region in profile.regions:
                peak_depth = region.peak / max(n_layers - 1, 1) * 100
                start_depth = region.start / max(n_layers - 1, 1) * 100
                end_depth = region.end / max(n_layers - 1, 1) * 100
                width = end_depth - start_depth

                # Find overlapping features: features whose layer_indices overlap
                # the CAZ region AND have concept alignment at those layers
                overlapping_ufs = []
                overlapping_fids = []

                for feat in features:
                    if feat.get("lifespan", 0) < 3:
                        continue
                    feat_layers = set(feat.get("layer_indices", []))
                    caz_layers = set(range(region.start, region.end + 1))

                    # Check overlap
                    if not feat_layers.intersection(caz_layers):
                        continue

                    # Check concept alignment in overlap region (from feature map if available)
                    # Use concept_alignment from the feature map
                    concept_align = feat.get("concept_alignment", {})
                    if concept_align.get(concept, 0) > 0.25:  # loose threshold
                        uf = feat.get("uf_id")
                        fid = feat.get("feature_id")
                        if uf and uf not in overlapping_ufs:
                            overlapping_ufs.append(uf)
                        if fid is not None and fid not in overlapping_fids:
                            overlapping_fids.append(fid)

                caz = CAZRegion(
                    model_id=model_id,
                    concept=concept,
                    start_layer=region.start,
                    peak_layer=region.peak,
                    end_layer=region.end,
                    n_layers_total=n_layers,
                    peak_depth_pct=round(peak_depth, 1),
                    start_depth_pct=round(start_depth, 1),
                    end_depth_pct=round(end_depth, 1),
                    width_depth_pct=round(width, 1),
                    caz_score=round(region.caz_score, 4),
                    functional_caz_score=round(region.functional_caz_score, 4),
                    peak_separation=round(region.peak_separation, 4),
                    caz_type=classify_caz(
                        region.peak, n_layers, region.caz_score,
                        functional_caz_score=region.functional_caz_score,
                        attention_paradigm=paradigm,
                    ),
                    overlapping_ufs=sorted(overlapping_ufs),
                    overlapping_feature_ids=sorted(overlapping_fids),
                )
                model_cazs.append(caz)

        all_cazs[model_id] = model_cazs
        log.info("  %d CAZes across %d concepts", len(model_cazs), len(CONCEPTS))

    return all_cazs


def save_caz_registry(
    all_cazs: dict[str, list[CAZRegion]],
    library_dir: Path,
) -> None:
    """Save CAZ registry to feature_library/cazs/."""

    cazs_root = library_dir / "cazs"
    cazs_root.mkdir(exist_ok=True)

    # Per-concept directories
    for concept in CONCEPTS:
        (cazs_root / concept).mkdir(exist_ok=True)

    # Write per-model×concept files and build index
    index = []

    for model_id, cazs in all_cazs.items():
        slug = _short(model_id)

        # Per-concept files
        for caz in cazs:
            concept_dir = cazs_root / caz.concept
            out_path = concept_dir / f"{slug}.json"

            # Load existing if present (multiple regions per model×concept)
            existing = []
            if out_path.exists():
                existing = json.loads(out_path.read_text())

            # Append or replace regions for this model
            region_data = {
                "model_id": model_id,
                "concept": caz.concept,
                "start_layer": caz.start_layer,
                "peak_layer": caz.peak_layer,
                "end_layer": caz.end_layer,
                "n_layers_total": caz.n_layers_total,
                "peak_depth_pct": caz.peak_depth_pct,
                "start_depth_pct": caz.start_depth_pct,
                "end_depth_pct": caz.end_depth_pct,
                "width_depth_pct": caz.width_depth_pct,
                "caz_score": caz.caz_score,
                "functional_caz_score": caz.functional_caz_score,
                "peak_separation": caz.peak_separation,
                "caz_type": caz.caz_type,
                "overlapping_ufs": caz.overlapping_ufs,
                "overlapping_feature_ids": caz.overlapping_feature_ids,
            }

            index.append({
                "model_id": model_id,
                "concept": caz.concept,
                "peak_depth_pct": caz.peak_depth_pct,
                "caz_score": caz.caz_score,
                "functional_caz_score": caz.functional_caz_score,
                "peak_separation": caz.peak_separation,
                "caz_type": caz.caz_type,
                "n_overlapping_ufs": len(caz.overlapping_ufs),
                "overlapping_ufs": caz.overlapping_ufs,
            })

        # Per-concept files: write all regions for this model grouped by concept
        concept_cazs: dict[str, list] = {}
        for caz in cazs:
            concept_cazs.setdefault(caz.concept, []).append({
                "start_layer": caz.start_layer,
                "peak_layer": caz.peak_layer,
                "end_layer": caz.end_layer,
                "n_layers_total": caz.n_layers_total,
                "peak_depth_pct": caz.peak_depth_pct,
                "start_depth_pct": caz.start_depth_pct,
                "end_depth_pct": caz.end_depth_pct,
                "width_depth_pct": caz.width_depth_pct,
                "caz_score": caz.caz_score,
                "functional_caz_score": caz.functional_caz_score,
                "peak_separation": caz.peak_separation,
                "caz_type": caz.caz_type,
                "overlapping_ufs": caz.overlapping_ufs,
                "overlapping_feature_ids": caz.overlapping_feature_ids,
            })

        for concept, regions in concept_cazs.items():
            out_path = cazs_root / concept / f"{slug}.json"
            out_path.write_text(json.dumps({
                "model_id": model_id,
                "concept": concept,
                "n_regions": len(regions),
                "regions": regions,
            }, indent=2, default=_jsonify))

        # Per-model cazs.json
        model_dir = library_dir / "models" / slug
        if model_dir.exists():
            model_caz_data = []
            for caz in cazs:
                model_caz_data.append({
                    "concept": caz.concept,
                    "start_layer": caz.start_layer,
                    "peak_layer": caz.peak_layer,
                    "end_layer": caz.end_layer,
                    "peak_depth_pct": caz.peak_depth_pct,
                    "caz_score": caz.caz_score,
                    "functional_caz_score": caz.functional_caz_score,
                    "peak_separation": caz.peak_separation,
                    "caz_type": caz.caz_type,
                    "overlapping_ufs": caz.overlapping_ufs,
                    "overlapping_feature_ids": caz.overlapping_feature_ids,
                })
            (model_dir / "cazs.json").write_text(json.dumps(model_caz_data, indent=2, default=_jsonify))

    # Write index
    (cazs_root / "_caz_index.json").write_text(json.dumps(index, indent=2, default=_jsonify))

    n_total = sum(len(v) for v in all_cazs.values())
    n_linked = sum(1 for entry in index if entry["n_overlapping_ufs"] > 0)
    log.info("CAZ registry saved: %d CAZes, %d linked to UFs", n_total, n_linked)


def update_atlas_with_cazs(
    all_cazs: dict[str, list[CAZRegion]],
    library_dir: Path,
) -> None:
    """Add overlapping_cazs field to each UF's profile.json."""

    # Build reverse map: UF -> list of (model, concept, score, depth)
    uf_to_cazs: dict[str, list[dict]] = {}
    for model_id, cazs in all_cazs.items():
        for caz in cazs:
            for uf_id in caz.overlapping_ufs:
                uf_to_cazs.setdefault(uf_id, []).append({
                    "model_id": model_id,
                    "concept": caz.concept,
                    "peak_depth_pct": caz.peak_depth_pct,
                    "caz_score": caz.caz_score,
                    "functional_caz_score": caz.functional_caz_score,
                    "peak_separation": caz.peak_separation,
                    "caz_type": caz.caz_type,
                })

    # Update each UF's profile
    uni_dir = library_dir / "universal"
    updated = 0
    for uf_dir in sorted(uni_dir.iterdir()):
        if not uf_dir.is_dir():
            continue
        profile_path = uf_dir / "profile.json"
        if not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text())
        uf_id = profile["uf_id"]
        profile["overlapping_cazs"] = uf_to_cazs.get(uf_id, [])
        profile_path.write_text(json.dumps(profile, indent=2, default=_jsonify))
        if uf_to_cazs.get(uf_id):
            updated += 1

    log.info("Updated %d UF profiles with CAZ crosslinks", updated)


def load_cazs_for_concept(library_dir: Path, concept: str) -> list[dict]:
    """Load all CAZ regions for a concept across all models."""
    cazs = []
    concept_dir = library_dir / "cazs" / concept
    if not concept_dir.exists():
        return cazs
    for f in sorted(concept_dir.glob("*.json")):
        data = json.loads(f.read_text())
        for region in data.get("regions", []):
            region["model_id"] = data["model_id"]
            region["concept"] = concept
            cazs.append(region)
    return cazs


def load_cazs_for_model(library_dir: Path, model_id: str) -> list[dict]:
    """Load all CAZ regions for one model."""
    slug = model_id.split("/")[-1]
    caz_path = library_dir / "models" / slug / "cazs.json"
    if not caz_path.exists():
        return []
    data = json.loads(caz_path.read_text())
    for entry in data:
        entry["model_id"] = model_id
    return data


if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build CAZ registry in feature library")
    parser.add_argument("--library", type=Path, required=True,
                        help="Path to feature_library/")
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to results/ directory with extraction results")
    args = parser.parse_args()

    log.info("Building CAZ registry...")
    all_cazs = build_caz_registry(args.library, args.results)

    n_total = sum(len(v) for v in all_cazs.values())
    log.info("Detected %d CAZ regions across %d models", n_total, len(all_cazs))

    log.info("Saving registry...")
    save_caz_registry(all_cazs, args.library)

    log.info("Updating UF profiles with CAZ crosslinks...")
    update_atlas_with_cazs(all_cazs, args.library)

    # Summary
    from collections import Counter
    concept_counts = Counter()
    linked = 0
    for cazs in all_cazs.values():
        for caz in cazs:
            concept_counts[caz.concept] += 1
            if caz.overlapping_ufs:
                linked += 1

    print(f"\n{n_total} CAZ regions across {len(all_cazs)} models")
    print(f"{linked} linked to at least one UF")
    print("\nPer concept:")
    for concept, count in sorted(concept_counts.items(), key=lambda x: -x[1]):
        print(f"  {concept:18s}  {count}")

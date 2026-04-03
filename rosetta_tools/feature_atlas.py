"""Universal Feature Library — maps model-specific features to universal features (UFs).

Two layers:
  - Universal: UF### entries with canonical fingerprint, profile, provenance
  - Model: model-specific instances with native-space directions, eigenvalues, etc.

Storage layout:
  feature_library/
    atlas.json
    universal/UF001/{canonical.npz, profile.json, provenance.json}
    models/pythia-6.9b/{rotation.npz, features.json, directions/F003.npz}
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .alignment import compute_procrustes_rotation, apply_rotation, cosine_similarity

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

N_DEPTH_BINS = 20  # 5% increments


@dataclass
class ModelFeatureEntry:
    """One tracked feature from one model."""

    model_id: str
    feature_id: int
    birth_layer: int
    death_layer: int
    lifespan: int
    n_layers_total: int
    layer_indices: list[int]
    eigenvalues: list[float]
    cos_chain: list[float]
    concept_alignment: dict[str, float]
    # Per-layer concept label from feature_labels.json (None = unlabeled at that layer)
    layer_labels: list[tuple[int, str | None, float]] | None = None  # [(layer, concept, cos)]
    # Depth-normalized profiles
    handoff_sequence: list[str | None] = field(default_factory=list)
    depth_eigenvalue_profile: list[float] = field(default_factory=list)
    # Ablation data (None if not ablated)
    ablation_impact: dict[str, float] | None = None  # concept -> retained_pct
    ablation_verdict: str | None = None
    # UF assignment
    uf_id: str | None = None

    @property
    def peak_eigenvalue(self) -> float:
        return max(self.eigenvalues) if self.eigenvalues else 0.0

    @property
    def peak_depth_pct(self) -> float:
        if not self.eigenvalues or not self.layer_indices:
            return 0.0
        peak_idx = int(np.argmax(self.eigenvalues))
        return self.layer_indices[peak_idx] / max(self.n_layers_total - 1, 1)

    @property
    def is_persistent(self) -> bool:
        return self.lifespan >= 5

    @property
    def handoff_label(self) -> str:
        """Human-readable handoff: 'causation -> certainty -> credibility'."""
        seen = []
        for concept in self.handoff_sequence:
            if concept is not None and (not seen or seen[-1] != concept):
                seen.append(concept)
        return " -> ".join(seen) if seen else "unlabeled"


@dataclass
class ModelRecord:
    """All features from one model plus alignment metadata."""

    model_id: str
    n_layers: int
    hidden_dim: int
    features: list[ModelFeatureEntry]
    deepdive_dir: str
    rotation: NDArray | None = None
    rotation_dim: int | None = None  # dimensionality of aligned space

    @property
    def slug(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def persistent_features(self) -> list[ModelFeatureEntry]:
        return [f for f in self.features if f.is_persistent]


@dataclass
class UniversalFeature:
    """One entry in the universal atlas."""

    uf_id: str
    handoff_template: list[str | None]
    description: str
    canonical_directions: dict[str, NDArray]  # depth_phase -> unit direction
    variance_profile: list[float]
    ablation_signature: dict[str, float]  # concept -> mean retained_pct
    provenance: list[dict]  # [{model_id, feature_id, handoff_label, alignment_cos}]
    n_models: int
    n_families: int


@dataclass
class FeatureAtlas:
    """Top-level container for the Universal Feature Library."""

    universal_features: list[UniversalFeature]
    model_records: dict[str, ModelRecord]
    reference_model: str
    aligned_dim: int
    n_depth_bins: int = N_DEPTH_BINS


# ---------------------------------------------------------------------------
# Depth normalization helpers
# ---------------------------------------------------------------------------


def _normalize_handoff(layer_labels: list[tuple[int, str | None, float]],
                       n_layers: int,
                       n_bins: int = N_DEPTH_BINS) -> list[str | None]:
    """Convert per-layer concept labels to depth-normalized bins."""
    bins: list[list[str | None]] = [[] for _ in range(n_bins)]
    for layer, concept, _cos in layer_labels:
        depth_pct = layer / max(n_layers - 1, 1)
        bin_idx = min(int(depth_pct * n_bins), n_bins - 1)
        bins[bin_idx].append(concept)

    result = []
    for bin_entries in bins:
        non_none = [c for c in bin_entries if c is not None]
        if non_none:
            result.append(Counter(non_none).most_common(1)[0][0])
        elif bin_entries:  # has entries but all None
            result.append(None)
        else:  # feature not alive in this bin
            result.append(None)
    return result


def _normalize_eigenvalue_profile(layer_indices: list[int],
                                  eigenvalues: list[float],
                                  n_layers: int,
                                  n_bins: int = N_DEPTH_BINS) -> list[float]:
    """Interpolate eigenvalue trajectory to fixed depth bins."""
    if not layer_indices or not eigenvalues:
        return [0.0] * n_bins

    depth_pcts = [l / max(n_layers - 1, 1) for l in layer_indices]
    bin_centers = np.linspace(0, 1, n_bins)
    interp = np.interp(bin_centers, depth_pcts, eigenvalues, left=0, right=0)
    peak = interp.max()
    if peak > 0:
        interp = interp / peak  # normalize to peak=1
    return interp.tolist()


def handoff_similarity(a: list[str | None], b: list[str | None]) -> float:
    """Fraction of mutually-alive bins where concept labels agree."""
    both_alive = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not both_alive:
        return 0.0
    agree = sum(1 for x, y in both_alive if x == y)
    return agree / len(both_alive)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _find_latest_deepdives(deepdive_root: Path) -> dict[str, Path]:
    """Find the latest deep dive directory per model (prefers feature_labels.json)."""
    model_dirs: dict[str, list[Path]] = {}
    for d in sorted(deepdive_root.glob("deepdive_*")):
        if not d.is_dir():
            continue
        fm = d / "feature_map.json"
        if not fm.exists():
            continue
        try:
            model_id = json.loads(fm.read_text())["model_id"]
        except (json.JSONDecodeError, KeyError):
            continue
        model_dirs.setdefault(model_id, []).append(d)

    latest: dict[str, Path] = {}
    for model_id, dirs in model_dirs.items():
        # Prefer dirs with feature_labels.json, then latest timestamp
        with_labels = [d for d in dirs if (d / "feature_labels.json").exists()]
        best = sorted(with_labels or dirs)[-1]
        latest[model_id] = best
    return latest


def _load_feature_labels(path: Path) -> dict[int, list[tuple[int, str | None, float]]]:
    """Load feature_labels.json -> {feature_id: [(layer, concept, cos), ...]}."""
    data = json.loads(path.read_text())
    features = data.get("features", {})
    result = {}
    for fid_str, entries in features.items():
        try:
            fid = int(fid_str)
        except ValueError:
            continue
        result[fid] = [
            (e["layer"], e.get("best_concept"), e.get("best_cos", 0.0))
            for e in entries
        ]
    return result


def _load_ablation(ablation_root: Path, model_id: str) -> dict[int, dict]:
    """Load ablation results for a model -> {feature_id: {concept: retained_pct, ...}}."""
    slug = model_id.replace("/", "_").replace("-", "_")
    # Try various naming patterns
    for pattern in [f"dark_ablation_{slug}", f"dark_ablation_{model_id.split('/')[-1]}"]:
        ablation_dir = ablation_root / pattern
        abl_file = ablation_dir / "dark_matter_ablation.json"
        if abl_file.exists():
            data = json.loads(abl_file.read_text())
            result = {}
            for r in data.get("results", []):
                impact = {}
                for concept, vals in r.get("concept_impact", {}).items():
                    impact[concept] = vals.get("retained_pct", 100.0)
                result[r["feature_id"]] = {
                    "impact": impact,
                    "verdict": r.get("verdict", ""),
                }
            return result
    # Try glob fallback
    for abl_dir in ablation_root.glob(f"dark_ablation_*{model_id.split('/')[-1]}*"):
        abl_file = abl_dir / "dark_matter_ablation.json"
        if abl_file.exists():
            data = json.loads(abl_file.read_text())
            result = {}
            for r in data.get("results", []):
                impact = {}
                for concept, vals in r.get("concept_impact", {}).items():
                    impact[concept] = vals.get("retained_pct", 100.0)
                result[r["feature_id"]] = {
                    "impact": impact,
                    "verdict": r.get("verdict", ""),
                }
            return result
    return {}


def _load_calibration_acts(convergence_root: Path, model_id: str) -> NDArray | None:
    """Load calibration activations from semantic_convergence, stack all concepts."""
    slug = model_id.replace("/", "_").replace("-", "_")
    # Find matching xarch directory
    for d in sorted(convergence_root.glob(f"xarch_{slug}*"), reverse=True):
        cal_files = sorted(d.glob("calibration_*.npy"))
        if cal_files:
            arrays = [np.load(f) for f in cal_files]
            return np.concatenate(arrays, axis=0)
    # Try with just the model name after /
    short = model_id.split("/")[-1].replace("-", "_")
    for d in sorted(convergence_root.glob(f"xarch_{short}*"), reverse=True):
        cal_files = sorted(d.glob("calibration_*.npy"))
        if cal_files:
            arrays = [np.load(f) for f in cal_files]
            return np.concatenate(arrays, axis=0)
    return None


def ingest_deepdives(
    deepdive_root: Path,
    ablation_root: Path | None = None,
    convergence_root: Path | None = None,
    reference_model: str = "openai-community/gpt2",
    min_lifespan: int = 5,
    n_depth_bins: int = N_DEPTH_BINS,
) -> dict[str, ModelRecord]:
    """Ingest deep dive results into ModelRecord objects.

    Parameters
    ----------
    deepdive_root : path to results/ with deepdive_* directories
    ablation_root : path to results/ with dark_ablation_* directories (same root is fine)
    convergence_root : path to semantic_convergence/results/ for calibration data
    reference_model : model whose space defines universal coordinates
    min_lifespan : minimum feature lifespan to include
    n_depth_bins : number of normalized depth bins
    """
    if ablation_root is None:
        ablation_root = deepdive_root

    latest = _find_latest_deepdives(deepdive_root)
    log.info("Found deep dives for %d models", len(latest))

    # Load reference model calibration for Procrustes
    ref_cal = None
    if convergence_root:
        ref_cal = _load_calibration_acts(convergence_root, reference_model)
        if ref_cal is not None:
            log.info("Reference calibration: %s (%s)", reference_model, ref_cal.shape)
        else:
            log.warning("No calibration data for reference model %s", reference_model)

    records: dict[str, ModelRecord] = {}

    for model_id, dd_dir in sorted(latest.items()):
        log.info("Ingesting %s from %s", model_id, dd_dir.name)

        # Load feature map
        fm_data = json.loads((dd_dir / "feature_map.json").read_text())
        n_layers = fm_data["n_layers"]
        hidden_dim = fm_data["hidden_dim"]

        # Load feature labels (if available)
        labels_path = dd_dir / "feature_labels.json"
        feature_labels = _load_feature_labels(labels_path) if labels_path.exists() else {}

        # Load ablation data
        ablation_data = _load_ablation(ablation_root, model_id)

        # Build feature entries
        features = []
        for feat in fm_data["features"]:
            fid = feat["feature_id"]
            lifespan = feat["lifespan"]
            if lifespan < min_lifespan:
                continue

            layer_labels = feature_labels.get(fid)
            handoff = _normalize_handoff(layer_labels, n_layers, n_depth_bins) if layer_labels else [None] * n_depth_bins
            eig_profile = _normalize_eigenvalue_profile(
                feat["layer_indices"], feat["eigenvalues"], n_layers, n_depth_bins
            )

            abl = ablation_data.get(fid)
            entry = ModelFeatureEntry(
                model_id=model_id,
                feature_id=fid,
                birth_layer=feat["birth_layer"],
                death_layer=feat["death_layer"],
                lifespan=lifespan,
                n_layers_total=n_layers,
                layer_indices=feat["layer_indices"],
                eigenvalues=feat["eigenvalues"],
                cos_chain=feat["cos_chain"],
                concept_alignment=feat.get("concept_alignment", {}),
                layer_labels=layer_labels,
                handoff_sequence=handoff,
                depth_eigenvalue_profile=eig_profile,
                ablation_impact=abl["impact"] if abl else None,
                ablation_verdict=abl["verdict"] if abl else None,
            )
            features.append(entry)

        # Compute Procrustes rotation if calibration data available
        rotation = None
        rotation_dim = None
        if convergence_root and ref_cal is not None and model_id != reference_model:
            target_cal = _load_calibration_acts(convergence_root, model_id)
            if target_cal is not None:
                rotation = compute_procrustes_rotation(ref_cal, target_cal)
                rotation_dim = rotation.shape[0]
                log.info("  Procrustes rotation: %s -> %s (%dd)",
                         model_id.split("/")[-1], reference_model.split("/")[-1], rotation_dim)
        elif model_id == reference_model:
            # Reference model: identity rotation
            if ref_cal is not None:
                rotation_dim = ref_cal.shape[1]
                rotation = np.eye(rotation_dim)
                log.info("  Reference model: identity rotation (%dd)", rotation_dim)

        records[model_id] = ModelRecord(
            model_id=model_id,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            features=features,
            deepdive_dir=str(dd_dir),
            rotation=rotation,
            rotation_dim=rotation_dim,
        )
        log.info("  %d features (%d persistent)", len(features), len([f for f in features if f.is_persistent]))

    return records


# ---------------------------------------------------------------------------
# Direction loading & alignment
# ---------------------------------------------------------------------------


def _load_feature_directions(dd_dir: Path, feature: ModelFeatureEntry) -> dict[int, NDArray]:
    """Load native-space direction vectors for a feature's layers."""
    directions = {}
    for layer_idx, pc_idx in zip(feature.layer_indices, feature.eigenvalues):
        # The eigenvalues list parallels layer_indices; pc_indices are separate
        pass  # need pc_indices

    # Actually: directions_L###.npy contains ALL top PCs for that layer.
    # We need the feature's pc_index at each layer.
    # The feature_map.json has pc_indices.
    # Load from the feature map directly.
    return directions


def _get_peak_direction_aligned(
    dd_dir: Path,
    feature: ModelFeatureEntry,
    pc_indices: list[int],
    rotation: NDArray | None,
) -> NDArray | None:
    """Get the Procrustes-aligned direction at the feature's peak layer."""
    if not feature.eigenvalues or not feature.layer_indices:
        return None

    peak_idx = int(np.argmax(feature.eigenvalues))
    peak_layer = feature.layer_indices[peak_idx]
    peak_pc = pc_indices[peak_idx]

    npy_path = Path(dd_dir) / f"directions_L{peak_layer:03d}.npy"
    if not npy_path.exists():
        return None

    dirs = np.load(npy_path)  # shape: [n_pcs, hidden_dim]
    if peak_pc >= len(dirs):
        return None

    direction = dirs[peak_pc].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    direction = direction / norm

    if rotation is not None:
        # Project to rotation dimensionality if needed
        if direction.shape[0] > rotation.shape[0]:
            direction = direction[:rotation.shape[0]]
        elif direction.shape[0] < rotation.shape[0]:
            return None  # can't align
        direction = apply_rotation(direction, rotation)

    return direction


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_universal_features(
    model_records: dict[str, ModelRecord],
    w_handoff: float = 0.5,
    w_variance: float = 0.2,
    w_direction: float = 0.3,
    distance_threshold: float = 0.4,
) -> tuple[list[UniversalFeature], dict[str, ModelRecord]]:
    """Cluster persistent features across models into universal features.

    Returns updated model records with UF assignments and the UF list.
    """
    # Collect all persistent features with their aligned directions
    pool: list[tuple[ModelFeatureEntry, NDArray | None, str]] = []  # (feature, aligned_dir, model_id)

    for model_id, record in model_records.items():
        fm_data = json.loads(Path(record.deepdive_dir, "feature_map.json").read_text())
        # Build pc_indices lookup
        pc_lookup: dict[int, list[int]] = {}
        for feat_data in fm_data["features"]:
            pc_lookup[feat_data["feature_id"]] = feat_data.get("pc_indices", [])

        for feat in record.persistent_features:
            pc_indices = pc_lookup.get(feat.feature_id, [])
            aligned_dir = None
            if record.rotation is not None and pc_indices:
                aligned_dir = _get_peak_direction_aligned(
                    Path(record.deepdive_dir), feat, pc_indices, record.rotation
                )
            pool.append((feat, aligned_dir, model_id))

    log.info("Clustering %d persistent features from %d models", len(pool), len(model_records))

    if len(pool) < 2:
        return [], model_records

    # Compute pairwise distance matrix
    n = len(pool)
    distances = np.ones((n, n))
    np.fill_diagonal(distances, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            fi, di, mi = pool[i]
            fj, dj, mj = pool[j]

            # Only cluster across different models
            if mi == mj:
                distances[i, j] = distances[j, i] = 1.0
                continue

            # Handoff similarity
            h_sim = handoff_similarity(fi.handoff_sequence, fj.handoff_sequence)

            # Variance profile correlation
            vi = np.array(fi.depth_eigenvalue_profile)
            vj = np.array(fj.depth_eigenvalue_profile)
            if vi.sum() > 0 and vj.sum() > 0:
                v_corr = float(np.corrcoef(vi, vj)[0, 1])
                v_corr = max(v_corr, 0.0)  # clamp negative correlations
            else:
                v_corr = 0.0

            # Direction similarity (if both have aligned directions)
            d_sim = 0.0
            if di is not None and dj is not None and di.shape == dj.shape:
                d_sim = abs(float(np.dot(di, dj)))

            dist = w_handoff * (1 - h_sim) + w_variance * (1 - v_corr) + w_direction * (1 - d_sim)
            distances[i, j] = distances[j, i] = dist

    # Agglomerative clustering
    condensed = squareform(distances)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Group into clusters
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # Build universal features
    ufs = []
    # Sort clusters: most models represented first, then by total eigenvalue
    def cluster_sort_key(cluster_indices):
        models = set(pool[i][2] for i in cluster_indices)
        total_eig = sum(pool[i][0].peak_eigenvalue for i in cluster_indices)
        return (-len(models), -total_eig)

    sorted_clusters = sorted(clusters.values(), key=cluster_sort_key)

    for uf_num, indices in enumerate(sorted_clusters, start=1):
        members = [pool[i] for i in indices]
        feats = [m[0] for m in members]
        dirs = [m[1] for m in members if m[1] is not None]
        models = set(m[2] for m in members)

        # Canonical handoff: majority vote per bin
        handoff = []
        n_bins = len(feats[0].handoff_sequence)
        for bin_idx in range(n_bins):
            concepts = [f.handoff_sequence[bin_idx] for f in feats if f.handoff_sequence[bin_idx] is not None]
            if concepts:
                handoff.append(Counter(concepts).most_common(1)[0][0])
            else:
                handoff.append(None)

        # Canonical directions per depth phase
        canonical_dirs = {}
        if dirs:
            mean_dir = np.mean(dirs, axis=0)
            norm = np.linalg.norm(mean_dir)
            if norm > 1e-12:
                canonical_dirs["peak"] = mean_dir / norm

        # Variance profile: mean across members
        var_profiles = np.array([f.depth_eigenvalue_profile for f in feats])
        mean_var = var_profiles.mean(axis=0)
        peak = mean_var.max()
        if peak > 0:
            mean_var = mean_var / peak
        variance_profile = mean_var.tolist()

        # Ablation signature: mean retained_pct
        ablation_sig = {}
        abl_members = [f for f in feats if f.ablation_impact]
        if abl_members:
            all_concepts = set()
            for f in abl_members:
                all_concepts.update(f.ablation_impact.keys())
            for concept in all_concepts:
                vals = [f.ablation_impact[concept] for f in abl_members if concept in f.ablation_impact]
                ablation_sig[concept] = round(sum(vals) / len(vals), 1)

        # Description
        seen = []
        for c in handoff:
            if c is not None and (not seen or seen[-1] != c):
                seen.append(c)
        desc = " -> ".join(seen) if seen else "unlabeled"

        # Families
        def _family(mid):
            if "pythia" in mid.lower(): return "Pythia"
            if "gpt2" in mid.lower() or "gpt-2" in mid.lower(): return "GPT-2"
            if "opt" in mid.lower(): return "OPT"
            if "qwen" in mid.lower(): return "Qwen"
            if "gemma" in mid.lower(): return "Gemma"
            if "llama" in mid.lower(): return "Llama"
            if "mistral" in mid.lower(): return "Mistral"
            if "phi" in mid.lower(): return "Phi"
            return mid.split("/")[0]

        families = set(_family(m) for m in models)

        # Provenance
        provenance = []
        for feat, aligned_dir, model_id in members:
            cos_to_canonical = 0.0
            if aligned_dir is not None and "peak" in canonical_dirs:
                cos_to_canonical = abs(float(np.dot(aligned_dir, canonical_dirs["peak"])))
            provenance.append({
                "model_id": model_id,
                "feature_id": feat.feature_id,
                "handoff_label": feat.handoff_label,
                "alignment_cos": round(cos_to_canonical, 4),
                "peak_eigenvalue": round(feat.peak_eigenvalue, 1),
                "lifespan": feat.lifespan,
            })

        uf_id = f"UF{uf_num:03d}"

        uf = UniversalFeature(
            uf_id=uf_id,
            handoff_template=handoff,
            description=desc,
            canonical_directions=canonical_dirs,
            variance_profile=variance_profile,
            ablation_signature=ablation_sig,
            provenance=provenance,
            n_models=len(models),
            n_families=len(families),
        )
        ufs.append(uf)

        # Assign UF IDs back to model features
        for feat, _, _ in members:
            feat.uf_id = uf_id

    log.info("Clustered into %d universal features", len(ufs))
    multi = [uf for uf in ufs if uf.n_families > 1]
    log.info("  %d cross-family UFs (present in 2+ architecture families)", len(multi))

    return ufs, model_records


# ---------------------------------------------------------------------------
# Matching new models
# ---------------------------------------------------------------------------


def match_features(
    atlas: FeatureAtlas,
    new_features: list[ModelFeatureEntry],
    aligned_directions: dict[int, NDArray] | None = None,
    w_handoff: float = 0.5,
    w_variance: float = 0.2,
    w_direction: float = 0.3,
    match_threshold: float = 0.4,
) -> list[dict]:
    """Match a new model's features against the universal atlas.

    Returns list of {feature_id, uf_id, distance, handoff_sim, direction_sim}.
    """
    results = []
    for feat in new_features:
        if not feat.is_persistent:
            continue

        best_uf = None
        best_dist = 999.0
        best_detail = {}

        for uf in atlas.universal_features:
            h_sim = handoff_similarity(feat.handoff_sequence, uf.handoff_template)

            vi = np.array(feat.depth_eigenvalue_profile)
            vj = np.array(uf.variance_profile)
            v_corr = 0.0
            if vi.sum() > 0 and vj.sum() > 0:
                v_corr = max(float(np.corrcoef(vi, vj)[0, 1]), 0.0)

            d_sim = 0.0
            if aligned_directions and feat.feature_id in aligned_directions and "peak" in uf.canonical_directions:
                ad = aligned_directions[feat.feature_id]
                cd = uf.canonical_directions["peak"]
                if ad.shape == cd.shape:
                    d_sim = abs(float(np.dot(ad, cd)))

            dist = w_handoff * (1 - h_sim) + w_variance * (1 - v_corr) + w_direction * (1 - d_sim)

            if dist < best_dist:
                best_dist = dist
                best_uf = uf.uf_id
                best_detail = {"handoff_sim": round(h_sim, 3), "variance_corr": round(v_corr, 3),
                               "direction_sim": round(d_sim, 3)}

        results.append({
            "feature_id": feat.feature_id,
            "uf_id": best_uf if best_dist < match_threshold else None,
            "distance": round(best_dist, 4),
            "matched": best_dist < match_threshold,
            **best_detail,
        })

    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_atlas(atlas: FeatureAtlas, output_dir: Path) -> None:
    """Save the complete atlas to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Atlas index
    index = {
        "reference_model": atlas.reference_model,
        "aligned_dim": atlas.aligned_dim,
        "n_depth_bins": atlas.n_depth_bins,
        "n_universal_features": len(atlas.universal_features),
        "n_models": len(atlas.model_records),
        "universal_features": [
            {
                "uf_id": uf.uf_id,
                "description": uf.description,
                "n_models": uf.n_models,
                "n_families": uf.n_families,
                "handoff_template": uf.handoff_template,
            }
            for uf in atlas.universal_features
        ],
    }
    (output_dir / "atlas.json").write_text(json.dumps(index, indent=2))

    # Universal features
    uni_dir = output_dir / "universal"
    for uf in atlas.universal_features:
        uf_dir = uni_dir / uf.uf_id
        uf_dir.mkdir(parents=True, exist_ok=True)

        # Canonical directions
        if uf.canonical_directions:
            np.savez(uf_dir / "canonical.npz", **uf.canonical_directions)

        # Profile
        profile = {
            "uf_id": uf.uf_id,
            "description": uf.description,
            "handoff_template": uf.handoff_template,
            "variance_profile": uf.variance_profile,
            "ablation_signature": uf.ablation_signature,
            "n_models": uf.n_models,
            "n_families": uf.n_families,
        }
        (uf_dir / "profile.json").write_text(json.dumps(profile, indent=2))

        # Provenance
        (uf_dir / "provenance.json").write_text(json.dumps(uf.provenance, indent=2))

    # Model records
    models_dir = output_dir / "models"
    for model_id, record in atlas.model_records.items():
        slug = record.slug
        model_dir = models_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        # Rotation
        if record.rotation is not None:
            np.savez(model_dir / "rotation.npz", rotation=record.rotation)

        # Features
        feat_list = []
        for f in record.features:
            feat_list.append({
                "feature_id": f.feature_id,
                "birth_layer": f.birth_layer,
                "death_layer": f.death_layer,
                "lifespan": f.lifespan,
                "layer_indices": f.layer_indices,
                "eigenvalues": [round(e, 4) for e in f.eigenvalues],
                "handoff_label": f.handoff_label,
                "handoff_sequence": f.handoff_sequence,
                "concept_alignment": f.concept_alignment,
                "ablation_impact": f.ablation_impact,
                "ablation_verdict": f.ablation_verdict,
                "uf_id": f.uf_id,
            })
        model_data = {
            "model_id": model_id,
            "n_layers": record.n_layers,
            "hidden_dim": record.hidden_dim,
            "deepdive_dir": record.deepdive_dir,
            "n_features": len(record.features),
            "n_persistent": len(record.persistent_features),
            "features": feat_list,
        }
        (model_dir / "features.json").write_text(json.dumps(model_data, indent=2))

        # Copy native directions for persistent features
        dd_dir = Path(record.deepdive_dir)
        dir_out = model_dir / "directions"
        dir_out.mkdir(exist_ok=True)
        # We store per-layer direction files as-is (they're small)
        for npy in dd_dir.glob("directions_L*.npy"):
            dest = dir_out / npy.name
            if not dest.exists():
                import shutil
                shutil.copy2(npy, dest)

    log.info("Atlas saved to %s", output_dir)
    log.info("  %d universal features, %d models", len(atlas.universal_features), len(atlas.model_records))


def load_atlas(atlas_dir: Path) -> FeatureAtlas:
    """Load a saved atlas from disk."""
    index = json.loads((atlas_dir / "atlas.json").read_text())

    # Load universal features
    ufs = []
    uni_dir = atlas_dir / "universal"
    for uf_entry in index["universal_features"]:
        uf_id = uf_entry["uf_id"]
        uf_dir = uni_dir / uf_id

        profile = json.loads((uf_dir / "profile.json").read_text())
        provenance = json.loads((uf_dir / "provenance.json").read_text())

        canonical = {}
        npz_path = uf_dir / "canonical.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                for key in data.files:
                    canonical[key] = data[key]

        ufs.append(UniversalFeature(
            uf_id=uf_id,
            handoff_template=profile["handoff_template"],
            description=profile["description"],
            canonical_directions=canonical,
            variance_profile=profile["variance_profile"],
            ablation_signature=profile.get("ablation_signature", {}),
            provenance=provenance,
            n_models=profile["n_models"],
            n_families=profile["n_families"],
        ))

    # Load model records
    records = {}
    models_dir = atlas_dir / "models"
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            feat_path = model_dir / "features.json"
            if not feat_path.exists():
                continue
            model_data = json.loads(feat_path.read_text())
            model_id = model_data["model_id"]

            rotation = None
            rotation_dim = None
            rot_path = model_dir / "rotation.npz"
            if rot_path.exists():
                with np.load(rot_path) as data:
                    rotation = data["rotation"]
                    rotation_dim = rotation.shape[0]

            features = []
            for fd in model_data["features"]:
                features.append(ModelFeatureEntry(
                    model_id=model_id,
                    feature_id=fd["feature_id"],
                    birth_layer=fd["birth_layer"],
                    death_layer=fd["death_layer"],
                    lifespan=fd["lifespan"],
                    n_layers_total=model_data["n_layers"],
                    layer_indices=fd["layer_indices"],
                    eigenvalues=fd["eigenvalues"],
                    cos_chain=[],
                    concept_alignment=fd.get("concept_alignment", {}),
                    handoff_sequence=fd.get("handoff_sequence", []),
                    ablation_impact=fd.get("ablation_impact"),
                    ablation_verdict=fd.get("ablation_verdict"),
                    uf_id=fd.get("uf_id"),
                ))

            records[model_id] = ModelRecord(
                model_id=model_id,
                n_layers=model_data["n_layers"],
                hidden_dim=model_data["hidden_dim"],
                features=features,
                deepdive_dir=model_data.get("deepdive_dir", ""),
                rotation=rotation,
                rotation_dim=rotation_dim,
            )

    return FeatureAtlas(
        universal_features=ufs,
        model_records=records,
        reference_model=index["reference_model"],
        aligned_dim=index.get("aligned_dim", 0),
        n_depth_bins=index.get("n_depth_bins", N_DEPTH_BINS),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_atlas_cli(
    deepdive_root: Path,
    ablation_root: Path | None = None,
    convergence_root: Path | None = None,
    output_dir: Path | None = None,
    reference_model: str = "openai-community/gpt2",
) -> FeatureAtlas:
    """Full pipeline: ingest -> cluster -> save."""
    if output_dir is None:
        output_dir = deepdive_root.parent / "feature_library"

    log.info("Building Universal Feature Atlas")
    log.info("  Deep dives: %s", deepdive_root)
    log.info("  Reference:  %s", reference_model)

    records = ingest_deepdives(
        deepdive_root=deepdive_root,
        ablation_root=ablation_root,
        convergence_root=convergence_root,
        reference_model=reference_model,
    )

    ufs, records = cluster_universal_features(records)

    # Determine aligned dimension
    aligned_dim = 0
    for r in records.values():
        if r.rotation_dim:
            aligned_dim = r.rotation_dim
            break

    atlas = FeatureAtlas(
        universal_features=ufs,
        model_records=records,
        reference_model=reference_model,
        aligned_dim=aligned_dim,
    )

    save_atlas(atlas, output_dir)
    return atlas


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build Universal Feature Atlas")
    parser.add_argument("--deepdive-root", type=Path, required=True,
                        help="Path to results/ directory containing deepdive_* dirs")
    parser.add_argument("--ablation-root", type=Path, default=None,
                        help="Path to results/ directory containing dark_ablation_* dirs")
    parser.add_argument("--convergence-root", type=Path, default=None,
                        help="Path to semantic_convergence/results/ for calibration data")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <deepdive_root>/../feature_library)")
    parser.add_argument("--reference-model", type=str, default="openai-community/gpt2",
                        help="Reference model for universal space (must have calibration data)")
    args = parser.parse_args()

    atlas = build_atlas_cli(
        deepdive_root=args.deepdive_root,
        ablation_root=args.ablation_root or args.deepdive_root,
        convergence_root=args.convergence_root,
        output_dir=args.output_dir,
        reference_model=args.reference_model,
    )

    print(f"\nAtlas built: {len(atlas.universal_features)} universal features from {len(atlas.model_records)} models")
    multi = [uf for uf in atlas.universal_features if uf.n_families > 1]
    print(f"Cross-family UFs: {len(multi)}")
    for uf in multi[:10]:
        print(f"  {uf.uf_id}: {uf.description} ({uf.n_models} models, {uf.n_families} families)")

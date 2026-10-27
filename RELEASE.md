# Releasing rosetta_tools

## Canonical location

Your local clone is the canonical stem of `rosetta_tools`. The GitHub repo at `github.com/jamesrahenry/Rosetta_Tools` is a publish target, not a source of truth. Do not develop directly against a secondary clone — develop in canonical, then publish.

## Why this matters

Downstream consumers (CIA, Rosetta_Program, anything else) pin to a tagged release pulled from GitHub. If unpinned `git+https://…/Rosetta_Tools.git` references exist in any consumer, every fresh install can pull a different `main` and silently change behaviour. A short-lived experimental commit on `main` (for example, the `dom.peak` change in `cfce0b2`, reverted by `c0801ea`) can leak into a downstream build during the window it sits at HEAD. Tag-pinning prevents that.

## Release procedure

When a change in canonical needs to be visible to consumers:

1. Commit the change here. Working tree must be clean.
2. Tag the commit with the next semver:
   ```
   git tag -a vX.Y.Z <sha> -m "vX.Y.Z — short summary"
   ```
   - Bump **patch** (Z) for fixes that don't change extraction or scoring semantics.
   - Bump **minor** (Y) for new functions, new optional parameters, or new probe-extraction methods that don't break existing call sites.
   - Bump **major** (X) for changes that break existing call sites or change the meaning of an existing artifact (for example, switching `extract_gem_probe` from handoff to peak — that is a major change, not a fix).
3. Push the tag to GitHub:
   ```
   git push origin vX.Y.Z
   ```
4. Update each consumer to point at the new tag:
   - Bump `rosetta-tools @ git+https://…/Rosetta_Tools.git@vX.Y.Z` in each consumer's `pyproject.toml`.
5. Note the change in this file's version log below.

## Do not

- Do not let any consumer reference unpinned `Rosetta_Tools.git` (no `@<ref>`). The unpinned form is a footgun.
- Do not develop or commit inside a consumer's submodule clone of `rosetta_tools`. Those clones must only ever check out an existing tag from canonical.
- Do not move tags after they are pushed. If a release is broken, cut a new tag.
- Do not create sibling clones of canonical. One canonical, one set of tags.

## Supported surface for downstream consumers

**`v2.2.0` is the supported release for deployment and new analysis** (Monitor, CIA,
Rosetta_Analysis). The papers' pins (`v1.3.1`; `v1.4.0` for the Gemma-2 rebuild) are frozen
reproducibility anchors, not something to upgrade. Unreleased branches are not a supported surface and cannot be pinned; when one is merged it
ships as a new tag and gets a version-log entry here.

A note on **GitHub Releases vs tags**: `v1.1.0` and `v1.3.1` exist as GitHub Releases (which is
what minted their Zenodo DOIs); `v1.2.0`, `v1.3.0`, `v1.4.0`, `v2.0.0`, `v2.1.0` and `v2.2.0` are bare
tags. Pinning by tag works either way. Create a GitHub Release only when a new DOI is wanted.

## Version log

- **v2.2.0** (2026-10-26) — minor: new API; defaults and existing call sites unchanged except
  where noted.
  - `probes`: `extract_probe` and `extract_gem_probe` take `direction_estimator="dom"|"logreg"|"lda"`
    (default `"dom"`, identical to every earlier release). The estimator is also used by the
    Fisher-weighted path, and `ProbeResult.direction_estimator` records which one produced a
    probe. Fitted estimators need the pair-aware split (v2.0.0+) to be compared fairly with
    `"dom"`.
  - GEM schema v2 (`GEM_SCHEMA_VERSION = "2"`): new module `gem_sublayer` and new types
    `GEMExtractionConfig`, `LayerProfile`, `LayerRecord`, `SublayerRecord`, `PermutationNull`,
    `ArchitectureFlags`, `Provenance`, `SubspaceSublayerDecomposition` and `SidecarRef`. Per-layer
    separation at every layer (Fisher and gauge-corrected projected, each with a permutation
    null), attention and MLP writes per block with an additivity check, the post-final-norm
    record at the last layer, a mixture-of-experts path, and a checksummed `.sidecar.npz` for
    large arrays. `save_gem` gains `arrays=`, `array_docs=` and `inline_directions=`;
    `load_gem_sidecar`, `sidecar_path_for` and `find_caz_regions_projected` are new.
    **Reader compatibility:** GEM JSON written with v2 fields must be read with ≥ 2.2.0 (v2.1.0's
    loader raises on the new keys); v1 files load unchanged. Fixes folded in before tagging:
    sublayer hooks are removed even when registration fails partway, the Phi `final_layernorm`
    is recognised as a final norm, and an empty stored sidecar sha256 is an error unless
    `verify=False`.
  - `dataset.pair_set_provenance()`: sha256 over the exact ordered pairs plus the
    Rosetta_Concept_Pairs commit, recorded in `Provenance.pair_set`, so runs that drew
    different samples from the same `load_concept_pairs()` call can be told apart.
  - Tests: `tests/conftest.py` shared helpers, `TestFindCazRegionsScored`,
    `TestDirectionEstimators`, `tests/test_gem.py`, `tests/test_gem_schema.py` and
    `tests/test_gem_sublayer_hooks.py`.
  - `bin/gpu_daemon.sh`: `gpu_jobs` removed from the `sync_repos` set.
- **v2.1.0** (2026-10-26) — minor: new API, no changes to existing call sites or to the
  meaning of existing artifacts.
  - `gem.extract_phase2_threads()` — the k>1 eigen-thread path that was previously a
    `NotImplementedError`; greedy bipartite matching of top-k eigenvectors across layers
    onto the existing `EigThread` type.
  - `gem.extract_settled_subspace()` / `attach_settled_subspace()` / `SettledSubspace` —
    INLP (iterative nullspace projection) at one layer with a binomial test on held-out
    accuracy as the stopping rule, for testing whether a concept's settled representation
    is rank-1. Wired into `build_concept_gem(..., h_pos_by_layer=, h_neg_by_layer=,
    compute_settled_subspace=)`; default off. **Marked experimental:** an independent
    review (2026-10-26) showed `k_effective` is an upper bound inflated by estimation
    noise when n ≪ d (exactly-rank-1 synthetic data returns 2–3), so rank > 1 claims need a
    matched rank-1 control. Two defects fixed before tagging: `attach_settled_subspace`
    built labels from `h_pos.shape[0]` for both classes (crashed on unequal sizes); the
    triage binomial null was 0.5 regardless of class balance (now the majority-class
    rate). `_topk_eigvecs` now clamps `k` to the available rank instead of raising
    `IndexError`. First tests for these paths: `tests/test_gem_phase2.py`.
  - `gem_to_dict` now emits a `settled_subspace` key (usually `null`) on every node and
    `gem_from_dict` reads it back. **Reader compatibility:** v2.0.0's loader passes node
    dicts straight to `GEMNode(**nd)` and raises `TypeError` on the new key, so GEM JSON
    written by ≥ 2.1.0 must be read with ≥ 2.1.0. GEM JSON written by ≤ 2.0.0 loads
    unchanged. `probes` (what Monitor/CIA consume) is unaffected.
  - Firm-up: `tests/test_models.py` rewritten against the YAML-backed registry (it had
    imported names removed by the April 2026 YAML refactor and failed collection on every
    tag since v1.1.0); `docs/api.md` now covers all 14 modules (previously omitted `probes`,
    `gem`, `compound_monitor`, `paths`, `viz_style` and the v2.0.0 additions);
    `CITATION.cff` version brought in line; `uv.lock` refreshed; `bin/README.md` marks the
    GPU-queue and provisioning scripts as retired legacy tooling; README gains a
    which-version-to-pin table.

- **v2.0.0** (2026-08-31) — major: migrated `extract_layer_wise_metrics()` and
  `build_caz_record()` (the caz-record schema assembler) from
  `Rosetta_Analysis/extraction/extract.py` into `rosetta_tools.extraction`, per
  the scope rule (rosetta_tools = reusable CAZ/GEM pipeline; Rosetta_Analysis =
  paper-writing support). New public API surface, so major rather than minor —
  see `Rosetta_Analysis/PIPELINE_MIGRATION_PLAN.md` for the full rationale.
  Also folds in three semantic-changing fixes that had accumulated unreleased
  since v1.4.0 and already qualify as major under this file's own policy
  (they change the meaning of existing artifacts, not just fix a bug):
  - `dataset.load_concept_pairs` now uses a stable SHA256-derived default seed
    instead of process-randomized `hash()` — pair sampling is now reproducible
    across processes/runs (previously every invocation drew a different subset).
  - `dataset.load_pairs` drops blank-text pairs instead of silently
    misaligning pos/neg rows.
  - `probes._split_indices` (via `extract_gem_probe`) is now pair-aware for
    contrastive datasets, closing a train/eval leak that inflated fitted
    estimators' (logreg/LDA) held-out AUROC relative to DOM. Already shifted
    a published P2 supervised-baseline artifact's numbers — correctly.
  There is no live v1.5.0/v1.6.0 — those version strings were set in-tree on
  unpublished commits and are superseded by this tag. (Commits `9d6d658` "(v1.5.0)" and
  `47b68d3` "(v1.6.0)" are both ancestors of `v2.0.0`; notes that cite "rosetta_tools
  v1.6.0" mean this tag.)
- **v1.4.0** (2026-07) — GPU host/queue matching by VRAM + host tags, Hopper
  `urgent` priority, multi-GPU `device_map='balanced'`/capped `max_memory`,
  `DirectionalShifter` additive ablation, gemma-2-9b-it-abliterated registered.
- **v1.3.1** — split `viz.py` into `viz_style` (no backend) + `viz` (Agg + file
  writers).
- **v1.3.0** — dependency restructuring for PyPI publication (base + optional
  extras), informative `ImportError` for GPU modules missing torch.
- **v1.2.0** — bidirectional host-to-host model/result sync tooling
  (`sync_models.sh`, `sync_results.sh`), GPU daemon disk-monitor + npy scratch
  purge, 8-bit/4-bit quantization support in `gpu_utils`, CITATION.cff + Zenodo
  DOI.
- **v1.1.0** (2026-04-28) — per-region assembly thresholds, Fisher-weighted GEM extraction, `calibration_percentile` parameter, CIA-derived routing and vigilance utilities. First tag cut under this release discipline.
- **v1.0.0** — pre-existing baseline.

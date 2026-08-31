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

## Version log

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
  unpublished commits and are superseded by this tag.
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

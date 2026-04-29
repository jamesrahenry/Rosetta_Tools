# Releasing rosetta_tools

## Canonical location

This repository (`~/Source/rosetta_tools/`) is the canonical stem of `rosetta_tools`. The GitHub repo at `github.com/jamesrahenry/Rosetta_Tools` is a publish target, not a source of truth. Do not develop directly against the GitHub clone — develop here, then publish.

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
   - **CIA** — bump `rosetta-tools @ git+https://…/Rosetta_Tools.git@vX.Y.Z` in `pyproject.toml`.
   - **Rosetta_Program** — `cd rosetta_tools && git checkout vX.Y.Z`, then commit the submodule pointer bump in the parent repo.
5. Note the change in this file's version log below.

## Do not

- Do not let any consumer reference unpinned `Rosetta_Tools.git` (no `@<ref>`). The unpinned form is a footgun.
- Do not develop or commit inside a consumer's clone of `rosetta_tools` (for example, inside `Rosetta_Program/rosetta_tools/`). Those clones must only ever check out an existing tag from canonical.
- Do not move tags after they are pushed. If a release is broken, cut a new tag.
- Do not create sibling clones of canonical (for example, in `~/Source/Rosetta_Tools_dev/`). One canonical, one set of tags.

## Version log

- **v1.1.0** (2026-04-28) — per-region assembly thresholds, Fisher-weighted GEM extraction, `calibration_percentile` parameter, CIA-derived routing and vigilance utilities. First tag cut under this release discipline.
- **v1.0.0** — pre-existing baseline.

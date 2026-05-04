#!/usr/bin/env bash
# sync_results.sh — Pull rosetta_data from multiple GPU hosts and merge.
#
# Reads hosts from <dest-parent>/rosetta_queue/sync_hosts.conf. Each line:
#   alias   rsync_source
# e.g.:
#   coder   coder@main.MI-Workspace.james-henry-git.coder:~/rosetta_data/
#   vec1    user@vec1.vectorinstitute.ai:~/rosetta_data/
#
# Blank lines and # comments are ignored.
#
# JSON result files are merged intelligently:
#   - Files with a list-valued "pair_results" or "results" key: lists are
#     unioned by record content (deduped), so each host's unique records survive.
#   - Files with a dict-valued "results" key: dict-unioned by model key.
#   - Other flat stat files: newer version wins (by "written" timestamp or mtime).
# Non-JSON files: newer mtime wins.
# .npy files are excluded (too large, not needed on dev machine).
#
# Usage:
#   bash sync_results.sh [--dry-run] [--dest DIR]
#
# Written: 2026-05-04 21:00 UTC

set -euo pipefail

DEST="${HOME}/Source/Rosetta_Program/rosetta_data"
CONF="$(dirname "$DEST")/rosetta_queue/sync_hosts.conf"
STAGING="${HOME}/.rosetta_sync_staging"
DRY_RUN=false
RSYNC_EXCLUDES=(--exclude='*.npy' --exclude='*.pt' --exclude='*.bin' --exclude='*.safetensors')

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --dest)    DEST="$2"; CONF="$(dirname "$2")/rosetta_queue/sync_hosts.conf"; shift 2 ;;
        -h|--help)
            echo "Usage: sync_results.sh [--dry-run] [--dest DIR]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

log() { echo "[sync $(date +%H:%M:%S)] $*"; }

if [[ ! -f "$CONF" ]]; then
    cat >&2 <<EOF
ERROR: $CONF not found.

Create it with one host per line:
  alias   user@host:~/rosetta_data/

Example:
  coder   coder@main.MI-Workspace.james-henry-git.coder:~/rosetta_data/
  vec1    user@vec1.vectorinstitute.ai:~/rosetta_data/
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Rsync each host into its own staging directory
# ---------------------------------------------------------------------------
declare -a ALIASES=()

while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    read -r alias source <<< "$line"
    [[ -z "$alias" || -z "$source" ]] && continue

    ALIASES+=("$alias")
    stage="${STAGING}/${alias}"
    mkdir -p "$stage"

    log "Pulling $alias ($source) → $stage"
    rsync -az "${RSYNC_EXCLUDES[@]}" "$source" "$stage/" \
        && log "  done" \
        || log "  ⚠ rsync failed for $alias — skipping"
done < "$CONF"

if [[ ${#ALIASES[@]} -eq 0 ]]; then
    log "No hosts in $CONF — nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Merge staged results into DEST
# ---------------------------------------------------------------------------
log "Merging ${#ALIASES[@]} host(s) → $DEST"

$DRY_RUN && log "(dry-run: Python merge skipped)" && exit 0

mkdir -p "$DEST"

python3 - "$STAGING" "$DEST" "${ALIASES[@]}" <<'PYEOF'
import sys, json, os, shutil
from pathlib import Path
from datetime import datetime, timezone

staging_root = Path(sys.argv[1])
dest_root    = Path(sys.argv[2])
aliases      = sys.argv[3:]

def read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def written_ts(obj):
    """Parse the 'written' field to a comparable datetime, or epoch."""
    w = obj.get("written") if isinstance(obj, dict) else None
    if not w:
        return datetime.min.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            dt = datetime.strptime(w, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)

def record_key(rec):
    """Stable hashable key for a result record dict."""
    if not isinstance(rec, dict):
        return repr(rec)
    return tuple(sorted((k, repr(v)) for k, v in rec.items()))

def merge_json(objs):
    """
    Merge a list of parsed JSON objects (all from the same relative path).
    Strategy:
      - list-valued 'pair_results' or 'results': union by record content
      - dict-valued 'results': union by key (later host wins on collision)
      - otherwise: newest 'written' timestamp wins; tie → last in list
    """
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    if len(objs) == 1:
        return objs[0]

    # Try list-union merge (pair_results or results as list)
    for key in ("pair_results", "results"):
        if all(isinstance(o.get(key), list) for o in objs):
            seen = {}
            merged_list = []
            for o in objs:
                for rec in o[key]:
                    k = record_key(rec)
                    if k not in seen:
                        seen[k] = True
                        merged_list.append(rec)
            # Use newest object as base, replace the list key
            base = max(objs, key=written_ts)
            base = dict(base)
            base[key] = merged_list
            base["written"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base["_merged_from"] = aliases
            return base

    # Try dict-union merge (results as dict keyed by model)
    if all(isinstance(o.get("results"), dict) for o in objs):
        merged = {}
        for o in objs:
            merged.update(o["results"])
        base = dict(max(objs, key=written_ts))
        base["results"] = merged
        base["written"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        base["_merged_from"] = aliases
        return base

    # Fallback: newest wins
    return max(objs, key=written_ts)


# Collect all relative paths across all staging dirs
all_rel_paths = set()
for alias in aliases:
    stage = staging_root / alias
    if not stage.exists():
        continue
    for p in stage.rglob("*"):
        if p.is_file():
            all_rel_paths.add(p.relative_to(stage))

merged = skipped = plain_copied = 0

for rel in sorted(all_rel_paths):
    dest_file = dest_root / rel
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    for alias in aliases:
        src = staging_root / alias / rel
        if src.exists():
            candidates.append(src)

    if not candidates:
        continue

    if rel.suffix == ".json":
        objs = [read_json(c) for c in candidates]
        # also include the existing dest file if present
        if dest_file.exists():
            objs.insert(0, read_json(dest_file))

        result = merge_json([o for o in objs if o is not None])
        if result is not None:
            dest_file.write_text(json.dumps(result, indent=2))
            print(f"  merged  {rel}  ({len(candidates)} host(s))")
            merged += 1
        else:
            skipped += 1
    else:
        # Non-JSON: copy newest by mtime
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        if not dest_file.exists() or newest.stat().st_mtime > dest_file.stat().st_mtime:
            shutil.copy2(newest, dest_file)
            print(f"  copied  {rel}  (from {newest.parent.name})")
            plain_copied += 1

print(f"\nDone — {merged} merged, {plain_copied} copied, {skipped} skipped")
PYEOF

log "Sync complete."

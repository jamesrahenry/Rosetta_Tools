#!/usr/bin/env bash
# sync_results.sh — Pull rosetta_data from multiple GPU hosts and merge.
#
# Reads hosts from <dest-parent>/rosetta_queue/sync_hosts.conf. Each line:
#   alias   rsync_source
# e.g.:
#   coder   coder@main.MI-Workspace.james-henry-git.coder:~/rosetta_data/
#   vec1    user@vec1.vectorinstitute.ai:~/rosetta_data/
#
# No staging copy is made. Non-JSON files rsync directly to DEST (newer wins).
# JSON files are fetched to a small per-host temp dir, merged with DEST, and
# written back — so only JSON files occupy temporary disk space.
#
# Merge strategy for JSON:
#   - list-valued "pair_results" / "results": union by record content (dedup)
#   - dict-valued "results": union by model key
#   - flat stat files: newest "written" timestamp wins
# Non-JSON, non-binary: rsync --update (newer mtime wins)
# Excluded: *.npy *.pt *.bin *.safetensors (large weight/tensor files)
#
# Usage:
#   bash sync_results.sh [--dry-run] [--dest DIR]
#
# Written: 2026-05-04 22:00 UTC

set -euo pipefail

DEST="${HOME}/Source/Rosetta_Program/rosetta_data"
CONF="$(dirname "$DEST")/rosetta_queue/sync_hosts.conf"
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

mkdir -p "$DEST"
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# ---------------------------------------------------------------------------
# Process each host
# ---------------------------------------------------------------------------
declare -a ALIASES=()

while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    read -r alias source <<< "$line"
    [[ -z "$alias" || -z "$source" ]] && continue

    ALIASES+=("$alias")
    tmpdir="${TMPROOT}/${alias}"
    mkdir -p "$tmpdir"

    log "Syncing $alias ($source)"

    if $DRY_RUN; then
        log "  (dry-run)"
        continue
    fi

    # 1. Non-JSON files: rsync directly to DEST, newer wins
    rsync -avz --update \
        --exclude='*.json' \
        "${RSYNC_EXCLUDES[@]}" \
        "$source" "$DEST/" 2>/dev/null \
        && log "  non-JSON files updated" \
        || log "  ⚠ rsync (non-JSON) failed for $alias"

    # 2. JSON files only: fetch to temp dir for merge
    rsync -avz \
        --include='*/' --include='*.json' --exclude='*' \
        "$source" "$tmpdir/" 2>/dev/null \
        && log "  JSON files fetched for merge" \
        || log "  ⚠ rsync (JSON) failed for $alias"

done < "$CONF"

if [[ ${#ALIASES[@]} -eq 0 ]]; then
    log "No hosts in $CONF — nothing to do."
    exit 0
fi

$DRY_RUN && log "dry-run complete." && exit 0

# ---------------------------------------------------------------------------
# Merge JSON files from all hosts into DEST
# ---------------------------------------------------------------------------
log "Merging JSON from ${#ALIASES[@]} host(s) → $DEST"

python3 - "$TMPROOT" "$DEST" "${ALIASES[@]}" <<'PYEOF'
import sys, json, shutil
from pathlib import Path
from datetime import datetime, timezone

tmp_root  = Path(sys.argv[1])
dest_root = Path(sys.argv[2])
aliases   = sys.argv[3:]

def read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def written_ts(obj):
    w = obj.get("written") if isinstance(obj, dict) else None
    if not w:
        return datetime.min.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            return datetime.strptime(w, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)

def record_key(rec):
    if not isinstance(rec, dict):
        return repr(rec)
    return tuple(sorted((k, repr(v)) for k, v in rec.items()))

def merge_json(objs):
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    if len(objs) == 1:
        return objs[0]

    # Top-level list (e.g. gem_adaptive_width results): union records by content
    if all(isinstance(o, list) for o in objs):
        seen, merged_list = {}, []
        for o in objs:
            for rec in o:
                k = record_key(rec)
                if k not in seen:
                    seen[k] = True
                    merged_list.append(rec)
        return merged_list

    # Mixed or non-dict: fall back to newest
    if not all(isinstance(o, dict) for o in objs):
        return max((o for o in objs if isinstance(o, dict)), key=written_ts, default=objs[-1])

    for key in ("pair_results", "results"):
        if all(isinstance(o.get(key), list) for o in objs):
            seen, merged_list = {}, []
            for o in objs:
                for rec in o[key]:
                    k = record_key(rec)
                    if k not in seen:
                        seen[k] = True
                        merged_list.append(rec)
            base = dict(max(objs, key=written_ts))
            base[key] = merged_list
            base["written"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base["_merged_from"] = aliases
            return base

    if all(isinstance(o.get("results"), dict) for o in objs):
        merged = {}
        for o in objs:
            merged.update(o["results"])
        base = dict(max(objs, key=written_ts))
        base["results"] = merged
        base["written"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        base["_merged_from"] = aliases
        return base

    return max(objs, key=written_ts)

all_rel = set()
for alias in aliases:
    d = tmp_root / alias
    if d.exists():
        for p in d.rglob("*.json"):
            all_rel.add(p.relative_to(d))

merged = skipped = 0
for rel in sorted(all_rel):
    dest_file = dest_root / rel
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    candidates = [tmp_root / a / rel for a in aliases if (tmp_root / a / rel).exists()]
    objs = ([read_json(dest_file)] if dest_file.exists() else []) + [read_json(c) for c in candidates]

    result = merge_json([o for o in objs if o is not None])
    if result is not None:
        dest_file.write_text(json.dumps(result, indent=2))
        print(f"  merged  {rel}  ({len(candidates)} host(s))")
        merged += 1
    else:
        skipped += 1

print(f"\nDone — {merged} JSON merged, {skipped} skipped")
PYEOF

log "Sync complete."

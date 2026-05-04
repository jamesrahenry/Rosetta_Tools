#!/usr/bin/env bash
# sync_models.sh — Bidirectional model run sync between GPU hosts.
#
# Run this on a GPU host after completing extractions to share results with
# peers so they don't regenerate models you've already computed.
#
# Reads peer hosts from ~/rosetta_queue/sync_peers.conf:
#   alias   hostname
# e.g.:
#   cia_host   coder-james-henry-git-jameshenry-workspace
#   coder      coder-james-henry-git-mi-workspace
#
# Uses --ignore-existing in both directions — model files are immutable once
# extracted, so we only fill gaps, never overwrite.
#
# Usage:
#   bash sync_models.sh [--pull-only] [--push-only] [--dry-run]
#
# Written: 2026-05-04 22:30 UTC

set -euo pipefail

CONF="${HOME}/rosetta_queue/sync_peers.conf"
LOCAL_MODELS="${HOME}/rosetta_data/models/"
DRY_RUN=false
DO_PULL=true
DO_PUSH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull-only) DO_PUSH=false; shift ;;
        --push-only) DO_PULL=false; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: sync_models.sh [--pull-only] [--push-only] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

log() { echo "[sync_models $(date +%H:%M:%S)] $*"; }

if [[ ! -f "$CONF" ]]; then
    cat >&2 <<EOF
ERROR: $CONF not found.

Create it with one peer per line:
  alias   hostname

Example:
  cia_host   coder-james-henry-git-jameshenry-workspace
  coder      coder-james-henry-git-mi-workspace
EOF
    exit 1
fi

mkdir -p "$LOCAL_MODELS"

MIN_FREE_GIB=10
RSYNC_FLAGS=(-avz --size-only --partial)
$DRY_RUN && RSYNC_FLAGS+=(--dry-run)

free_gib() {
    df -BG "${1:-$HOME}" | awk 'NR==2 {gsub("G",""); print $4}'
}

while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    read -r alias host <<< "$line"
    [[ -z "$alias" || -z "$host" ]] && continue

    PEER_MODELS="${host}:${LOCAL_MODELS}"

    if $DO_PULL; then
        avail=$(free_gib "$LOCAL_MODELS")
        if [[ "$avail" -lt "$MIN_FREE_GIB" ]]; then
            log "⚠ Skipping pull from $alias — only ${avail} GiB free (min ${MIN_FREE_GIB} GiB)"
        else
            log "Pull $alias → local  (${avail} GiB free)"
            rsync "${RSYNC_FLAGS[@]}" "$PEER_MODELS" "$LOCAL_MODELS" \
                && log "  pull done" \
                || log "  ⚠ pull failed for $alias (disk full?)"
        fi
    fi

    if $DO_PUSH; then
        log "Push local → $alias"
        rsync "${RSYNC_FLAGS[@]}" "$LOCAL_MODELS" "$PEER_MODELS" \
            && log "  push done" \
            || log "  ⚠ push failed for $alias"
    fi

done < "$CONF"

log "Done."

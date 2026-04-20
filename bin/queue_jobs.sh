#!/usr/bin/env bash
# queue_jobs.sh — Load a jobs file into the Hopper gpu-job queue.
#
# Reads a jobs file (one shell command per line, # comments ignored) and
# creates a Hopper task for each. Tasks are tagged gpu-job and ordered by
# creation time so the daemon runs them in file order.
#
# Usage:
#   bash scripts/queue_jobs.sh jobs/prh_full.txt
#   bash scripts/queue_jobs.sh --dry-run jobs/prh_full.txt
#   bash scripts/queue_jobs.sh --priority high jobs/prh_full.txt
#
# Written: 2026-04-20 UTC

set -euo pipefail

DRY_RUN=false
PRIORITY="medium"
JOBS_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --priority)  PRIORITY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: queue_jobs.sh [--dry-run] [--priority <p>] JOBS_FILE"
            exit 0 ;;
        *)  JOBS_FILE="$1"; shift ;;
    esac
done

if [[ -z "$JOBS_FILE" || ! -f "$JOBS_FILE" ]]; then
    echo "ERROR: jobs file not found: ${JOBS_FILE:-<none>}"
    exit 1
fi

# Count non-blank, non-comment lines
TOTAL=$(grep -cve '^\s*$\|^\s*#' "$JOBS_FILE" || true)
echo "Queuing $TOTAL jobs from $JOBS_FILE (priority=$PRIORITY, dry_run=$DRY_RUN)"
echo ""

JOB_NUM=0
LAST_COMMENT=""

while IFS= read -r line || [[ -n "$line" ]]; do
    # Track the most recent comment as a human-readable title
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*(.*) ]]; then
        LAST_COMMENT="${BASH_REMATCH[1]}"
        continue
    fi
    [[ -z "$line" || "$line" =~ ^[[:space:]]*$ ]] && continue

    JOB_NUM=$(( JOB_NUM + 1 ))

    # Use comment as title, fall back to truncated command
    if [[ -n "$LAST_COMMENT" ]]; then
        title="$LAST_COMMENT"
    else
        title="${line:0:60}..."
    fi
    LAST_COMMENT=""  # consume the comment

    echo "[$JOB_NUM/$TOTAL] $title"
    echo "  cmd: $line"

    if $DRY_RUN; then
        echo "  (dry run — skipped)"
    else
        task_id=$(hopper task add "$title" \
            --description "$line" \
            --tag gpu-job \
            --priority "$PRIORITY" \
            --non-interactive 2>&1 \
            | grep -o 't[a-f0-9]\{8\}' | head -1)
        echo "  → $task_id"
    fi
    echo ""
done < "$JOBS_FILE"

if ! $DRY_RUN; then
    hopper sync 2>/dev/null || true
    echo "Synced. Check queue: hopper task list --tag gpu-job --status open"
fi

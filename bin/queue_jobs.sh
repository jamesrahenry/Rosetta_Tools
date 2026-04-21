#!/usr/bin/env bash
# queue_jobs.sh — Load a jobs file into the Hopper gpu-job queue.
#
# Reads a jobs file (one shell command per line, # comments ignored) and
# creates a Hopper task for each. Tasks are tagged gpu-job and ordered by
# creation time so the daemon runs them in file order.
#
# Dependency barriers:
#   A line containing only "# ---" separates phases. Tasks after a barrier
#   are added as `blocked` with the IDs of all preceding tasks embedded in
#   their description. The daemon unblocks them once all predecessors complete.
#
# Usage:
#   bash queue_jobs.sh jobs/prh_full.txt
#   bash queue_jobs.sh --dry-run jobs/prh_full.txt
#   bash queue_jobs.sh --priority high jobs/prh_full.txt
#
# Written: 2026-04-20 UTC

set -uo pipefail

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

TOTAL=$(grep -cve '^\s*$\|^\s*#' "$JOBS_FILE" || true)
echo "Queuing $TOTAL jobs from $JOBS_FILE (priority=$PRIORITY, dry_run=$DRY_RUN)"
echo ""

JOB_NUM=0
LAST_COMMENT=""
COMPLETED_IDS=()   # IDs of all tasks in phases before the last barrier
BARRIER_IDS=()     # IDs of tasks in the current phase (reset at each barrier)
AFTER_BARRIER=false

add_task() {
    local title="$1" cmd="$2" status="$3" deps="$4"
    local description="$cmd"
    [[ -n "$deps" ]] && description="$cmd ### depends:$deps"

    hopper task add "$title" \
        --description "$description" \
        --tag gpu-job \
        --priority "$PRIORITY" \
        --status "$status" \
        --non-interactive 2>&1 \
        | grep -o 't[a-f0-9]\{8,\}' | head -1
}

while IFS= read -r line || [[ -n "$line" ]]; do
    # Dependency barrier — "# ---"
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*---[[:space:]]*$ ]]; then
        echo "[barrier] tasks below depend on ${#BARRIER_IDS[@]} tasks above"
        COMPLETED_IDS+=("${BARRIER_IDS[@]}")
        BARRIER_IDS=()
        AFTER_BARRIER=true
        echo ""
        continue
    fi

    # Track comments as human-readable titles
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*(.*) ]]; then
        LAST_COMMENT="${BASH_REMATCH[1]}"
        continue
    fi
    [[ -z "$line" || "$line" =~ ^[[:space:]]*$ ]] && continue

    JOB_NUM=$(( JOB_NUM + 1 ))
    title="${LAST_COMMENT:-${line:0:60}...}"
    LAST_COMMENT=""

    if $AFTER_BARRIER && [[ ${#COMPLETED_IDS[@]} -gt 0 ]]; then
        deps=$(IFS=,; echo "${COMPLETED_IDS[*]}")
        status="blocked"
    else
        deps=""
        status="open"
    fi

    echo "[$JOB_NUM/$TOTAL] $title  [status=$status]"
    echo "  cmd: $line"
    [[ -n "$deps" ]] && echo "  deps: $deps"

    if $DRY_RUN; then
        echo "  (dry run — skipped)"
        BARRIER_IDS+=("DRY_RUN_ID_$JOB_NUM")
    else
        task_id=$(add_task "$title" "$line" "$status" "$deps")
        if [[ -n "$task_id" ]]; then
            echo "  → $task_id"
            BARRIER_IDS+=("$task_id")
        else
            echo "  ✗ failed to create task — aborting"
            exit 1
        fi
    fi
    echo ""

done < "$JOBS_FILE"

if ! $DRY_RUN; then
    hopper sync 2>/dev/null || true
    echo "Synced. Check queue:"
    echo "  hopper task list --tag gpu-job --compact"
fi

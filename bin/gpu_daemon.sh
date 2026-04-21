#!/usr/bin/env bash
# gpu_daemon.sh — Hopper-driven GPU job daemon.
#
# Polls for open gpu-job tasks, runs them one at a time, marks complete.
# After each completion, scans blocked gpu-job tasks and unblocks any whose
# dependencies (listed as "### depends:ID1,ID2,..." in the description) are
# all completed.
#
# Jobs are added from any machine via queue_jobs.sh:
#   bash ~/rosetta_tools/bin/queue_jobs.sh jobs/prh_full.txt
#
# Run in tmux so you can detach:
#   tmux new-session -d -s gpu 'bash ~/rosetta_tools/bin/gpu_daemon.sh'
#   tmux attach -t gpu
#
# Written: 2026-04-20 UTC

set -uo pipefail

IDENTITY="gpu:$(hostname)"
POLL_INTERVAL=30      # seconds between polls when idle
LOG_DIR="${HOME}/gpu_runs"
MIN_DISK_GIB=15

mkdir -p "$LOG_DIR"
DAEMON_LOG="$LOG_DIR/daemon.log"

# Tee all daemon output to daemon.log — survives tmux detach/reattach
exec > >(tee -a "$DAEMON_LOG") 2>&1

export MLFLOW_TRACKING_URI="file:///dev/null"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "$(date +%H:%M:%S) [daemon] $*"; }

sync_hopper() { hopper sync 2>/dev/null || true; }

free_gib() {
    local target="${HF_HOME:-$HOME/.cache/huggingface}"
    mkdir -p "$target"
    df -BG "$target" | awk 'NR==2 {gsub("G",""); print $4}'
}

purge_hf_cache() {
    local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    if [[ -d "$cache_dir" ]]; then
        local before; before=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        rm -rf "$cache_dir"/models--*
        log "HF cache cleared ($before freed) — $(free_gib) GiB free"
    fi
}

sync_repos() {
    local repos=("$HOME/semantic_convergence" "$HOME/rosetta_tools")
    for repo in "${repos[@]}"; do
        [[ -d "$repo/.git" ]] || continue
        if git -C "$repo" pull --ff-only --autostash --quiet 2>/dev/null; then
            log "  pulled $(basename $repo)"
        else
            log "  ⚠ pull skipped (diverged): $(basename $repo)"
        fi
        if [[ -f "$repo/pyproject.toml" ]]; then
            pip install -q -e "$repo" 2>/dev/null && log "  reinstalled $(basename $repo)"
        fi
    done
    python -c "import rosetta_tools" 2>/dev/null || { log "rosetta_tools missing — installing"; pip install -q -e "$HOME/rosetta_tools"; }
}

get_full_description() {
    hopper task get "$1" 2>/dev/null \
        | awk '/^Description:/{found=1; next} found && /^(Tags:|Assigned|Created|Updated|Body:|$)/{exit} found{print}' \
        | sed 's/^[[:space:]]*//'
}

get_command() {
    # Strip the "### depends:..." trailer if present
    get_full_description "$1" | sed 's/ *### depends:.*$//'
}

get_deps() {
    # Extract comma-separated dep IDs from "### depends:ID1,ID2,..."
    get_full_description "$1" | grep -o '### depends:[^ ]*' | sed 's/### depends://' || true
}

task_status() {
    hopper task get "$1" 2>/dev/null | awk '/^Status:/{print $2; exit}'
}

# ---------------------------------------------------------------------------
# Dependency resolution — call after each completed task
# ---------------------------------------------------------------------------

resolve_blocked() {
    local blocked_ids
    blocked_ids=$(hopper task list --tag gpu-job --status blocked --ids-only 2>/dev/null || true)
    [[ -z "$blocked_ids" ]] && return

    while IFS= read -r tid; do
        [[ -z "$tid" ]] && continue
        local deps; deps=$(get_deps "$tid")
        [[ -z "$deps" ]] && continue

        local all_done=true
        IFS=',' read -ra dep_list <<< "$deps"
        for dep in "${dep_list[@]}"; do
            [[ -z "$dep" ]] && continue
            local s; s=$(task_status "$dep")
            if [[ "$s" != "completed" ]]; then
                all_done=false
                break
            fi
        done

        if $all_done; then
            log "Unblocking $tid (all dependencies completed)"
            hopper task status "$tid" open -f 2>/dev/null || true
        fi
    done <<< "$blocked_ids"
}

# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------

run_job() {
    local task_id="$1"
    local cmd; cmd=$(get_command "$task_id")

    if [[ -z "$cmd" ]]; then
        log "⚠ Task $task_id has no command in description — marking blocked"
        hopper task status "$task_id" blocked -f
        sync_hopper
        return
    fi

    log "Claiming $task_id"
    hopper task status "$task_id" in_progress --assign "$IDENTITY" -f
    sync_hopper

    sync_repos

    local avail; avail=$(free_gib)
    if [[ "$avail" -lt "$MIN_DISK_GIB" ]]; then
        log "Low disk (${avail} GiB) — purging HF cache before run"
        purge_hf_cache
    fi

    local ts; ts=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${ts}_${task_id}.log"

    ln -sfn "$log_file" "$LOG_DIR/current.log"  # always points to active job

    log "Command: $cmd"
    log "Log:     $log_file"

    {
        echo "=== Task:    $task_id ==="
        echo "=== Command: $cmd ==="
        echo "=== Started: $(date) ==="
        echo ""
    } > "$log_file"

    local start; start=$(date +%s)

    # Heartbeat in background — every 10 min, syncs so remote sees it
    (
        while sleep 600; do
            hopper task heartbeat "$task_id" 2>/dev/null || true
            sync_hopper
        done
    ) &
    local hb_pid=$!

    set +e
    bash -c "$cmd" >> "$log_file" 2>&1
    local exit_code=$?
    set -e

    kill "$hb_pid" 2>/dev/null || true

    local elapsed=$(( $(date +%s) - start ))
    local duration="$(( elapsed/60 ))m$(( elapsed%60 ))s"

    {
        echo ""
        echo "=== Exit: $exit_code  Duration: $duration ==="
    } >> "$log_file"

    if [[ $exit_code -eq 0 ]]; then
        log "✓ $task_id complete ($duration)"
        hopper task status "$task_id" completed -f
        sync_hopper
        resolve_blocked
        sync_hopper
    else
        log "✗ $task_id FAILED (exit=$exit_code, ${duration}) — see $log_file"
        hopper task status "$task_id" blocked -f
        hopper task update "$task_id" \
            --description "$cmd  [FAILED exit=$exit_code $duration log=$(basename $log_file)]" \
            2>/dev/null || true
        sync_hopper
    fi

    purge_hf_cache
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

log "Starting — identity: $IDENTITY"
log "Polling every ${POLL_INTERVAL}s for open gpu-job tasks"
log "Logs → $LOG_DIR"
log "Daemon log → $DAEMON_LOG  (tail -f to follow)"
echo ""

while true; do
    {
        sync_hopper

        task_id=$(hopper task list --tag gpu-job --status open --ids-only 2>/dev/null | tail -1)
        log "Next task: ${task_id:-<none>}"

        if [[ -n "$task_id" ]]; then
            run_job "$task_id" || log "⚠ run_job exited non-zero for $task_id — continuing"
        else
            log "Queue empty — sleeping ${POLL_INTERVAL}s"
            sleep "$POLL_INTERVAL"
        fi
    } || { log "⚠ loop iteration failed — sleeping ${POLL_INTERVAL}s before retry"; sleep "$POLL_INTERVAL"; }
done

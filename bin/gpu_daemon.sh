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
MAX_RETRIES=2         # auto-retry failed jobs this many times before marking blocked
RETRY_DELAY=60        # seconds before re-queuing unknown/infrastructure failures
RETRY_DELAY_FAST=15   # seconds before re-queuing transient network failures

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
    # repo_path:clone_url pairs; rosetta_analysis is a symlink → Rosetta_Analysis
    declare -A REPO_URLS=(
        ["$HOME/rosetta_tools"]="https://github.com/jamesrahenry/rosetta_tools.git"
        ["$HOME/Rosetta_Analysis"]="https://github.com/jamesrahenry/Rosetta_Analysis.git"
        ["$HOME/Rosetta_Concept_Pairs"]="https://github.com/jamesrahenry/Rosetta_Concept_Pairs.git"
        ["$HOME/Concept_Integrity_Auditor"]="https://github.com/james-henry-git/Concept_Integrity_Auditor.git"
    )

    for repo in "${!REPO_URLS[@]}"; do
        local url="${REPO_URLS[$repo]}"
        local name; name=$(basename "$repo")

        # Resolve symlink if path doesn't exist but a symlink target does
        if [[ ! -d "$repo" && -L "$repo" ]]; then
            repo=$(readlink -f "$repo")
        fi

        # Clone if missing
        if [[ ! -d "$repo/.git" ]]; then
            log "  Cloning $name from $url"
            git clone --quiet "$url" "$repo" 2>/dev/null \
                && log "  cloned $name" \
                || log "  ⚠ clone failed: $name"
        fi

        [[ -d "$repo/.git" ]] || continue

        if git -C "$repo" pull --ff-only --autostash --quiet 2>/dev/null; then
            log "  pulled $name"
        else
            log "  ⚠ pull skipped (diverged): $name"
        fi
        # Skip auto-pip-install for CIA — it owns its own uv-managed .venv
        # which the job command provisions via `uv sync`. Forcing pip install
        # into the daemon's env risks dependency conflicts and is unnecessary.
        if [[ -f "$repo/pyproject.toml" && "$name" != "Concept_Integrity_Auditor" ]]; then
            pip install -q -e "$repo" 2>/dev/null && log "  reinstalled $name"
        fi
    done

    # Ensure ~/rosetta_analysis symlink exists
    if [[ ! -e "$HOME/rosetta_analysis" && -d "$HOME/Rosetta_Analysis" ]]; then
        ln -s "$HOME/Rosetta_Analysis" "$HOME/rosetta_analysis"
        log "  created symlink rosetta_analysis → Rosetta_Analysis"
    fi

    python -c "import rosetta_tools" 2>/dev/null || { log "rosetta_tools missing — installing"; pip install -q -e "$HOME/rosetta_tools"; }
}

get_full_description() {
    # Use JSON to get the raw description — text output wraps long lines
    # which breaks grep patterns like "### depends:..."
    hopper --json task get "$1" 2>/dev/null | jq -r '.description // empty'
}

get_command() {
    # Strip all ### markers (depends, retries, …) — they are always appended at the end
    get_full_description "$1" | sed 's/ *### .*$//'
}

get_deps() {
    get_full_description "$1" | grep -o '### depends:[^ ]*' | sed 's/### depends://' || true
}

get_retry_count() {
    get_full_description "$1" | grep -o '### retries:[0-9]*' | grep -o '[0-9]*' || echo 0
}

set_retry_count() {
    local tid="$1" count="$2"
    local desc; desc=$(get_full_description "$tid")
    local new_desc
    if echo "$desc" | grep -q '### retries:'; then
        new_desc=$(echo "$desc" | sed "s/### retries:[0-9]*/### retries:$count/")
    else
        new_desc="$desc ### retries:$count"
    fi
    hopper task update "$tid" --description "$new_desc" 2>/dev/null || true
}

task_status() {
    hopper --json task get "$1" 2>/dev/null | jq -r '.status // empty'
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
# Failure classification — scan log to decide retry behaviour
# Returns: "no_retry" | "retry_fast" | "retry_slow"
# ---------------------------------------------------------------------------

classify_failure() {
    local log_file="$1"
    [[ -f "$log_file" ]] || { echo "retry_slow"; return; }

    # OOM or disk-full — needs hardware/config change, not a re-run
    if grep -q "OutOfMemoryError\|CUDA out of memory\|No space left on device" "$log_file"; then
        echo "no_retry"; return
    fi

    # Hard code/logic errors — re-running won't change the outcome
    if grep -q "SyntaxError\|ImportError\|ModuleNotFoundError\|AttributeError\|TypeError\|NameError" "$log_file"; then
        echo "no_retry"; return
    fi

    # Missing data — re-running won't produce the data
    if grep -q "FileNotFoundError\|No GEM data\|No concept pairs\|concept.*not found\|No extraction dir" "$log_file"; then
        echo "no_retry"; return
    fi

    # Transient network / download blip — retry quickly
    if grep -q "ConnectTimeout\|ConnectionResetError\|ReadTimeout\|ChunkedEncodingError\|requests\.exceptions" "$log_file"; then
        echo "retry_fast"; return
    fi

    # HF hub errors — could be rate-limit or transient 5xx, retry with delay
    if grep -q "huggingface_hub\|RepositoryNotFoundError\|EntryNotFoundError\|HTTPError" "$log_file"; then
        echo "retry_slow"; return
    fi

    # Default: unknown failure, retry with delay
    echo "retry_slow"
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
        local failure_type; failure_type=$(classify_failure "$log_file")
        local retries; retries=$(get_retry_count "$task_id")

        if [[ "$failure_type" == "no_retry" ]]; then
            log "✗ $task_id FAILED (exit=$exit_code, ${duration}, $failure_type) — not retrying — blocked — see $log_file"
            hopper task status "$task_id" blocked -f
            sync_hopper
        elif [[ "$retries" -lt "$MAX_RETRIES" ]]; then
            local next=$(( retries + 1 ))
            local delay=$RETRY_DELAY
            [[ "$failure_type" == "retry_fast" ]] && delay=$RETRY_DELAY_FAST
            log "✗ $task_id FAILED (exit=$exit_code, ${duration}, $failure_type) — retry $next/$MAX_RETRIES in ${delay}s — see $log_file"
            set_retry_count "$task_id" "$next"
            sync_hopper
            sleep "$delay"
            hopper task status "$task_id" open -f 2>/dev/null || true
            sync_hopper
        else
            log "✗ $task_id FAILED (exit=$exit_code, ${duration}, $failure_type) — exhausted $MAX_RETRIES retries — blocked — see $log_file"
            hopper task status "$task_id" blocked -f
            sync_hopper
        fi
    fi

    local avail_post; avail_post=$(free_gib)
    if [[ "$avail_post" -lt "$MIN_DISK_GIB" ]]; then
        purge_hf_cache
    else
        log "HF cache retained — ${avail_post} GiB free (threshold: ${MIN_DISK_GIB} GiB)"
    fi
}

# ---------------------------------------------------------------------------
# Startup: reclaim any in_progress tasks left by a previous daemon crash
# ---------------------------------------------------------------------------

reclaim_interrupted() {
    local ids
    ids=$(hopper task list --tag gpu-job --status in_progress --ids-only 2>/dev/null || true)
    [[ -z "$ids" ]] && return

    while IFS= read -r tid; do
        [[ -z "$tid" ]] && continue
        local assigned
        assigned=$(hopper --json task get "$tid" 2>/dev/null | jq -r '.assigned_to // empty')
        if [[ "$assigned" == "$IDENTITY" ]]; then
            log "Reclaiming interrupted task $tid → open"
            hopper task status "$tid" open -f 2>/dev/null || true
            hopper task update "$tid" --unassign 2>/dev/null || true
        fi
    done <<< "$ids"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

log "Starting — identity: $IDENTITY"
log "Polling every ${POLL_INTERVAL}s for open gpu-job tasks"
log "Logs → $LOG_DIR"
log "Daemon log → $DAEMON_LOG  (tail -f to follow)"
echo ""

sync_hopper
reclaim_interrupted
resolve_blocked
sync_hopper

while true; do
    {
        sync_hopper

        task_id=$(hopper --json task list --tag gpu-job --status open 2>/dev/null \
            | jq -r '
                map(. + {_pri: (
                    if .priority == "critical" then 0
                    elif .priority == "high"   then 1
                    elif .priority == "medium" then 2
                    else 3 end
                )})
                | sort_by([._pri, .created_at])
                | first | .id // empty
            ')
        log "Next task: ${task_id:-<none>}"

        if [[ -n "$task_id" ]]; then
            run_job "$task_id" || log "⚠ run_job exited non-zero for $task_id — continuing"
        else
            log "Queue empty — sleeping ${POLL_INTERVAL}s"
            sleep "$POLL_INTERVAL"
        fi
    } || { log "⚠ loop iteration failed — sleeping ${POLL_INTERVAL}s before retry"; sleep "$POLL_INTERVAL"; }
done

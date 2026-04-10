#!/usr/bin/env bash
# gpu_runner.sh — Sequential GPU job runner with disk cleanup and logging.
#
# Reads a job file (one command per line) and runs each sequentially.
# Cleans HF cache between jobs, logs stdout/stderr per job, tracks status.
#
# Usage:
#   # Create a jobs file
#   cat > jobs.txt << 'EOF'
#   cd ~/caz_scaling && python src/probe_shallow_deep.py --all
#   cd ~/semantic_convergence && python src/extract.py --all
#   cd ~/semantic_convergence && python src/align_depth_matched.py --all
#   EOF
#
#   # Run it (in tmux so you can detach)
#   bash gpu_runner.sh jobs.txt
#
#   # Or with options
#   bash gpu_runner.sh --no-clean jobs.txt    # skip HF cache cleanup
#   bash gpu_runner.sh --dry-run jobs.txt     # print jobs without running
#
# Each job gets:
#   - Its own log file in ./gpu_runs/<timestamp>/job_NNN.log
#   - A status file tracking pass/fail
#   - HF cache purge after completion (unless --no-clean)
#
# The runner is idempotent: if you re-run the same jobs file, completed
# jobs are skipped (based on the status file).

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
CLEAN_CACHE=true
DRY_RUN=false
JOBS_FILE=""
MIN_DISK_GIB=10   # Abort job if free disk falls below this threshold

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-clean)  CLEAN_CACHE=false; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: gpu_runner.sh [--no-clean] [--dry-run] JOBS_FILE"
            exit 0
            ;;
        *)           JOBS_FILE="$1"; shift ;;
    esac
done

if [[ -z "$JOBS_FILE" ]]; then
    echo "ERROR: No jobs file specified."
    echo "Usage: gpu_runner.sh [--no-clean] [--dry-run] JOBS_FILE"
    exit 1
fi

if [[ ! -f "$JOBS_FILE" ]]; then
    echo "ERROR: Jobs file not found: $JOBS_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
RUN_DIR="./gpu_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# Symlink gpu_runs/current → this run for easy access
ln -sfn "$(basename "$RUN_DIR")" ./gpu_runs/current

# Copy jobs file for reference
cp "$JOBS_FILE" "$RUN_DIR/jobs.txt"

# Status file: tracks which jobs completed
STATUS_FILE="$RUN_DIR/status.txt"
touch "$STATUS_FILE"

# Count jobs (skip blank lines and comments)
TOTAL=$(grep -cve '^\s*$\|^\s*#' "$JOBS_FILE" || true)

echo "=========================================="
echo "GPU Job Runner"
echo "=========================================="
echo "Jobs file:    $JOBS_FILE"
echo "Total jobs:   $TOTAL"
echo "Run dir:      $RUN_DIR"
echo "Clean cache:  $CLEAN_CACHE"
echo "Dry run:      $DRY_RUN"
echo "Started:      $(date)"
echo "=========================================="
echo ""

# Disable MLflow tracking entirely
export MLFLOW_TRACKING_URI="file:///dev/null"

# Disable HuggingFace Xet download backend (2026-04-10)
# huggingface_hub >=1.5.0 enables Xet by default.  Xet changes download
# behavior and has been observed pulling entire model repos (tflite, onnx,
# flax, rust weights) instead of just safetensors + config + tokenizer.
# Falls back to standard HTTPS downloads which respect allow_patterns.
export HF_HUB_DISABLE_XET=1

# Increase download timeout for large models on flaky connections.
# Default is 10s which can cause 0-byte failures on 3GB+ weight files.
export HF_HUB_DOWNLOAD_TIMEOUT=120

# ---------------------------------------------------------------------------
# Sync repos — pull latest code before running anything
# ---------------------------------------------------------------------------

# Repos that must exist: pull if present, clone if absent.
# Format: "local_path|remote_url"
# Remote URLs use the github-personal SSH alias (set up in ~/.ssh/config).
declare -A REPO_REMOTES=(
    ["$HOME/semantic_convergence"]=""
    ["$HOME/caz_scaling"]="git@github-personal:jamesrahenry/caz_scaling.git"
    ["$HOME/rosetta_tools"]="git@github-personal:jamesrahenry/Rosetta_Tools.git"
    ["$HOME/Rosetta_Manifold"]=""
    ["$HOME/Activation_Manifold_Cartography"]=""
    ["$HOME/Concept_Allocation_Zone"]=""
    ["$HOME/Rosetta_Concept_Pairs"]="git@github-personal:jamesrahenry/Rosetta_Concept_Pairs.git"
    ["$HOME/Rosetta_Feature_Library"]="git@github-personal:jamesrahenry/Rosetta_Feature_Library.git"
)

echo "Syncing repos..."
for repo in "${!REPO_REMOTES[@]}"; do
    remote="${REPO_REMOTES[$repo]}"
    if [[ -d "$repo/.git" ]]; then
        echo -n "  $repo: "
        if git -C "$repo" pull --ff-only --quiet 2>/dev/null; then
            echo "✓"
        else
            echo "⚠ pull failed (dirty or diverged?) — continuing with local state"
        fi
        # Reinstall if it's an editable package (rosetta_tools)
        if [[ -f "$repo/pyproject.toml" ]] && pip show "$(basename "$repo")" &>/dev/null; then
            pip install -q -e "$repo" 2>/dev/null && echo "    ↳ reinstalled editable package"
        fi
    elif [[ -n "$remote" ]]; then
        echo -n "  $repo: not found — cloning from $remote ... "
        if git clone "$remote" "$repo" 2>&1 | sed 's/^/    /'; then
            echo "  ✓ cloned"
        else
            echo "  ✗ clone failed — jobs requiring $repo will fail"
        fi
    fi
done

# Validate rosetta_tools is importable — ephemeral GPU boxes lose pip installs
# between sessions, so reinstall if needed regardless of pip show status.
if ! python -c "import rosetta_tools" 2>/dev/null; then
    echo "rosetta_tools not importable — installing..."
    pip install -q -e "$HOME/rosetta_tools"
    echo "  ↳ rosetta_tools installed"
else
    echo "rosetta_tools OK"
fi
echo ""

# ---------------------------------------------------------------------------
# Disk space helpers
# ---------------------------------------------------------------------------

free_gib() {
    # Returns available disk space in GiB on the filesystem containing $HOME
    df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $4}'
}

check_and_recover_disk() {
    local avail
    avail=$(free_gib)
    if [[ "$avail" -lt "$MIN_DISK_GIB" ]]; then
        echo "  ⚠ Low disk: ${avail} GiB free (threshold: ${MIN_DISK_GIB} GiB) — clearing HF cache..."
        local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub"
        if [[ -d "$cache_dir" ]]; then
            local before
            before=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
            rm -rf "$cache_dir"/models--*
            echo "  🗑 Cleared HF cache ($before freed)"
        fi
        avail=$(free_gib)
        if [[ "$avail" -lt "$MIN_DISK_GIB" ]]; then
            echo "  ✗ Still only ${avail} GiB free after cache clear — skipping job to avoid mid-run crash"
            return 1
        fi
        echo "  ✓ Disk recovered: ${avail} GiB free"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Run jobs
# ---------------------------------------------------------------------------
JOB_NUM=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    JOB_NUM=$((JOB_NUM + 1))
    LOG_FILE="$RUN_DIR/job_$(printf '%03d' $JOB_NUM).log"

    echo "[$JOB_NUM/$TOTAL] $(date +%H:%M:%S) ▶ $line"

    if $DRY_RUN; then
        echo "  (dry run — skipped)"
        continue
    fi

    # Re-validate rosetta_tools before each job — long-running jobs can drift
    if ! python -c "import rosetta_tools" 2>/dev/null; then
        echo "  rosetta_tools lost — reinstalling..."
        pip install -q -e "$HOME/rosetta_tools"
    fi

    # Check disk space — clear HF cache if low, skip job if still insufficient
    echo "  Disk: $(free_gib) GiB free"
    if ! check_and_recover_disk; then
        echo "$JOB_NUM|FAIL(no-disk)|0m0s|$line" >> "$STATUS_FILE"
        echo ""
        continue
    fi

    # Run the job, capturing output
    START_TIME=$(date +%s)
    echo "=== JOB $JOB_NUM: $line ===" > "$LOG_FILE"
    echo "=== Started: $(date) ===" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    set +e
    # Use bash -c so cd and && chains work
    bash -c "$line" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    echo "" >> "$LOG_FILE"
    echo "=== Finished: $(date) ===" >> "$LOG_FILE"
    echo "=== Exit code: $EXIT_CODE  Duration: ${MINUTES}m${SECONDS}s ===" >> "$LOG_FILE"

    if [[ $EXIT_CODE -eq 0 ]]; then
        STATUS="PASS"
        echo "  ✓ ${MINUTES}m${SECONDS}s"
    else
        STATUS="FAIL(exit=$EXIT_CODE)"
        echo "  ✗ FAILED (exit=$EXIT_CODE) — see $LOG_FILE"
    fi

    echo "$JOB_NUM|$STATUS|${MINUTES}m${SECONDS}s|$line" >> "$STATUS_FILE"

    # Clean HF cache between jobs to prevent disk exhaustion
    if $CLEAN_CACHE && [[ $JOB_NUM -lt $TOTAL ]]; then
        CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
        if [[ -d "$CACHE_DIR" ]]; then
            CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
            rm -rf "$CACHE_DIR"/models--*
            echo "  🗑 Cleared HF cache ($CACHE_SIZE)"
        fi
    fi

    echo ""

done < "$JOBS_FILE"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "COMPLETE — $(date)"
echo "=========================================="
echo ""

PASS_COUNT=$(grep -c "|PASS|" "$STATUS_FILE" || true)
FAIL_COUNT=$(grep -c "|FAIL" "$STATUS_FILE" || true)

echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed out of $TOTAL"
echo ""

cat "$STATUS_FILE" | while IFS='|' read -r num status duration cmd; do
    if [[ "$status" == "PASS" ]]; then
        echo "  ✓ [$num] $duration — $cmd"
    else
        echo "  ✗ [$num] $status $duration — $cmd"
    fi
done

echo ""
echo "Logs: $RUN_DIR/"
echo "Status: $STATUS_FILE"

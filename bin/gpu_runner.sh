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

# ---------------------------------------------------------------------------
# MLflow tracking server
# ---------------------------------------------------------------------------
MLFLOW_STORE="$HOME/mlflow"
MLFLOW_PORT=5111
MLFLOW_URI="http://127.0.0.1:$MLFLOW_PORT"

mkdir -p "$MLFLOW_STORE"

if command -v mlflow &>/dev/null; then
    # Check if already running
    if ! curl -s "$MLFLOW_URI/health" &>/dev/null; then
        echo "Starting MLflow server..."
        mlflow server \
            --backend-store-uri "sqlite:///$MLFLOW_STORE/mlflow.db" \
            --default-artifact-root "$MLFLOW_STORE/artifacts" \
            --host 0.0.0.0 \
            --port "$MLFLOW_PORT" \
            >> "$MLFLOW_STORE/server.log" 2>&1 &
        MLFLOW_PID=$!
        # Wait for it to come up
        for i in $(seq 1 10); do
            sleep 0.5
            curl -s "$MLFLOW_URI/health" &>/dev/null && break
        done
        echo "  MLflow UI: $MLFLOW_URI"
    else
        echo "MLflow server already running at $MLFLOW_URI"
    fi
    export MLFLOW_TRACKING_URI="$MLFLOW_URI"
    echo "  Tracking URI: $MLFLOW_TRACKING_URI"
    echo "  Tunnel:  ssh -L $MLFLOW_PORT:localhost:$MLFLOW_PORT <this-host>"
    echo ""
else
    echo "mlflow not installed — tracking disabled"
    echo ""
fi

# ---------------------------------------------------------------------------
# Sync repos — pull latest code before running anything
# ---------------------------------------------------------------------------
PROJECT_REPOS=(
    "$HOME/semantic_convergence"
    "$HOME/caz_scaling"
    "$HOME/rosetta_tools"
    "$HOME/Rosetta_Manifold"
    "$HOME/Activation_Manifold_Cartography"
    "$HOME/Concept_Assembly_Zone"
)

echo "Syncing repos..."
for repo in "${PROJECT_REPOS[@]}"; do
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

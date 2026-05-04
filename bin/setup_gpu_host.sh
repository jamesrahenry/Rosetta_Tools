#!/usr/bin/env bash
# setup_gpu_host.sh — Bootstrap a new GPU host for the Rosetta job queue.
#
# Run this script ONCE on each new GPU host. It:
#   1. Verifies / installs Hopper
#   2. Creates a .hopper/ config in the chosen instance directory
#   3. Generates a DID key for this host if one doesn't exist
#   4. Prompts for a short host alias (shown in gpu_queue.sh beside running jobs)
#   5. Clones rosetta_tools
#   6. Prints the commands to run on the dev machine to approve this host
#
# Instance directory (--instance-dir):
#   Where .hopper/ is created. Default is ~ (global ~/.hopper — hopper then
#   works from any directory with no cd needed). Pass a custom path to keep the
#   instance self-contained, e.g. ~/rosetta_queue or ~/chicken.
#
# Usage:
#   bash setup_gpu_host.sh [--instance-dir DIR] [--upstream-server URL]
#
# Written: 2026-05-04 20:30 UTC

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
UPSTREAM_SERVER="https://hopper.henrynet.ca"
ROSETTA_TOOLS_URL="https://github.com/jamesrahenry/rosetta_tools.git"
QUEUE_DIR="${HOME}/rosetta_queue"   # always used for host_alias / hopper_dir
INSTANCE_DIR="${HOME}"              # where .hopper/ lives; default = global

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-dir)    INSTANCE_DIR="$2"; shift 2 ;;
        --upstream-server) UPSTREAM_SERVER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: setup_gpu_host.sh [--instance-dir DIR] [--upstream-server URL]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Expand ~ in INSTANCE_DIR if passed as a literal string
INSTANCE_DIR="${INSTANCE_DIR/#\~/$HOME}"
HOPPER_DIR="${INSTANCE_DIR}/.hopper"

log() { echo "[setup $(date +%H:%M:%S)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

HOSTNAME="$(hostname)"

# ---------------------------------------------------------------------------
# 1. Verify Hopper is installed
# ---------------------------------------------------------------------------
if ! command -v hopper &>/dev/null; then
    log "Hopper not found — installing via pip..."
    pip install --quiet hopper-ai || die "pip install hopper-ai failed. Install manually then re-run."
fi
log "Hopper $(hopper --version 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------------------
# 2. Configure Hopper instance in INSTANCE_DIR
# ---------------------------------------------------------------------------
mkdir -p "$HOPPER_DIR"
log "Hopper instance → ${HOPPER_DIR}"

python3 - <<PYEOF
import yaml, pathlib

p = pathlib.Path("${HOPPER_DIR}") / "config.yaml"
cfg = yaml.safe_load(p.read_text()) if p.exists() else {}

cfg.setdefault("instance", {}).update({"id": "Rosetta_Program", "name": "Rosetta_Program"})
cfg.setdefault("storage",  {}).update({"type": "markdown", "path": "${HOPPER_DIR}"})
cfg.setdefault("profiles", {}).setdefault("default", {}) \
   .setdefault("upstream", {}).update({
       "server":       "${UPSTREAM_SERVER}",
       "did_key_path": "${HOPPER_DIR}/did.key",
       "enabled":      True,
   })

p.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"  wrote {p}")
PYEOF

# ---------------------------------------------------------------------------
# 3. Generate DID key if not already present
# ---------------------------------------------------------------------------
if [[ ! -f "${HOPPER_DIR}/did.key" ]]; then
    log "Generating DID key for ${HOSTNAME}..."
    (cd "$INSTANCE_DIR" && hopper upstream init)
else
    log "DID key already exists at ${HOPPER_DIR}/did.key"
fi

DID=$(cd "$INSTANCE_DIR" && hopper upstream whoami 2>/dev/null || echo "<could not read DID>")
log "This host's DID: $DID"

# ---------------------------------------------------------------------------
# 4. Short alias for this host (shown in gpu_queue.sh beside running jobs)
# ---------------------------------------------------------------------------
mkdir -p "$QUEUE_DIR"

DEFAULT_ALIAS="$(hostname | cut -d. -f1 | sed 's/-[0-9]*$//' | cut -c1-10)"
if [[ -f "${QUEUE_DIR}/host_alias" ]]; then
    EXISTING_ALIAS=$(cat "${QUEUE_DIR}/host_alias")
    log "Existing host alias: $EXISTING_ALIAS"
    read -rp "  New alias (enter to keep '$EXISTING_ALIAS'): " HOST_ALIAS
    HOST_ALIAS="${HOST_ALIAS:-$EXISTING_ALIAS}"
else
    read -rp "Short name for this host [default: $DEFAULT_ALIAS]: " HOST_ALIAS
    HOST_ALIAS="${HOST_ALIAS:-$DEFAULT_ALIAS}"
fi
echo "$HOST_ALIAS" > "${QUEUE_DIR}/host_alias"
log "Host alias: '$HOST_ALIAS'"

# Write instance dir for the daemon (only needed when non-global)
if [[ "$INSTANCE_DIR" != "$HOME" ]]; then
    echo "$INSTANCE_DIR" > "${QUEUE_DIR}/hopper_dir"
    log "Instance dir recorded → ${QUEUE_DIR}/hopper_dir"
else
    rm -f "${QUEUE_DIR}/hopper_dir"   # global config; daemon needs no cd
fi

# ---------------------------------------------------------------------------
# 5. Clone / update rosetta_tools
# ---------------------------------------------------------------------------
if [[ ! -d "${HOME}/rosetta_tools/.git" ]]; then
    log "Cloning rosetta_tools → ~/rosetta_tools ..."
    git clone --quiet "$ROSETTA_TOOLS_URL" "${HOME}/rosetta_tools"
else
    log "rosetta_tools already present — pulling ..."
    git -C "${HOME}/rosetta_tools" pull --ff-only --quiet 2>/dev/null \
        || log "  ⚠ pull skipped (diverged) — run manually if needed"
fi

pip install --quiet -e "${HOME}/rosetta_tools" \
    || log "⚠ rosetta_tools pip install failed — continue manually"

# ---------------------------------------------------------------------------
# 6. Print approval steps
# ---------------------------------------------------------------------------

# Build the cd prefix for manual hopper commands (empty if global)
if [[ "$INSTANCE_DIR" != "$HOME" ]]; then
    CD_PREFIX="cd ${INSTANCE_DIR} && "
    CD_NOTE=" (must be in ${INSTANCE_DIR})"
else
    CD_PREFIX=""
    CD_NOTE=" (works from any directory)"
fi

echo ""
echo "======================================================================"
echo "  NEXT STEPS — approve this host from the dev machine"
echo "======================================================================"
echo ""
echo "  DEV machine (~/Source/Rosetta_Program):"
echo "    hopper upstream invite create -n Rosetta_Program"
echo ""
echo "  THIS host — after receiving the token${CD_NOTE}:"
echo "    ${CD_PREFIX}hopper upstream redeem <TOKEN>"
echo "    ${CD_PREFIX}hopper sync"
echo ""
echo "  Verify:"
echo "    ${CD_PREFIX}hopper task list --tag gpu-job --compact"
echo ""
echo "======================================================================"
echo ""
echo "  START DAEMON:"
echo "    tmux new-session -d -s gpu 'bash ~/rosetta_tools/bin/gpu_daemon.sh'"
echo "    tmux attach -t gpu"
echo ""
echo "======================================================================"

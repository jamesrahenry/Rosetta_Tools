#!/usr/bin/env bash
# setup_gpu_host.sh — Bootstrap a new GPU host for the Rosetta job queue.
#
# Run this script ONCE on each new GPU host. It:
#   1. Verifies / installs Hopper
#   2. Configures the GLOBAL ~/.hopper config with the Rosetta_Program
#      instance ID and upstream server — so 'hopper' works from any directory
#   3. Generates a DID key for this host if one doesn't exist
#   4. Prompts for a short host alias (shown in gpu_queue.sh beside running jobs)
#   5. Clones rosetta_tools
#   6. Prints the commands to run on the dev machine to approve this host
#
# ~/rosetta_queue/ is created only for host_alias and job logs.
# No embedded .hopper/ — the global config handles everything.
#
# Usage:
#   bash setup_gpu_host.sh [--upstream-server URL]
#
# Written: 2026-05-04 20:15 UTC

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — override via flags
# ---------------------------------------------------------------------------
UPSTREAM_SERVER="https://hopper.henrynet.ca"
ROSETTA_TOOLS_URL="https://github.com/jamesrahenry/rosetta_tools.git"
QUEUE_DIR="${HOME}/rosetta_queue"
GLOBAL_HOPPER_DIR="${HOME}/.hopper"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upstream-server) UPSTREAM_SERVER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: setup_gpu_host.sh [--upstream-server URL]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

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
# 2. Configure global Hopper with Rosetta_Program instance + upstream server
#
# We patch ~/.hopper/config.yaml in-place so we don't clobber other fields
# the user may have set. Python is guaranteed present (hopper requires it).
# ---------------------------------------------------------------------------
mkdir -p "$GLOBAL_HOPPER_DIR"

# Ensure a base global config exists
if [[ ! -f "${GLOBAL_HOPPER_DIR}/config.yaml" ]]; then
    log "Initialising global Hopper config..."
    hopper init --non-interactive 2>/dev/null || true
fi

log "Configuring global Hopper: instance=Rosetta_Program, upstream=${UPSTREAM_SERVER}"
python3 - <<PYEOF
import yaml, pathlib, sys

p = pathlib.Path.home() / ".hopper/config.yaml"
cfg = yaml.safe_load(p.read_text()) if p.exists() else {}

# Instance
cfg.setdefault("instance", {})["id"]   = "Rosetta_Program"
cfg.setdefault("instance", {})["name"] = "Rosetta_Program"

# Upstream in the default profile
cfg.setdefault("profiles", {}).setdefault("default", {}) \
   .setdefault("upstream", {}).update({
       "server":       "${UPSTREAM_SERVER}",
       "did_key_path": str(pathlib.Path.home() / ".hopper/did.key"),
       "enabled":      True,
   })

p.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"  wrote {p}")
PYEOF

# ---------------------------------------------------------------------------
# 3. Generate DID key for this host if not already present
# ---------------------------------------------------------------------------
if [[ ! -f "${GLOBAL_HOPPER_DIR}/did.key" ]]; then
    log "Generating DID key for ${HOSTNAME}..."
    hopper upstream init
else
    log "DID key already exists at ${GLOBAL_HOPPER_DIR}/did.key"
fi

DID=$(hopper upstream whoami 2>/dev/null || echo "<could not read DID>")
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
log "Host alias: '$HOST_ALIAS' → ${QUEUE_DIR}/host_alias"

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
echo ""
echo "======================================================================"
echo "  NEXT STEPS — approve this host from the dev machine"
echo "======================================================================"
echo ""
echo "  DEV machine (~/Source/Rosetta_Program):"
echo "    hopper upstream invite create -n Rosetta_Program"
echo ""
echo "  THIS host — after receiving the token:"
echo "    hopper upstream redeem <TOKEN>"
echo "    hopper sync"
echo ""
echo "  Verify:"
echo "    hopper task list --tag gpu-job --compact"
echo ""
echo "======================================================================"
echo ""
echo "  START DAEMON:"
echo "    tmux new-session -d -s gpu 'bash ~/rosetta_tools/bin/gpu_daemon.sh'"
echo "    tmux attach -t gpu"
echo ""
echo "======================================================================"

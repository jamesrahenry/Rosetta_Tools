#!/usr/bin/env bash
# setup_gpu_host.sh — Bootstrap a new GPU host for the Rosetta job queue.
#
# Run this script ONCE on each new GPU host. It:
#   1. Verifies / installs Hopper
#   2. Creates a ~/rosetta_queue/.hopper/ project instance with the right
#      instance ID (Rosetta_Program) and upstream configured
#   3. Generates a DID key for this host if one doesn't exist
#   4. Prints the commands James must run on the DEV machine to approve this host
#   5. Clones rosetta_tools
#   6. Prints the tmux start command
#
# Usage:
#   bash setup_gpu_host.sh [--upstream-server URL]
#
# Written: 2026-05-04 18:30 UTC

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — override via flags
# ---------------------------------------------------------------------------
UPSTREAM_SERVER="https://hopper.henrynet.ca"
ROSETTA_TOOLS_URL="https://github.com/jamesrahenry/rosetta_tools.git"
QUEUE_DIR="${HOME}/rosetta_queue"
HOPPER_DIR="${QUEUE_DIR}/.hopper"

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
HOPPER_VERSION=$(hopper --version 2>/dev/null || echo "unknown")
log "Hopper version: $HOPPER_VERSION"

# ---------------------------------------------------------------------------
# 2. Create rosetta_queue project directory with embedded .hopper
# ---------------------------------------------------------------------------
log "Creating queue directory: $QUEUE_DIR"
mkdir -p "$QUEUE_DIR"

# Short name for this host — shown in gpu_queue.sh next to running jobs.
# Defaults to the first label of the hostname (strips domain + cluster suffix).
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

if [[ ! -f "${HOPPER_DIR}/config.yaml" ]]; then
    log "Initialising Hopper project at $QUEUE_DIR ..."
    # hopper init creates .hopper/ in the current directory
    (cd "$QUEUE_DIR" && hopper init --non-interactive 2>/dev/null || true)
fi

# ---------------------------------------------------------------------------
# 3. Write the project config (overwrite to ensure correct values)
# ---------------------------------------------------------------------------
mkdir -p "$HOPPER_DIR"
cat > "${HOPPER_DIR}/config.yaml" <<YAML
instance:
  id: Rosetta_Program
  name: Rosetta_Program
  scope: personal
storage:
  type: markdown
  path: ${HOPPER_DIR}
sync:
  enabled: true
  server_url: null
  sync_patterns: true
  sync_episodes: false
defaults:
  priority: medium
  status: pending
active_profile: default
profiles:
  default:
    mode: local
    local:
      path: ${HOPPER_DIR}
      auto_detect_embedded: true
    upstream:
      server: ${UPSTREAM_SERVER}
      did_key_path: ${HOPPER_DIR}/did.key
      enabled: true
YAML
log "Wrote ${HOPPER_DIR}/config.yaml (instance=Rosetta_Program, upstream=${UPSTREAM_SERVER})"

# ---------------------------------------------------------------------------
# 4. Generate DID key for this host if not already present
# ---------------------------------------------------------------------------
if [[ ! -f "${HOPPER_DIR}/did.key" ]]; then
    log "Generating DID key for ${HOSTNAME}..."
    (cd "$QUEUE_DIR" && hopper upstream init)
else
    log "DID key already exists at ${HOPPER_DIR}/did.key"
fi

DID=$(cd "$QUEUE_DIR" && hopper upstream whoami 2>/dev/null || echo "<could not read DID>")
log "This host's DID: $DID"

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

pip install --quiet -e "${HOME}/rosetta_tools" || log "⚠ rosetta_tools pip install failed — continue manually"

# ---------------------------------------------------------------------------
# 6. Print approval commands for James (dev machine)
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  NEXT STEPS — run these on the DEV machine to approve this host"
echo "======================================================================"
echo ""
echo "  cd ~/Source/Rosetta_Program"
echo ""
echo "  # Generate an invite token (single-use):"
echo "  hopper upstream invite create -n Rosetta_Program"
echo ""
echo "  # Send the token to ${HOSTNAME}, then on THIS host run:"
echo "  cd ~/rosetta_queue"
echo "  hopper upstream set-server ${UPSTREAM_SERVER}"
echo "  hopper upstream redeem <TOKEN>"
echo ""
echo "  # Verify the host is recognised:"
echo "  hopper upstream status"
echo ""
echo "  # Pull all current tasks to this host (must be in rosetta_queue):"
echo "  cd ~/rosetta_queue && hopper sync"
echo ""
echo "======================================================================"
echo ""
echo "  START DAEMON (on ${HOSTNAME}, after invite/redeem):"
echo ""
echo "  tmux new-session -d -s gpu \\"
echo "    'cd ~/rosetta_queue && bash ~/rosetta_tools/bin/gpu_daemon.sh'"
echo "  tmux attach -t gpu"
echo ""
echo "======================================================================"

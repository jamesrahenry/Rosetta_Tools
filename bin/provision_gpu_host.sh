#!/usr/bin/env bash
# provision_gpu_host.sh — DEV-MACHINE wrapper: blank GPU VM -> queue-ready in one command.
#
# Closes the four manual gaps so spinning a VM up is "at will":
#   1. Ensures ONE reusable read-only staging deploy key exists on the dev machine
#      and is registered on Rosetta_Analysis-staging (created+registered once;
#      reused on every host so rebuilds never touch GitHub — the key "sticks").
#   2. Ships that key + setup_gpu_host.sh to the VM.
#   3. Runs setup with the HF token (auto-read from the dev cache unless --hf-token).
#   4. Registers the host in the dev machine's sync_hosts.conf + ~/.ssh/config.
#
# After this, the only interactive bits left are the hopper enrol handshake and
# starting the daemon — both printed at the end.
#
# Usage:
#   bash provision_gpu_host.sh <ssh-target> --alias <name> [options]
#     <ssh-target>   an ~/.ssh/config alias (e.g. linode-2x) OR user@host
#
# Options:
#   --alias NAME     queue alias for the host (required)
#   --hf-token TOK   HF token (default: read ~/.cache/huggingface/token on dev)
#   --ip IP          results-sync IP (default: resolved from ssh -G <target>)
#   --                pass any following flags straight through to setup_gpu_host.sh
#
# Written: 2026-06-03 04:12 UTC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_gpu_host.sh"
PROGRAM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"          # ~/Source/Rosetta_Program
SYNC_CONF="${PROGRAM_DIR}/rosetta_queue/sync_hosts.conf"

DEPLOY_KEY="${HOME}/.ssh/rosetta_staging_deploy"
STAGING_REPO="jamesrahenry/Rosetta_Analysis-staging"

log()  { echo "[provision $(date +%H:%M:%S)] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

[[ $# -ge 1 ]] || die "usage: provision_gpu_host.sh <ssh-target> --alias <name> [options]"
TARGET="$1"; shift

ALIAS=""; HF_TOKEN=""; IP=""; PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --alias)    ALIAS="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --ip)       IP="$2"; shift 2 ;;
        --)         shift; PASSTHRU=("$@"); break ;;
        *) die "unknown flag: $1" ;;
    esac
done
[[ -n "$ALIAS" ]] || die "--alias is required"
[[ -f "$SETUP_SCRIPT" ]] || die "setup_gpu_host.sh not found at $SETUP_SCRIPT"

# --- 1. Ensure the reusable staging deploy key exists + is registered ---------
if [[ ! -f "$DEPLOY_KEY" ]]; then
    log "Creating reusable staging deploy key ($DEPLOY_KEY) ..."
    ssh-keygen -t ed25519 -N "" -C "rosetta-staging-deploy-ro" -f "$DEPLOY_KEY"
fi
if command -v gh &>/dev/null; then
    if ! gh repo deploy-key list --repo "$STAGING_REPO" 2>/dev/null \
         | grep -q "rosetta-staging-deploy-ro"; then
        log "Registering read-only deploy key on $STAGING_REPO ..."
        gh repo deploy-key add "${DEPLOY_KEY}.pub" --repo "$STAGING_REPO" \
           --title "rosetta-staging-deploy-ro (reusable, GPU hosts)" \
           || log "  (deploy-key add failed — may already be registered)"
    else
        log "Deploy key already registered on $STAGING_REPO"
    fi
else
    log "gh not found — ensure ${DEPLOY_KEY}.pub is a deploy key on $STAGING_REPO"
fi

# --- 2. HF token (default: dev cache) -----------------------------------------
if [[ -z "$HF_TOKEN" && -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
    log "Using HF token from dev cache (must be the personal james-ra-henry token)"
fi
[[ -n "$HF_TOKEN" ]] || log "WARN: no HF token — gated models will fail; pass --hf-token"

# --- 3. Resolve sync IP -------------------------------------------------------
if [[ -z "$IP" ]]; then
    IP="$(ssh -G "$TARGET" 2>/dev/null | awk '/^hostname /{print $2; exit}')"
fi
[[ -n "$IP" ]] || log "WARN: could not resolve IP for $TARGET — sync line not added"

SSH_OPTS="-o StrictHostKeyChecking=accept-new"

# --- 4. Ship key + script, then run setup -------------------------------------
log "Shipping deploy key + setup script to $TARGET ..."
ssh $SSH_OPTS "$TARGET" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
scp $SSH_OPTS "$DEPLOY_KEY"   "$TARGET:~/.ssh/id_ed25519_staging"
scp $SSH_OPTS "$SETUP_SCRIPT" "$TARGET:~/setup_gpu_host.sh"

log "Running setup_gpu_host.sh on $TARGET ..."
ssh $SSH_OPTS "$TARGET" \
    "bash ~/setup_gpu_host.sh --alias '$ALIAS' --hf-token '$HF_TOKEN' \
     --staging-key ~/.ssh/id_ed25519_staging ${PASSTHRU[*]:-}"

# --- 5. Register host in dev sync_hosts.conf ----------------------------------
if [[ -n "$IP" ]]; then
    line="${ALIAS}  root@${IP}:~/rosetta_data/"
    if ! grep -q "^${ALIAS}[[:space:]]" "$SYNC_CONF" 2>/dev/null; then
        echo "$line" >> "$SYNC_CONF"
        log "Added to sync_hosts.conf: $line"
    else
        log "sync_hosts.conf already has '$ALIAS'"
    fi
fi

cat <<DONE

======================================================================
  $ALIAS provisioned. Two interactive steps remain:
======================================================================
  1. Hopper enrol:
       (dev)    hopper upstream invite create -n Rosetta_Program
       (target) ssh $TARGET 'hopper upstream redeem --server https://hopper.henrynet.ca <TOKEN> && hopper sync'
  2. Start daemon:
       ssh $TARGET "tmux new-session -d -s gpu 'bash ~/rosetta_tools/bin/gpu_daemon.sh'"
======================================================================
DONE

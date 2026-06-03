#!/usr/bin/env bash
# setup_gpu_host.sh — Bootstrap a new GPU host for the Rosetta job queue.
#
# Tested on Ubuntu 24.04 + NVIDIA RTX 4000 Ada (driver 595, CUDA 13.2).
# Assumes a clean root environment (Linode/Akamai GPU nodes, do-droplet, etc.)
#
# What it does:
#   1. Installs NVIDIA driver if needed (detects recommended via ubuntu-drivers)
#   2. Creates ~/venv (Python 3.12) — Ubuntu 24.04 enforces PEP 668
#   3. Installs PyTorch cu128, core ML stack, huggingface_hub, hf_transfer
#   4. Installs the reusable staging deploy key so the daemon pulls private
#      staging UNATTENDED (no SSH agent forwarding) — keys "stick" across rebuilds
#   5. Caches the HF token to ~/.cache/huggingface/token (gated models)
#   6. Clones rosetta_tools (public) + rosetta_analysis (private STAGING)
#   7. Installs hopper from source (github.com/apathy-ca/hopper)
#   8. Configures hopper instance + generates DID key
#   9. Writes host alias and rosetta_queue dir
#  10. Prints approval steps for the dev machine
#
# Usage:
#   # Easiest: drive it from the dev machine with the wrapper, which scp's the
#   # deploy key + this script and passes the HF token automatically:
#   bash ~/Source/Rosetta_Program/rosetta_tools/bin/provision_gpu_host.sh <ssh-host> --alias <name>
#
#   # Or run directly on the host (after scp'ing this script + the staging key):
#   scp ~/.ssh/rosetta_staging_deploy root@<host>:~/.ssh/id_ed25519_staging
#   scp ~/Source/Rosetta_Program/rosetta_tools/bin/setup_gpu_host.sh root@<host>:~
#   ssh root@<host> "bash ~/setup_gpu_host.sh --alias <name> --hf-token hf_xxx \
#                    --staging-key ~/.ssh/id_ed25519_staging"
#
# Options:
#   --alias NAME           Short host alias shown in gpu_queue.sh (required unless
#                          host_alias already exists in ~/rosetta_queue/)
#   --hf-token TOKEN       Personal james-ra-henry HF token (NOT the TELUS token).
#                          Cached to ~/.cache/huggingface/token for the daemon.
#   --staging-key PATH     Private deploy key for private staging pulls. Installed
#                          to ~/.ssh/id_ed25519_staging. Defaults to that path if a
#                          key was already scp'd there.
#   --upstream-server URL  Hopper server (default: https://hopper.henrynet.ca)
#   --skip-driver          Skip NVIDIA driver detection/install (already installed)
#   --skip-torch           Skip PyTorch install (already in venv)
#   --skip-repos           Skip rosetta_tools/rosetta_analysis clone steps
#
# Updated: 2026-06-03 04:10 UTC

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
UPSTREAM_SERVER="https://hopper.henrynet.ca"
ROSETTA_TOOLS_URL="https://github.com/jamesrahenry/rosetta_tools.git"
# Private STAGING (not public main): the daemon pulls staging via the deploy key
# installed below. Public main is the released baseline; net-new code lands here.
ROSETTA_ANALYSIS_URL="git@github.com:jamesrahenry/Rosetta_Analysis-staging.git"
HOPPER_SRC_URL="https://github.com/apathy-ca/hopper.git"
QUEUE_DIR="${HOME}/rosetta_queue"
STAGING_KEY_DST="${HOME}/.ssh/id_ed25519_staging"
HOST_ALIAS=""
HF_TOKEN=""
STAGING_KEY=""
SKIP_DRIVER=0
SKIP_TORCH=0
SKIP_REPOS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alias)           HOST_ALIAS="$2"; shift 2 ;;
        --hf-token)        HF_TOKEN="$2"; shift 2 ;;
        --staging-key)     STAGING_KEY="$2"; shift 2 ;;
        --upstream-server) UPSTREAM_SERVER="$2"; shift 2 ;;
        --skip-driver)     SKIP_DRIVER=1; shift ;;
        --skip-torch)      SKIP_TORCH=1; shift ;;
        --skip-repos)      SKIP_REPOS=1; shift ;;
        -h|--help)
            echo "Usage: setup_gpu_host.sh [--alias NAME] [--hf-token TOKEN]"
            echo "       [--staging-key PATH] [--upstream-server URL]"
            echo "       [--skip-driver] [--skip-torch] [--skip-repos]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

HOPPER_DIR="${HOME}/.hopper"

log()  { echo "[setup $(date +%H:%M:%S)] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN ] $*" >&2; }

HOSTNAME_SHORT="$(hostname | cut -d. -f1)"

# ---------------------------------------------------------------------------
# 1. NVIDIA driver (Ubuntu 24.04 — skip if already loaded)
# ---------------------------------------------------------------------------
if [[ $SKIP_DRIVER -eq 0 ]]; then
    if nvidia-smi &>/dev/null; then
        log "NVIDIA driver already loaded: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    else
        log "NVIDIA driver not loaded — checking for GPU ..."
        if lspci | grep -qi nvidia; then
            log "NVIDIA GPU found. Installing ubuntu-drivers-common ..."
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common
            RECOMMENDED=$(ubuntu-drivers devices 2>/dev/null \
                | grep recommended | grep -oP 'nvidia-driver-\S+' | head -1)
            if [[ -n "$RECOMMENDED" ]]; then
                log "Installing $RECOMMENDED ..."
                DEBIAN_FRONTEND=noninteractive apt-get install -y "$RECOMMENDED"
                log "Driver installed. You may need to reboot and re-run this script."
                log "After reboot: bash $0 --skip-driver $*"
                exit 0
            else
                warn "No recommended driver found — install manually then re-run with --skip-driver"
            fi
        else
            log "No NVIDIA GPU detected — skipping driver install"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 2. Python venv  (Ubuntu 24.04 enforces PEP 668; never pip install --break-system-packages)
# ---------------------------------------------------------------------------
if [[ ! -d "${HOME}/venv" ]]; then
    log "Creating Python venv at ~/venv ..."
    apt-get install -y python3-pip python3-dev python3-venv -qq 2>/dev/null || true
    python3 -m venv "${HOME}/venv"
fi
source "${HOME}/venv/bin/activate"
grep -q 'source ~/venv/bin/activate' ~/.bashrc 2>/dev/null \
    || echo 'source ~/venv/bin/activate' >> ~/.bashrc
log "venv: $(python --version)"

# ---------------------------------------------------------------------------
# 3. PyTorch + core ML stack
# ---------------------------------------------------------------------------
if [[ $SKIP_TORCH -eq 0 ]]; then
    log "Installing PyTorch cu128 + core ML deps ..."
    pip install --quiet --upgrade pip
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install --quiet \
        numpy scipy scikit-learn tqdm \
        transformers accelerate safetensors \
        huggingface_hub hf_transfer sentencepiece protobuf \
        datasets
    pip install --quiet "nvidia-cusparselt-cu12>=0.6" \
        || warn "nvidia-cusparselt-cu12 install failed — some CKA ops may error"
    CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    log "PyTorch $(python -c 'import torch; print(torch.__version__)') — CUDA: $CUDA_OK"
fi

# ---------------------------------------------------------------------------
# 4. Staging deploy key — makes UNATTENDED private pulls "stick"
#    Without this the daemon (running in tmux, no SSH agent) cannot pull the
#    private Rosetta_Analysis-staging repo. The key is the SAME reusable
#    read-only deploy key on every host, so rebuilds never touch GitHub.
# ---------------------------------------------------------------------------
mkdir -p "${HOME}/.ssh"; chmod 700 "${HOME}/.ssh"
# If --staging-key points at a separate path, install it to the canonical dst.
if [[ -n "$STAGING_KEY" && "$STAGING_KEY" != "$STAGING_KEY_DST" ]]; then
    cp "$STAGING_KEY" "$STAGING_KEY_DST"
fi
if [[ -f "$STAGING_KEY_DST" ]]; then
    chmod 600 "$STAGING_KEY_DST"
    # github.com over SSH → use ONLY this key (IdentitiesOnly), and ignore any
    # forwarded agent (IdentityAgent none) so a stray agent key can't shadow it.
    if ! grep -q 'id_ed25519_staging' "${HOME}/.ssh/config" 2>/dev/null; then
        cat >> "${HOME}/.ssh/config" <<SSHCFG

Host github.com
  IdentityFile ~/.ssh/id_ed25519_staging
  IdentitiesOnly yes
  IdentityAgent none
SSHCFG
    fi
    ssh-keyscan -t ed25519 github.com >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
    sort -u "${HOME}/.ssh/known_hosts" -o "${HOME}/.ssh/known_hosts" 2>/dev/null || true
    log "Staging deploy key installed → unattended staging pulls enabled"
else
    warn "No staging key at $STAGING_KEY_DST — private staging pulls will FAIL."
    warn "  scp ~/.ssh/rosetta_staging_deploy root@<host>:$STAGING_KEY_DST  (or pass --staging-key)"
fi

# ---------------------------------------------------------------------------
# 5. Hugging Face token — REQUIRED for gated models (gemma/llama).
#    Must live at ~/.cache/huggingface/token; env/.bashrc/tmux don't reach the
#    daemon, so a missing token => gated downloads fail SILENTLY ("completed"
#    with no output). Use the personal james-ra-henry token, NOT the TELUS one.
# ---------------------------------------------------------------------------
if [[ -n "$HF_TOKEN" ]]; then
    mkdir -p "${HOME}/.cache/huggingface"
    printf '%s' "$HF_TOKEN" > "${HOME}/.cache/huggingface/token"
    chmod 600 "${HOME}/.cache/huggingface/token"
    log "HF token cached → gated model downloads enabled"
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    log "HF token already present at ~/.cache/huggingface/token"
else
    warn "No HF token — gated models (gemma/llama) will fail silently."
    warn "  re-run with --hf-token hf_xxx  (personal james-ra-henry token)"
fi

# ---------------------------------------------------------------------------
# 6. Repos
# ---------------------------------------------------------------------------
if [[ $SKIP_REPOS -eq 0 ]]; then
    # rosetta_tools (public HTTPS)
    if [[ ! -d "${HOME}/rosetta_tools/.git" ]]; then
        log "Cloning rosetta_tools ..."
        git clone --quiet "$ROSETTA_TOOLS_URL" "${HOME}/rosetta_tools"
    else
        log "rosetta_tools present — pulling ..."
        git -C "${HOME}/rosetta_tools" pull --ff-only --quiet 2>/dev/null \
            || warn "rosetta_tools pull skipped (diverged)"
    fi
    pip install --quiet -e "${HOME}/rosetta_tools" \
        || warn "rosetta_tools pip install failed"

    # rosetta_analysis (private STAGING via deploy key installed above)
    if [[ ! -d "${HOME}/rosetta_analysis/.git" && ! -d "${HOME}/Rosetta_Analysis/.git" ]]; then
        log "Cloning rosetta_analysis (private staging) ..."
        git clone --quiet "$ROSETTA_ANALYSIS_URL" "${HOME}/Rosetta_Analysis" \
            || warn "rosetta_analysis clone failed — check staging deploy key"
    else
        ra_dir="${HOME}/Rosetta_Analysis"; [[ -d "${HOME}/rosetta_analysis/.git" ]] && ra_dir="${HOME}/rosetta_analysis"
        log "rosetta_analysis present — repointing origin to staging + pulling ..."
        git -C "$ra_dir" remote set-url origin "$ROSETTA_ANALYSIS_URL" 2>/dev/null || true
        git -C "$ra_dir" pull --ff-only --quiet 2>/dev/null \
            || warn "rosetta_analysis pull skipped"
    fi
fi

# ---------------------------------------------------------------------------
# 7. Hopper from source
# ---------------------------------------------------------------------------
if ! command -v hopper &>/dev/null; then
    log "Cloning hopper from source (github.com/apathy-ca/hopper) ..."
    if [[ ! -d "${HOME}/hopper/.git" ]]; then
        git clone --quiet "$HOPPER_SRC_URL" "${HOME}/hopper"
    fi
    pip install --quiet -e "${HOME}/hopper" \
        || die "hopper install failed — check ${HOME}/hopper for errors"
fi
log "Hopper: $(hopper --version 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------------------
# 8. Hopper instance config + DID key
# ---------------------------------------------------------------------------
mkdir -p "$HOPPER_DIR"

python3 - <<PYEOF
import yaml, pathlib

p = pathlib.Path("${HOPPER_DIR}") / "config.yaml"
cfg = yaml.safe_load(p.read_text()) if p.exists() else {}

cfg.setdefault("instance", {}).update({"id": "Rosetta_Program", "name": "Rosetta_Program"})
cfg.setdefault("storage",  {}).update({"type": "markdown", "path": "${HOPPER_DIR}"})
profile = cfg.setdefault("profiles", {}).setdefault("default", {})
# Force local mode — fresh installs default to 'server' which breaks hopper sync.
profile["mode"] = "local"
profile.setdefault("upstream", {}).update({
    "server":       "${UPSTREAM_SERVER}",
    "did_key_path": "${HOPPER_DIR}/did.key",
    "enabled":      True,
})

p.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"  wrote {p}")
PYEOF

if [[ ! -f "${HOPPER_DIR}/did.key" ]]; then
    log "Generating DID key for ${HOSTNAME_SHORT} ..."
    hopper upstream init
else
    log "DID key already present at ${HOPPER_DIR}/did.key"
fi

DID=$(hopper upstream whoami 2>/dev/null || echo "<could not read DID>")
log "This host's DID: $DID"

# ---------------------------------------------------------------------------
# 9. Host alias + rosetta_queue dir
# ---------------------------------------------------------------------------
mkdir -p "$QUEUE_DIR"

if [[ -z "$HOST_ALIAS" ]]; then
    if [[ -f "${QUEUE_DIR}/host_alias" ]]; then
        HOST_ALIAS=$(cat "${QUEUE_DIR}/host_alias")
        log "Using existing host alias: $HOST_ALIAS"
    else
        HOST_ALIAS="${HOSTNAME_SHORT}"
        log "No --alias provided; defaulting to '$HOST_ALIAS'"
    fi
fi
echo "$HOST_ALIAS" > "${QUEUE_DIR}/host_alias"
log "Host alias: '$HOST_ALIAS'"

# ---------------------------------------------------------------------------
# 10. Data dirs
# ---------------------------------------------------------------------------
mkdir -p "${HOME}/rosetta_data/"{models,results,paper_n250}
mkdir -p "${HOME}/gpu_runs"

# ---------------------------------------------------------------------------
# 11. Print next steps
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "  AUTOMATED: driver, venv, torch, staging deploy key, HF token,"
echo "             repos (analysis on STAGING), hopper, DID, alias, data dirs"
echo "======================================================================"
echo "  REMAINING (one interactive handshake + start daemon):"
echo ""
echo "  1. Hopper enrol — create invite on DEV, redeem here:"
echo "       (dev)  hopper upstream invite create -n Rosetta_Program"
echo "       (here) hopper upstream redeem --server ${UPSTREAM_SERVER} <TOKEN> && hopper sync"
echo ""
echo "  2. Start daemon in tmux (self-activates venv):"
echo "       tmux new-session -d -s gpu 'bash ~/rosetta_tools/bin/gpu_daemon.sh'"
echo ""
echo "  NOTE: the provision_gpu_host.sh wrapper registers this host in the dev"
echo "  machine's sync_hosts.conf + ~/.ssh/config automatically. If you ran this"
echo "  script directly, add the results sync line yourself:"
echo "       echo '${HOST_ALIAS}  root@<HOST_IP>:~/rosetta_data/' \\"
echo "           >> ~/Source/Rosetta_Program/rosetta_queue/sync_hosts.conf"
echo "======================================================================"

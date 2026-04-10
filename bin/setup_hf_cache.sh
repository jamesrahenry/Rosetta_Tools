#!/usr/bin/env bash
# setup_hf_cache.sh — Split HF model cache across /home and /tmp.
#
# Problem: GPU box has 62G on /home, 48G on /tmp, and ~235G of models.
# We can't cache everything, but we can:
#   1. Use /home/coder/hf_cache as the persistent HF_HOME (survives reboots)
#   2. Symlink small model dirs (≤2GB) to /tmp for speed + to save /home space
#   3. Let the runner's disk pressure cleanup handle the rest
#
# Run once on the GPU box, then gpu_runner.sh picks up HF_HOME automatically.
#
# Usage:
#   bash ~/rosetta_tools/bin/setup_hf_cache.sh
#
# 2026-04-10 UTC

set -euo pipefail

HF_PERSISTENT="$HOME/hf_cache"
HF_VOLATILE="/tmp/hf_cache_models"

echo "Setting up split HF cache..."
echo "  Persistent: $HF_PERSISTENT (on /home, survives reboot)"
echo "  Volatile:   $HF_VOLATILE (on /tmp, fast but lost on reboot)"

# Create directories
mkdir -p "$HF_PERSISTENT/hub"
mkdir -p "$HF_VOLATILE"

# Small models (≤2GB) — cheap to re-download, put on /tmp
SMALL_MODELS=(
    "models--EleutherAI--pythia-70m"
    "models--EleutherAI--pythia-160m"
    "models--EleutherAI--pythia-410m"
    "models--Qwen--Qwen2.5-0.5B"
    "models--Qwen--Qwen2.5-0.5B-Instruct"
    "models--facebook--opt-125m"
    "models--facebook--opt-350m"
    "models--openai-community--gpt2"
    "models--openai-community--gpt2-medium"
)

echo ""
echo "Creating symlinks for small models → /tmp..."
for model_dir in "${SMALL_MODELS[@]}"; do
    target="$HF_VOLATILE/$model_dir"
    link="$HF_PERSISTENT/hub/$model_dir"

    # If real directory exists, move it to /tmp
    if [[ -d "$link" && ! -L "$link" ]]; then
        echo "  Moving $model_dir → /tmp"
        mv "$link" "$target"
    else
        mkdir -p "$target"
    fi

    # Create symlink if not already there
    if [[ ! -L "$link" ]]; then
        ln -sfn "$target" "$link"
        echo "  ✓ $model_dir → /tmp"
    else
        echo "  ✓ $model_dir (already linked)"
    fi
done

# Write HF_HOME to profile so all processes pick it up
PROFILE_LINE="export HF_HOME=\"$HF_PERSISTENT\""
if ! grep -q "HF_HOME" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# HF model cache — persistent on /home, small models symlinked to /tmp" >> "$HOME/.bashrc"
    echo "$PROFILE_LINE" >> "$HOME/.bashrc"
    echo ""
    echo "Added HF_HOME to ~/.bashrc"
else
    echo ""
    echo "HF_HOME already in ~/.bashrc — verify it points to $HF_PERSISTENT"
fi

echo ""
echo "Done. Disk layout:"
echo "  /home models: large (>2GB), persistent, ~62G available"
echo "  /tmp models:  small (≤2GB), volatile (~8G total), re-download on reboot"
echo ""
echo "NOTE: If /home fills up, the gpu_runner's disk pressure check will"
echo "purge the largest cached models automatically (threshold: 10G free)."
echo ""
echo "To activate now:  export HF_HOME=$HF_PERSISTENT"

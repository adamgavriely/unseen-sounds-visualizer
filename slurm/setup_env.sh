#!/bin/bash
# One-time environment build on the BIU cluster login node.
#   bash slurm/setup_env.sh
# Creates conda env "msproj" with CUDA torch + the GPU backends (diffusers, Qwen2.5-VL).
set -euo pipefail

ENV_NAME=msproj
PY_VER=3.11

# --- locate conda (miniconda in $HOME, or a module) ---------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v module >/dev/null 2>&1 && module load miniconda 2>/dev/null; then
    source "$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
else
    echo "conda not found. Installing miniconda to \$HOME/miniconda3 ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$HOME/miniconda3"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# --- create / reuse the env ---------------------------------------------------
# conda-forge only, with --override-channels: Anaconda's own default channels
# (repo.anaconda.com/pkgs/main and /pkgs/r) now require accepting a commercial
# Terms of Service, which an academic install has no reason to agree to. Everything
# needed here is on conda-forge.
conda config --system --add channels conda-forge 2>/dev/null || true
conda config --system --remove channels defaults 2>/dev/null || true

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" "python=${PY_VER}"         -c conda-forge --override-channels
fi
conda activate "$ENV_NAME"

# ffmpeg (Stage 1) — no root needed via conda-forge
conda install -y -c conda-forge --override-channels ffmpeg

# --- python deps --------------------------------------------------------------
pip install --upgrade pip
# CUDA torch (cu121 works on L4 / A100 / H200 driver stacks)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
# GPU-only backends (commented out in requirements.txt for the CPU laptop)
pip install "accelerate>=0.34.0" "diffusers>=0.30.0" qwen-vl-utils safetensors sentencepiece

mkdir -p logs
echo
echo "env '${ENV_NAME}' ready. Verify with:  sbatch slurm/job_smoke.sh"

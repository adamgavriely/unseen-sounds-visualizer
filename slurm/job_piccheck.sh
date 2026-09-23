#!/bin/bash
#SBATCH --job-name=piccheck
#SBATCH --output=logs/pic_%j.out
#SBATCH --error=logs/pic_%j.err
#SBATCH --partition=A100-4h,L40s-4h,L4-12h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Does the generated picture actually SHOW what its label says? The per-sound metric scores the
# label and the moment, never the image, so this is the only test of the generator itself -- and it
# is what decides whether a different image stack (ComfyUI, a different model) would buy anything.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi
python benchmark/gold/error_taxonomy.py --tag ${TAG:-v4b4} --pictures

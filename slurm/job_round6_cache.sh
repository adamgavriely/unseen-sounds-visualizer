#!/bin/bash
#SBATCH --job-name=r6cache
#SBATCH --output=logs/r6cache_%j.out
#SBATCH --error=logs/r6cache_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Detector round 6 (docs/prereg_round6_dasm.md, 2026-09-28): MGA-CLAP text embeddings of the 215 FlexSED queries, then
# DASM frame scores on the 280 (calib) and the 415 (heldout) into <WIN>/dasm_cache (new folders; nothing overwritten).
# Weights were fetched on the login node (official sources), so the node runs offline.
#   sbatch slurm/job_round6_cache.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in ${STEPS:-embed calib heldout}; do
  if [ "$st" = embed ]; then
    python benchmark/detector_round6.py embed
  else
    python benchmark/detector_round6.py cache --set "$st"
  fi
done
echo "DONE round6 cache"

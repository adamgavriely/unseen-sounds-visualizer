#!/bin/bash
#SBATCH --job-name=devcand_c
#SBATCH --output=logs/devcand_c_%j.out
#SBATCH --error=logs/devcand_c_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# DEV check of the detector candidates (docs/dev_candidates_check_2026-09-28.md), report only, DEV only.
# Caches on the 49 DEV audio.wav files: EAT-large (round 5), DASM (round 6; weights fetched on the login node from the
# official HF repo CPF2/detect_any_sound into ~/Transformer4SED), I6 scene-prior answers (Qwen3.8-27B, round 8).
#   sbatch slurm/job_devcand_caches.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in ${STEPS:-eat dasm vlm}; do
  python benchmark/gold/dev_candidates_check.py "$st"
done
echo "DONE devcand caches"

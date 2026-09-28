#!/bin/bash
#SBATCH --job-name=devcand_s
#SBATCH --output=logs/devcand_s_%j.out
#SBATCH --error=logs/devcand_s_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# DEV check of the detector candidates (docs/dev_candidates_check_2026-09-28.md), report only, DEV only.
# Stage 4 of every arm (gate D0; occlusion onsets only for new spans), stage 5 of every arm and system on the H200 (the
# scored run's card class; gate answers of the scored run reused, other questions memoised), then the score (gate D5,
# table, bootstrap, Holm, ship rule) -> benchmark/gold/dev_candidates_check.json. Resumable (per clip).
#   sbatch --dependency=afterok:<caches job> slurm/job_devcand_stages.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in ${STEPS:-stage4 stage5 score}; do
  python benchmark/gold/dev_candidates_check.py "$st"
done
echo "DONE devcand stages"

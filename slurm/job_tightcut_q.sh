#!/bin/bash
#SBATCH --job-name=tcq
#SBATCH --output=logs/tcq_%A_%a.out
#SBATCH --error=logs/tcq_%A_%a.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#
# Round 58 TIGHT-CUT (docs/prereg_round13_detector_push.md): Qwen3-Omni V4 asks. Task 0 = held-out 415 subsample (padded +
# tight), task 1 = DEV / DEV2 refused band items (tight). Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then python benchmark/gold/tightcut.py held q; else python benchmark/gold/tightcut.py devask q; fi
echo "DONE tcq $SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=ctxq
#SBATCH --output=logs/ctxq_%A_%a.out
#SBATCH --error=logs/ctxq_%A_%a.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#
# Round 59 CONTEXT (docs/prereg_round13_detector_push.md): Qwen3-Omni +-5 s audio + video context asks. Tasks 0/1 = held-out
# 415 (Round 58 sample, two halves), task 2 = DEV, task 3 = DEV2 refused candidates. Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$SLURM_ARRAY_TASK_ID" = "2" ]; then python benchmark/gold/context.py ask dev
elif [ "$SLURM_ARRAY_TASK_ID" = "3" ]; then python benchmark/gold/context.py ask dev2
else python benchmark/gold/context.py ask 415 "$SLURM_ARRAY_TASK_ID"; fi
echo "DONE ctxq $SLURM_ARRAY_TASK_ID"

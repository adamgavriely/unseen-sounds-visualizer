#!/bin/bash
#SBATCH --job-name=judge_pp
#SBATCH --output=logs/judgepp_%j.out
#SBATCH --error=logs/judgepp_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Per-picture judge PP-1, answers phase (docs/prereg_per_picture_judge.md). ARGS passed through.
#   ARGS="--tags v4b4 dev_monocap_v31 --bench-fresh" sbatch slurm/job_judge_pp.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="${PY:-$HOME/venvs/judge/bin/python}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" benchmark/gold/judge_per_picture.py answers ${ARGS:-}
echo "DONE"

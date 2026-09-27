#!/bin/bash
#SBATCH --job-name=repaint
#SBATCH --output=logs/repaint_%j.out
#SBATCH --error=logs/repaint_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Repaint a scored render with the shipped picture setup (scripts/repaint_shipped.py): same sounds and times,
# V3.1 subjects (Qwen3.8 VLM, then unloaded) and Qwen-Image-2512 pictures. Display only, never scored.
#   TAG=dev_monocap_v31 sbatch slurm/job_repaint_shipped.sh
#   TAG=test_final_v33 SHARD=0 OF=2 sbatch slurm/job_repaint_shipped.sh
#   TAG=dev_monocap_v31 LIMIT=2 sbatch slurm/job_repaint_shipped.sh      (smoke test)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/repaint_shipped.py --tag "${TAG:?TAG required}" --shard "${SHARD:-0}" --of "${OF:-1}" \
    ${LIMIT:+--limit "$LIMIT"} ${CLIPS:+--clips $CLIPS} ${PHASE:+--phase "$PHASE"}
echo "DONE repaint $TAG ${SHARD:-0}/${OF:-1}"

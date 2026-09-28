#!/bin/bash
#SBATCH --job-name=repaint_v
#SBATCH --output=logs/repaintv_%j.out
#SBATCH --error=logs/repaintv_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# The shipped repaint with the picture check on (scripts/repaint_shipped.py --verify): Qwen-Image-2512 and the
# Qwen3.8-27B checker resident together (~115 GB, one H200), EasyOCR; into data/work/shipped_v_<TAG>/.
#   TAG=dev_monocap_v31 sbatch slurm/job_repaint_verify.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/repaint_shipped.py --verify --phase draw --tag "${TAG:?TAG required}" --shard "${SHARD:-0}" \
    --of "${OF:-1}" ${LIMIT:+--limit "$LIMIT"} ${CLIPS:+--clips $CLIPS}
echo "DONE repaint_verify $TAG ${SHARD:-0}/${OF:-1}"

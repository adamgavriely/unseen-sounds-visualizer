#!/bin/bash
#SBATCH --job-name=redo_pic
#SBATCH --output=logs/redopic_%j.out
#SBATCH --error=logs/redopic_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Redraw only the listed shipped pictures (scripts/repaint_shipped.py --verify --only), 28 Sept: PICTURE_MAKER and/or
# PICTURE_LOOK_VLM, into data/work/<DEST>_<TAG>/.
#   TAG=sliceB_v32 ONLY="clip:1 clip2:0" FLAGS="--maker --look-vlm" DEST=shipped_v2 sbatch slurm/job_redo_pictures.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/repaint_shipped.py --verify --tag "${TAG:?TAG required}" --only ${ONLY:?ONLY required} \
    --dest "${DEST:-shipped_v2}" ${FLAGS:-}
echo "DONE redo_pictures $TAG"

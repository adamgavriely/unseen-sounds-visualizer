#!/bin/bash
#SBATCH --job-name=trailmedia
#SBATCH --output=logs/trailmedia_%j.out
#SBATCH --error=logs/trailmedia_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Decision Inspector videos of the frozen D' (benchmark/gold/render_trail_media.py): Qwen-Image pictures + picture check
# (one H200: generator and checker VLM together) + composite, merged DEV and merged TEST, one shard per job.
# Submit from ~/MscProj_tg:  SHARD=0/4 sbatch --export=ALL slurm/job_trail_media.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/render_trail_media.py --arm "${ARM:-SHIP8+MD3+WW5+SL}" --shard "${SHARD:-0/1}" ${EXTRA:-}
echo "DONE trailmedia $SHARD"

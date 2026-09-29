#!/bin/bash
#SBATCH --job-name=r14xq
#SBATCH --output=logs/r14xq_%j.out
#SBATCH --error=logs/r14xq_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 14 amendment D on DEV (docs/prereg_round13_detector_push.md): the extra FlexSED queries (group a) as arms XQ,
# XQ + R13-1 and <best round-14 listener arm> + XQ, given in XQ_ARMS by the caller (sbatch --export=ALL,XQ_ARMS="...").
# GPU for the occlusion onsets of BEATs spans whose start the new FlexSED twins moved, and for stage 5. Resumable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "arms: $XQ_ARMS"
python benchmark/gold/round13_dev.py stage4 --arms $XQ_ARMS
python benchmark/gold/round13_dev.py stage5 --arms $XQ_ARMS
python benchmark/gold/round13_dev.py score
echo "DONE r14xq"

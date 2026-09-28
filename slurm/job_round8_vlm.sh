#!/bin/bash
#SBATCH --job-name=r8vlm
#SBATCH --output=logs/r8vlm_%j.out
#SBATCH --error=logs/r8vlm_%j.err
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#
# Detector round 8, cell I6: Qwen3.8-27B (thinking off, greedy) lists the plausible families for each clip from 4 frames,
# into benchmark/audioset_<set>_windows/round8_vlm.json (resumable: done clips are skipped).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for s in ${SETS:-calib heldout}; do
  python benchmark/detector_round8.py vlm --set $s
done
echo "DONE round8 vlm"

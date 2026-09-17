#!/bin/bash
#SBATCH --job-name=flam_det
#SBATCH --output=logs/flam_%j.out
#SBATCH --error=logs/flam_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,L4-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# FLAM as the detector (benchmark/flam_detector.py, docs/prereg_flam.md): score DCASE gold and
# the dev clips with the 527 AudioSet names as queries, then the declared evaluation.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
conda activate sota   && python -m benchmark.flam_detector --cache dcase --cache dev; conda deactivate
conda activate msproj && python -m benchmark.flam_detector --eval
echo DONE

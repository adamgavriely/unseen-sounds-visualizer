#!/bin/bash
#SBATCH --job-name=flam_v2
#SBATCH --output=logs/flam_v2_%j.out
#SBATCH --error=logs/flam_v2_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# FLAM, second attempt (benchmark/flam_v2.py, docs/prereg_v4.md section 4): descriptive
# vocabulary, per-query calibration on DCASE dev-train-tau, the five declared bars on
# dev-test-tau and the dev detections. Run `python -m benchmark.flam_v2 --fetch` on the
# login node first (the compute nodes have no internet).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
conda activate sota   && python -m benchmark.flam_v2 --cache; conda deactivate
conda activate msproj && python -m benchmark.flam_v2 --calibrate --eval
echo DONE

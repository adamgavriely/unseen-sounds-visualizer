#!/bin/bash
#SBATCH --job-name=det_calib
#SBATCH --output=logs/det_calib_%j.out
#SBATCH --error=logs/det_calib_%j.err
#SBATCH --partition=L4-4h,L4-12h,A100-4h,RTX6000-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Detector bars on the AudioSet-Strong calibration set (benchmark/detector_calib.py).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"
conda activate msproj && python -m benchmark.detector_calib --cache-beats; conda deactivate
conda activate psed   && python -m benchmark.detector_calib --cache-psed;  conda deactivate
conda activate msproj && python -m benchmark.detector_calib --choose
echo DONE

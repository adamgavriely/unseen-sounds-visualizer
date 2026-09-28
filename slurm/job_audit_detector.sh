#!/bin/bash
#SBATCH --job-name=det_audit
#SBATCH --output=logs/det_audit_%j.out
#SBATCH --error=logs/det_audit_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Detector audit (2026-09-28): false-span and miss audit of the shipped stage-4 stack on the 280 and the 415.
# Reads the saved caches only; runs no model; nothing on DEV/TEST.
#   sbatch slurm/job_audit_detector.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/detector_audit.py
echo "DONE detector audit"

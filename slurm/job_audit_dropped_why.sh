#!/bin/bash
#SBATCH --job-name=dropwhy
#SBATCH --output=logs/dropwhy_%j.out
#SBATCH --error=logs/dropwhy_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Why are "heard but dropped" sounds low? (2026-09-28): features of dropped events and FlexSED 0.4-0.8 band spans on
# the 280 and the 415. Reads caches + audio; runs no model; nothing on DEV/TEST.
#   sbatch slurm/job_audit_dropped_why.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/dropped_why.py
echo "DONE dropped why"

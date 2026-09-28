#!/bin/bash
#SBATCH --job-name=setup_audit
#SBATCH --output=logs/setup_audit_cpu_%j.out
#SBATCH --error=logs/setup_audit_cpu_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Setup audit (2026-09-28): BEATs stamps, the bark-run check and the baseline re-scores on the 280 ONLY, from the saved
# caches. Runs no model; nothing on the 415 / fresh / DEV / TEST.
#   sbatch slurm/job_setup_audit_cpu.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/setup_audit.py caches
echo "DONE setup audit caches"

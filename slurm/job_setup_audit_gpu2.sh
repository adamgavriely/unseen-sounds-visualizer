#!/bin/bash
#SBATCH --job-name=setup_audit_gpu
#SBATCH --output=logs/setup_audit_gpu2_%j.out
#SBATCH --error=logs/setup_audit_gpu2_%j.err
#SBATCH --partition=L4-4h,A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Setup audit (2026-09-28): the real stage-4 detect_events (committed local __init__.py incl. the BEATs self-veto, loaded
# from a copy, the cluster src/ untouched) under config.use_shipped() vs the harness stack, 20 clips of the 280 that have a scored consequential event.
#   sbatch slurm/job_setup_audit_gpu2.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/setup_audit.py pipeline --stage4-file "$HOME/setup_audit_tmp/stage4_local.py" --n 20 --pick conseq
echo "DONE setup audit pipeline"

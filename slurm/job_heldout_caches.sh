#!/bin/bash
#SBATCH --job-name=heldcache
#SBATCH --output=logs/hc_%j.out
#SBATCH --error=logs/hc_%j.err
#SBATCH --partition=H200-4h,A100-4h,generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Amendment 24: BEATs and PANNs frame caches on the held-out AudioSet-Strong set. Caches only -- no score is printed
# (the held-out set is read once, for the one picked cell, by benchmark/detector_round2.py heldout).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python -m benchmark.audioset_detector_eval --cache beats --set heldout
python -c "from benchmark import audioset_stage4_report as R; R.use_set('heldout'); R.panns_cache('cuda'); print('panns done')"

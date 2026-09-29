#!/bin/bash
#SBATCH --job-name=motts
#SBATCH --output=logs/motts_%j.out
#SBATCH --error=logs/motts_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#
# Round 14 amendment J: J4 item lists (benchmark/gold/listener_timestamps.py pool) and the J3 motion-energy cache for
# DEV + TEST (benchmark/gold/motion_energy.py). CPU only. No gold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/listener_timestamps.py pool
python benchmark/gold/motion_energy.py both

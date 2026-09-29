#!/bin/bash
#SBATCH --job-name=test2final
#SBATCH --output=logs/test2final_%j.out
#SBATCH --error=logs/test2final_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Confirmation set 1, REVISED: THE ONE TEST2 SCORING (benchmark/gold/test2_final.py). Submit from ~/MscProj_tg.
#   sbatch slurm/job_test2_final.sh                 # B0r, B1, C1 = TO1+F7F8
#   sbatch slurm/job_test2_final.sh --c2 NAME       # + C2 (only if ruled in)
#   sbatch slurm/job_test2_final.sh --dry-run       # everything up to the tag read (stubbed to raise)
# Refuses to start if benchmark/gold/test2_final.json or .started exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/test2_final.py
python $P stage4 "$@"
python $P stage5 "$@"
python $P score "$@"
echo "DONE test2final"

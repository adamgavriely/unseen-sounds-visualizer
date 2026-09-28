#!/bin/bash
#SBATCH --job-name=sense_arm
#SBATCH --output=logs/sensearm_%j.out
#SBATCH --error=logs/sensearm_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Picture sense test (docs/picture_sense_test_2026-09-28.md): the arms given as arguments (A, B, C), one after another,
# each with the fixed union checker. Needs items.json + union.json from job_sense_prep.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$HOME" | tail -1
for arm in "$@"; do
  python scripts/picture_sense_test.py --phase arm --arm "$arm"
done
df -h "$HOME" | tail -1
echo "DONE sense_arm $*"

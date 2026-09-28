#!/bin/bash
#SBATCH --job-name=sense_prep
#SBATCH --output=logs/senseprep_%j.out
#SBATCH --error=logs/senseprep_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Picture sense test (docs/picture_sense_test_2026-09-28.md): prep (frozen subjects, swaps, slot forms, B look-alikes;
# text only) then mistake mining (4 pictures per label, Qwen-Image-2512) and the union checker options.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$HOME" | tail -1
python scripts/picture_sense_test.py --phase prep
python scripts/picture_sense_test.py --phase mine
df -h "$HOME" | tail -1
echo "DONE sense_prep"

#!/bin/bash
#SBATCH --job-name=dasmrc
#SBATCH --output=logs/dasmrc_%j.out
#SBATCH --error=logs/dasmrc_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# DASM restore check (benchmark/gold/dasm_restore_check.py): 4 DEV clips re-scored with the re-fetched DASM must match
# the old DEV cache (max |diff| < 1e-3). Submit from ~/MscProj_tg; chain job_tagger.sh dasm and arms after it (afterok).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/dasm_restore_check.py
echo "DONE dasmrc"

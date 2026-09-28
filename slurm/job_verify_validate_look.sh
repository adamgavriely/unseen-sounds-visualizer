#!/bin/bash
#SBATCH --job-name=vfy_look
#SBATCH --output=logs/vfylook_%j.out
#SBATCH --error=logs/vfylook_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# The checker validation (scripts/verify_validate.py --phase check) with the VLM-written look-alikes
# (PICTURE_LOOKALIKE_VLM), into data/work/verify_validation/results_lookalike_vlm.json (round 2 stays in results.json).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/verify_validate.py --phase check --lookalike-vlm --out-name results_lookalike_vlm.json
echo "DONE verify_validate_look"

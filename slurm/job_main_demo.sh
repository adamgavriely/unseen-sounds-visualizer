#!/bin/bash
#SBATCH --job-name=maindemo
#SBATCH --output=logs/maindemo_%j.out
#SBATCH --error=logs/maindemo_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#
# End-to-end check of the shipped pipeline on one clip: main.py (use_shipped = TO1+F7F8, on-the-spot listener inputs,
# Qwen-Image pictures). Submit from ~/MscProj:  sbatch slurm/job_main_demo.sh path/to/clip.mp4
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python main.py --input "${1:?clip}" --device cuda --generator diffusion
echo "DONE maindemo"

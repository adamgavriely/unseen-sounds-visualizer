#!/bin/bash
#SBATCH --job-name=picsim
#SBATCH --output=logs/picsim_%j.out
#SBATCH --error=logs/picsim_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Round 30 PIC-SIM (docs/prereg_round13_detector_push.md): gate picture-vs-frames SigLIP-2 similarity, DEV screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/pic_sim_gate.py run
python benchmark/gold/pic_sim_gate.py score
echo "DONE picsim"

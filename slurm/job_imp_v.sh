#!/bin/bash
#SBATCH --job-name=impv
#SBATCH --output=logs/impv_%j.out
#SBATCH --error=logs/impv_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#
# Round 37 IMP-V (docs/prereg_round13_detector_push.md): gate VLM names the maker of a DASM impact peak; from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/imp_v_screen.py run
python benchmark/gold/imp_v_screen.py score
echo "DONE impv"

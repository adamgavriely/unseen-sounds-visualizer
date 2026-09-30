#!/bin/bash
#SBATCH --job-name=gategem
#SBATCH --output=logs/gategem_%j.out
#SBATCH --error=logs/gategem_%j.err
#SBATCH --partition=H200-4h,A100-4h,RTX6000-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
# Round 24: Gemma-4-31B asked the gate's questions on the DEV judge clips (gate_gold.py), then scored. From ~/MscProj.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$HOME/venvs/judge/bin/python" benchmark/gold/gate_gold.py --model google/gemma-4-31B-it --dev-only
python benchmark/gold/gate_gold.py --score --subsets dev54
echo "DONE gategem"

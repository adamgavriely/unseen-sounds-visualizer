#!/bin/bash
#SBATCH --job-name=gate_gold
#SBATCH --output=logs/gate_gold_%j.out
#SBATCH --error=logs/gate_gold_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Gate accuracy on the gold sounds (benchmark/gold/gate_gold.py). MODEL=<hf id> or OWL=1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${OWL:-}" = "1" ]; then python benchmark/gold/gate_gold.py --owl; fi
if [ -n "${MODEL:-}" ]; then python benchmark/gold/gate_gold.py --model "$MODEL"; fi
echo DONE

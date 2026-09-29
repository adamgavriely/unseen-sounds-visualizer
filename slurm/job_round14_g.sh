#!/bin/bash
#SBATCH --job-name=r14g
#SBATCH --output=logs/r14g_%j.out
#SBATCH --error=logs/r14g_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 14 amendment G (rule audit) on DEV: each loosened rule on B0r and on the best amendment-F / H arm, score, then
# one stacked arm of the candidates on that base (round13_dev.py gplan / gstack). Resumable. DEV only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
G=$(python benchmark/gold/round13_dev.py gplan | tail -1)
echo "G arms: $G"
python benchmark/gold/round13_dev.py stage4 --arms $G
python benchmark/gold/round13_dev.py stage5 --arms $G
python benchmark/gold/round13_dev.py score
ST=$(python benchmark/gold/round13_dev.py gstack | tail -1)
echo "stack: ${ST:-none}"
if [ -n "$ST" ]; then
  python benchmark/gold/round13_dev.py stage4 --arms GSTACK
  python benchmark/gold/round13_dev.py stage5 --arms GSTACK
  python benchmark/gold/round13_dev.py score
fi
echo "DONE r14g"

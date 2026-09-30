#!/bin/bash
#SBATCH --job-name=v4b
#SBATCH --output=logs/v4b_%j.out
#SBATCH --error=logs/v4b_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
# Round 27: Qwen V4b answers on the P2/PV cuts of all four parts. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=$HOME/MscProj_tg/benchmark/gold; R=$HOME/MscProj_r13/benchmark/gold; W=$HOME/MscProj/data/work
python benchmark/gold/listener_v4b.py "$R/dev_listener_v.json:$W/devcand/wav16:$G/dev_listener_v4b.json" \
  "$G/dev2_listener_v.json:$W/r13dev2/wav16:$G/dev2_listener_v4b.json" \
  "$G/test_listener_v.json:$W/r13test/wav16:$G/test_listener_v4b.json" \
  "$G/test2_listener_v.json:$W/r13test2/wav16:$G/test2_listener_v4b.json"
echo "DONE v4b"

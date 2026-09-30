#!/bin/bash
#SBATCH --job-name=ptc
#SBATCH --output=logs/ptc_%j.out
#SBATCH --error=logs/ptc_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
# Round 31 PTC: peak-tight-cut V4 answers (Qwen3-Omni, then AF-Next) on the long-cut P2/PV candidates of merged DEV. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
G=$HOME/MscProj_tg/benchmark/gold; R=$HOME/MscProj_r13/benchmark/gold; W=$HOME/MscProj/data/work
python benchmark/gold/listener_ptc.py "$R/dev_listener_v.json:$W/devcand/wav16:$G/dev_listener_ptc.json" \
  "$G/dev2_listener_v.json:$W/r13dev2/wav16:$G/dev2_listener_ptc.json"
echo "DONE ptc"

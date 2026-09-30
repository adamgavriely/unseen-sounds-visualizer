#!/bin/bash
#SBATCH --job-name=p1v4
#SBATCH --output=logs/p1v4_%j.out
#SBATCH --error=logs/p1v4_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 17 R1: Qwen3-Omni V4 on P1 cuts + family parse of Qwen and AF V4 (benchmark/gold/listener_p1v4.py). Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=$HOME/MscProj_tg/benchmark/gold; R=$HOME/MscProj_r13/benchmark/gold; W=$HOME/MscProj/data/work
python benchmark/gold/listener_p1v4.py \
  "dev:$R/dev_listener_v.json:$R/dev_listener_afn.json:$W/devcand/wav16:$G/dev_listener_p1v4.json" \
  "dev2:$G/dev2_listener_v.json:$G/dev2_listener_afn.json:$W/r13dev2/wav16:$G/dev2_listener_p1v4.json" \
  "test:$G/test_listener_v.json:$G/test_listener_afn.json:$W/r13test/wav16:$G/test_listener_p1v4.json" \
  "test2:$G/test2_listener_v.json:$G/test2_listener_afn.json:$W/r13test2/wav16:$G/test2_listener_p1v4.json"
echo "DONE p1v4"

#!/bin/bash
#SBATCH --job-name=contrast
#SBATCH --output=logs/contrast_%j.out
#SBATCH --error=logs/contrast_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
# Round 39 CONTRAST: Qwen3-Omni forced choice (A vs B vs neither) on TIER-rejected P2/PV cuts of merged DEV. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=$HOME/MscProj_tg/benchmark/gold; R=$HOME/MscProj_r13/benchmark/gold; W=$HOME/MscProj/data/work; T=$HOME/MscProj_tg/data/work
python benchmark/gold/listener_contrast.py \
  "$R/dev_listener_v.json:$G/dev_listener_afn.json:$W/devcand/wav16:$T/j2_dev_beats:$T/devcand/dasm_cache:$G/dev_listener_contrast.json" \
  "$G/dev2_listener_v.json:$G/dev2_listener_afn.json:$W/r13dev2/wav16:$T/j2_dev2_beats:$T/dasm_dev2:$G/dev2_listener_contrast.json"
python benchmark/gold/contrast_screen.py
echo "DONE contrast"

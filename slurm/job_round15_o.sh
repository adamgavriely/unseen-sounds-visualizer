#!/bin/bash
#SBATCH --job-name=r15o
#SBATCH --output=logs/r15o_%j.out
#SBATCH --error=logs/r15o_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 15 amendment O (docs/prereg_round13_detector_push.md): Whisper-AT caches (DEV, DEV2, TEST2; ~/venvs/wat), then arm
# TO1F7F8+O on DEV (round13_dev.py in ~/MscProj_r13). Submit from ~/MscProj_r13.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for S in dev dev2 test2; do "$HOME/venvs/wat/bin/python" "$HOME/MscProj/benchmark/gold/wat_cache.py" --split $S; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python benchmark/gold/round13_dev.py stage4 --arms TO1F7F8+O
python benchmark/gold/round13_dev.py stage5 --arms TO1F7F8+O
python benchmark/gold/round13_dev.py score
echo "DONE r15o"

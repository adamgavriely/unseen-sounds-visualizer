#!/bin/bash
#SBATCH --job-name=expect
#SBATCH --output=logs/expect_%j.out
#SBATCH --error=logs/expect_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#
# Round 40 EXPECT (docs/prereg_round13_detector_push.md): VLM proposes off-screen families, Omni V4 confirms, shipped gate at the
# weak-bar onset. Five stages in one job, each its own process (models never co-resident). From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/expect_screen.py propose
python benchmark/gold/expect_screen.py listen
python benchmark/gold/expect_screen.py cands
python benchmark/gold/expect_screen.py gate
python benchmark/gold/expect_screen.py score
echo "DONE expect"

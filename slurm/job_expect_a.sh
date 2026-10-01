#!/bin/bash
#SBATCH --job-name=expecta
#SBATCH --output=logs/expecta_%j.out
#SBATCH --error=logs/expecta_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#
# Round 40b EXPECT-A (docs/prereg_round13_detector_push.md): Omni list -> word map -> weak-bar onset -> shipped gate. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/expect_a_screen.py listen
python benchmark/gold/expect_a_screen.py cands
python benchmark/gold/expect_a_screen.py gate
python benchmark/gold/expect_a_screen.py score
echo "DONE expecta"

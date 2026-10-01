#!/bin/bash
#SBATCH --job-name=syncgate
#SBATCH --output=logs/syncgate_%j.out
#SBATCH --error=logs/syncgate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Round 38 E1 SYNC (docs/prereg_round13_detector_push.md): Synchformer p(offset = 0) per gate stretch, all 139 cached
# gold clips (DEV judge = screen, non-judge = threshold calibration). Scoring runs on the laptop against current gold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SYNCHFORMER_DIR="$HOME/Synchformer"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/sync_gate.py run --device cuda
echo "DONE syncgate"

#!/bin/bash
#SBATCH --job-name=humangate
#SBATCH --output=logs/humangate_%j.out
#SBATCH --error=logs/humangate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 43 HUMAN (docs/prereg_round13_detector_push.md): onset-dense frames, name the source, is it acting now? DEV screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum benchmark/gold/annotations/gold_AG.json
python benchmark/gold/human_gate.py run
python benchmark/gold/human_gate.py score
echo "DONE humangate"

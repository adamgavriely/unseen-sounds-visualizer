#!/bin/bash
#SBATCH --job-name=gateav
#SBATCH --output=logs/gateav_%j.out
#SBATCH --error=logs/gateav_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 15 GA (docs/prereg_round13_detector_push.md): Set-of-Mark crop vote for the gate, DEV screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/gate_audio_veto.py run
python benchmark/gold/gate_audio_veto.py score
echo "DONE gateav"

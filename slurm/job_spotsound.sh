#!/bin/bash
#SBATCH --job-name=spot
#SBATCH --output=logs/spot_%j.out
#SBATCH --error=logs/spot_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Round 31 SPOT (docs/prereg_round13_detector_push.md): SpotSound-A on listener cuts. From ~/MscProj_tg. Arg: smoke|run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/spotsound_screen.py "${1:-run}" "benchmark/gold/spotsound_raw_${1:-run}.json"
echo "DONE spot"

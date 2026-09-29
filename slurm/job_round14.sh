#!/bin/bash
#SBATCH --job-name=r14dev
#SBATCH --output=logs/r14dev_%j.out
#SBATCH --error=logs/r14dev_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 14 on DEV (docs/prereg_round13_detector_push.md, "Round 14" + addendum; benchmark/gold/round13_dev.py). DEV only.
# The 8 filter arms (F1, F4, F1+F4, F1+F4+F3 on LR-V4+R13-1 and LR-V12+R13-1), score, then the best of them + F5, + F6,
# + F5+F6, score; then the best round-14 arm so far + F7, + F8, + F7+F8 (amendment B), score. Every arm starts from the B0 stage-4 env of round13_dev.BASE. Resumable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/round13_dev.py stage4
python benchmark/gold/round13_dev.py stage5
python benchmark/gold/round13_dev.py score
EXTRA=$(python benchmark/gold/round13_dev.py best14 | tail -1)
echo "addendum arms: $EXTRA"
python benchmark/gold/round13_dev.py stage4 --arms $EXTRA
python benchmark/gold/round13_dev.py stage5 --arms $EXTRA
python benchmark/gold/round13_dev.py score
EXTRA=$(python benchmark/gold/round13_dev.py best14 --which F7F8 | tail -1)
echo "amendment B arms: $EXTRA"
python benchmark/gold/round13_dev.py stage4 --arms $EXTRA
python benchmark/gold/round13_dev.py stage5 --arms $EXTRA
python benchmark/gold/round13_dev.py score
echo "DONE r14dev"

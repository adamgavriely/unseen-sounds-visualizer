#!/bin/bash
#SBATCH --job-name=depicttest
#SBATCH --output=logs/depicttest_%j.out
#SBATCH --error=logs/depicttest_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
# Round 57 TEST read (reported only): DEPICT answers for the merged-TEST SHIP8+MD3 renders, into WORK_DIR/depict_answers.json
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m src.stage6_visual_augmentation.depict "data/work/r16final/SHIP8+MD3_proposed"
python -m src.stage6_visual_augmentation.depict "data/work/r13test2/SHIP8+MD3_proposed"
echo "DONE depicttest"

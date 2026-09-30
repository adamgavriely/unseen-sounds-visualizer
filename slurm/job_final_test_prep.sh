#!/bin/bash
#SBATCH --job-name=ftprep
#SBATCH --output=logs/ftprep_%j.out
#SBATCH --error=logs/ftprep_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Merged TEST, gold-free preparation (benchmark/gold/final_test.py): DASM for the old TEST, old TEST stage 4/5/gates of
# B0r and the arm (data/work/r16final), tagger TEST part stage 4/5/gates via tagger_prep. NO scoring. Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${ARM:-TO1+F7F8}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/final_test.py dasm
python benchmark/gold/final_test.py stage --arm "$ARM"
P=benchmark/gold/tagger_prep.py
python $P --split test2 stage4 --arms B0r "$ARM"
python $P --split test2 stage5 --arms B0r "$ARM"
python $P --split test2 gates --arms B0r "$ARM"
echo "DONE ftprep"

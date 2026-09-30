#!/bin/bash
#SBATCH --job-name=tgarms
#SBATCH --output=logs/tgarms_%j.out
#SBATCH --error=logs/tgarms_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Extra arms (TG_ARMS) on the tagger DEV part, then merged-DEV scoring of base + these arms. Submit from ~/MscProj_tg:
#   TG_ARMS="arm1 arm2" sbatch --export=ALL slurm/job_tagger_arms.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/tagger_prep.py
python $P --split dev2 stage4 --arms B0r "TO1+F7F8" $TG_ARMS
python $P --split dev2 stage5 --arms B0r "TO1+F7F8" $TG_ARMS
python $P --split dev2 gates --arms B0r "TO1+F7F8" $TG_ARMS
python benchmark/gold/merged_dev.py --arms B0r "TO1+F7F8" $TG_ARMS
echo "DONE tgarms"

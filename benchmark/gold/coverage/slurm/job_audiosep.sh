#!/bin/bash
#SBATCH --job-name=audiosep
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:55:00
#SBATCH --output=logs/audiosep_%j.out
# Step 14 features (resumes)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate msproj
export PYTHONUNBUFFERED=1
cd ~/AudioSep
~/venvs/audiosep/bin/python ~/MscProj_tg/benchmark/gold/coverage/audiosep_feats.py

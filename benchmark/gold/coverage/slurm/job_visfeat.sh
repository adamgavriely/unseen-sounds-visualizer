#!/bin/bash
#SBATCH --job-name=visfeat
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:55:00
#SBATCH --output=logs/visfeat_%j.out
# Step 12 v1 features (resumes). SHARD=i/n
cd ~/MscProj_tg
export PYTHONUNBUFFERED=1
~/venvs/denseav/bin/python benchmark/gold/coverage/vis_features.py run $SHARD

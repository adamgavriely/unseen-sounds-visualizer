#!/bin/bash
#SBATCH --job-name=earcache
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ear_cache_%j.out
# Step 5 (benchmark/gold/coverage/PREREG_step5_detector_finetune.md): ARGS="NAME BACKBONE [ckpt] --set dev|heldout"
cd ~/MscProj_tg
export PATH=~/miniconda3/envs/sota/bin:$PATH PYTHONUNBUFFERED=1
for a in "${JOBS[@]:-}"; do :; done
~/venvs/psed2/bin/python benchmark/gold/coverage/ear_cache.py $ARGS

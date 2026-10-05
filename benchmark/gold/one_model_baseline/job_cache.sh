#!/bin/bash
#SBATCH --job-name=m2d_cache
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dsi/adamg/MscProj_tg/benchmark/gold/one_model_baseline/job_cache_%j.log
cd ~/MscProj_tg
export PATH=~/miniconda3/envs/sota/bin:$PATH
~/venvs/psed2/bin/python benchmark/gold/one_model_baseline/cache_m2d.py $(cat benchmark/gold/one_model_baseline/missing.txt)

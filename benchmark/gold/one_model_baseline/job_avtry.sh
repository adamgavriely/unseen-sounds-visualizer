#!/bin/bash
#SBATCH --job-name=omb_avtry
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/home/dsi/adamg/MscProj_tg/benchmark/gold/one_model_baseline/job_avtry_%j.log
set -euo pipefail
cd ~/MscProj_tg
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj; source ~/venvs/qomni_av/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python benchmark/gold/one_model_baseline/qwen_omni_av.py 4

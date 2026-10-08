#!/bin/bash
#SBATCH --job-name=s5train
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/s5_train_%j.out
# Step 5: fine-tune (resumes from its last epoch). Runs only when the mixtures are complete.
set -euo pipefail
cd ~/MscProj_tg
n=$(ls ~/open_data/mix/train_*.npz 2>/dev/null | wc -l)
[ "$n" -ge 40 ] && ls ~/open_data/mix/val_*.npz >/dev/null || { echo "mixtures incomplete ($n)"; exit 1; }
export PATH=~/miniconda3/envs/sota/bin:$PATH PYTHONUNBUFFERED=1
~/venvs/psed2/bin/python benchmark/gold/coverage/train_finetune.py ATST-F atstf_ft_s0 0

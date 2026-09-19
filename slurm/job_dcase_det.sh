#!/bin/bash
#SBATCH --job-name=dcase_det
#SBATCH --partition=L40s-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:50:00
#SBATCH --output=logs/dcase_det_%j.out
#SBATCH --error=logs/dcase_det_%j.err
# stage-2 swap check: SAM 3 vs OWLv2 on the DCASE visibility set (docs/prereg_v4.md)
set -uo pipefail
cd ~/MscProj
source ~/miniconda3/etc/profile.d/conda.sh; conda activate msproj
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[job] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -m benchmark.eval_dcase_visibility_det --backend sam3
python -m benchmark.eval_dcase_visibility_det --backend owlv2
echo "[job] done"

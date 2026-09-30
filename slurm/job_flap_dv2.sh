#!/bin/bash
#SBATCH --job-name=flapdv2
#SBATCH --output=logs/flapdv2_%j.out
#SBATCH --error=logs/flapdv2_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 32 amendment DV2 arm (SHIP6 + FineLAP second clip veto) on both parts, then merged-DEV scoring + CV over all round-32 arms.

# benchmark/gold/finelap_full.json (slurm/job_finelap_full.sh). Submit from ~/MscProj_tg:
#   sbatch --dependency=afterany:<flapjoint job> slurm/job_flap_dv2.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FJ_ARMS="SHIP6+DV2"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cp benchmark/gold/finelap_full.json "$HOME/MscProj_r13/benchmark/gold/finelap_full.json"
( cd "$HOME/MscProj_r13" && python benchmark/gold/flap_joint_arms.py dev )
python benchmark/gold/flap_joint_arms.py dev2
python benchmark/gold/flap_joint_arms.py merged
echo "DONE flapdv2"

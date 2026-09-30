#!/bin/bash
#SBATCH --job-name=flapjoint
#SBATCH --output=logs/flapjoint_%j.out
#SBATCH --error=logs/flapjoint_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 32 arms (docs/prereg_round13_detector_push.md): SHIP6+FLR and SHIP6+FLR+F1 on DEV 49 (~/MscProj_r13 harness) and
# on the tagger DEV part (~/MscProj_tg), then merged-DEV scoring + CV selection. Needs data/work/finelap_as_dasm and
# benchmark/gold/finelap_full.json (slurm/job_finelap_full.sh). Submit from ~/MscProj_tg:
#   sbatch --dependency=afterok:<flapfull job> slurm/job_flap_joint.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cp benchmark/gold/finelap_full.json "$HOME/MscProj_r13/benchmark/gold/finelap_full.json"
( cd "$HOME/MscProj_r13" && python benchmark/gold/flap_joint_arms.py dev )
python benchmark/gold/flap_joint_arms.py dev2
python benchmark/gold/flap_joint_arms.py merged
echo "DONE flapjoint"

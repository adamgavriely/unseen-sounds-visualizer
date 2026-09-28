#!/bin/bash
#SBATCH --job-name=devcand_x
#SBATCH --output=logs/devcand_x_%j.out
#SBATCH --error=logs/devcand_x_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# DEV check, amendment 1 (docs/dev_candidates_check_2026-09-28.md): the 3 extra arms R1, R6, R7 (round 10), frozen values.
# FlexSED paraphrase scores on the DEV audio with round 10's own worker (R2), stage 4 and stage 5 of R1/R6/R7 (same memo
# of stage-5 answers as job 31330563; DEV DASM cache of job 31330562), then the score of all arms. Resumable.
#   sbatch --dependency=afterok:31330563 slurm/job_devcand_extra.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/dev_candidates_check.py paralist
python benchmark/round10_flexsed.py --work "$SLURM_SUBMIT_DIR/data/work/devcand/para_work_dev.json" --shard 0 --of 1 --batch 24
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python benchmark/gold/dev_candidates_check.py stage4 --arms R1 R6 R7
python benchmark/gold/dev_candidates_check.py stage5 --arms R1 R6 R7
python benchmark/gold/dev_candidates_check.py score
echo "DONE devcand extra"

#!/bin/bash
#SBATCH --job-name=r13final
#SBATCH --output=logs/r13final_%j.out
#SBATCH --error=logs/r13final_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 13: THE ONE TEST EXPOSURE (docs/prereg_round13_detector_push.md; benchmark/gold/r13_test_final.py).
# Usage (from ~/MscProj):  sbatch slurm/job_r13_test_final.sh R13-1 [extra r13_test_final.py args, e.g. --flags '{...}']
#   dry run (gold stubbed to raise, outputs in data/work/r13final_dry):  sbatch slurm/job_r13_test_final.sh R13-1 --dry-run
# B0r, B1 and the arm: stage 4 -> stage 5 -> gates (D0, D5, completeness) -> score (TEST gold read only here).
# The script refuses to start if benchmark/gold/r13_test_final.json or .started exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${1:?arm name}"; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/r13_test_final.py
python $P stage4 --arm "$ARM" "$@"
python $P stage5 --arm "$ARM" "$@"
python $P score --arm "$ARM" "$@"
echo "DONE r13final $ARM"

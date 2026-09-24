#!/bin/bash
#SBATCH --job-name=pic_bench
#SBATCH --output=logs/picbench_%j.out
#SBATCH --error=logs/picbench_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Picture quality bench (docs/picture_quality_prereg.md). STEPS is a space-separated list of
# "phase[:arm]", run in order, each in its own process so one large model is resident at a time.
#   STEPS="specs eval:shipped report" sbatch slurm/job_picture_bench.sh
#   PBARGS="--tag picfresh_v32 --bench data/work/picture_bench_fresh" adds arguments to every step
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in ${STEPS:?STEPS required}; do
  phase="${step%%:*}"; arm=""
  [[ "$step" == *:* ]] && arm="${step#*:}"
  echo "=== $phase ${arm}"
  python benchmark/gold/picture_bench.py "$phase" ${arm:+--arm "$arm"} ${PBARGS:-}
done
echo "DONE $STEPS"

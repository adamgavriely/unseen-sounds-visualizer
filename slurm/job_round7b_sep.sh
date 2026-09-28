#!/bin/bash
#SBATCH --job-name=r7bsep
#SBATCH --output=logs/r7bsep_%j.out
#SBATCH --error=logs/r7bsep_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Detector round 7b (retry: descriptive queries, predict_spans=True, erase check) (docs/prereg_round7_samaudio.md): SAM-Audio large removes "a person talking" then "background music" (predict_spans=True); writes the arithmetic
# residual, the model's residual stem and the identity route (16 kHz wavs) to data/work/r7b_samaudio/<set>/.
# Env sota (holds sam_audio since 20 Sept; nothing installed). Weights were fetched on the login node: offline here.
#   sbatch --export=ALL,SET=calib,SHARD=0,OF=2 slurm/job_round7b_sep.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sota
export R7_TAG=7b PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/detector_round7.py sep --set "${SET:?}" --shard "${SHARD:-0}" --of "${OF:-1}" ${LIMIT:+--limit $LIMIT}
echo "DONE round7b sep ${SET} ${SHARD:-0}/${OF:-1}"

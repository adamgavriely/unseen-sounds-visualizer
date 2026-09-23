#!/bin/bash
#SBATCH --job-name=sam3vote
#SBATCH --output=logs/sam3_%j.out
#SBATCH --error=logs/sam3_%j.err
#SBATCH --partition=A100-4h,L40s-4h,H200-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# SAM 3 as a per-stretch silencing vote (amendment 18c).
#
# OWLv2 in the same harness landed at exactly 2.00 removed per sound lost, against a break-even of
# 2.0 -- it failed by nothing. SAM 3 scores more than double OWLv2's cgF1 on the open-vocabulary
# SA-Co benchmark, where OWLv2 is the paper's own strongest baseline, so this asks whether a better
# concept detector clears the line OWLv2 tied. The concept phrases are OWLv2's, so the two runs are
# comparable one for one.
#
#   BACKEND=sam3 BAR=0.5 sbatch slurm/job_sam3.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
# SAM 3's weights are gated on the Hub; the token is read the same way every other job reads it
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'") || true
    export HF_TOKEN
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/owl_per_stretch.py --half dev \
    --backend "${BACKEND:-sam3}" --bar "${BAR:-0.5}"

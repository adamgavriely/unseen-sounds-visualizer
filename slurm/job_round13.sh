#!/bin/bash
#SBATCH --job-name=r13dev
#SBATCH --output=logs/r13dev_%j.out
#SBATCH --error=logs/r13dev_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 13 detector push on DEV (docs/prereg_round13_detector_push.md; benchmark/gold/round13_dev.py). DEV only.
# Runs from a separate code tree (~/MscProj_r13: the round-13 source; data/ and ckpts/ are links to ~/MscProj) so no file
# of a running job in ~/MscProj is touched. Arms: B0r (flags off; gates D0 + conf-equality, D5), R13-1, R13-2 (b 0.8, 0.7),
# R13-5, R13-6 (+ R13-6ev), STACK12 and STACK1256 at both b. Every arm starts from the B0 stage-4 env set explicitly
# in round13_dev.BASE (FLEXSED_BAR 0.8, FLEXSED_VETO 0.3, PANNS_VETO 0.05, BEATS_SELF_VETO 0, MONO, no cap). Resumable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/round13_dev.py panns
python benchmark/gold/round13_dev.py stage4
python benchmark/gold/round13_dev.py stage5
python benchmark/gold/round13_dev.py score
echo "DONE r13dev"

#!/bin/bash
#SBATCH --job-name=psed_ens
#SBATCH --output=logs/psed_ens_%j.out
#SBATCH --error=logs/psed_ens_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,L4-4h,L4-12h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# The five-backbone average (benchmark/psed_ensemble.py, docs/prereg_psed_ensemble.md).
# The four extra checkpoints must already be in ~/PretrainedSED/resources (login node).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"
conda activate psed   && python -m benchmark.psed_ensemble --cache; conda deactivate
conda activate msproj && python -m benchmark.psed_ensemble --eval
echo DONE

#!/bin/bash
#SBATCH --job-name=sep_witness
#SBATCH --output=logs/sep_%j.out
#SBATCH --error=logs/sep_%j.err
#SBATCH --partition=L4-4h,L4-12h,A100-4h,RTX6000-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# Speech removal as a second witness (benchmark/sep_witness.py): SAM-Audio views, PSED on them.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"
SETS="${SETS:-calib sliceB}"
conda activate samaudio
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/samaudio/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
for S in $SETS; do python -m benchmark.sep_witness --separate --set $S; done
conda deactivate
conda activate psed
for S in $SETS; do python -m benchmark.sep_witness --psed --set $S; done
conda deactivate
conda activate msproj && python -m benchmark.sep_witness --analyse
echo DONE

#!/bin/bash
#SBATCH --job-name=ref_v2
#SBATCH --output=logs/ref_v2_%j.out
#SBATCH --error=logs/ref_v2_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Independent reference for the v2 run's 100 clips, then re-judge v2's cached
# descriptions against it (results tagged v2_indep). No re-render, no re-describe.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
GEN=diffusion TAG=v2 PHASE=reference bash slurm/job_protocol.sh
GEN=diffusion TAG=v2_indep DESC_TAG=v2 PHASE=judge INDEP=1 bash slurm/job_protocol.sh
echo "DONE -> benchmark/protocol_results_v2_indep.json"

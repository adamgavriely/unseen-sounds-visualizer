#!/bin/bash
#SBATCH --job-name=xref_judge
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:50:00
#SBATCH --output=logs/xref_%j.out
#SBATCH --error=logs/xref_%j.err
# Same-answer-sheet check for the detector swap (docs/prereg_v4.md): each row's cached
# descriptions judged against the OTHER row's grounded references, Mistral judge.
#   v4ab2 descriptions vs v4b2 references -> protocol_results_v4ab2_xref_v4b2_grounded.json
#   v4b2 descriptions vs v4ab2 references -> protocol_results_v4b2_xref_v4ab2_grounded.json
set -uo pipefail
cd ~/MscProj
J=mistralai/Mistral-7B-Instruct-v0.3
env V4=58 GEN=diffusion LIMIT=100 TAG=v4ab2_xref_v4b2_grounded DESC_TAG=v4ab2 REF_TAG=v4b2 JUDGE=$J PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh
env V4=58 GEN=diffusion LIMIT=100 TAG=v4b2_xref_v4ab2_grounded DESC_TAG=v4b2 REF_TAG=v4ab2 JUDGE=$J PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh
echo "[xref] done"

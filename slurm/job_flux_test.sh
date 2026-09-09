#!/bin/bash
#SBATCH --job-name=fluxtest
#SBATCH --output=logs/fluxtest_%j.out
#SBATCH --error=logs/fluxtest_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# A batch job gets a non-interactive shell, so HF_TOKEN set in .bashrc is not visible
# and gated repos 401. Sourcing .bashrc is not an option either: it pulls in /etc/bashrc,
# which is not safe under `set -u` and killed this job in three seconds with no message.
# So read just the one line, and never fail the job over it.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc"                | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi
if [ -n "${HF_TOKEN:-}" ]; then echo "[auth] HF_TOKEN present (${#HF_TOKEN} chars)"; else echo "[auth] no HF_TOKEN - gated repos will fail"; fi

export GEN_TEST_MODEL="${GEN_TEST_MODEL:-black-forest-labs/FLUX.1-schnell}"
python scripts/flux_test.py
ls -la data/output/flux_test/

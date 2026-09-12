#!/bin/bash
#SBATCH --job-name=fluxtest
#SBATCH --output=logs/fluxtest_%j.out
#SBATCH --error=logs/fluxtest_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
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
# reduces fragmentation, which is what turned "90 MiB short" into a hard failure
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

# Both generators, same twelve prompts, one job: the shipping PixArt-Sigma first, then
# FLUX.1-schnell (the proposal's named model). Sheets land in data/output/gen_compare/.
for M in "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" "black-forest-labs/FLUX.1-schnell"; do
    echo "=== $M ==="
    GEN_TEST_MODEL="$M" python scripts/flux_test.py || echo "  ! $M failed"
done
ls -la data/output/gen_compare/*/SHEET*.png

#!/bin/bash
#SBATCH --job-name=comfyui
#SBATCH --output=logs/comfy_%j.out
#SBATCH --error=logs/comfy_%j.err
#SBATCH --partition=L4-12h,A100-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#
# The ComfyUI demo server (docs/comfyui_demo_plan.md).
#
# It runs on a GPU node because triton, which ComfyUI pulls in, refuses to initialise without a
# CUDA driver -- the login node has none. The job prints the one command a viewer needs to reach
# the GUI from a laptop, then serves the graph until the walltime runs out.
#
#   sbatch slurm/job_comfy.sh          then read logs/comfy_<jobid>.out for the tunnel command
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export MSCPROJ_ROOT="$HOME/MscProj"
# FLUX is gated; read the token the same way every other job does (never printed)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'") || true
    export HF_TOKEN
fi
PORT="${PORT:-8188}"
echo "=================================================================="
echo " ComfyUI demo is starting on $(hostname), port ${PORT}"
echo ""
echo " From your laptop, run:"
echo "   ssh -N -L ${PORT}:$(hostname):${PORT} adamg@slurm-login1.lnx.biu.ac.il"
echo " then open:  http://127.0.0.1:${PORT}"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python comfyui_nodes/comfy_start.py --listen 0.0.0.0 --port "$PORT" --disable-auto-launch --disable-cuda-malloc

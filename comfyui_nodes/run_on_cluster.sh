#!/bin/bash
#SBATCH --job-name=comfyui
#SBATCH --output=logs/comfy_%j.out
#SBATCH --error=logs/comfy_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#
# The ComfyUI server (comfyui_nodes/README.md). H200: the frozen pipeline needs the VLM next to Qwen-Image on one card.
#
# It runs on a GPU node because triton, which ComfyUI pulls in, refuses to initialise without a
# CUDA driver -- the login node has none. The job prints the one command a viewer needs to reach
# the GUI from a laptop, then serves the graph until the walltime runs out.
#
#   sbatch comfyui_nodes/run_on_cluster.sh   then read logs/comfy_<jobid>.out for the tunnel command
# It activates a conda environment named `msproj` that holds the project's packages (main README.md,
# Installation); edit the two `conda` lines if yours has another name or place.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
# uploads and temp files go to home: a shared node's /tmp can be full
export TMPDIR="$HOME/tmp_comfy"; mkdir -p "$TMPDIR"
export MSCPROJ_ROOT="$HOME/MscProj"
# Gated Hugging Face models need a token: export HF_TOKEN=<your token> before sbatch (Slurm passes the
# environment on to the job). The token is never printed.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "note: HF_TOKEN is not set; gated models will not download" >&2
fi
PORT="${PORT:-8188}"
echo "=================================================================="
echo " ComfyUI demo is starting on $(hostname), port ${PORT}"
echo ""
echo " From your laptop, run:"
echo "   ssh -N -L ${PORT}:$(hostname):${PORT} <user>@<login-node>"
echo " then open:  http://127.0.0.1:${PORT}"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python comfyui_nodes/comfy_start.py --listen 0.0.0.0 --port "$PORT" --disable-auto-launch --disable-cuda-malloc &
COMFY_PID=$!
# the simple public page (comfyui_nodes/public_page.py): its https://....gradio.live link is printed in
# logs/comfy_<jobid>.err ("Running on public URL"). PUBLIC=0 skips it. GRADIO_PYTHON is the Python of an
# environment with gradio, requests and websockets.
GRADIO_PYTHON="${GRADIO_PYTHON:-$HOME/venv_gradio/bin/python}"
if [ "${PUBLIC:-1}" = "1" ] && [ -x "$GRADIO_PYTHON" ]; then
    until curl -s -o /dev/null "http://127.0.0.1:${PORT}/object_info/MscAugmentVideo"; do sleep 10; done
    "$GRADIO_PYTHON" comfyui_nodes/public_page.py --comfy "http://127.0.0.1:${PORT}" --share &
fi
wait "$COMFY_PID"

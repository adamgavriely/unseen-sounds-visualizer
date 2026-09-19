#!/bin/bash
# One-shot setup of the RunPod pod (msproj-h100). Everything persistent goes under /workspace
# (the container disk is erased on Stop). Idempotent: re-run after a Start to repair.
#   bash /workspace/MscProj/slurm/runpod_setup.sh 2>&1 | tee -a /workspace/setup.log
# Gated weights (google/gemma-4-31B-it, facebook/sam3) need HF_TOKEN in the environment; the
# script skips them when it is absent and says so.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
# RunPod injects the pod's env vars (HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID) into PID 1 only;
# an ssh session does not inherit them, so import the ones we need (values are never printed)
if [ -r /proc/1/environ ]; then
    while IFS= read -r -d "" kv; do
        case "$kv" in HF_TOKEN=*|RUNPOD_API_KEY=*|RUNPOD_POD_ID=*) export "$kv";; esac
    done < /proc/1/environ
fi
W=/workspace
export HF_HOME=$W/hf
mkdir -p $HF_HOME $W/MscProj

echo "=== [1] system packages"
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ffmpeg git tmux htop >/dev/null 2>&1 && echo ffmpeg ok

echo "=== [2] miniconda under /workspace"
if [ ! -x $W/miniconda3/bin/conda ]; then
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh && bash /tmp/mc.sh -b -p $W/miniconda3 >/dev/null
fi
source $W/miniconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base false >/dev/null 2>&1
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1
grep -q "miniconda3/etc/profile.d/conda.sh" ~/.bashrc || cat >> ~/.bashrc <<'EOF'
source /workspace/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
EOF

echo "=== [3] env msproj (the pipeline: torch cu128, transformers 5.x, diffusers)"
if ! conda env list | grep -q "^msproj "; then
    conda create -y -q -n msproj python=3.11 >/dev/null
    conda activate msproj
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -q "transformers>=5.8" diffusers accelerate safetensors sentencepiece librosa soundfile numpy scipy scikit-learn matplotlib pillow faster-whisper huggingface_hub[cli] timm einops opencv-python-headless sentence-transformers
    conda deactivate
fi
echo "=== [4] env psed (PretrainedSED: numpy<2)"
if ! conda env list | grep -q "^psed "; then
    conda create -y -q -n psed python=3.10 >/dev/null
    conda activate psed
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -q "numpy<2" librosa soundfile timm einops huggingface_hub
    [ -d $W/PretrainedSED ] || git clone -q https://github.com/fschmid56/PretrainedSED.git $W/PretrainedSED
    grep -v sed_scores_eval $W/PretrainedSED/requirements.txt > /tmp/psed_req.txt; pip install -q -r /tmp/psed_req.txt 2>/dev/null
    conda deactivate
fi
ln -sfn $W/PretrainedSED ~/PretrainedSED

echo "=== [5] weights -> $HF_HOME (ungated first)"
conda activate msproj
# ~/.bashrc returns early for non-interactive shells, so read the token line directly (as on BIU)
if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.bashrc ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' ~/.bashrc | tail -1 | tr -d "\"'") || true
    export HF_TOKEN
fi
for m in Qwen/Qwen-Image-2512 Qwen/Qwen3.8-27B ibm-granite/granite-speech-4.1-2b WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 black-forest-labs/FLUX.1-schnell mistralai/Mistral-7B-Instruct-v0.3 google/owlv2-base-patch16-ensemble google/siglip-base-patch16-224 Systran/faster-whisper-base sentence-transformers/all-MiniLM-L6-v2; do
    echo "--- $m"; hf download "$m" >/dev/null 2>&1 && echo "   ok" || echo "   FAILED (gated or network)"
done
if [ -n "${HF_TOKEN:-}" ]; then
    for m in google/gemma-4-31B-it facebook/sam3; do echo "--- $m (gated)"; hf download "$m" >/dev/null 2>&1 && echo "   ok" || echo "   FAILED"; done
else
    echo "!!! HF_TOKEN not set: skipped gated google/gemma-4-31B-it and facebook/sam3 -- set the token and re-run [5]"
fi
# PretrainedSED checkpoint (GitHub release, auto-downloaded on first load); pull it now
conda deactivate; conda activate psed
( cd $W/PretrainedSED && python - <<'PY'
import sys; sys.path.insert(0, ".")
from models.prediction_wrapper import PredictionsWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1"); print("PSED BEATs_strong_1 ok")
PY
) 2>&1 | tail -1
conda deactivate
du -sh $HF_HOME

echo "=== [6] idle watcher: stop the pod after 20 min without GPU work"
cat > $W/idle_stop.sh <<'EOF'
#!/bin/bash
# stops this pod when GPU utilisation stays under 5% for 20 minutes (checked every minute)
while IFS= read -r -d "" kv; do case "$kv" in RUNPOD_API_KEY=*|RUNPOD_POD_ID=*) export "$kv";; esac; done < /proc/1/environ
idle=0
while true; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ "${u:-0}" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
    if [ $idle -ge 20 ]; then echo "$(date) idle 20 min -> stop" >> /workspace/idle_stop.log; runpodctl stop pod "$RUNPOD_POD_ID"; idle=0; fi
    sleep 60
done
EOF
chmod +x $W/idle_stop.sh
pgrep -f "idle_stop.sh" >/dev/null || setsid nohup bash $W/idle_stop.sh >/dev/null 2>&1 < /dev/null &
sleep 1; echo "idle watcher running (pid $(pgrep -f "bash /workspace/idle_stop.sh" | head -1))"
echo "=== done"

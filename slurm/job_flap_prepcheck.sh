#!/bin/bash
#SBATCH --job-name=flapprep
#SBATCH --output=logs/flapprep_%j.out
#SBATCH --error=logs/flapprep_%j.err
#SBATCH --partition=H200-4h,A100-4h,L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
CLIP=tg_d007
OUT=$HOME/flapchk_out; rm -rf $OUT
cd $HOME/MscProj_tg
$HOME/venv_flap/bin/python benchmark/gold/finelap_screen.py split dev2 --out $OUT $CLIP
python - <<PY
import numpy as np
a=np.load("$OUT/$CLIP.npz"); b=np.load("data/work/finelap_cache/$CLIP.npz")
print("labels", list(a["labels"])==list(b["labels"]), list(a["labels"]))
for k in ("fs","fe","scores"): print(k, a[k].shape, b[k].shape, np.allclose(a[k],b[k],atol=1e-5), float(np.abs(a[k]-b[k]).max()))
PY
cd $HOME/MscProj
ls data/work/finelap_live_as_explosion_XJ8lc3I6 2>/dev/null || true
python - <<PY
from pathlib import Path
import config
from src.listener_prep import ensure_listener_inputs
config.use_shipped()
s = ensure_listener_inputs(Path("data/input/tagger_live_as_explosion_XJ8lc3I6/as_explosion_XJ8lc3I6.mp4"))
config.set_listener_split(s)
print("split", s, "FINELAP_VETO", config.FINELAP_VETO, "DIR", config.FINELAP_DIR)
import numpy as np
z = np.load(Path(config.FINELAP_DIR) / "as_explosion_XJ8lc3I6.npz"); print("live npz", list(z["labels"]), z["scores"].shape)
from src.stage4_audio_event_detection import _require_caches
_require_caches(Path("data/work/x/as_explosion_XJ8lc3I6/audio.wav")); print("REQUIRE_CACHES OK")
PY
echo DONE

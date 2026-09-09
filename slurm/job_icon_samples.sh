#!/bin/bash
#SBATCH --job-name=icons
#SBATCH --output=logs/icons_%j.out
#SBATCH --error=logs/icons_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Generate a handful of augmentation panels and nothing else, so the pictogram prompt
# can be judged by eye before a full run is spent on it. The prompt asks SDXL for flat
# icon art; whether it complies is an empirical question, and the subject-tracking sound
# glyph depends on it complying (a photographic result fills the frame and the mark
# falls back to the corner).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

python - <<'PY'
from pathlib import Path
import config
config.DEVICE = "cuda"
from src.stage6_visual_augmentation import (icon_prompt, _diffusion_image, _sound_glyph,
                                            _caption, _fit_on_white, _subject_bbox)
from PIL import Image

out = Path("data/output/icon_samples"); out.mkdir(parents=True, exist_ok=True)
subjects = ["Bird", "Fire engine", "Rain", "Dog", "Train", "Glass", "Thunder", "Footsteps"]
for s in subjects:
    raw = out / f"{s.replace(' ', '_')}_raw.png"
    ok = _diffusion_image(raw, icon_prompt(s), (768, 768),
                          model="stabilityai/stable-diffusion-xl-base-1.0", device="cuda")
    if not ok:
        print(f"  ! {s}: generation failed", flush=True); continue
    img = _fit_on_white(Image.open(raw).convert("RGB"), (720, 405))
    box = _subject_bbox(img)
    panel = _caption(_sound_glyph(img, 0.7), s)
    panel.save(out / f"{s.replace(' ', '_')}_panel.png")
    print(f"  {s:14} subject bbox={box}  "
          f"{'ICON (mark goes on the subject)' if box else 'photo-like (mark falls back to corner)'}",
          flush=True)
print("done", flush=True)
PY
ls data/output/icon_samples/*_panel.png | wc -l

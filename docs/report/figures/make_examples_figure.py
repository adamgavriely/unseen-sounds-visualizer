"""Example figure of the report: real outputs of the final system, several per outcome.

Frames are taken from the rendered output videos of the final system (the decision inspector's media, which are not
tracked in git because of their size). Each frame shows the input video on the left and the picture panel on the right.
Run from the repository root, pointing MEDIA at the folder that holds media/bysig/{DEV,TEST}:
    python docs/report/figures/make_examples_figure.py <media folder>
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MEDIA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", "inspector")
DATA = os.path.join(ROOT, "..", "..", "decision_trail", "data.js")
TMP = os.path.join(ROOT, "_frames")

# (row title, [(clip, kind, label, time of the picture or of the sound[, frame offset])]); the frame is taken OFFSET
# seconds after that time unless the item gives its own offset
ROWS = [
    ("Hit: right picture, source off screen", [
        ("ly_ambulance_(siren)_-yPSgCn", "pic", "Siren", 0.0),
        ("mv_protest_scene_movie", "pic", "Glass", 10.75),
        ("w8_kids_fire_alarm_school_1b", "pic", "Alarm", 0.0)]),
    ("Missed: object in frame but not making the sound", [
        ("bell_miami", "gold", "Bell", 0.2),
        ("b3_pet_shop", "gold", "Bird", 0.1),
        ("tg_d078", "gold", "Vehicle horn", 5.2)]),
    ("Wrong: source is on screen", [
        ("london_protest_01", "pic", "Vehicle", 0.25),
        ("tg_d128", "pic", "Laughter", 3.08, 0.8),
        ("w8_dashcam_ambulance_behind_1a", "pic", "Siren", 0.22)]),
    ("Wrong: another sound is heard", [
        ("mv_protest_scene_movie", "pic", "Glass", 4.75, 0.75),    # a baby cries (visible); no glass is heard
        ("tg_d001", "pic", "Honk", 5.12),
        ("m4_live_fire_26a", "pic", "Gunshot", 2.5)]),
]
OFFSET = 1.2
RATIO = 2000 / 720          # most output videos are 2000 x 720; others are padded to this shape


def letterbox(path):
    im = Image.open(path).convert("RGB")
    if im.size[1] == 720 and 2000 < im.size[0] <= 2004:   # a few videos are 2002 x 720: crop, do not pad
        im = im.crop((0, 0, 2000, 720))
    w, h = im.size
    W, H = (w, round(w / RATIO)) if w / h > RATIO else (round(h * RATIO), h)
    a = np.asarray(im)
    # pad with the tile's own colour, not white: side bars take the picture panel's colour (right edge)
    if W > w:
        c1 = c2 = np.median(a[:, -4:].reshape(-1, 3), 0)
    else:
        c1, c2 = np.median(a[:4].reshape(-1, 3), 0), np.median(a[-4:].reshape(-1, 3), 0)
    canvas = np.empty((H, W, 3), np.uint8)
    x0, y0 = (W - w) // 2, (H - h) // 2
    if W > w:
        canvas[:, :x0 + 1] = c1; canvas[:, x0:] = c2
    else:
        canvas[:y0 + 1] = c1; canvas[y0:] = c2
    canvas[y0:y0 + h, x0:x0 + w] = a
    return canvas


def clips():
    s = open(DATA, encoding="utf-8").read()
    d = json.loads(s[s.index("{"):s.rstrip().rstrip(";").rindex("}") + 1])
    return {c["clip"]: c for c in d["clips"]}


def frame(video, t, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-q:v", "3", out],
                   check=True)


def main():
    C = clips()
    os.makedirs(TMP, exist_ok=True)
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(10.5, 1.75 * len(ROWS)))
    for r, (title, items) in enumerate(ROWS):
        for k, item in enumerate(items):
            clip, kind, label, t = item[:4]
            off = item[4] if len(item) > 4 else OFFSET
            c = C[clip]
            video = os.path.join(MEDIA, c["video"].replace("../inspector/", ""))
            host = video.replace(os.path.join("media", "bysig"), os.path.join("media", "bysig_host")).replace(
                "media/bysig/", "media/bysig_host/")
            if os.path.exists(host):           # re-rendered with the alarm-type picture rule (same display spans)
                video = host
            out = os.path.join(TMP, f"{r}_{k}.jpg")
            frame(video, t + off, out)
            ax = axes[r][k]
            ax.imshow(letterbox(out))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{label} at {t:.1f} s ({'development' if c['split'] == 'DEV' else 'test'} set)", fontsize=7.5)
        axes[r][0].set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    fig.savefig(os.path.join(ROOT, "examples.pdf"), dpi=200)
    fig.savefig(os.path.join(ROOT, "examples.png"), dpi=110)
    print("examples.pdf")


if __name__ == "__main__":
    main()

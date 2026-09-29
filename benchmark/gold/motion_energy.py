"""Round 14 amendment J3 feature cache (docs/prereg_round13_detector_push.md, "Round 14 amendment J"): per clip, the video's
frame-difference energy. No gold is read.

  frames  ffmpeg (the msproj env's), fps=10, scaled to 160 px width (height keeps the aspect, even), grayscale (0-255)
  energy  energy[i] = mean |frame[i+1] - frame[i]| over pixels; times[i] = (i + 1) / fps (the later frame of the pair)
  video   the clip's mp4 as the DEV/TEST harness finds it: media.json "video_path" in
          data/work/protocol_proposed_<tag>/<stem>/ (dev_monocap_v31 for DEV, test_final_v33 for TEST)
  out     data/work/motion_{dev,test}/<stem>.npz  (energy, times, fps, median, width, height, n_frames, video)

    python benchmark/gold/motion_energy.py [dev|test|both]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
FPS, WIDTH = 10, 160
SPLITS = {"dev": ("dev_stems.txt", "protocol_proposed_dev_monocap_v31", 49),
          "test": ("test_stems.txt", "protocol_proposed_test_final_v33", 60)}


def stems(split):
    f, _, n = SPLITS[split]
    s = sorted(x.strip() for x in (GOLD / f).read_text(encoding="utf-8").splitlines() if x.strip())
    assert len(s) == n, (split, len(s))
    return s


def video_of(split, st):
    return json.loads((WORK / SPLITS[split][1] / st / "media.json").read_text(encoding="utf-8"))["video_path"]


def dims(video):
    """display width/height (rotation 90/270 swaps them; ffmpeg auto-rotates on decode)"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:stream_tags=rotate", "-of", "json", video],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if "rotate" in (s.get("tags") or {}):
        rot = int(s["tags"]["rotate"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def frames(video):
    w, h = dims(video)
    oh = max(2, int(round(WIDTH * h / w / 2.0)) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-an", "-vf", f"fps={FPS},scale={WIDTH}:{oh},format=gray",
                          "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], check=True, capture_output=True).stdout
    n = len(raw) // (WIDTH * oh)
    assert n * WIDTH * oh == len(raw), (video, len(raw), WIDTH, oh)
    return np.frombuffer(raw, np.uint8).reshape(n, oh, WIDTH), oh


def build(split):
    out = WORK / f"motion_{split}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, st in enumerate(stems(split), 1):
        dst = out / f"{st}.npz"
        if dst.exists():
            continue
        v = video_of(split, st)
        f, oh = frames(v)
        f = f.astype(np.float32)
        e = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)).astype(np.float32)
        t = (np.arange(len(e), dtype=np.float64) + 1.0) / FPS
        med = float(np.median(e)) if len(e) else 0.0
        np.savez_compressed(dst, energy=e, times=t, fps=np.float32(FPS), median=np.float32(med), width=WIDTH, height=oh,
                            n_frames=len(f), video=v)
        print(f"[motion] {split} {k} {st}: {len(f)} frames {WIDTH}x{oh}, median {med:.3f}, max {float(e.max()) if len(e) else 0:.2f}"
              f" ({time.time() - t0:.0f} s)", flush=True)
    print(f"[motion] {split}: {sum((out / f'{s}.npz').exists() for s in stems(split))} / {len(stems(split))} -> {out}",
          flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for sp in (("dev", "test") if which == "both" else (which,)):
        build(sp)

"""Step 11 F: per-frame mean luminance of each DEV clip at full frame rate; flashes = frames > median(previous 1 s) +
5 x robust sigma (1.4826 MAD, floor 2 grey levels), runs of <= 4 frames. Cluster CPU (msproj, OpenCV).
    python benchmark/gold/coverage/flash_scan.py -> scratch_cov/flashes_dev.json  {stem: {"fps":, "flashes": [t, ...]}}"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ROOT = Path.cwd()


def scan(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ys = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ys.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2YUV)[:, :, 0].mean()))
    y = np.array(ys)
    w = max(1, int(round(fps)))
    hot = np.zeros(len(y), bool)
    for i in range(w, len(y)):
        base = y[i - w:i]
        med = np.median(base)
        sig = max(2.0, 1.4826 * np.median(np.abs(base - med)))
        hot[i] = y[i] > med + 5 * sig
    fl, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if j - i <= 4:
                fl.append(round(i / fps, 3))
            i = j
        else:
            i += 1
    return fps, fl, len(y)


def main():
    vf = os.environ.get("FLASH_VIDEOS")
    it = json.loads(Path(vf).read_text(encoding="utf-8")) if vf else json.loads((HERE / "verify_items_dev.json").read_text(encoding="utf-8"))["videos"]
    out = {}
    for st, rel in sorted(it.items()):
        fps, fl, n = scan(_ROOT / rel)
        out[st] = {"fps": fps, "frames": n, "flashes": fl}
        print(st, fps, n, len(fl), flush=True)
    p = _ROOT / "scratch_cov" / os.environ.get("FLASH_OUT", "flashes_dev.json")
    p.write_text(json.dumps(out, indent=0), encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    main()

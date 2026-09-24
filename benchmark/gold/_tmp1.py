"""When does the panel actually light up, next to when the spec says the sound starts?"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

SITE = Path("data/work/error_site/clips")
AUG  = Path(sys.argv[1])          # scratchpad/aug_dump
FPS  = 10.0

def panel_series(mp4):
    p = subprocess.run(["ffprobe","-v","error","-select_streams","v",
                        "-show_entries","stream=width,height","-of","csv=p=0",str(mp4)],
                       capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(p[0]), int(p[1])
    out = subprocess.run(["ffmpeg","-v","error","-i",str(mp4),
        "-vf", f"crop={h}:{h}:{w-h}:0,fps={FPS},scale=16:16","-f","rawvideo","-pix_fmt","gray","-"],
        capture_output=True).stdout
    a = np.frombuffer(out, dtype=np.uint8)
    n = len(a)//256
    return a[:n*256].reshape(n,256).mean(axis=1), 1.0/FPS

rows=[]
for mp4 in sorted(SITE.glob("*.mp4")):
    stem = mp4.stem
    f = AUG/stem/"augmentations.json"
    if not f.exists(): continue
    specs=[s for s in json.loads(f.read_text()) if s.get("augment") and s.get("image_path")]
    if not specs: continue
    ser, dt = panel_series(mp4)
    if len(ser)<5: continue
    dark = ser.min()
    lit  = ser > (dark + 20)                 # panel background is (16,18,24)
    if not lit.any(): continue
    first_lit = float(np.argmax(lit))*dt
    spec_start = min(min([sp[0] for sp in (s.get("spans") or [[s["start"],s["end"]]])]) for s in specs)
    rows.append((stem, first_lit, spec_start, first_lit-spec_start))
    print(f"{stem[:34]:34s} panel lights {first_lit:6.2f}s   spec says {spec_start:6.2f}s   diff {first_lit-spec_start:+.2f}s")
d=np.array([r[3] for r in rows])
print(f"\n{len(rows)} clips: panel-vs-spec median {np.median(d):+.2f}s  mean {d.mean():+.2f}s  min {d.min():+.2f} max {d.max():+.2f}")

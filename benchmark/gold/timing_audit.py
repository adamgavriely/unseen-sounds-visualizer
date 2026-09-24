"""Where does the picture's timing go wrong? (2026-09-24)

Adam, watching the rendered videos: "even on correct pictures the sound comes 1-2 seconds early".
A picture can be out of step with its sound in three different places, and only one of them is worth
fixing, so this measures all three separately.

  A  does the RENDER move anything?   composite audio against source audio, and composite left-half
     brightness against source brightness. Both must land at zero lag.
  B  does the PANEL follow the spec?  when the right-hand panel first lights up, against the time
     the spec says the sound starts.
  C  does the SPEC follow the SOUND?  our start (and end) against the annotator's, for every drawn
     sound matched to a gold sound of the same family. This is the one that matters.

    python benchmark/gold/timing_audit.py --specs data/work/protocol_proposed_dev_symgen_v30

--specs is the run's work root; it needs augmentations.json per clip. --site is the error page,
whose clips/ folder holds the rendered composites that checks A and B read.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

FPS = 25.0


# --------------------------------------------------------------------- signals
def brightness(path, crop=None):
    vf = f"{crop + ',' if crop else ''}fps={FPS},scale=32:32"
    v = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf,
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    n = len(v) // 1024
    return np.frombuffer(v, dtype=np.uint8)[:n * 1024].reshape(n, 1024).mean(axis=1).astype(np.float64)


def envelope(path):
    a = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(a, dtype=np.float32).astype(np.float64)
    k = int(16000 / FPS)
    m = len(y) // k
    return np.abs(y[:m * k].reshape(m, k)).max(axis=1)


def lag(a, b, maxs=3.0):
    """shift, in seconds, that best aligns b onto a (positive means b happens later)"""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best, bl = -9.0, 0
    for s in range(-int(maxs * FPS), int(maxs * FPS) + 1):
        x, y = (a[:n - s], b[s:]) if s >= 0 else (a[-s:], b[:n + s])
        if len(x) < 20:
            continue
        c = float((x * y).mean())
        if c > best:
            best, bl = c, s
    return bl / FPS, best


def source_of(stem):
    g = glob.glob(f"data/input/benchmark/*/{stem}.*") + glob.glob(f"data/input/gold139/all/{stem}.*")
    return g[0] if g else None


# ------------------------------------------------------------------ the checks
def check_render(clips, limit=8):
    print("A. does the render move anything?  (both columns must read 0.00)")
    for comp in sorted(clips.glob("*.mp4"))[:limit]:
        src = source_of(comp.stem)
        if not src:
            continue
        la, ra = lag(envelope(src), envelope(comp))
        lv, rv = lag(brightness(src), brightness(comp, crop="crop=in_h*16/9:in_h:0:0"))
        print(f"   {comp.stem[:32]:32s} audio {la:+.2f}s (r {ra:.2f})   video {lv:+.2f}s (r {rv:.2f})")


def panel_series(mp4):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
                       capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(p[0]), int(p[1])
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                          "-vf", f"crop={h}:{h}:{w - h}:0,fps=10,scale=16:16",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    n = len(out) // 256
    return np.frombuffer(out, dtype=np.uint8)[:n * 256].reshape(n, 256).mean(axis=1), 0.1


def check_panel(clips, specs):
    print("\nB. does the panel follow the spec?")
    d = []
    for mp4 in sorted(clips.glob("*.mp4")):
        f = specs / mp4.stem / "augmentations.json"
        if not f.exists():
            continue
        sp = [s for s in json.loads(f.read_text(encoding="utf-8"))
              if s.get("augment") and s.get("image_path")]
        if not sp:
            continue
        ser, dt = panel_series(mp4)
        if len(ser) < 5:
            continue
        lit = ser > (ser.min() + 20)            # the empty panel is (16, 18, 24)
        if not lit.any():
            continue
        first = float(np.argmax(lit)) * dt
        says = min(min(x[0] for x in (s.get("spans") or [[s["start"], s["end"]]])) for s in sp)
        d.append(first - says)
    d = np.array(d)
    print(f"   {len(d)} clips: panel lights {np.median(d):+.2f}s (median) from what the spec says "
          f"[{d.min():+.2f}, {d.max():+.2f}]")


def check_spec(specs, out_json=None):
    print("\nC. does the spec follow the sound?")
    gold = S.load_gold([GOLD])
    rows = []
    for stem, snds in gold.items():
        f = specs / stem / "augmentations.json"
        if not f.exists():
            continue
        for sp in json.loads(f.read_text(encoding="utf-8")):
            if not sp.get("augment"):
                continue
            spans = sp.get("spans") or [[sp["start"], sp["end"]]]
            ours, ourend = min(x[0] for x in spans), max(x[1] for x in spans)
            cand = [s for s in snds if S.same_family(sp["event_label"], s["label"])]
            if not cand:
                continue
            g = min(cand, key=lambda s: abs(s["start"] - ours))
            if abs(g["start"] - ours) > 6.0:            # a different sound, not a timing error
                continue
            rows.append({"clip": stem, "label": sp["event_label"], "detail": sp.get("detail", ""),
                         "ours": round(ours, 2), "gold": round(g["start"], 2),
                         "d_start": round(ours - g["start"], 2),
                         "d_end": round(ourend - g["end"], 2)})
    d = np.array([r["d_start"] for r in rows])
    e = np.array([r["d_end"] for r in rows])
    print(f"   {len(rows)} drawn sounds matched to a gold sound of the same family")
    print(f"   start: median {np.median(d):+.2f}s  mean {d.mean():+.2f}s   "
          f"early(<-0.5) {int((d < -0.5).sum())}  on time {int((abs(d) <= 0.5).sum())}  "
          f"late(>0.5) {int((d > 0.5).sum())}   p10 {np.percentile(d, 10):+.2f}s")
    print(f"   end:   median {np.median(e):+.2f}s  (negative means the picture leaves first)")
    print("   worst early:")
    for r in sorted(rows, key=lambda r: r["d_start"])[:6]:
        print(f"      {r['clip'][:30]:30s} {r['label'][:12]:12s} ours {r['ours']:6.2f} "
              f"gold {r['gold']:6.2f}  {r['d_start']:+.2f}   (drawn as {r['detail'][:24]})")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print("   ->", out_json)


def check_display(clips, specs, pattern="{stem}.mp4"):
    """The window the VIEWER actually gets, against the window the sound actually occupies.

    check_spec reads the spec's spans, but the panel adds a minimum dwell and caps a picture at
    config.MAX_SPAN, so the two are not the same window. Only clips showing exactly one picture
    are used, because with two pictures the lit panel cannot be attributed to either.
    """
    print("\nD. the window the viewer gets, against the sound")
    gold = S.load_gold([GOLD])
    rows = []
    for mp4 in sorted(clips.glob("*.mp4")):
        stem = mp4.stem[:-len("_augmented")] if mp4.stem.endswith("_augmented") else mp4.stem
        f = specs / stem / "augmentations.json"
        if not f.exists():
            continue
        sp = [s for s in json.loads(f.read_text(encoding="utf-8"))
              if s.get("augment") and s.get("image_path")]
        if len(sp) != 1:
            continue
        ser, dt = panel_series(mp4)
        lit = ser > (ser.min() + 20)
        if not lit.any():
            continue
        idx = np.where(lit)[0]
        shown = (float(idx[0]) * dt, float(idx[-1] + 1) * dt)
        s0 = sp[0]
        spans = s0.get("spans") or [[s0["start"], s0["end"]]]
        first = min(x[0] for x in spans)
        cand = [g for g in gold.get(stem, []) if S.same_family(s0["event_label"], g["label"])]
        if not cand:
            continue
        g = min(cand, key=lambda g: abs(g["start"] - first))
        rows.append((stem, shown, (g["start"], g["end"])))
        print(f"   {stem[:30]:30s} shown {shown[0]:5.1f}-{shown[1]:5.1f}   "
              f"sound {g['start']:5.1f}-{g['end']:5.1f}   end {shown[1] - g['end']:+5.1f}")
    d = np.array([r[1][1] - r[2][1] for r in rows])
    capped = sum(1 for r in rows if abs((r[1][1] - r[1][0]) - 8.0) < 0.25)
    print(f"   {len(rows)} one-picture clips: displayed end vs sound end median {np.median(d):+.2f}s "
          f"mean {d.mean():+.2f}s;  {capped} of them are exactly 8.0 s long (config.MAX_SPAN)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="data/work/error_site")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--render", default="", help="a data/output/protocol_* folder of composites; "
                                                 "use this instead of the error page's clips/")
    a = ap.parse_args()
    specs = Path(a.specs)
    clips = Path(a.render) if a.render else Path(a.site) / "clips"
    if not a.skip_render:
        check_render(clips)
    check_panel(clips, specs)
    check_spec(specs, out_json=_ROOT / "benchmark" / "gold" / f"timing_audit_{specs.name}.json")
    check_display(clips, specs)


if __name__ == "__main__":
    main()

"""Is our "can the viewer see the source?" check right? Gold from DCASE 2025 Task 3.

The visibility question is the decision the whole system rests on, and until now the
only gold for it was Adam's clip-level tags. DCASE 2025 Task 3 (audiovisual track)
labels every sound event in 30,000 five-second clips as onscreen (1) or offscreen (0),
at 100 ms resolution (Shimada et al. 2025; MIT). We take THEIR events -- class and time
-- show our VLM the frames, ask our question, and count agreement. No detector in the
loop, so this measures the visibility check alone.

Caveat, stated up front: their label is geometric -- the source's direction of arrival
falls inside the camera's field of view -- not "you can see it making the sound". A
phone ringing in the pocket of a person in shot is onscreen to them and, correctly, not
visible to us. So agreement is reported against their label as given, and disagreements
in that direction are expected and are not all errors.

    python -m benchmark.eval_dcase_visibility --n 300 --split dev-test-tau
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

CLASSES = ["Female speech, woman speaking", "Male speech, man speaking", "Clapping",
           "Telephone", "Laughter", "Domestic sounds", "Walk, footsteps",
           "Door, open or close", "Music", "Musical instrument", "Water tap, faucet",
           "Bell", "Knock"]
# what we ask the VLM about; their names are AudioSet-like already, trimmed to a noun
ASK_AS = {0: "Speech", 1: "Speech", 2: "Clapping", 3: "Telephone", 4: "Laughter",
          5: "Domestic sounds", 6: "Footsteps", 7: "Door", 8: "Music",
          9: "Musical instrument", 10: "Water tap", 11: "Bell", 12: "Knock"}
FPS = 10.0    # their metadata is one row per 100 ms


def events_from_csv(path: Path):
    """(class_idx, start, end, onscreen_fraction) per contiguous (class, source) run."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[(int(r["class"]), int(r["source"]))].append((int(r["frame"]), int(r["onscreen"])))
    out = []
    for (c, src), fr in by.items():
        fr.sort()
        run = [fr[0]]
        for f in fr[1:]:
            if f[0] - run[-1][0] <= 2:
                run.append(f)
            else:
                out.append((c, run[0][0] / FPS, (run[-1][0] + 1) / FPS, sum(x[1] for x in run) / len(run)))
                run = [f]
        out.append((c, run[0][0] / FPS, (run[-1][0] + 1) / FPS, sum(x[1] for x in run) / len(run)))
    return [e for e in out if e[2] - e[1] >= 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_ROOT / "data" / "dcase2025_task3"))
    ap.add_argument("--split", default="dev-test-tau")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(_ROOT / "benchmark" / "eval_dcase_visibility.json"))
    a = ap.parse_args()
    root = Path(a.root)
    metas = sorted((root / "metadata_dev" / a.split).glob("*.csv"))
    random.seed(a.seed)
    random.shuffle(metas)

    # gather events, stratified: aim for balance between on and off, and across classes
    pool = defaultdict(list)     # (class, onscreen) -> [(clip, start, end)]
    for m in metas:
        for c, s, e, frac in events_from_csv(m):
            if 0.2 < frac < 0.8:
                continue          # mixed within the event: ambiguous gold, skip
            pool[(c, int(frac >= 0.5))].append((m.stem, s, e))
    per_cell = max(3, a.n // (len(CLASSES) * 2))
    chosen = []
    for key, items in pool.items():
        chosen += [(key, x) for x in items[:per_cell]]
    random.shuffle(chosen)
    chosen = chosen[:a.n]
    print(f"[dcase] {len(chosen)} events from {len(metas)} clips; cells:",
          {f"{CLASSES[c][:10]}/{'on' if o else 'off'}": len(v) for (c, o), v in sorted(pool.items())})

    from src.stage5_cross_modal_analysis import reason as R
    from src.stage2_video_understanding import _sample_frames_at
    mdl, proc = R._load(config.VLM_MODEL, config.DEVICE)

    done = {}
    out = Path(a.out)
    if out.exists():
        done = {tuple(r["key"]): r for r in json.loads(out.read_text("utf-8")).get("records", [])}
    records = list(done.values())
    for (c, gold), (stem, s, e) in chosen:
        key = (stem, c, round(s, 1))
        if key in done:
            continue
        video = root / "video_dev" / a.split / (stem + ".mp4")
        n = 6
        lo, hi = max(0.0, s - 1.0), min(5.0, e + 1.0)
        times = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
        frames = _sample_frames_at(video, times)
        seen, named = R._sound_is_visible(ASK_AS[c], frames, mdl, proc, config.DEVICE)
        rec = {"key": list(key), "clip": stem, "class": CLASSES[c], "ask": ASK_AS[c],
               "start": s, "end": e, "gold_onscreen": gold, "pred_visible": int(bool(seen)),
               "named": named}
        records.append(rec)
        print(f"  {CLASSES[c][:16]:16s} gold={'on ' if gold else 'off'} pred={'vis' if seen else 'not'} "
              f"{'OK' if gold == int(bool(seen)) else '--'}  {stem} {s:.1f}-{e:.1f} ({named})", flush=True)
        out.write_text(json.dumps({"records": records}, indent=1), encoding="utf-8")

    # ---- report
    n = len(records)
    agree = sum(r["gold_onscreen"] == r["pred_visible"] for r in records)
    tp = sum(r["gold_onscreen"] == 1 and r["pred_visible"] == 1 for r in records)
    fn = sum(r["gold_onscreen"] == 1 and r["pred_visible"] == 0 for r in records)
    fp = sum(r["gold_onscreen"] == 0 and r["pred_visible"] == 1 for r in records)
    tn = sum(r["gold_onscreen"] == 0 and r["pred_visible"] == 0 for r in records)
    print(f"\n=== visibility vs DCASE 2025 gold ({n} events) ===")
    print(f"  agreement {agree / n:.1%}   on-screen recall {tp / max(1, tp + fn):.1%}   "
          f"off-screen recall {tn / max(1, tn + fp):.1%}")
    print(f"  gold on -> we said visible {tp}, not {fn}   |   gold off -> we said visible {fp}, not {tn}")
    print("  per class (agreement, n):")
    per = defaultdict(list)
    for r in records:
        per[r["class"]].append(r["gold_onscreen"] == r["pred_visible"])
    for cname, v in sorted(per.items(), key=lambda kv: -len(kv[1])):
        print(f"    {cname[:28]:28s} {sum(v) / len(v):5.1%}  ({len(v)})")
    summary = {"n": n, "agreement": agree / n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
               "per_class": {k: [sum(v), len(v)] for k, v in per.items()}}
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()

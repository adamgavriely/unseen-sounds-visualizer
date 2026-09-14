"""When does the picture come up, against DCASE 2025 Task 3's 100 ms onsets.

The onset fix of 2026-09-14 (hysteresis + prefix-silencing occlusion) was measured on
seven hand-labelled onsets over three clips: mean error +0.9 s -> +0.3 s. That is an
anecdote. DCASE 2025 Task 3 gives every sound event in 30,000 five-second clips a start
frame at 100 ms resolution (Shimada et al. 2025; MIT), so the same question can be put
to a few hundred events.

Design. Take their events for the classes our detector can name (door, knock, bell,
telephone, clapping, laughter, footsteps, water tap, domestic sounds; speech and music
are not shown by the system). Keep only events that START inside the clip (gold onset
>= MIN_ONSET), since an event already sounding at frame 0 says nothing about onsets.
Run BEATs once per clip and stamp events three ways from the same scores:

    stamp     sliding-window stamp, flat threshold (what shipped before 09-14)
    hyst      + hysteresis (extend through low scores)
    occl      + hysteresis + occlusion refinement (what ships now)

A gold event is MATCHED by a method if that method produced an event whose label is in
the gold class's AudioSet family and whose onset is within MATCH_SEC of the gold onset.
Signed error = ours - gold (positive = late). Reported per method: n matched, mean and
median signed error, MAE, fraction within 0.25 s and 0.5 s. The recall of the detector
itself is not the question here (their classes and ours differ), only where a matched
event is stamped -- so the "n matched" column is the same population for hyst and occl
(occlusion only moves onsets) and may differ for stamp (hysteresis can merge two spans).

Audio. The DCASE video files carry no audio track; the stereo audio ships separately as
stereo_dev.zip (10 GB, Zenodo record 15559774). Only the ~150 clips holding the sampled
events are needed, so they are fetched by HTTP range request from inside the zip
(remotezip) into data/dcase2025_task3/stereo_dev/<split>/ -- about 0.5 MB and half a
second each. Run the fetch on a node with internet before the GPU job:

    python -m benchmark.eval_dcase_onset --n 300 --fetch-only      # login node
    python -m benchmark.eval_dcase_onset --n 300 --split dev-test-tau
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.eval_dcase_visibility import CLASSES, events_from_csv
from src.labels import is_descendant

# DCASE class index -> AudioSet labels that count as the same sound (label itself or any
# descendant in the ontology). Speech (0, 1), music (8) and instrument (9) are not shown
# by the system and are not scored.
FAMILY = {
    2: ["Clapping", "Applause"],
    3: ["Telephone"],
    4: ["Laughter"],
    5: ["Domestic sounds, home sounds"],
    6: ["Walk, footsteps"],
    7: ["Door"],
    10: ["Water tap, faucet", "Sink (filling or washing)"],
    11: ["Bell", "Church bell", "Jingle bell", "Bicycle bell"],
    12: ["Knock", "Tap"],
}
MIN_ONSET = 0.3      # gold events starting earlier are already sounding at the cut
MAX_ONSET = 4.0      # leave room for a 0.5 s event before the clip ends
MATCH_SEC = 1.5      # our onset must fall this close to the gold onset to count
THR = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
MIN_DUR = float(getattr(config, "AED_MIN_DUR", 0.5))


def _in_family(label: str, c: int) -> bool:
    return any(label == f or is_descendant(label, f) for f in FAMILY[c])


ZENODO_ZIP = "https://zenodo.org/records/15559774/files/stereo_dev.zip?download=1"


def select_events(root: Path, split: str, n: int, seed: int):
    """(class, (stem, start, end)) gold events with an onset inside the clip, balanced
    across classes; deterministic in the seed so --fetch-only and the run agree."""
    metas = sorted((root / "metadata_dev" / split).glob("*.csv"))
    random.seed(seed)
    random.shuffle(metas)
    pool = defaultdict(list)
    for m in metas:
        for c, s, e, _frac in events_from_csv(m):
            if c in FAMILY and MIN_ONSET <= s <= MAX_ONSET:
                pool[c].append((m.stem, s, e))
    per_class = max(5, n // len(FAMILY))
    chosen = [(c, x) for c, items in pool.items() for x in items[:per_class]]
    random.shuffle(chosen)
    return chosen[:n], pool, len(metas)


def fetch_audio(root: Path, split: str, stems, url: str = ZENODO_ZIP) -> int:
    """Pull just these clips' wavs out of the remote zip; returns how many were fetched."""
    from remotezip import RemoteZip
    dest = root / "stereo_dev" / split
    dest.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if not (dest / f"{s}.wav").exists()]
    if not todo:
        return 0
    with RemoteZip(url) as z:
        for i, s in enumerate(todo, 1):
            z.extract(f"stereo_dev/{split}/{s}.wav", root)   # lands under root/stereo_dev/split/
            if i % 25 == 0 or i == len(todo):
                print(f"[dcase-onset] fetched {i}/{len(todo)} wavs", flush=True)
    return len(todo)


def _wav(root: Path, split: str, stem: str, out_dir: Path) -> Path:
    src = root / "stereo_dev" / split / f"{stem}.wav"
    if not src.exists():
        raise FileNotFoundError(f"{src} -- run with --fetch-only on a node with internet first")
    wav = out_dir / f"{stem}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    str(wav), "-loglevel", "error"], check=True)
    return wav


def stamp_three_ways(wav: Path, device: str):
    """(stamp, hyst, occl) event lists from one BEATs pass."""
    from src.stage4_audio_event_detection import _extract_events, _refine_onsets_cam
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    fw, times, labels = infer_beats(wav, device)
    stamp = _extract_events(fw, times, labels, THR, None, MIN_DUR)
    low = THR * float(getattr(config, "AED_HYSTERESIS", 1.0))
    hyst = _extract_events(fw, times, labels, THR, None, MIN_DUR, low=low)
    occl = _refine_onsets_cam(wav, list(hyst), labels, device)
    return {"stamp": stamp, "hyst": hyst, "occl": occl}


def nearest_onset(events, c: int, gold_start: float):
    cands = [e.start for e in events if _in_family(e.label, c)
             and abs(e.start - gold_start) <= MATCH_SEC]
    return min(cands, key=lambda s: abs(s - gold_start)) if cands else None


def summarise(records):
    out = {}
    for m in ("stamp", "hyst", "occl"):
        err = np.array([r["err"][m] for r in records if r["err"].get(m) is not None])
        if len(err) == 0:
            out[m] = {"n": 0}
            continue
        out[m] = {"n": int(len(err)), "mean": float(err.mean()), "median": float(np.median(err)),
                  "mae": float(np.abs(err).mean()),
                  "within_0.25": float((np.abs(err) <= 0.25).mean()),
                  "within_0.5": float((np.abs(err) <= 0.5).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_ROOT / "data" / "dcase2025_task3"))
    ap.add_argument("--split", default="dev-test-tau")
    ap.add_argument("--n", type=int, default=300, help="gold events to score")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(_ROOT / "benchmark" / "eval_dcase_onset.json"))
    ap.add_argument("--fetch-only", action="store_true",
                    help="download the sampled clips' audio from Zenodo and exit")
    a = ap.parse_args()
    root = Path(a.root)
    chosen, pool, n_metas = select_events(root, a.split, a.n, a.seed)
    print(f"[dcase-onset] {len(chosen)} gold events from {n_metas} clips; per class:",
          {CLASSES[c][:12]: len(v) for c, v in sorted(pool.items())}, flush=True)
    stems = sorted({stem for _c, (stem, _s, _e) in chosen})
    if a.fetch_only:
        got = fetch_audio(root, a.split, stems)
        print(f"[dcase-onset] {len(stems)} clips, {got} fetched now -> {root / 'stereo_dev' / a.split}")
        return

    out = Path(a.out)
    done = {}
    if out.exists():
        done = {tuple(r["key"]): r for r in json.loads(out.read_text("utf-8")).get("records", [])}
    records = list(done.values())
    cache = {}
    with tempfile.TemporaryDirectory() as td:
        for i, (c, (stem, s, e)) in enumerate(chosen, 1):
            key = (stem, c, round(s, 2))
            if key in done:
                continue
            if stem not in cache:
                cache[stem] = stamp_three_ways(_wav(root, a.split, stem, Path(td)), config.DEVICE)
            evs = cache[stem]
            err = {m: (None if (o := nearest_onset(evs[m], c, s)) is None else round(o - s, 3))
                   for m in evs}
            records.append({"key": list(key), "class": CLASSES[c], "gold_start": s,
                            "gold_end": e, "err": err})
            if i % 20 == 0 or i == len(chosen):
                print(f"[dcase-onset] {i}/{len(chosen)}  matched so far: "
                      + " ".join(f"{m}={sum(1 for r in records if r['err'].get(m) is not None)}"
                                 for m in ("stamp", "hyst", "occl")), flush=True)
                out.write_text(json.dumps({"records": records, "summary": summarise(records)},
                                          indent=1), encoding="utf-8")
    summary = summarise(records)
    out.write_text(json.dumps({"records": records, "summary": summary,
                               "settings": {"threshold": THR, "min_dur": MIN_DUR,
                                            "match_sec": MATCH_SEC, "split": a.split}},
                              indent=1), encoding="utf-8")
    print("[dcase-onset] summary (signed error = ours - gold, s):")
    for m, v in summary.items():
        if v["n"]:
            print(f"  {m:6s} n={v['n']:3d}  mean {v['mean']:+.2f}  median {v['median']:+.2f}  "
                  f"MAE {v['mae']:.2f}  <=0.25s {v['within_0.25']:.0%}  <=0.5s {v['within_0.5']:.0%}")
        else:
            print(f"  {m:6s} n=0")
    print("[dcase-onset] ->", out)


if __name__ == "__main__":
    main()

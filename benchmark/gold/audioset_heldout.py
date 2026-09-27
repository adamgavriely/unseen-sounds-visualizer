"""Amendment 24 (docs/prereg_v4.md): a NEW held-out AudioSet-Strong set for the detector round, rich in complex scenes.

Disjoint from the 280 calibration clips (+ their missing ids), slice B (+ missing) and every gold clip named by an
AudioSet segment id. 300 COMPLEX + 200 RANDOM, seed 23:
  complex = Speech or Music covers >= 50 % of the 10-s clip AND at least one labelled non-speech, non-music event lies
            >= half under Speech/Music (masked);
  random  = drawn from the remaining pool (complex clips can appear here too; it keeps quiet clips, where false alarms show).
The id list is written with --dry and committed before any download; the download then fetches exactly that list
(clips gone from YouTube are listed as missing, never replaced).

    python benchmark/gold/audioset_heldout.py --dry     # selection only -> audioset_heldout.json (ids, strata)
    python benchmark/gold/audioset_heldout.py           # download the committed list (needs internet), write events
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold.audioset_slice import load_labels, is_cover, CONSEQUENTIAL, fetch as _fetch
from src.labels import canonical
import benchmark.gold.audioset_slice as S

VIDEOS = _ROOT / "data" / "input" / "audioset_heldout"
OUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
SEED, N_COMPLEX, N_RANDOM = 23, 300, 200
_SEG = re.compile(r"^[A-Za-z0-9_-]{11}_\d+$")


def _cover_frac(events, dur=10.0):
    iv = sorted((max(0.0, s), min(dur, e)) for name, s, e in events if is_cover(name) and e > s)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot / dur


def _masked_nonspeech(events):
    covers = [(s, e) for name, s, e in events if is_cover(name)]
    for name, s, e in events:
        if is_cover(name) or e - s <= 0:
            continue
        covered = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in covers)
        if covered >= 0.5 * (e - s):
            return True
    return False


def is_complex(events) -> bool:
    return _cover_frac(events) >= 0.5 and _masked_nonspeech(events)


def excluded_ids() -> set[str]:
    ex = set()
    for f in ("audioset_calib.json", "audioset_slice.json"):
        d = json.loads((_ROOT / "benchmark" / "gold" / f).read_text(encoding="utf-8"))
        ex |= {c["id"] for c in d["clips"]} | set(d.get("missing", []))
    gold = json.loads((_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json").read_text(encoding="utf-8"))
    for c in gold["clips"]:
        if isinstance(c, dict):
            stem = Path(c["clip"]).stem if c["clip"].endswith((".mp4", ".wav")) else c["clip"]
            if _SEG.match(stem):
                ex.add(stem)
    return ex


def select(clips):
    ex = excluded_ids()
    pool = sorted(s for s in clips if s not in ex)
    rng = random.Random(SEED)
    cx = [s for s in pool if is_complex(clips[s])]
    rng.shuffle(cx)
    complex_ids = cx[:N_COMPLEX]
    rest = [s for s in pool if s not in set(complex_ids)]
    rng.shuffle(rest)
    random_ids = rest[:N_RANDOM]
    return complex_ids, random_ids, {"pool": len(pool), "complex_pool": len(cx), "excluded": len(ex)}


def records(events):
    covers = [(s, e) for name, s, e in events if is_cover(name)]
    recs = []
    for name, s, e in events:
        covered = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in covers)
        recs.append({"label": name, "start": round(s, 2), "end": round(e, 2),
                     "masked": bool(not is_cover(name) and e - s > 0 and covered >= 0.5 * (e - s)),
                     "consequential": bool(name in CONSEQUENTIAL or canonical(name) in CONSEQUENTIAL)})
    return sorted(recs, key=lambda r: r["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="write the id list only (commit it before downloading)")
    a = ap.parse_args()
    clips = load_labels()
    if a.dry:
        cx, rd, info = select(clips)
        OUT.write_text(json.dumps({"seed": SEED, "info": info, "strata": {s: "complex" for s in cx} | {s: "random" for s in rd},
                                   "clips": [], "missing": []}, indent=1), encoding="utf-8")
        n_cx_in_rd = sum(is_complex(clips[s]) for s in rd)
        print(f"[heldout] pool {info['pool']} (excluded {info['excluded']}), complex pool {info['complex_pool']}; "
              f"chosen {len(cx)} complex + {len(rd)} random ({n_cx_in_rd} of the random are complex too) -> {OUT}")
        return
    d = json.loads(OUT.read_text(encoding="utf-8"))
    strata = d["strata"]
    assert strata, "run --dry and commit the list first"
    VIDEOS.mkdir(parents=True, exist_ok=True)
    S.VIDEOS = VIDEOS                      # fetch() writes into this folder
    done = {c["id"] for c in d["clips"]} | set(d["missing"])
    for i, seg in enumerate(strata, 1):
        if seg in done:
            continue
        if _fetch(seg):
            d["clips"].append({"id": seg, "stratum": strata[seg], "src": f"../../data/input/audioset_heldout/{seg}.mp4",
                               "duration": 10.0, "events": records(clips[seg])})
        else:
            d["missing"].append(seg)
        if i % 20 == 0:
            print(f"[heldout] {i}/{len(strata)}: {len(d['clips'])} fetched, {len(d['missing'])} gone", flush=True)
            OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"[heldout] done: {len(d['clips'])} clips, {len(d['missing'])} gone -> {OUT}")


if __name__ == "__main__":
    main()

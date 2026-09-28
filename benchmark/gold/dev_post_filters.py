"""DEV-only check of two no-training display rules on the scored DEV renders (Adam, 28 Sept; rules fixed before running):

  merge   one picture per sound: consecutive pictures of the same sound family whose gap is <= GAP s are merged into one
          (the first picture stays up until the last one ends; no new picture is drawn). GAP = 2 s (primary), 5 s (check).
  floor   a picture is shown only if its detector confidence is >= FLOOR. FLOOR = 0.45 (primary), 0.40 / 0.50 (check).

Scored with the official scorer on DEV for ours and blind, onset rule (primary) and the "during the sound" rule
(sensitivity). Writes benchmark/gold/dev_post_filters.json.

    python benchmark/gold/dev_post_filters.py      # cluster (needs the DEV renders)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from src.labels import canonical

TAG = "dev_monocap_v31"
WORK = _ROOT / "data" / "work"
OUT = _ROOT / "benchmark" / "gold" / "dev_post_filters.json"


def merge(specs, gap):
    shown = sorted([s for s in specs if s.get("augment")], key=lambda s: s["start"])
    last = {}
    for s in shown:
        f = canonical(s["event_label"])
        p = last.get(f)
        if p is not None and s["start"] - p["end"] <= gap:
            p["end"] = max(p["end"], s["end"])
            p["spans"] = [[p["spans"][0][0] if p.get("spans") else p["start"], p["end"]]]
            s["augment"] = False
            s["reason"] = (s.get("reason") or "") + " | merged into the previous picture of the same sound"
        else:
            last[f] = s
    return specs


def floor(specs, thr):
    for s in specs:
        if s.get("augment") and float(s.get("confidence", 0)) < thr:
            s["augment"] = False
    return specs


def during_hits(gold, pics):
    """sensitivity: a wrong same-family picture starting in [onset - 0.5, end] of a missed needed sound becomes a hit"""
    r = S.score_clip(gold, pics)
    return r


def score(root, system):
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    dev = sorted(S.subsets_of(gold)["dev"])
    rows = []
    for stem in dev:
        pics = S.load_pictures(root, stem, system) or []
        rows.append(S.score_clip(gold[stem], pics))
    a = S.aggregate(rows)
    return {k: a[k] for k in ("hits", "misses", "visible", "cross", "phantom", "dup", "P", "R", "F1", "viewer_cost")}


def variant(system, fn):
    src = WORK / f"protocol_{system}_{TAG}"
    tmp = Path(tempfile.mkdtemp(prefix="devpost_"))
    for d in src.iterdir():
        if not (d / "augmentations.json").exists():
            continue
        (tmp / d.name).mkdir()
        for f in ("augmentations.json", "media.json"):
            if (d / f).exists():
                shutil.copy(d / f, tmp / d.name / f)
        specs = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
        for s in specs:                                  # keep image paths valid (pictures live in the scored dir)
            if s.get("image_path") and not Path(s["image_path"]).exists():
                c = d / "augmentations" / Path(s["image_path"]).name
                if c.exists():
                    s["image_path"] = str(c)
        (tmp / d.name / "augmentations.json").write_text(json.dumps(fn(specs), indent=1), encoding="utf-8")
    res = score(tmp, system)
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def main():
    out = {}
    for system in ("proposed", "blind_a2i"):
        out[system] = {"scored": score(WORK / f"protocol_{system}_{TAG}", system)}
        for g in (2.0, 5.0):
            out[system][f"merge {g:g}s"] = variant(system, lambda sp, g=g: merge(sp, g))
        for t in (0.40, 0.45, 0.50):
            out[system][f"floor {t:.2f}"] = variant(system, lambda sp, t=t: floor(sp, t))
        out[system]["merge 2s + floor 0.45"] = variant(system, lambda sp: floor(merge(sp, 2.0), 0.45))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for system, rows in out.items():
        print(system)
        for k, v in rows.items():
            print(f"  {k:24s} hits {v['hits']:2d}/{v['hits'] + v['misses']}  wrong {v['visible'] + v['cross'] + v['phantom']:2d} "
                  f"(vis {v['visible']}, diff {v['cross']}, none {v['phantom']})  dup {v['dup']}  F1 {v['F1']:.3f}  cost {v['viewer_cost']:.2f}")


if __name__ == "__main__":
    main()

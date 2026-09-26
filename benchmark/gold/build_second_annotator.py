"""Week plan M6 / D4: the second-annotator packet (30 clips, ~1 hour, someone other than Adam).

Design signed by the panel (docs/WEEK_PLAN_2026-09-26.md, docs/panel_2026-09-26_plan.md B.7): 30 clips stratified by the
scorer's own halves (15 DEV + 15 TEST, no slice B — Adam: "not slice B videos because they are weird"), the mixed
category over-represented; Adam's family list per clip pre-filled, start and end LEFT BLANK for the second person to
mark; ticks heard / visible / obvious / importance with Adam's own definitions (benchmark/gold/tool_template.html).
Reported later: Cohen's kappa on `needed` and `visible`, weighted kappa on importance, median |start difference|, and
one sensitivity row — never merged into the gold, never a gate. Selection is fixed by the seed below, before anyone
labels anything.

    python benchmark/gold/build_second_annotator.py --out data/work/second_annotator
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.detector_dry import clip_path

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
SEED = 20260927
PER_HALF = {"mixed": 5, "unseen": 4, "seen": 4, "no_ambient": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data" / "work" / "second_annotator"))
    a = ap.parse_args()
    raw = {Path(c["clip"]).stem: c for c in json.loads(GOLD.read_text(encoding="utf-8"))["clips"]
           if isinstance(c, dict) and c.get("done") and not c.get("bad")}
    gold = S.load_gold([str(GOLD)])
    subs = S.subsets_of(gold)
    rng = random.Random(SEED)
    chosen = []
    for half in ("dev", "test_bench"):
        pool = {}
        for st in sorted(subs[half]):
            if st in raw and raw[st].get("sounds") and clip_path(raw[st]["clip"]) is not None:
                pool.setdefault(S.category(gold[st]), []).append(st)
        take = []
        for cat, n in PER_HALF.items():
            c = sorted(pool.get(cat, [])); rng.shuffle(c); take += [(half, cat, st) for st in c[:n]]
        short = 15 - len(take)
        if short > 0:                                   # fill from mixed, then unseen
            rest = [st for cat in ("mixed", "unseen") for st in sorted(pool.get(cat, [])) if all(st != t[2] for t in take)]
            rng.shuffle(rest); take += [(half, "fill", st) for st in rest[:short]]
        chosen += take
    order = chosen[:]; rng.shuffle(order)
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    clips = []
    for n, (half, cat, st) in enumerate(order, 1):
        src = clip_path(raw[st]["clip"]); cid = f"C{n:02d}"
        shutil.copy2(src, out / "clips" / f"{cid}{src.suffix}")
        fams = []
        for s in raw[st]["sounds"]:
            f = s.get("family") or s.get("label")
            if f and f not in fams:
                fams.append(f)
        clips.append({"id": cid, "video": f"clips/{cid}{src.suffix}", "sounds": fams})
    key = [{"id": c["id"], "stem": st, "half": half, "category": cat} for c, (half, cat, st) in zip(clips, order)]
    (_ROOT / "benchmark" / "gold" / "second_annotator_selection.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    html = (Path(__file__).resolve().parent / "second_annotator_template.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html.replace("/*CLIPS*/[]", json.dumps(clips)), encoding="utf-8")
    print(f"{len(clips)} clips ({sum(k['half'] == 'dev' for k in key)} DEV, {sum(k['half'] == 'test_bench' for k in key)} TEST) -> {out}")
    for k in key:
        print(" ", k["id"], k["half"], k["category"], k["stem"])


if __name__ == "__main__":
    main()

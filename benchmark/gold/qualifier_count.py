"""D8 of docs/panel_2026-09-26_plan.md: of the frozen confirmation subjects, how many say more than the family word a
text tag would show? Reads only the subjects file and specs (written before the sitting), never answers or keys.

    python benchmark/gold/qualifier_count.py --bench data/work/picture_bench_confirm --subjects V31G
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import label_names  # noqa: E402

STOP = set("a an the is are of in on at with and its it to from by being making sound sounds".split())


def words(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP}


def names(label):
    return set().union(words(label), *[words(n) for n in (label_names(label) or [])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/work/picture_bench_confirm")
    ap.add_argument("--subjects", default="V31G")
    a = ap.parse_args()
    b = _ROOT / a.bench
    subj = json.loads((b / f"subjects_{a.subjects}.json").read_text(encoding="utf-8"))
    specs = {str(s["i"]): s for s in json.loads((b / "specs.json").read_text(encoding="utf-8"))}
    finer, extra = [], []
    for k, v in subj.items():
        fam = specs[k]["label"]; src = v.get("source") or fam
        famw, srcw = names(fam), names(src)
        if src != fam and not srcw <= famw:
            finer.append(f"{fam} -> {src}")
        new = words(v["subject"]) - famw - srcw
        if new:
            extra.append(f"{fam}: {v['subject']}  +{sorted(new)}")
    print(f"subjects: {len(subj)}")
    print(f"source label finer than the family word: {len(finer)}")
    for e in finer:
        print("  ", e)
    print(f"subject adds words beyond the family and source names: {len(extra)}")
    for e in extra:
        print("  ", e)


if __name__ == "__main__":
    main()

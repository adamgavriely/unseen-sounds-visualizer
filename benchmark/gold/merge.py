"""Merge the annotators' exports into the gold set and report agreement.

    python benchmark/gold/merge.py            -> benchmark/gold/gold_set.json + a printed report

Per clip the gold set carries: every annotator's sounds (label, visible, masked, start, end),
their sentence, their picture-due verdict; the MAJORITY picture-due verdict where two or more
annotators agree, else "disputed"; Cohen's kappa on picture-due between each pair of
annotators, and agreement with Adam's original clip label (unseen/mixed = due).

Slice B (AudioSet-Strong clips, tag "audioset_strong" in the export) has no original label:
those clips are kept in the gold set with "slice": "audioset_strong" and left out of the
kappa against Adam's labels.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gate_dev_sweep import load, SHOULD_SHOW

ann = {}
for p in sorted((HERE / "annotations").glob("gold_*.json")):
    j = json.loads(p.read_text(encoding="utf-8"))
    name = j.get("annotator") or p.stem.replace("gold_", "")
    ann[name] = {c["clip"]: c for c in j.get("clips", []) if c.get("done")}
if not ann:
    sys.exit("no annotator files in benchmark/gold/annotations/")
orig = {r["clip"]: (r["tag"] in SHOULD_SHOW) for r in load("test")}
audioset = {c for d in ann.values() for c, v in d.items() if v.get("tag") == "audioset_strong" or c not in orig}


def slice_of(c):
    return "audioset_strong" if c in audioset else "benchmark"


def kappa(a, b):
    keys = [k for k in a if k in b]
    if not keys:
        return float("nan"), 0
    po = sum(a[k] == b[k] for k in keys) / len(keys)
    pa = sum(a[k] for k in keys) / len(keys); pb = sum(b[k] for k in keys) / len(keys)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), len(keys)


due = {n: {c: bool(v.get("picture_due")) for c, v in d.items()} for n, d in ann.items()}
print("annotators:", {n: len(d) for n, d in ann.items()})
for a, b in combinations(due, 2):
    k, n = kappa(due[a], due[b]); print(f"  picture-due kappa {a} vs {b}: {k:.2f} on {n} clips")
for a in due:
    k, n = kappa(due[a], {c: orig[c] for c in due[a] if c in orig and c not in audioset})
    skipped = sum(c in audioset for c in due[a])
    print(f"  {a} vs Adam's original label: kappa {k:.2f} on {n} clips" + (f" ({skipped} AudioSet-Strong clips left out)" if skipped else ""))

clips = defaultdict(dict)
for n, d in ann.items():
    for c, v in d.items():
        clips[c][n] = {"sounds": v.get("sounds", []), "sentence": v.get("sentence", ""), "picture_due": bool(v.get("picture_due")),
                       "unsure": bool(v.get("unsure")), "note": v.get("note", ""), "opened_candidates": bool(v.get("opened_candidates"))}
gold = []
for c, per in sorted(clips.items()):
    votes = [v["picture_due"] for v in per.values()]
    maj = None
    if len(votes) >= 2:
        yes = sum(votes)
        maj = True if yes > len(votes) / 2 else False if yes < len(votes) / 2 else None
    elif len(votes) == 1:
        maj = votes[0]
    gold.append({"clip": c, "slice": slice_of(c), "annotators": per, "picture_due_majority": maj, "disputed": maj is None,
                 "original_label_due": orig.get(c) if c not in audioset else None, "n_annotators": len(per)})
(HERE / "gold_set.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False), encoding="utf-8")
n2 = sum(g["n_annotators"] >= 2 for g in gold); disputed = sum(g["disputed"] for g in gold)
nb = sum(g["slice"] == "audioset_strong" for g in gold)
print(f"gold set: {len(gold)} clips ({n2} with two or more annotators, {disputed} disputed, {nb} AudioSet-Strong) -> {HERE / 'gold_set.json'}")

"""The per-sound answer sheet for the blind picture round (docs/picture_v3_prereg.md, amendment).

Written after the V3 subjects exist and committed BEFORE Adam's answers are opened. For every drawn
sound it lists which words in a free-text answer count as

  * correct  — the source's own names (the AudioSet label split at its commas) plus hand synonyms;
  * narrower — any ontology descendant of the source (counted correct, reported separately); when the
               source is a tie's common parent, the tied children are here;
  * vague    — any ancestor of the source, the family included when the source is more specific;
  * wrong    — the source's siblings and their descendants (fire engine for ambulance).

Everything is built mechanically from the AudioSet ontology the detector was trained on; the only hand
input is `answer_synonyms.json` (plain words people use: "rooster" for "Crowing, cock-a-doodle-doo"),
written from the list of sources alone. Anything an answer says that is on none of the lists is
"unclassified" and is reported, never silently decided.

    python benchmark/gold/answer_sheet.py --bench data/work/picture_bench_fresh \
        --out benchmark/gold/pictures/answer_sheet_picfresh_v32.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import _parents, ancestors, canonical  # noqa: E402

SYN = Path(__file__).resolve().parent / "pictures" / "answer_synonyms.json"


def _children() -> dict:
    kids = {}
    for c, p in _parents().items():
        kids.setdefault(p, []).append(c)
    return kids


def descendants(label: str, kids: dict) -> list:
    out, todo, seen = [], list(kids.get(label, [])), {label}
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        todo.extend(kids.get(x, []))
    return out


def names(label: str, syn: dict) -> list:
    """the words a person could use for this label: its comma parts, then hand synonyms. A bracket is
    AudioSet's disambiguation ("Ambulance (siren)"), not a name: "siren" alone does not say ambulance."""
    parts = [p.strip().lower() for p in label.split("(")[0].split(",")]
    return sorted({p for p in parts if p} | {s.lower() for s in syn.get(label, [])})


def sheet_for(source: str, family: str, syn: dict, kids: dict) -> dict:
    par = _parents().get(source)
    sibs = [s for s in kids.get(par, []) if s != source] if par else []
    wrong = set()
    for s in sibs:
        wrong |= {s, *descendants(s, kids)}
    vague = list(ancestors(source))
    if family != source and family not in vague:
        vague.insert(0, family)
    narrower = descendants(source, kids)
    # never let one word sit on two lists: correct beats narrower beats vague beats wrong
    correct_w = set(names(source, syn))
    narrow_w = {w for x in narrower for w in names(x, syn)} - correct_w
    vague_w = {w for x in vague for w in names(x, syn)} - correct_w - narrow_w
    wrong_w = {w for x in wrong for w in names(x, syn)} - correct_w - narrow_w - vague_w
    return {"source": source, "family": family, "parent": par or "",
            "correct": sorted(correct_w), "narrower": sorted(narrow_w),
            "vague": sorted(vague_w), "wrong": sorted(wrong_w),
            "vague_labels": vague, "sibling_labels": sorted(sibs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="V3", help="which subjects file carries the sources (V3, V31G, ...)")
    a = ap.parse_args()
    bench = Path(a.bench) if Path(a.bench).is_absolute() else _ROOT / a.bench
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
    v3 = json.loads((bench / f"subjects_{a.subjects}.json").read_text(encoding="utf-8"))
    syn = json.loads(SYN.read_text(encoding="utf-8")) if SYN.exists() else {}
    syn = {k: v for k, v in syn.items() if not k.startswith("_")}
    kids = _children()
    out = {}
    for it in specs:
        fam = canonical(it["label"])
        src = v3.get(str(it["i"]), {}).get("source", fam)
        row = sheet_for(src, fam, syn, kids)
        row.update({"clip": it["clip"], "label": it["label"], "start": it["start"],
                    "candidates": v3.get(str(it["i"]), {}).get("candidates", [])})
        out[str(it["i"])] = row
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    srcs = sorted({r["source"] for r in out.values()})
    print(f"{len(out)} sounds, {len(srcs)} distinct sources -> {a.out}")
    for s in srcs:
        print(f"   {s:40s} correct: {', '.join(names(s, syn))}")


if __name__ == "__main__":
    main()

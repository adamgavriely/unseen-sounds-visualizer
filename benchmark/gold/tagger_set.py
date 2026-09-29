"""Second gold set from Adam's own tagging in the tagger tool (tagger/index.html), kept apart from DEV/TEST.

Takes a tagger export (tagger_<name>_<date>.json), keeps the clips marked done and not broken, and:
  - writes benchmark/gold/annotations/tagger_<annotator>.json: the export's done clips, clip ids renamed tg_dNNN,
    every sound given a `family` resolved from its free-text label (TAGGER_ALIASES first, then
    score_per_sound.resolve_label); the raw text stays in `label`. Loadable with score_per_sound.load_gold.
    NOT merged into gold_AG.json (that file's "test" subset is gold minus DEV, so any clip added there would count
    as TEST).
  - moves those videos out of the tagger package: tagger/videos/dNNN.mp4 -> data/input/tagger_set/tg_dNNN.mp4
    (clips marked broken -> data/input/tagger_set/_bad/), and rewrites tagger/src/clips.js without them, so the
    annotator package only lists clips still to tag.
Re-run with each newer export: clips already moved stay in the set and keep their latest tags.

    python benchmark/gold/tagger_set.py D:/Downloads/tagger_AG_2026-09-29_2248.json
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

TAGGER = _ROOT / "tagger"
CLIPS_JS = TAGGER / "src" / "clips.js"
MEDIA = _ROOT / "data" / "input" / "tagger_set"
OUT_DIR = _ROOT / "benchmark" / "gold" / "annotations"

# free text in the tagger -> ontology name, where resolve_label would pick a wrong or no name
# (kept here, not in score_per_sound.ALIASES, so the frozen DEV/TEST scoring is untouched)
TAGGER_ALIASES = {
    "car honk": "Vehicle horn, car horn, honking",      # resolve_label -> "Honk" (a goose)
    "fire alert ring": "Fire alarm",                     # -> "Fire"
    "screams": "Screaming",                              # unresolved
    "car winker": "Tick",                                # a car's turn indicator ticking; unresolved
    "soazzle (sand raking)": "Scrape",                   # unresolved
}


def family_of(text: str) -> str:
    t = (text or "").strip()
    if t.lower() in TAGGER_ALIASES:
        return TAGGER_ALIASES[t.lower()]
    return S.resolve_label(t)


def tg(clip: str) -> str:
    return "tg_" + Path(clip).stem


def main(export: str):
    d = json.loads(Path(export).read_text(encoding="utf-8"))
    who = (d.get("annotator") or "anon").strip()
    out = OUT_DIR / f"tagger_{re.sub(r'[^A-Za-z0-9_-]+', '_', who)}.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"clips": []}
    keep = {c["clip"]: c for c in prev["clips"]}
    names = set(S._ontology_names())
    moved, bad, unresolved = [], [], []
    MEDIA.mkdir(parents=True, exist_ok=True)
    (MEDIA / "_bad").mkdir(exist_ok=True)
    for c in d["clips"]:
        if not c.get("done"):
            continue
        src = TAGGER / "videos" / c["clip"]
        if c.get("bad"):
            dst = MEDIA / "_bad" / (tg(c["clip"]) + ".mp4")
            bad.append(c["clip"])
        else:
            dst = MEDIA / (tg(c["clip"]) + ".mp4")
            cc = dict(c, clip=tg(c["clip"]) + ".mp4", source_clip=c["clip"])
            cc["sounds"] = []
            for s in c.get("sounds", []):
                s = dict(s)
                fam = family_of(s.get("family") or s.get("label"))
                if fam not in names:
                    unresolved.append((c["clip"], s.get("label")))
                s["family"] = fam
                cc["sounds"].append(s)
            keep[cc["clip"]] = cc
            moved.append(c["clip"])
        if src.exists():
            shutil.move(str(src), str(dst))
    res = {"annotator": who, "source_exports": sorted(set(prev.get("source_exports", [])) | {Path(export).name}),
           "note": "second gold set from the tagger tool (not DEV/TEST); built by benchmark/gold/tagger_set.py",
           "clips": [keep[k] for k in sorted(keep)]}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    # the annotator package lists only clips whose video is still in tagger/videos
    txt = CLIPS_JS.read_text(encoding="utf-8")
    clips = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
    left = [c for c in clips if (TAGGER / c["src"]).exists()]
    CLIPS_JS.write_text("window.DELEGATION_CLIPS = " + json.dumps(left, separators=(",", ":")) + ";\n", encoding="utf-8")
    gold = S.load_gold([out])
    n_snd = sum(len(v) for v in gold.values())
    n_need = sum(1 for v in gold.values() for s in v if s["needed"] and s["importance"] >= 2)
    print(f"set {out.relative_to(_ROOT)}: {len(gold)} clips, {n_snd} scored sounds, {n_need} needed (importance >= 2)")
    print(f"moved this run: {len(moved)} good -> {MEDIA.relative_to(_ROOT)}, {len(bad)} broken -> _bad; "
          f"tagger now lists {len(left)} clips")
    print("unresolved labels:", unresolved or "none")


if __name__ == "__main__":
    main(sys.argv[1])

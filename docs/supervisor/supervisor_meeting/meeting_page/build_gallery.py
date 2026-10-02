"""Build the gallery block of page.html from the videos present in demos/v3 and the v3
results: for each clip its scenario, the judge's scores for the three systems, the
reference sentence, what the describer wrote for our panel, the judge's reason, and what the
detector heard with the gate's verdict. Then rewrite the <script id="galleryData"> block.

    python docs/supervisor_meeting/meeting_page/build_gallery.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
sys.path.insert(0, str(root))
from benchmark.gate_dev_sweep import decide, min_confidence

results = json.loads((root / "benchmark" / "protocol_results_v3_grounded.json").read_text(encoding="utf-8"))
by = {}
for r in results:
    by.setdefault(r["clip"], {})[r["system"]] = r
gate = json.loads((root / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
TAGNAME = {"unseen_ambient": "unseen", "mixed": "mixed", "seen_ambient": "seen", "no_ambient": "no ambient"}
DUE = {"unseen_ambient", "mixed"}

items = []
for f in sorted((here / "demos" / "v3" / "clean").glob("*_augmented.mp4")):
    stem = f.name[:-len("_augmented.mp4")]
    clip = next((c for c in by if Path(c).stem == stem), None)
    rec = by.get(clip, {})
    p, b, c = rec.get("proposed"), rec.get("blind_a2i"), rec.get("audio_caption")
    votes_p = root / "benchmark" / "gate_votes" / "test" / (clip + ".json") if clip else None
    heard = []
    if votes_p and votes_p.exists():
        v = json.loads(votes_p.read_text(encoding="utf-8"))
        d = decide(v, gate["bar"], gate["rule"], gate["kinds"])
        for s in v["sounds"]:
            if s["confidence"] >= min_confidence(s["label"], gate["bar"]):
                heard.append({"label": s["label"], "conf": round(s["confidence"], 2),
                              "verdict": "shown" if s["label"] in d["shown"] else ("silenced" if s["label"] in d["silenced"] else "dropped")})
    tag = (p or {}).get("human_tag") or ""
    items.append({
        "stem": stem, "tag": TAGNAME.get(tag, tag or "dev"), "due": tag in DUE,
        "scores": {"ours": (p or {}).get("score"), "blind": (b or {}).get("score"), "caption": (c or {}).get("score")},
        "n_pictures": (p or {}).get("n_augmentations"),
        "reference": (p or {}).get("reference", ""), "ours_says": (p or {}).get("description", ""),
        "judge": (p or {}).get("why", ""), "heard": heard,
        "debug": (here / "demos" / "v3" / "debug" / f.name).exists(),
    })

# order: mixed and unseen with pictures first, then the rest; within a group, worst-to-best is
# more useful for discussion than alphabetical
items.sort(key=lambda x: (not x["due"], x["tag"], -(x["scores"]["ours"] if x["scores"]["ours"] is not None else -1)))
block = '<script id="galleryData" type="application/json">' + json.dumps(items, ensure_ascii=False) + '</script>'
page = here / "page.html"
s = page.read_text(encoding="utf-8")
s2, n = re.subn(r'<script id="galleryData" type="application/json">.*?</script>', lambda m: block, s, flags=re.S)
if n == 0:
    s2 = s.replace("<script>\n(function(){", block + "\n<script>\n(function(){", 1)
page.write_text(s2, encoding="utf-8")
print(f"gallery: {len(items)} clips ({sum(i['due'] for i in items)} with a picture due; {sum(i['debug'] for i in items)} with a debug view)")

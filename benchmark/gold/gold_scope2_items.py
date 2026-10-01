"""Round 46b GOLD-SCOPE-2 (1 Oct, Adam offered to investigate the mistakes): blind rating of (W) the current DEV wrong pictures
not rated in Round 46, (L) the 2 hits Round 53 WEAK-WITNESS loses, (C) 15 random current hits as controls. Same page and
questions as Round 46 plus a free-text "what do you actually hear?". Uses (pre-registered): gold corrections only by the
established blind-with-controls process and only on Adam's say-so; the thesis error taxonomy. Never TEST.

    python benchmark/gold/gold_scope2_items.py build | score
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gold_scope_items as G1

G = _ROOT / "benchmark" / "gold"
KEY, ANS = G / "gold_scope2_items.json", G / "gold_scope2_answers.json"
HTML, MEDIA = _ROOT / "docs" / "review" / "gold_scope2_recheck.html", _ROOT / "docs" / "review" / "gold_scope2_media"
RATED = {("mv_protest_scene_movie", 4.75), ("b3_crossing_bells", 0.22), ("un_hair_dryer_drying_WWu24rJs", 11.25),
         ("b3_laundromat", 1.0), ("tg_d088", 10.75), ("b3_golf_course", 18.84)}          # rated in Round 46
DROPPED = {("b3_crossing_bells", 0.22)}                                                  # Round 57 removes it
LOST53 = [("tg_d032", "Thunder", 13.75), ("tg_d075", "Alarm", 0.14)]


def build():
    rng = random.Random(1)
    L = json.loads((G / "ledger_SHIP8_MD3.json").read_text(encoding="utf-8"))
    W = [{"set": "W", "clip": w["clip"], "family": w["picture"], "at": w["at"], "cls": w["type"]} for w in L["wrong"]
         if (w["clip"], w["at"]) not in RATED | DROPPED]
    Ls = [{"set": "L", "clip": c, "family": f, "at": a, "cls": "hit (lost by Round 53)"} for c, f, a in LOST53]
    miss = {(m["clip"], round(m["at"], 2)) for m in L["misses"]}
    pool = [(c["clip"], g) for c in json.loads((G / "grp" / "pairs_dev.json").read_text(encoding="utf-8"))["clips"]
            for g in c["gold"] if g["needed"] and (c["clip"], round(g["start"], 2)) not in miss
            and (c["clip"], g["label"]) not in {(x[0], x[1]) for x in LOST53}]
    C = [{"set": "C", "clip": c, "family": g["label"], "at": float(g["start"]), "cls": "hit (control)"}
         for c, g in rng.sample(pool, 15)]
    items = W + Ls + C
    rng.shuffle(items)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(items, 1):
        x["id"] = f"h{i:02d}"
        s = max(0.0, x["at"] - G1.PRE)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-i", str(G1.video(x["clip"])), "-t",
                        f"{G1.PRE + G1.POST:.2f}", "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf",
                        "28", "-c:a", "aac", "-b:a", "128k", str(MEDIA / f"{x['id']}.mp4")], check=True)
    KEY.write_text(json.dumps({"items": items}, indent=1), encoding="utf-8")
    pub = [{"id": x["id"], "file": f"{x['id']}.mp4", "family": x["family"]} for x in items]
    html = G1.TEMPLATE.replace("__ITEMS__", json.dumps(pub)).replace("gold_scope_media/", "gold_scope2_media/")
    html = html.replace("gold_scope_r46", "gold_scope_r46b").replace("gold_scope_answers.json", "gold_scope2_answers.json")
    html = html.replace("(41 short moments)", f"round 2 ({len(items)} short moments)")
    html = html.replace('<div class="q">Note (optional)</div>', '<div class="q">What do you actually hear here? (optional)</div>')
    HTML.write_text(html, encoding="utf-8")
    print(f"{len(items)} items: W {len(W)}, L {len(Ls)}, C {len(C)} -> {HTML}")


def score():
    items = {x["id"]: x for x in json.loads(KEY.read_text(encoding="utf-8"))["items"]}
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    ans = ans.get("answers", ans)
    for s in ("C", "L", "W"):
        print(f"== {s}")
        for i, x in sorted(items.items(), key=lambda t: t[1]["clip"]):
            if x["set"] == s:
                print(f"   {i} {x['clip']} {x['family']} {x['at']} {x['cls']}: {ans.get(i, {})}")


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()

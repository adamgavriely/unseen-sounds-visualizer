"""Round 41 NAMED-VETO (docs/prereg_round13_detector_push.md): drop a placed SHIP8 picture when the gate stretch(es) covering
its onset name a DIFFERENT depictable maker. CPU, saved SHIP8 pictures of merged DEV, rescored with score_per_sound.

    TG_ARMS=SHIP8 python benchmark/gold/named_veto_screen.py   -> benchmark/gold/named_veto_screen.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import gate_gold as G
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from benchmark.gold.dbr_screen import passes
from benchmark.gold import dbr_screen as DBR
from src import labels as L

B.ARM = "SHIP8"
DBR.BASE.update({"hits": 28, "wrong": 21, "cost": 2.282})
CACHE = G.OUT_DIR / "Qwen38-27B"
OUT = _ROOT / "benchmark" / "gold" / "named_veto_screen.json"

# declared noun table (prereg Round 41 NAMED-VETO, step 2): things AudioSet hides or lacks
NOUNS = {
    "locomotive": "Train", "tram": "Train", "streetcar": "Train", "taxi": "Car", "suv": "Car", "sedan": "Car",
    "van": "Truck", "pickup": "Truck", "ambulance": "Ambulance (siren)", "rifle": "Gunshot, gunfire",
    "shotgun": "Gunshot, gunfire", "ak-47": "Gunshot, gunfire", "gun": "Gunshot, gunfire",
    "tank main gun": "Artillery fire", "tank": "Artillery fire", "cellphone": "Cellphone buzz, vibrating alert",
    "phone": "Cellphone buzz, vibrating alert", "fountain": "Water", "waves": "Water", "cockpit": "Helicopter",
    "loudspeaker": "Loudspeaker",
}


def _words(text):
    return re.findall(r"[a-z0-9\-]+", text.lower())


def _whole(phrase, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def resolve_named(text):
    """-> ontology label or None (prereg steps 1-3; depictability is step 4, checked separately)"""
    low = (text or "").strip().lower()
    if not low or low == "nothing":
        return None
    if low in S.ALIASES:
        return S.ALIASES[low]
    for k in sorted(NOUNS, key=len, reverse=True):
        if _whole(k, low) or _whole(k + "s", low):
            return NOUNS[k]
    names = S._ontology_names()
    cands = []
    for n in names:
        for part in [n.lower()] + [p.strip().lower() for p in n.split(",")]:
            if len(part) < 3:
                continue
            if _whole(part, low) or _whole(part + "s", low) or (part.endswith("s") and _whole(part[:-1], low)):
                cands.append((n not in L.ACTION_LABELS, len(part), n))
                break
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][2]


def depictable(label) -> bool:
    if not label:
        return False
    with R.flags({"LABEL_FILTER": "depictable"}):
        return bool(L.is_salient_nonspeech(label))


def main():
    P = B.parts()
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures")
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (28, 21) and abs(Bm["merged"]["cost"] - 2.282) < 0.001, Bm
    DBR.BASE["cost"] = Bm["merged"]["cost"]
    cache = {}
    for f in CACHE.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        cache[f.stem] = [(s["label"], float(st["start"]), float(st["end"]), st.get("named"))
                         for s in d["sounds"] for st in s["stretches"]]
    res = {}
    for variant in ("primary", "any"):
        rows = {"dev": [], "dev2": []}; dropped = []; lost = []
        cnt = {"pictures": 0, "no cached clip": 0, "no covering stretch": 0, "covered, no depictable name": 0,
               "covered, own family named": 0, "covered, other family only": 0, "dropped": 0}
        for pt, st, g, pics in P:
            cl = classify(g, [p[:3] for p in pics])
            before = {(l, round(a, 3)): k for l, a, b, k, _ in cl}
            keep = []
            for l, a, b, resc in pics:
                cnt["pictures"] += 1
                strs = cache.get(st)
                if strs is None:
                    cnt["no cached clip"] += 1; keep.append((l, a, b)); continue
                cov = [(lab, s0, s1, nm) for lab, s0, s1, nm in strs if s0 - 1e-6 <= a <= s1 + 1e-6]
                if not cov:
                    cnt["no covering stretch"] += 1; keep.append((l, a, b)); continue
                named = [(nm, resolve_named(nm)) for *_x, nm in cov]
                fams = [(nm, r) for nm, r in named if r and depictable(r)]
                other = [r for nm, r in fams if not S.same_family(l, r)]
                own = [r for nm, r in fams if S.same_family(l, r)]
                if not fams:
                    cnt["covered, no depictable name"] += 1; keep.append((l, a, b)); continue
                fire = bool(other) and (not own if variant == "primary" else True)
                if own and other:
                    cnt["covered, own family named"] += 1
                elif other:
                    cnt["covered, other family only"] += 1
                else:
                    cnt["covered, own family named"] += 1
                if fire:
                    cnt["dropped"] += 1
                    dropped.append({"part": pt, "clip": st, "label": l, "start": round(a, 2), "end": round(b, 2), "rescued": resc,
                                    "before": before[(l, round(a, 3))],
                                    "covering": [{"sound": lab, "stretch": [round(s0, 2), round(s1, 2)], "named": nm, "family": r}
                                                 for (lab, s0, s1, nm), (_n, r) in zip(cov, named)]})
                else:
                    keep.append((l, a, b))
            r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, keep)
            rows[pt].append(r1)
            oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
            for d in dropped:
                if d["part"] == pt and d["clip"] == st and "clip_after" not in d:
                    d["clip_before"], d["clip_after"] = oc(r0), oc(r1)
            if r1["hit"] < r0["hit"]:
                lost.append((pt, st, r0["hit"] - r1["hit"]))
        X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
        ln = sum(x[2] for x in lost)
        old, few = passes(X["merged"], lost, ln)
        verdict = "GO (main rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"
        print(f"\n[{variant}] merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  hits lost {lost} -> {verdict}")
        print("   ", cnt)
        for d in dropped:
            print(f"   {d['part']:4s} {d['clip']} {d['label']} {d['start']}-{d['end']}: {d['before']} dropped  clip {d['clip_before']} -> {d['clip_after']}")
            for c in d["covering"]:
                print(f"        stretch {c['sound']} {c['stretch']} named={c['named']!r} -> {c['family']}")
        res[variant] = {"rows": X, "counts": cnt, "dropped": dropped, "hits_lost": lost, "main_rule": old, "fewer_pictures": few,
                        "verdict": verdict}
    OUT.write_text(json.dumps({"base": Bm, "nouns": NOUNS, **res}, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

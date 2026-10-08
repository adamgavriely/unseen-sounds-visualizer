"""Step 11 (PREREG_step11_display_policies.md): texture-parent ban (T), common-ancestor picture (A), visible flash (F)
on top of a base set of DEV pictures. CPU, local.
    python benchmark/gold/coverage/step11_policies.py [PICS_JSON CELL]   -> step11_dev.md
Default base: step2_pics_dev.json, cell SHIP8+MD3+WW5+SL|AB-m."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.inspector_data import classify
from src.labels import is_descendant, canonical

HERE = Path(__file__).resolve().parent
TEXTURE = {"Vehicle", "Water", "Engine", "Rain", "Wind", "Liquid", "Mechanisms", "Domestic sounds, home sounds"}
GROUPS = [({"Gunshot, gunfire", "Gunshot", "Explosion", "Artillery fire", "Machine gun"}, "Explosion"),
          ({"Siren", "Shofar", "Alarm", "Air horn, truck horn"}, "Alarm"),
          ({"Honk", "Gull, seagull", "Bird", "Duck"}, "Bird")]
BLAST = {"Explosion", "Fireworks", "Gunshot, gunfire", "Gunshot", "Machine gun", "Artillery fire"}


def in_group(lab, grp):
    return lab in grp or canonical(lab) in grp


def provenance(gold):
    """{(clip, gold index in S.load_gold order): from_detector} by matching the raw export on label / family + times"""
    raw = json.loads(V.GOLD.read_text(encoding="utf-8"))
    out = {}
    for c in raw["clips"]:
        if not isinstance(c, dict):
            continue
        st = Path(c.get("clip", "")).stem
        if st not in gold:
            continue
        for i, g in enumerate(gold[st]):
            m = [s for s in c.get("sounds", []) if s.get("start") is not None and abs(float(s["start"]) - g["start"]) < 1e-6
                 and abs(float(s["end"]) - g["end"]) < 1e-6]
            out[(st, i)] = bool(m and m[0].get("from_detector"))
    return out


def policy(st, pics, which, af, cands, flashes):
    out = []
    for lab, a, b in pics:
        if "T" in which and lab in TEXTURE:
            names = [f for x in af.get(st, []) if x["end"] > a and x["start"] < b for f in (x.get("af_fams") or [])]
            if not any(f != lab and is_descendant(f, lab) for f in names):
                continue
        if "A" in which:
            for grp, parent in GROUPS:
                if in_group(lab, grp) and any(in_group(c["label"], grp) and canonical(c["label"]) != canonical(lab)
                                              and abs(c["start"] - a) <= 1.0 for c in cands.get(st, [])):
                    lab = parent
                    break
        if "F" in which and flashes is not None:
            fl = flashes.get(st, {}).get("flashes", [])
            if canonical(lab) == "Thunder" or lab == "Thunder":
                if any(a - 2.0 <= t <= b for t in fl):
                    continue
            if lab in BLAST or canonical(lab) in BLAST:
                if any(abs(t - a) <= 0.3 for t in fl):
                    continue
        out.append((lab, a, b))
    return out


def main():
    pj = sys.argv[1] if len(sys.argv) > 1 else "step2_pics_dev.json"
    cell = sys.argv[2] if len(sys.argv) > 2 else "SHIP8+MD3+WW5+SL|AB-m"
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    base = {st: [tuple(p) for p in json.loads((HERE / pj).read_text(encoding="utf-8"))["cells"][cell]["clips"][st]["pics_none"]] for st in dev}
    af = {}
    for f in ("p1v4_dev_old.json", "p1v4_dev2_old.json"):
        for x in json.loads((HERE / f).read_text(encoding="utf-8"))["items"]:
            af.setdefault(x["clip"], []).append(x)
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    cands = {c["clip"]: c["cands"] for c in dj["clips"] if c["split"] == "DEV"}
    fp = HERE / "flashes_dev.json"
    flashes = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
    prov = provenance(gold)
    L = [f"# Step 11: display policies on top of {cell} (DEV)", "",
         "| policy | hits | wrong (vis / other / none) | onset cost | cost_cov | hits on detector-suggested / plain human labels |",
         "|---|---|---|---|---|---|"]
    runs = ["", "T", "A", "TA"] + (["F", "TAF"] if flashes is not None else [])
    for w in runs:
        rows, hd, hh = [], 0, 0
        for st in dev:
            pics = policy(st, base[st], w, af, cands, flashes)
            rows.append(V.score_clip_v2(gold[st], pics))
            per_sound, _ = classify(gold[st], pics)
            for i, o in enumerate(per_sound):
                if o["outcome"] == "hit":
                    if prov.get((st, i)):
                        hd += 1
                    else:
                        hh += 1
        A = V.aggregate(rows)
        v = sum(r["visible"] for r in rows); c = sum(r["cross"] for r in rows); p = sum(r["phantom"] for r in rows)
        L.append(f"| {w or 'base'} | {A['hits']} | {A['wrong']} ({v} / {c} / {p}) | {A['onset_cost']:.3f} | {A['cost_cov']:.3f} | {hd} / {hh} |")
    if flashes is not None:
        L += ["", "Flashes per DEV clip (non-zero): " + ", ".join(f"{st} {len(v['flashes'])}" for st, v in sorted(flashes.items()) if v["flashes"])]
    (HERE / "step11_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

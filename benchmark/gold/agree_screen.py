"""Round 41 AGREE (docs/prereg_round13_detector_push.md): a shipped-majority "seen" stretch flips to not-seen only if BOTH
the Round 38 (b) SYNC veto and Round 38 BOX-2 flip it. Stretch level; sound seen iff every stretch seen. DEV judge clips.

    python benchmark/gold/agree_screen.py          # CPU -> benchmark/gold/gate_gold/agree_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gate_gold as G
from benchmark.gold import sync_gate as SG
from benchmark.gold import box_gate as BG

SUMMARY = G.OUT_DIR / "agree_summary.json"
BASE = (16, 41, 33, 38)


def _key(stem, s, st):
    return (stem, s["label"], round(float(s["start"]), 2), round(float(st["start"]), 2))


def veto_flip(st, t) -> bool:
    yes, no = SG._votes(st)
    v = SG.sync_of(st)
    return (yes > no) and yes != 3 and (v is None or v < t)


def box_flip(st) -> bool:
    return BG.majority(st) and not BG.seen_box(st, "BOX2")


def main():
    t = json.loads(SG.SUMMARY.read_text(encoding="utf-8"))["t"]
    gold = BG.gold_index()
    judge = set(G.JUDGE100.read_text().split())
    box = {}
    for f in sorted(BG.OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            for st in s["stretches"]:
                box[_key(f.stem, s, st)] = st
    rows = []
    for f in sorted(SG.OUT.glob("*.json")):
        if f.stem not in judge:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            s["_seen"] = bool(g["seen"]); s["_stem"] = f.stem
            for st in s["stretches"]:
                st["_box"] = box.get(_key(f.stem, s, st))
            rows.append(s)
    n_box = sum(st["_box"] is not None for s in rows for st in s["stretches"])
    print(f"DEV judge sounds {len(rows)}, stretches {sum(len(s['stretches']) for s in rows)}, with box record {n_box}; t = {t:.5f}")

    def seen(s, rule):
        for st in s["stretches"]:
            maj = BG.majority(st)
            if rule == "majority":
                ok = maj
            elif rule == "veto":
                ok = maj and not veto_flip(st, t)
            elif rule == "box2":
                ok = maj and not (st["_box"] is not None and box_flip(st["_box"]))
            else:   # agree
                ok = maj and not (veto_flip(st, t) and st["_box"] is not None and box_flip(st["_box"]))
            if not ok:
                return False
        return True

    out = {"t": t, "dev": {}, "flips": {}, "stretch_flips": {}}
    base = {id(s): seen(s, "majority") for s in rows}
    for rule in ("majority", "veto", "box2", "agree"):
        a = na = b = nb = 0
        flips = []
        for s in rows:
            p = seen(s, rule)
            if s["_seen"]:
                na += 1; a += int(p)
            else:
                nb += 1; b += int(not p)
            if p != base[id(s)]:
                flips.append({"clip": s["_stem"], "label": s["label"], "start": s["start"], "gold_seen": s["_seen"],
                              "base_seen": base[id(s)], "new_seen": p,
                              "votes": [SG._votes(st) for st in s["stretches"]], "sync": [SG.sync_of(st) for st in s["stretches"]],
                              "box": [(st["_box"] or {}).get("box", {}).get("status") if st["_box"] else None for st in s["stretches"]]})
        out["dev"][rule] = {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb}
        out["flips"][rule] = flips
        print(f"DEV judge  {rule:9s} seen silenced {a}/{na}  needed kept {b}/{nb}  flips {len(flips)}")
        for x in flips:
            tag = "good" if x["new_seen"] == x["gold_seen"] else "BAD "
            print(f"   {tag} {x['clip']} {x['label']} {x['start']:.1f}s gold_seen={x['gold_seen']} {x['base_seen']}->{x['new_seen']} "
                  f"votes={x['votes']} sync={x['sync']} box={x['box']}")
    # stretch-level agreement counts
    nv = nb2 = nag = 0
    for s in rows:
        for st in s["stretches"]:
            v = veto_flip(st, t); b2 = st["_box"] is not None and box_flip(st["_box"])
            nv += v; nb2 += b2; nag += v and b2
    out["stretch_flips"] = {"veto": nv, "box2": nb2, "both": nag}
    print("stretch flips: veto", nv, "box2", nb2, "both", nag)
    assert (out["dev"]["majority"]["seen_silenced"], out["dev"]["majority"]["seen"],
            out["dev"]["majority"]["needed_kept"], out["dev"]["majority"]["needed"]) == BASE, out["dev"]["majority"]
    r = out["dev"]["agree"]
    go = (r["seen_silenced"] >= 19 and r["needed_kept"] >= 32) or (r["needed_kept"] >= 35 and r["seen_silenced"] >= 15)
    out["GO"] = go
    print("Round 41 AGREE:", "GO" if go else "STOP")
    if go:
        try:
            out["pictures"] = SG.pictures_on_flips(out["flips"]["agree"])
            print("SHIP8 pictures on agree flips:", out["pictures"])
        except Exception as e:
            out["pictures_error"] = repr(e)[:300]; print("pictures:", repr(e)[:300])
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(SUMMARY)


if __name__ == "__main__":
    main()

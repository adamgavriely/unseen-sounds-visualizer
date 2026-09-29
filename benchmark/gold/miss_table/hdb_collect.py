"""per-needed-sound chain evidence on DEV for the best arm TO1+F7F8 (TIER + ONCE + R13-1 + F7 + F8); raw rows -> hdb_raw.json
(adapted from hd_collect.py: real arm outputs copied from the cluster, no local filter replay)"""
import json, sys
from pathlib import Path

ROOT = Path("P:/MscProj")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hd_collect as H
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import score_per_sound as S
from src.labels import canonical

OUTD = Path(__file__).resolve().parent
ARM = "TO1+F7F8"
R13 = ROOT / "data/work/r13"
S4 = R13 / "stage4_r14.json"             # cluster ~/MscProj/data/work/r13/stage4.json, copied read-only
PDIR = R13 / f"{ARM}_proposed"           # cluster folder, copied read-only


def ovl(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0) > -1e-9


def v4txt(t, n=3):
    seen = []
    for x in (t or "").split("\n"):
        x = x.strip()
        if x and x != "Assistant" and x not in seen and len(x) > 2:
            seen.append(x)
    return "/".join(seen[:n])


def main():
    gold, stems = C.dev_stems()
    rd = json.loads((ROOT / "benchmark/gold/round13_dev.json").read_text(encoding="utf-8"))
    d4 = json.loads(S4.read_text(encoding="utf-8"))
    s4 = d4["arms"]
    LV = json.loads((ROOT / "benchmark/gold/dev_listener_v.json").read_text(encoding="utf-8"))["items"]
    LA = json.loads((ROOT / "benchmark/gold/dev_listener_afn.json").read_text(encoding="utf-8"))["items"]
    afk = {(x["clip"], x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2)): x for x in LA}
    P0 = {st: S.load_pictures(R13 / "B0r_proposed", st, "proposed") or [] for st in stems}
    PB = {st: S.load_pictures(PDIR, st, "proposed") or [] for st in stems}
    chk = C.metrics([S.score_clip(gold[st], PB[st]) for st in stems])
    ref = rd["rows"]["proposed"][ARM]
    print("arm pictures:", chk["hits"], chk["wrong"], "ref:", ref["hits"], ref["wrong"])
    assert chk["hits"] == ref["hits"] and chk["wrong"] == ref["wrong"]
    nc = rd["needed_changes"]["proposed"][ARM]
    out, wrong = [], []
    for st in stems:
        k = f"{ARM}|proposed|{st}"
        rows = s4[f"{ARM}|proposed"][st]
        drp = d4["r14_dropped"].get(k) or {}
        lst = d4["listener"].get(k) or {}
        f7 = d4["f7"].get(k) or []
        mir = d4["mirror"].get(k) or []
        specs = json.loads((PDIR / st / "augmentations.json").read_text(encoding="utf-8"))
        gp = PDIR / st / "gate_votes.json"
        votes = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else []
        fr = {"beats": C.load_fr(H.BEATS / f"{st}.npz"), "flex": C.load_fr(H.FLEX / f"{st}.npz"),
              "panns": C.load_fr(H.PANNS / f"{st}.npz"), "xq": H.load_xq(st)}

        def tier(x):
            a = afk.get((st, x["pool"], x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2)))
            v4 = bool((x.get("accept") or {}).get("V4", False))
            av4 = bool(((a or {}).get("accept") or {}).get("V4", False))
            pk = x.get("peak")
            pk = 1.0 if pk is None else float(pk)
            ok = v4 and (pk >= 0.6 or av4)
            return {"pool": x["pool"], "label": x["label"], "family": x["family"], "start": round(x["start"], 2),
                    "end": round(x["end"], 2), "peak": round(pk, 3), "qwen_v4": v4, "qwen_v4_text": v4txt(x.get("v4_text")),
                    "qwen_yes": [f for f in ("V1", "V2", "V3", "V4", "V12") if (x.get("accept") or {}).get(f)],
                    "af_v4": av4 if a else None, "af_v4_text": (a or {}).get("afn_v4_text", ""), "af_yn": (a or {}).get("afn_yn_x"),
                    "tier": ok, "tier_fail": None if ok else ("Qwen V4 no" if not v4 else "peak < 0.6 and AF V4 no")}
        for g, h in C.needed_hit(gold[st], PB[st]):
            lab, on, en = g["label"], g["start"], g["end"]
            wa, wb = on - S.EARLY, on + S.LATE
            key = [st, lab, on]
            h0 = any(S.same_family(l, lab) and S.in_window(a, on, S.EARLY, S.LATE) for l, a, b in P0[st])
            r = {"clip": st, "label": lab, "onset": on, "end": en, "importance": g["importance"], "depictable": H.dep(lab),
                 "hit_B0r": bool(h0), "hit_best": bool(h),
                 "chain_ok": bool(h) == ((h0 or key in nc["gained"]) and key not in nc["lost"])}
            ev = {}
            for kk, f in fr.items():
                v, l, t = H.best(f, lab, wa, wb)
                ev[kk] = {"score": round(v, 3), "label": l, "t": t}
            r["ev"] = ev
            r["heard"] = {kk: ev[kk]["score"] >= H.BARS[kk] for kk in H.BARS}
            r["heard_any"] = any(r["heard"].values())
            r["runs"] = sorted(H.flex_runs(fr["flex"], lab, wa, wb, "215") + H.flex_runs(fr["xq"], lab, wa, wb, "xq"),
                               key=lambda x: -x["peak"])
            fam = lambda l: S.same_family(l, lab)
            r["lis"] = [tier(x) for x in LV if x.get("clip") == st and x.get("pool") in ("P2", "PV")
                        and (fam(x["label"]) or fam(x["family"])) and ovl(x["start"], x["end"], wa, wb)]
            r["lis_p1"] = [{"label": x["label"], "start": round(x["start"], 2), "end": round(x["end"], 2),
                            "yes": [f for f in ("V1", "V2", "V12") if (x.get("accept") or {}).get(f)], "yn": x.get("cached_score")}
                           for x in LV if x.get("clip") == st and x.get("pool") == "P1" and fam(x["label"])
                           and ovl(x["start"], x["end"], wa, wb)]
            r["a_added"] = [x for x in lst.get("a_added", []) if fam(x[0]) and ovl(x[1], x[2], on - 1.0, en + 1.0)]
            r["b_kept"] = [x for x in lst.get("b_kept", []) if fam(x[0]) and ovl(x[1], x[2], on - 1.0, en + 1.0)]
            dd = {f: [x for x in v if fam(x[0]) and ovl(x[1], x[2], on - 1.0, en + 1.0)] for f, v in drp.items()}
            r["dropped"] = {f: v for f, v in dd.items() if v}
            r["mirror_dropped"] = [x for x in mir if fam(x[0]) and ovl(x[1], x[2], wa, wb)]
            r["f7"] = [x for x in f7 if fam(x[0]) and ovl(x[1], x[2], wa, wb)]
            r["s4"] = [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"], bool(x.get("rescued")),
                        S.in_window(x["start"], on, S.EARLY, S.LATE)] for x in rows
                       if fam(x["label"]) and ovl(x["start"], x["end"], on - 1.0, en + 1.0)]
            r["s4_B0r"] = [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"], False,
                            S.in_window(x["start"], on, S.EARLY, S.LATE)] for x in s4["B0r|proposed"][st]
                           if fam(x["label"]) and ovl(x["start"], x["end"], on - 1.0, en + 1.0)]
            r["spec"] = [[s["event_label"], round(s["start"], 2), round(s["end"], 2), round(s.get("confidence", 0), 3),
                          bool(s.get("augment")), s.get("reason", ""), bool(s.get("rescued", False))]
                         for s in specs if fam(s["event_label"]) and ovl(s["start"], s["end"], on - 1.0, en + 1.0)]
            r["gate_votes"] = [[v.get("label"), v.get("stretch"), v.get("seen"), v.get("named", "")] for v in votes
                               if fam(v.get("label", ""))]
            r["pics"] = [[l, round(a, 2), round(b, 2)] for l, a, b in PB[st] if fam(l) and ovl(a, b, on - 1.0, en + 1.0)]
            r["pics_B0r"] = [[l, round(a, 2), round(b, 2)] for l, a, b in P0[st] if fam(l) and ovl(a, b, on - 1.0, en + 1.0)]
            out.append(r)
        # wrong pictures (miss_feats.py: the class of each picture = the change of score_clip's counts when it is added)
        ps = sorted(PB[st], key=lambda p: p[1])
        prev = S.score_clip(gold[st], [])
        f7ok = [x for x in f7 if x[3]]
        for i, p in enumerate(ps):
            cur = S.score_clip(gold[st], ps[:i + 1])
            cls = [c for c in ("hit", "visible", "cross", "phantom", "dup", "dontcare", "collision") if cur[c] != prev[c]]
            prev = cur
            if not any(c in cls for c in ("visible", "cross", "phantom")):
                continue
            typ = [c for c in cls if c in ("visible", "cross", "phantom")][0]
            l, a, b = p
            src = sorted([x for x in rows if S.same_family(x["label"], l) and min(b, x["end"]) - max(a, x["start"]) > -0.01],
                         key=lambda x: abs(x["start"] - a))
            s0 = src[0] if src else None
            if s0 is None:
                origin = "?"
            elif s0.get("rescued"):
                origin = "rescued"
            elif s0["origin"] == "flex":
                origin = "FlexSED"
            else:
                origin = "BEATs"
            kept7 = bool(s0 and s0["origin"] == "tagger" and any(
                canonical(x[0]) == canonical(s0["label"]) and abs(x[1] - s0["start"]) < 0.3 for x in f7ok))
            coll = [[gg["label"], round(gg["start"], 2), round(gg["end"], 2),
                     "needed" if gg["needed"] else ("visible" if gg["visible"] else "obvious")]
                    for gg in gold[st] if S.in_window(a, gg["start"], S.EARLY, S.LATE) or (gg["start"] <= a <= gg["end"])]
            in_b0r = any(q[0] == l and abs(q[1] - a) < 0.01 and abs(q[2] - b) < 0.01 for q in P0[st])
            fb, bb, pb = (H.best(fr[x], l, a - 0.5, a + 1.0) for x in ("flex", "beats", "panns"))
            wrong.append({"clip": st, "label": l, "start": round(a, 2), "end": round(b, 2), "type": typ, "origin": origin,
                          "f7_kept": kept7, "in_B0r": in_b0r,
                          "stage4_src": [[x["label"], round(x["start"], 2), round(x["end"], 2), round(x["conf"], 3), x["origin"],
                                          bool(x.get("rescued"))] for x in src[:3]],
                          "beats_at": [round(bb[0], 3), bb[1]], "flex_at": [round(fb[0], 3), fb[1]],
                          "panns_at": [round(pb[0], 3), pb[1]], "gold_at_time": coll})
    (OUTD / "hdb_raw.json").write_text(json.dumps({"needed": out, "wrong": wrong}, indent=1, default=str), encoding="utf-8")
    print(len(out), "needed;", sum(r["hit_best"] for r in out), "hits;", sum(not r["chain_ok"] for r in out), "chain mismatches;",
          len(wrong), "wrong", {t: sum(w["type"] == t for w in wrong) for t in ("visible", "cross", "phantom")},
          {o: sum(w["origin"] == o for w in wrong) for o in ("BEATs", "FlexSED", "rescued", "?")})


if __name__ == "__main__":
    main()

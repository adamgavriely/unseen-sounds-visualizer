"""Round 32 screens (docs/prereg_round13_detector_push.md, "Round 32"), CPU only, on the saved SHIP6 state of merged DEV
(stage-4 traces, r14_dropped, augmentations.json). Nothing in src/ or config.py is edited.
  (A) FLAP-F8 at arm level: SHIP6 F8-dropped rescued spans with FineLAP span max >= SPAN_BAR restored, ONCE re-run over
      kept + restored rescued spans, gate verdict of a restored span from the F8-less arm TO1+F7 (same family within
      0.5 s; none -> drawn), pictures re-placed through _display_spans and rescored. OR = restore only; R = also drop
      kept rescued spans with FineLAP < SPAN_BAR (FineLAP alone, the arm's rule).
  (B) DV seat at candidate level: SHIP6 non-rescued rows the FineLAP clip bar would drop (no Qwen / AF P1 keep) and the
      DASM-DV-dropped rows (SHIP2+KV4 rows absent from SHIP3+DV) it would keep back, with gold classes.
  (C) SHIP6 ONCE-dropped spans with gold classes (record only).
  CV: per-clip costs of SHIP6 / OR / R, split-half (10 seeds) and 5-fold selection as cv_select.py.

    python benchmark/gold/flap_joint_sim.py          # from ~/MscProj_tg on the cluster (msproj)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import btp_screen as B
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import dev_listener as L
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold.finelap_screen import PARTS, CACHE, item_score
from benchmark.gold.finelap_full import CACHE2, OUT as CALIB, P4
from src.labels import canonical
from src.stage4_audio_event_detection import listener_p1_lookup, _af_p1_accepts

G = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
SPAN_BAR = 0.32922908663749695
BASE = {"hits": 25, "wrong": 28, "cost": 2.479}
PROXY = "TO1+F7"
OUT = G / "flap_joint_sim.json"
PLACEHOLDER = str(_ROOT / "README.md")


def load_fl(clip):
    for d in (CACHE2, CACHE):
        f = d / f"{clip}.npz"
        if f.exists():
            return np.load(f, allow_pickle=True)
    return None


def fl_span(z, label, a, b):
    return None if z is None else item_score(z, {"label": label, "start": a, "end": b})


def fl_clip(z):
    if z is None:
        return {}
    labs = [str(v) for v in z["labels"]]
    return {canonical(l): float(z["scores"][:, i].max()) for i, l in enumerate(labs)}


def parts():
    """(part, stem, gold, out_dir, stage4 dict, listener caches) per clip, DEV then tagger DEV (as btp_screen.parts)"""
    gold, stems = DCC.dev_stems()
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    cfg = R.arm_cfg("SHIP6")
    P = [("dev", st, gold[st], R.R13, s4, cfg) for st in stems]
    import os
    os.environ.setdefault("TG_ARMS", "SHIP6")        # configure remaps the listener caches only for arms named here
    from benchmark.gold import tagger_prep as T
    DCC2, R2, stems2 = T.configure("dev2")
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    s42 = json.loads(R2.STAGE4.read_text(encoding="utf-8"))
    cfg2 = R2.arm_cfg("SHIP6")                       # the dev2-remapped caches (configure rewrote the arm's paths)
    P += [("dev2", st, g2[st], T.out("dev2"), s42, cfg2) for st in stems2 if st in g2]
    return P


def peaks(part):
    """(clip, family, start, end) -> listener-pool peak of P2 / PV / P4 items (the conf a restored rescued span carries)"""
    out = {}
    for f in (PARTS[part]["v"], P4[part]):
        if not Path(f).exists():
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for x in (d["items"] if isinstance(d, dict) else d):
            if x.get("pool") in ("P2", "PV", "P4") and x.get("peak") is not None:
                out.setdefault((x["clip"], canonical(x.get("family") or x["label"]), round(float(x["start"]), 2),
                                round(float(x["end"]), 2)), float(x["peak"]))
    return out


def find_peak(pk, clip, label, a, b):
    k = (clip, canonical(label), round(a, 2), round(b, 2))
    if k in pk:
        return pk[k]
    c = [(abs(s - a) + abs(e - b), v) for (cl, f, s, e), v in pk.items() if cl == clip and f == k[1] and abs(s - a) <= 0.05 and abs(e - b) <= 0.05]
    return min(c)[1] if c else None


def specs_of(root, st):
    f = root / st / "augmentations.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def place(specs, root, st):
    m = root / st / "media.json"
    dur = (float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None) if m.exists() else None
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = []
    for x in specs:
        objs.append(AugmentationSpec(index=x.get("index", 0), event_label=x["event_label"], start=float(x["start"]), end=float(x["end"]),
                                     augment=bool(x.get("augment")), confidence=float(x.get("confidence", 0)), image_path=x.get("image_path"),
                                     talked_about=bool(x.get("talked_about")), spans=[tuple(y) for y in x.get("spans", [])],
                                     breaks=[tuple(y) for y in x.get("breaks", [])]))
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    pl, _ = _assign_rows(_display_spans(objs, d, require_image=True))
    return [(lab, float(a), float(b)) for _, lab, a, b, _sp in pl]


def gate_proxy(root, st, label, a):
    """the F8-less arm's verdict on the same span: (augment, found)"""
    c = [x for x in specs_of(root / f"{PROXY}_proposed", st) if canonical(x["event_label"]) == canonical(label) and abs(float(x["start"]) - a) <= 0.5]
    if not c:
        return True, False
    return bool(min(c, key=lambda x: abs(float(x["start"]) - a)).get("augment")), True


def sim_a(P, pk):
    """-> per variant: rows (part -> score_clip list), log"""
    res = {v: {"rows": {"dev": [], "dev2": []}, "restored": [], "displaced": [], "newly_dropped": [], "no_conf": 0, "no_gate": 0}
           for v in ("OR", "R")}
    base = {"dev": [], "dev2": []}
    for part, st, g, root, s4, cfg in P:
        rows = s4["arms"]["SHIP6|proposed"][st]
        drop = s4.get("r14_dropped", {}).get(f"SHIP6|proposed|{st}", {})
        z = load_fl(st)
        specs = specs_of(root / "SHIP6_proposed", st)
        base[part].append(S.score_clip(g, place(specs, root / "SHIP6_proposed", st)))
        kept = [(r["label"], float(r["start"]), float(r["end"]), float(r["conf"])) for r in rows if r.get("rescued")]
        for v in ("OR", "R"):
            X = res[v]
            rest = []
            for lab, a, b, _d in drop.get("F8", []):
                s = fl_span(z, lab, a, b)
                if s is not None and s >= SPAN_BAR:
                    c = find_peak(pk[part], st, lab, a, b)
                    if c is None:
                        X["no_conf"] += 1; c = 0.5
                    rest.append((lab, float(a), float(b), c))
            keep2 = list(kept)
            if v == "R":
                for lab, a, b, c in kept:
                    s = fl_span(z, lab, a, b)
                    if s is not None and s < SPAN_BAR:
                        keep2.remove((lab, a, b, c))
                        X["newly_dropped"].append([part, st, lab, round(a, 2), round(b, 2), L.gold_class(g, lab, a, b)])
            first, surv = {}, []
            for e in sorted(keep2 + rest, key=lambda e: (e[1], e[2])):
                fam = canonical(e[0])
                if fam in first:
                    continue
                first[fam] = e; surv.append(e)
            sp = [x for x in specs]
            for lab, a, b, c in kept:
                if (lab, a, b, c) not in surv and (lab, a, b, c) in keep2:
                    sp = [x for x in sp if not (x.get("rescued") and canonical(x["event_label"]) == canonical(lab) and abs(float(x["start"]) - a) <= 0.05)]
                    X["displaced"].append([part, st, lab, round(a, 2), L.gold_class(g, lab, a, b)])
                elif (lab, a, b, c) not in keep2:
                    sp = [x for x in sp if not (x.get("rescued") and canonical(x["event_label"]) == canonical(lab) and abs(float(x["start"]) - a) <= 0.05)]
            for lab, a, b, c in surv:
                if (lab, a, b, c) in kept:
                    continue
                aug, found = gate_proxy(root, st, lab, a)
                X["no_gate"] += not found
                cls = L.gold_class(g, lab, a, b)
                X["restored"].append([part, st, lab, round(a, 2), round(b, 2), round(c, 3), "drawn" if aug else "gated", cls, "proxy" if found else "no-proxy"])
                if aug and c >= float(getattr(config, "PICTURE_MIN_CONF", 0.40) or 0.40):
                    sp.append({"event_label": lab, "start": a, "end": b, "augment": True, "confidence": c, "image_path": PLACEHOLDER,
                               "spans": [[a, b]], "rescued": True})
            X["rows"][part].append(S.score_clip(g, place(sp, root / "SHIP6_proposed", st)))
    return base, res


def sim_b(P, bar):
    out = {"would_drop": [], "kept_back": [], "unqueried": 0}
    for part, st, g, root, s4, cfg in P:
        z = load_fl(st); dpk = fl_clip(z)
        look = listener_p1_lookup(cfg.get("LISTENER_VCACHE"), cfg.get("LISTENER_CACHE"), st)
        config.LISTENER_AFCACHE = cfg.get("LISTENER_AFCACHE"); config._CURRENT_CLIP = st
        from src.types import AudioEvent
        disp = 0.40

        def keep(lab, a, b):
            return bool(look(lab, a, b)[0]) or _af_p1_accepts(AudioEvent(lab, a, b, 1.0))
        for r in s4["arms"]["SHIP6|proposed"][st]:
            if r.get("rescued") or r["conf"] < disp:
                continue
            fam = canonical(r["label"])
            if fam not in dpk:
                out["unqueried"] += 1; continue
            if dpk[fam] < bar and not keep(r["label"], r["start"], r["end"]):
                out["would_drop"].append([part, st, r["label"], round(r["start"], 2), round(dpk[fam], 3), L.gold_class(g, r["label"], r["start"], r["end"])])
        sig = lambda r: (r["label"], round(r["end"], 2), round(r["conf"], 3))
        have = {sig(r) for r in arm_rows(s4, ("SHIP3+DV", "SHIP4"), st)}
        for r in arm_rows(s4, ("SHIP2+KV4", "SHIP3"), st):
            if r.get("rescued") or r["conf"] < disp or sig(r) in have:
                continue
            fam = canonical(r["label"])
            if dpk.get(fam, 1.0) >= bar:
                out["kept_back"].append([part, st, r["label"], round(r["start"], 2), round(dpk.get(fam, 1.0), 3), L.gold_class(g, r["label"], r["start"], r["end"])])
    return out


def arm_rows(s4, names, st):
    for n in names:
        if f"{n}|proposed" in s4["arms"]:
            return s4["arms"][f"{n}|proposed"].get(st, [])
    return []


def sim_d(P, bar):
    """amendment DV2: SHIP6 pictures of the (B) would-drop rows removed, rescored"""
    Bres = sim_b(P, bar)
    wd = {}
    for part, st, lab, a, _m, _c in Bres["would_drop"]:
        wd.setdefault((part, st), []).append((canonical(lab), a))
    rows = {"dev": [], "dev2": []}; base = {"dev": [], "dev2": []}; gone = []; lost = []
    for part, st, g, root, s4, cfg in P:
        specs = specs_of(root / "SHIP6_proposed", st)
        pics = place(specs, root / "SHIP6_proposed", st)
        ends = {}
        for r in s4["arms"]["SHIP6|proposed"][st]:
            ends.setdefault((canonical(r["label"]), round(r["start"], 2)), float(r["end"]))
        keep = []
        for lab, a, b in pics:
            hit = any(f == canonical(lab) and s - 1.5 <= a <= ends.get((f, s), s) for f, s in wd.get((part, st), []))
            (gone if hit else keep).append((lab, a, b))
            if hit:
                gone[-1] = [part, st, lab, round(a, 2), round(b, 2), L.gold_class(g, lab, a, b)]
        r0 = S.score_clip(g, pics); r1 = S.score_clip(g, keep)
        base[part].append(r0); rows[part].append(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((part, st, r0["hit"] - r1["hit"]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"])}
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm
    BASE["cost"] = Bm["merged"]["cost"]
    M = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    verdict = passes(M["merged"], lost)
    print(f"DV2 (clip bar {bar:.4f}): merged {B.fmt(M['merged'])} | DEV {B.fmt(M['dev'])} | DEV2 {B.fmt(M['dev2'])}  removed {len(gone)} hits lost {lost} -> {verdict}")
    for x in gone:
        print("    removed:", x)
    return {"rows": M, "removed": gone, "hits_lost": lost, "verdict": verdict}


def sim_c(P):
    return [[part, st, lab, round(a, 2), L.gold_class(g, lab, a, b)]
            for part, st, g, root, s4, cfg in P
            for lab, a, b, *_w in s4.get("r14_dropped", {}).get(f"SHIP6|proposed|{st}", {}).get("ONCE", [])]


def passes(X, lost_parts):
    gain = X["hits"] - BASE["hits"]; lost = BASE["hits"] - X["hits"]
    old = X["hits"] >= BASE["hits"] and X["wrong"] <= BASE["wrong"] + 2 * max(gain, 0) and X["cost"] < BASE["cost"] and not lost_parts
    few = X["cost"] < BASE["cost"] and X["wrong"] <= BASE["wrong"] - 3 * max(lost, 0) and X["hits"] >= BASE["hits"] - 3
    return "GO (old rule)" if old else "GO (fewer-pictures clause)" if few else "STOP"


def cv(C, part, ship, seeds=10):
    """C: cell -> per-clip cost array. split-half (train on one half, test on the other, both ways) and 5-fold selection"""
    cells = list(C)
    n = len(part)
    out = {}
    for name, K in (("half", 2), ("5fold", 5)):
        proc, fixed, picks = [], [], {}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            fold = np.empty(n, int)
            for p in (0, 1):
                idx = np.where(part == p)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % K
            for k in range(K):
                tr, te = fold != k, fold == k
                best = min(cells, key=lambda c: (round(C[c][tr].mean(), 9), c != ship))
                picks[best] = picks.get(best, 0) + 1
                proc.append(C[best][te].mean()); fixed.append(C[ship][te].mean())
        out[name] = {"procedure": float(np.mean(proc)), "fixed_ship": float(np.mean(fixed)), "picks": picks}
    out["full_argmin"] = min(cells, key=lambda c: (round(C[c].mean(), 9), c != ship))
    return out


def main():
    calib = json.loads(CALIB.read_text(encoding="utf-8")) if CALIB.exists() else None
    clip_bar = calib["clip_bar"] if calib else None
    P = parts()
    if len(sys.argv) > 1 and sys.argv[1] == "dv2":
        for k in R.DISPLAY_KEYS:
            setattr(config, k, R.arm_cfg("SHIP6")[k])
        res = sim_d(P, clip_bar)
        (G / "flap_dv2_screen.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        return
    pk = {"dev": peaks("dev"), "dev2": peaks("dev2")}
    disp = {k: R.arm_cfg("SHIP6")[k] for k in R.DISPLAY_KEYS}
    for k, v in disp.items():
        setattr(config, k, v)                        # the arm's display flags, as the scorers set them
    base, A = sim_a(P, pk)
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP6 (re-placed): merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    BASE["cost"] = Bm["merged"]["cost"]
    res = {"base": Bm, "span_bar": SPAN_BAR, "clip_bar": clip_bar, "A": {}, "B": None, "C": sim_c(P), "cv": None}
    C = {"SHIP6": np.array([DCC.clip_cost(r) for r in base["dev"] + base["dev2"]])}
    part = np.array([0] * len(base["dev"]) + [1] * len(base["dev2"]))
    for v, X in A.items():
        rows = X["rows"]
        M = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
        lost = [(pt, i) for pt in ("dev", "dev2") for i, (r0, r1) in enumerate(zip(base[pt], rows[pt])) if r1["hit"] < r0["hit"]]
        verdict = passes(M["merged"], lost)
        nr = [x for x in X["restored"] if x[7] == "hit_needed"]
        print(f"A-{v}: merged {B.fmt(M['merged'])} | DEV {B.fmt(M['dev'])} | DEV2 {B.fmt(M['dev2'])}  restored {len(X['restored'])} "
              f"(needed {len(nr)}, drawn {sum(x[6] == 'drawn' for x in X['restored'])}, no-proxy {X['no_gate']}, no-conf {X['no_conf']}) "
              f"displaced {len(X['displaced'])} newly dropped {len(X['newly_dropped'])} hits lost {lost} -> {verdict}")
        for x in X["restored"]:
            print("   ", x)
        for x in X["displaced"]:
            print("    displaced:", x)
        for x in X["newly_dropped"]:
            print("    newly dropped:", x)
        res["A"][v] = {"rows": M, "verdict": verdict, "hits_lost": lost, **{k: X[k] for k in ("restored", "displaced", "newly_dropped", "no_conf", "no_gate")}}
        C[v] = np.array([DCC.clip_cost(r) for r in rows["dev"] + rows["dev2"]])
    res["cv"] = cv(C, part, "SHIP6")
    print("CV (SHIP6 / OR / R):", json.dumps(res["cv"]))
    if clip_bar is not None:
        res["B"] = sim_b(P, clip_bar)
        wd, kb = res["B"]["would_drop"], res["B"]["kept_back"]
        cls = lambda L_: {c: sum(x[5] == c for x in L_) for c in ("hit_needed", "other_gold", "none")}
        print(f"B (clip bar {clip_bar:.4f}): would drop {len(wd)} {cls(wd)}; kept back {len(kb)} {cls(kb)}; unqueried {res['B']['unqueried']}")
        for x in wd:
            print("    drop:", x)
        for x in kb:
            print("    keep back:", x)
    print(f"C ONCE-dropped {len(res['C'])}: {res['C']}")
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

"""Opus F (11 Oct, logging only): can a CONJUNCTION of signals bring back v1.7 misses that a stage-4 step dropped,
without many new wrongs? Each rule keeps some dropped candidates (decision trail features, opusF_feats.py); kept
candidates of one family that overlap or sit within 1 s are merged into one picture (earliest start), a picture is
skipped when v1.7 already shows the same family overlapping it, and the clip is re-scored with score_per_sound.
  worst case : every kept candidate survives the later steps and the on-screen gate and is drawn.
  estimated  : per added picture (each candidate is shown through the v1.7 display: bans, onset rules, merging), P(drawn) = 0 if the clip's stored gate checked that family on a stretch covering
               it and voted every stretch seen; 1 if it voted some stretch not seen; otherwise the observed rate at
               which stage-5 sounds drawn by plan B (no gate) stay drawn after the v1.7 on-screen decision, split by hit / visible wrong / other wrong.
               Later stage-4 steps and the v1.5-1.7 display rules are not modelled for these candidates.
Thresholds: clip-grouped 5-fold CV (random.Random(0) shuffle of all 158 stems, folds sh[k::5]), chosen on the
training folds by onset cost; "off" is always in the grid. Also the stage-5 pool (B draws, v1.7 silent) is tested
exactly through decide()/pictures(), and each v1.5-v1.7 display flag is switched off alone (exact).
    python benchmark/gold/coverage/opusF_rules.py   -> opusF_rules.md"""
import copy, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, IN
from benchmark.gold.coverage.opusF_feats import load

HERE = Path(__file__).resolve().parent
gold, pics, miss, cands, allc = load()
ST5 = json.loads((IN / "stage5_specs.json").read_text()); DUR = json.loads((IN / "durations.json").read_text())
WR = lambda r: r["visible"] + r["cross"] + r["phantom"]
BASE = {st: S.score_clip(gold[st], pics[st]) for st in allc}


# ---------- survival estimate: stage-5 plan B drawn -> v1.7 drawn ----------
def _kind(r):
    return "hit" if r["hit"] else "visible" if r["visible"] else "other" if (r["cross"] + r["phantom"]) else None


def _surv_rates():
    """gate only (the display is applied to injected candidates already): a stage-5 sound plan B draws, kept drawn by
    the v1.7 on-screen decision (decide(), before the display), by what B's picture alone would score as."""
    n = {"hit": [0, 0], "visible": [0, 0], "other": [0, 0]}
    for st in allc:
        for b, q in zip(ST5[st]["B"], P0[st]):
            if not b.get("augment"):
                continue
            k = _kind(S.score_clip(gold[st], [(b["event_label"], float(b["start"]), float(b["end"]))]))
            if k is None:
                continue
            n[k][0] += bool(q.get("augment")); n[k][1] += 1
    return {k: (v[0] / v[1] if v[1] else 1.0, v) for k, v in n.items()}


def gate_verdict(st, lab, a, b):
    for g in ST5[st]["gate"]:
        if S.same_family(g["label"], lab) and g["start"] - 0.5 <= a <= g["end"] and g["stretches"]:
            return 0.0 if all((v["seen"] if v.get("ab") is None else bool(v["ab"])) for v in g["stretches"]) else 1.0
    return None


# ---------- injection through the v1.7 display (bans, onset rules, merging, rows) ----------
def _cfg():
    config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    config.ONSET_CURVES = str(IN / "detector_curves.json")


_cfg()
P0 = {st: decide(ST5[st]["P"], ST5[st]["B"], ST5[st]["gate"]) for st in allc}
SURV = _surv_rates()


def merged(st, keep):
    ks = sorted([f for f in cands[st] if keep(f)], key=lambda f: f["start"]); out = []
    for f in ks:
        for o in out:
            if S.same_family(o[0], f["label"]) and f["start"] <= o[2] + 1.0:
                o[2] = max(o[2], f["end"]); break
        else:
            out.append([f["label"], f["start"], f["end"]])
    return [tuple(o) for o in out if not any(S.same_family(l, o[0]) and a < o[2] and o[1] < b for l, a, b in pics[st])]


def show(st, add):
    sp = [{"index": 900 + i, "event_label": l, "start": a, "end": b, "augment": True, "confidence": 0.5, "image_path": "x",
           "spans": [[a, b]], "breaks": [], "gate_doubt": False} for i, (l, a, b) in enumerate(add)]
    return pictures(P0[st] + sp, float(DUR[st] or 10), st)


def clip_delta(st, keep):
    add = merged(st, keep)
    if not add:
        return 0, 0, 0.0, 0.0, []
    r = S.score_clip(gold[st], show(st, add)); dh, dw = r["hit"] - BASE[st]["hit"], WR(r) - WR(BASE[st])
    eh = ew = 0.0; det = []
    for p in add:                                    # marginal of each added picture, weighted by P(drawn)
        r1 = S.score_clip(gold[st], show(st, [p])); h1, w1 = r1["hit"] - BASE[st]["hit"], WR(r1) - WR(BASE[st])
        v = gate_verdict(st, *p)
        sw = SURV["visible" if r1["visible"] > BASE[st]["visible"] else "other"][0]
        eh += h1 * (SURV["hit"][0] if v is None else v); ew += w1 * (sw if v is None else v)
        if h1 or w1:
            det.append((p[0], round(p[1], 2), h1, w1, v))
    return dh, dw, eh, ew, det


def cost_of(sts, delta):
    return sum(4 * (BASE[s]["miss"] - delta[s][0]) + 2 * (WR(BASE[s]) + delta[s][1]) for s in sts) / len(sts)


# ---------- rules: (name, description, grid of params, keep(f, t)) ----------
both = lambda f: f["qwen"] and f["af"]
RULES = [
    ("A", "dropped at dasm_vote, both listeners name it, FineLAP >= t", [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9],
     lambda f, t: f["at"] == "dasm_vote" and both(f) and (f["finelap"] or 0) >= t),
    ("B", "dropped anywhere after the listeners were asked, both listeners name it, DASM (span +- 0.5 s) >= t",
     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], lambda f, t: f["at"] not in ("gate", "display_bar", "depict_event", "rescue_once") and both(f) and (f["dasm"] or 0) >= t),
    ("C", "dropped at band_rescue, Audio Flamingo names it, FlexSED run peak >= t", [0.6, 0.65, 0.7, 0.75],
     lambda f, t: f["at"] == "band_rescue" and f["af"] and (f["peak"] or 0) >= t),
    ("D", "dropped at band_rescue / dasm_rescue / family_merge / continuation_veto, one listener names it, DASM >= t",
     [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], lambda f, t: f["at"] in ("band_rescue", "dasm_rescue", "family_merge", "continuation_veto")
     and (f["qwen"] or f["af"]) and (f["dasm"] or 0) >= t),
    ("E", "dropped at mirror_veto, BEATs peak >= t (no listener asked there)", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
     lambda f, t: f["at"] == "mirror_veto" and (f["peak"] or 0) >= t),
]


def evaluate(rule):
    name, desc, grid, kp = rule
    D = {t: {st: clip_delta(st, lambda f, t=t: kp(f, t)) for st in allc} for t in grid}
    D[None] = {st: (0, 0, 0.0, 0.0, []) for st in allc}
    full = []
    for t in grid:
        d = D[t]; full.append((t, sum(x[0] for x in d.values()), sum(x[1] for x in d.values()),
                                sum(x[2] for x in d.values()), sum(x[3] for x in d.values()), cost_of(allc, d)))
    sh = allc[:]; random.Random(0).shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [s for s in allc if s not in te]
        best = min([None] + grid, key=lambda t: (cost_of(tr, D[t]), t is not None))
        ch.append(best)
        for s in te:
            oof[s] = D[best][s]
    cv = (ch, sum(oof[s][0] for s in allc), sum(oof[s][1] for s in allc), cost_of(allc, oof))
    return full, cv, D



def stage5_pool():
    """B draws, v1.7 silent: force B's plan for that sound only (exact through decide + display)."""
    _cfg()
    rows = []
    for st in allc:
        r = ST5[st]; P0 = decide(r["P"], r["B"], r["gate"])
        for i, (p, b) in enumerate(zip(P0, r["B"])):
            if not b.get("augment") or p.get("augment"):
                continue
            P1 = copy.deepcopy(P0); P1[i] = dict(b); P1[i]["gate_doubt"] = False
            r1 = S.score_clip(gold[st], pictures(P1, float(DUR[st] or 10), st))
            g = [x for x in r["gate"] if x["label"] == b["event_label"] and abs(x["start"] - b["start"]) < 0.011]
            sv = [v for x in g for v in x["stretches"]]
            votes = sum(int(bool(v.get(q))) for v in sv for q in ("name", "ab", "desc")); nv = 3 * len(sv)
            rows.append((st, b["event_label"], round(float(b["start"]), 2), r1["hit"] - BASE[st]["hit"], WR(r1) - WR(BASE[st]),
                         votes, nv, str(p.get("reason", ""))[:40]))
    return rows


def flags_off():
    out = []
    flags = {"VISIBILITY_RULE": "majority", "FLASH_RULE": False, "PICTURE_BAN": None, "HOLD_FLEXSED": None, "REPEAT_LOCK": False,
             "UNVERIFIABLE_BAN": None, "COONSET_CONTEST": None, "WEAK_NO_RISE": None, "BEATS_NO_RISE": False, "GATE_DOUBT_DASM": None}
    for k, v in flags.items():
        _cfg()
        setattr(config, k, v)
        dh = dw = 0; which = []
        for st in allc:
            r = ST5[st]; r1 = S.score_clip(gold[st], pictures(decide(r["P"], r["B"], r["gate"]), float(DUR[st] or 10), st))
            h, w = r1["hit"] - BASE[st]["hit"], WR(r1) - WR(BASE[st]); dh += h; dw += w
            if h > 0:
                which.append(st)
        out.append((k, dh, dw, which))
    return out


def main():
    L = ["# Opus F: conjunction rules to recover v1.7 misses, all 158 clips", "",
         f"v1.7: {sum(r['hit'] for r in BASE.values())} hits, {sum(WR(r) for r in BASE.values())} wrong, cost {cost_of(allc, {s: (0, 0) for s in allc}):.3f}. "
         f"Misses {sum(len(v) for v in miss.values())}; with an in-window dropped candidate of the family "
         f"{sum(1 for st in allc for g in miss[st] if any(f['real'] and S.same_family(f['label'], g['label']) and S.in_window(f['start'], g['start'], S.EARLY, S.LATE) for f in cands[st]))}.",
         "", "Gate survival (plan B drawn -> still drawn after the v1.7 on-screen decision): " + ", ".join(f"{k} {v[0]:.2f} {v[1]}" for k, v in SURV.items()) + " (an added wrong uses 'visible' or 'other' by its own type).",
         "Bar: a rule pays when it gains >= 2 hits per extra wrong (cost: 1 hit = 4, 1 wrong = 2).", ""]
    for rule in RULES:
        full, cv, D = evaluate(rule)
        L += [f"## Rule {rule[0]}: {rule[1]}", "", "| t | +hits | +wrong (worst) | +hits est | +wrong est | cost (worst) |", "|---|---|---|---|---|---|"]
        L += [f"| {t} | {h} | {w} | {eh:.1f} | {ew:.1f} | {c:.3f} |" for t, h, w, eh, ew, c in full]
        L += ["", f"CV choices {cv[0]}; out of fold +hits {cv[1]}, +wrong {cv[2]}, cost {cv[3]:.3f}", ""]
        t0 = max(rule[2], key=lambda t: (2 * sum(x[0] for x in D[t].values()) - sum(x[1] for x in D[t].values()), t))   # best 2 x hits - wrongs (= cost)
        det = [(st,) + d for st in allc for d in D[t0][st][4]]
        L += [f"Pictures that change the score at t = {t0} (label, start, +hit, +wrong, gate verdict):", ""] + [f"- {d}" for d in det] + [""]
        print("\n".join(L[-30:]), flush=True)
    rows = stage5_pool()
    L += ["## Stage-5 pool: sounds plan B draws and v1.7 does not (exact)", "",
          "| clip | label | start | +hit | +wrong | gate yes votes / asked | v1.7 reason |", "|---|---|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {e} | {f}/{g} | {h} |" for a, b, c, d, e, f, g, h in sorted(rows, key=lambda r: r[5] / max(1, r[6]))]
    for k in (0.2, 0.34, 0.5, 0.67):
        sel = [r for r in rows if r[6] and r[5] / r[6] <= k]
        L.append(f"- rule G (yes share <= {k}): +hits {sum(r[3] for r in sel)}, +wrong {sum(r[4] for r in sel)} (n {len(sel)})")
    L += ["", "## Each v1.5-v1.7 display rule switched off alone (exact)", "", "| flag | +hits | +wrong | clips gaining |", "|---|---|---|---|"]
    L += [f"| {k} | {h} | {w} | {', '.join(c)} |" for k, h, w, c in flags_off()]
    (HERE / "opusF_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()

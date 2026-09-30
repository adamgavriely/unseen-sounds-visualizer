"""Ceiling analysis of the shipped SHIP7 pipeline on merged DEV (71 clips): why can't we do better?
Each ORACLE row fixes ONE stage perfectly with the gold and leaves every other stage as shipped; the saved SHIP7 pictures
(arm `SHIP6+FLAP|proposed`) are edited in memory and rescored with score_per_sound exactly as the BTP / CONT / CPO screens.
Nothing in src/ or config.py is edited; no picture is changed on disk; CPU only.

  O1 gate      drop every "visible" picture; add back the gate-silenced needed sounds (their spec exists, augment False)
  O2 listener  every P2/PV rescue candidate of a needed sound accepted, every other rescue rejected (ONCE, F8, FLAP, gate as shipped)
  O3 vetoes    put back needed sounds a veto / filter removed (F8, ONCE, CONT, DV, FLAP, B0 clip veto) if conf >= the 0.40 floor
  O4 timing    move a cross picture of the right family to its gold onset
  O5 family    relabel a cross picture to the gold sound whose onset window holds it
  O6 recall    misses no model hears at all (BEATs / FlexSED / DASM / FineLAP all under their lowest bar)
  O7 doubt     needed / visible calls of the gold that are doubtful (annotator flags + the prereg readings)
  waterfall    shipped -> each oracle alone -> all together (4 -> 5 -> 5+4 joint -> 1 -> 3 -> 2 with filters lifted) -> residual

    TG_ARMS="SHIP6+FLAP SHIP7 ..." python benchmark/gold/ceiling_ship7.py      # from ~/MscProj_tg (msproj)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TG_ARMS", "SHIP6+FLAP SHIP7 B0r TO1+F7F8 TO1F7F8+N2b SHIP+DR2 SHIP2+KV4 SHIP3+DV SHIP4+BTP SHIP5+CONT")
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import ship7_errors as E            # sets B.ARM = "SHIP6+FLAP", loads SpotSound raw (unused here)
from benchmark.gold import btp_screen as B
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from benchmark.gold.flap_joint_sim import load_fl, fl_span
from src.labels import canonical
from src.stage4_audio_event_detection import _cache_items

G = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
LEDGER = json.loads((G / "ship7_errors.json").read_text(encoding="utf-8"))
OUT = G / "ceiling_ship7.json"
FLOOR = 0.40                # PICTURE_MIN_CONF under use_shipped
F8_BAR, FLAP_BAR = 0.575, 0.329
BARS = {"beats": 0.175, "flexsed": 0.3, "dasm": 0.084, "finelap": 0.329}     # the LOWEST bar each model has in the pipeline
ROOTS = {"dev": WORK / "r13", "dev2": WORK / "r13dev2"}
N_CLIPS = 71
BASE = (25, 27, 2.451)


def in_win(a, on):
    return S.in_window(a, on, S.EARLY, S.LATE)


def scored_needed(g):
    return g["needed"] and g["importance"] >= S.MIN_IMPORTANCE


def r3(x):
    return None if x is None else round(float(x), 3)


# ----------------------------------------------------------------------------- state
class Clip:
    def __init__(self, pt, st, gold, pics):
        self.pt, self.st, self.gold = pt, st, gold
        self.pics = [(l, float(a), float(b)) for l, a, b, _ in pics]
        self.resc = {(l, round(float(a), 4)) for l, a, b, r in pics if r}
        self.cfg = E.set_clip(pt, st)
        fx = B.FLEX_DIR / f"{st}.npz"; self.flex = DCC.load_fr(fx) if fx.exists() else None
        bx = self.cfg["beats"] / f"{st}.npz"; self.beats = DCC.load_fr(bx) if bx.exists() else None
        dx = self.cfg["dasm"] / f"{st}.npz"; self.dasm = DCC.load_fr(dx) if dx.exists() else None
        self.fl = load_fl(st)
        self.specs = E.specs_of(ROOTS[pt] / "SHIP6+FLAP_proposed", st)
        self.specs_f7 = E.specs_of(ROOTS[pt] / "TO1+F7_proposed", st)        # the F8-less arm: gate votes for restored spans

    def classes(self, pics):
        return CG.classify(self.gold, pics)

    def matched(self, pics):
        """indices (in start-sorted gold) of gold sounds a picture matches (hit / covered / visible / dup), as score_clip"""
        gs = sorted(self.gold, key=lambda g: g["start"]); tk = [False] * len(gs); out = set()
        for lab, a, b in sorted(pics, key=lambda p: p[1]):
            cands = [i for i, g in enumerate(gs) if S.same_family(lab, g["label"]) and in_win(a, g["start"])]
            free = [i for i in cands if not tk[i]]
            if not free:
                continue
            i = min(free, key=lambda i: abs(a - gs[i]["start"])); tk[i] = True; out.add(i)
            for j in cands:
                if not tk[j] and S.same_family(gs[j]["label"], gs[i]["label"]):
                    tk[j] = True; out.add(j)
        return gs, out

    def open_needed(self, pics):
        gs, m = self.matched(pics)
        return [g for i, g in enumerate(gs) if scored_needed(g) and i not in m]

    def gate_proxy(self, lab, a):
        """the shipped gate's verdict on a span that never reached stage 5: the augment flag of a same-family spec within
        0.5 s in the SHIP7 or the F8-less TO1+F7 arm (the harness memo would reuse the vote); none -> drawn (pessimistic)"""
        for specs in (self.specs, self.specs_f7):
            c = [x for x in specs if S.same_family(x["event_label"], lab) and abs(float(x["start"]) - a) <= 0.5]
            if c:
                x = min(c, key=lambda x: abs(float(x["start"]) - a))
                return bool(x.get("augment")), "spec %s %.2f augment=%s reason=%s" % (x["event_label"], float(x["start"]), x.get("augment"), (x.get("reason") or "")[:60])
        return True, "no same-family spec within 0.5 s: treated as drawn"

    def f8(self, lab, a, b):
        return CG.fam_peak(self.dasm, canonical(lab), a, b, 0.5)

    def flap(self, lab, a, b):
        return fl_span(self.fl, lab, a, b)


def summ_rows(rows_by_part):
    X = {pt: B.summ(v) for pt, v in rows_by_part.items()}
    X["merged"] = B.summ(rows_by_part["dev"] + rows_by_part["dev2"])
    return X


def score(clips, pics_of):
    rows = {"dev": [], "dev2": []}
    for c in clips:
        rows[c.pt].append(S.score_clip(c.gold, pics_of[c.st]))
    return summ_rows(rows)


def brief(X):
    m = X["merged"]
    return {"hits": m["hits"], "n": m["n"], "wrong": m["wrong"], "visible": m["visible"], "cross": m["cross"], "phantom": m["phantom"],
            "cost": round(m["cost"], 3), "dev": (X["dev"]["hits"], X["dev"]["wrong"]), "dev2": (X["dev2"]["hits"], X["dev2"]["wrong"])}


def fmt(X):
    return B.fmt(X["merged"])


# ----------------------------------------------------------------------------- oracles (each: pics_of -> new pics_of, log)
def ledger(kind, pt, st):
    return [x for x in LEDGER[kind] if x["part"] == pt and x["clip"] == st]


def o1_gate(clips, pics_of):
    new, log = {}, []
    for c in clips:
        keep = sorted(pics_of[c.st], key=lambda p: p[1])
        for _ in range(6):                      # a dropped visible picture can expose a same-family "dup" as visible: repeat
            cl = c.classes(keep)
            vis = [p for p, k in zip(keep, cl) if k[3] == "visible"]
            if not vis:
                break
            for p in vis:
                log.append({"clip": c.st, "act": "drop visible picture", "label": p[0], "start": round(p[1], 2)})
            keep = [p for p, k in zip(keep, cl) if k[3] != "visible"]
        # restore gate-silenced needed sounds: the spec exists in the onset window with augment False
        for g in c.open_needed(keep):
            spw = [x for x in c.specs if S.same_family(x["event_label"], g["label"]) and in_win(float(x["start"]), g["start"]) and not x.get("augment")]
            if spw:
                x = min(spw, key=lambda x: abs(float(x["start"]) - g["start"]))
                keep.append((x["event_label"], float(x["start"]), float(x["end"])))
                log.append({"clip": c.st, "act": "restore gate-silenced", "label": x["event_label"], "start": float(x["start"]), "end": float(x["end"]),
                            "conf": r3(x.get("confidence")), "has_image": bool(x.get("image_path")), "gate_reason": (x.get("reason") or "")[:80], "gold": g["label"], "gold_start": g["start"]})
                continue
            # display merge: a SHIP7 row of the family in the window whose spec was merged into a silenced spec of the family
            for m in ledger("misses", c.pt, c.st):
                if m["label"] == g["label"] and abs(m["start"] - g["start"]) < 0.01 and m["removed_by"].startswith("stage-4 row in window but no spec") and not m["rescue_filter_dropped"]:
                    sil = [x for x in c.specs if S.same_family(x["event_label"], g["label"]) and not x.get("augment")]
                    rows = m["rows_in_window_by_arm"].get("SHIP6+FLAP", [])
                    if sil and rows:
                        r = max(rows, key=lambda r: r[3])
                        keep.append((r[0], float(r[1]), float(r[2])))
                        log.append({"clip": c.st, "act": "restore gate-silenced (display merge into the silenced spec)", "label": r[0], "start": r[1], "end": r[2], "conf": r[3], "has_image": False, "gold": g["label"], "gold_start": g["start"]})
        new[c.st] = keep
    return new, log


def cand_items(c, g, items):
    fam_ok = lambda x: S.same_family(x.get("family") or x["label"], g["label"]) or S.same_family(x["label"], g["label"])
    return [x for x in items if x.get("clip") == c.st and x.get("pool") in ("P2", "PV") and fam_ok(x) and in_win(float(x.get("run_start", x["start"])), g["start"])]


def o2_listener(clips, pics_of, caches, lift_filters=False):
    new, log = {}, []
    for c in clips:
        pics = sorted(pics_of[c.st], key=lambda p: p[1]); cl = c.classes(pics)
        keep = []
        for p, k in zip(pics, cl):
            if (p[0], round(p[1], 4)) in c.resc and k[3] in ("visible", "cross", "phantom"):
                log.append({"clip": c.st, "act": "reject wrong rescue", "label": p[0], "start": round(p[1], 2), "class": k[3]})
            else:
                keep.append(p)
        resc_fams = {canonical(p[0]) for p in keep if (p[0], round(p[1], 4)) in c.resc}
        added_fams = set()
        for g in sorted(c.open_needed(keep), key=lambda g: g["start"]):
            li = cand_items(c, g, caches[c.pt])
            if not li:
                continue
            x = min(li, key=lambda x: abs(float(x.get("run_start", x["start"])) - g["start"]))
            a = float(x.get("run_start", x["start"])); b = max(float(x.get("run_end", x["end"])), a + 1.0)
            fam = canonical(x["label"])
            ds, fl = c.f8(x["label"], a, b), c.flap(x["label"], a, b)
            drawn, why = c.gate_proxy(x["label"], a)
            block = []
            if fam in resc_fams or fam in added_fams:
                block.append("ONCE (family already rescued in the clip)")
            if not lift_filters:
                if ds is None or ds < F8_BAR:
                    block.append("F8 (DASM %s < %.3f)" % (r3(ds), F8_BAR))
                if fl is not None and fl < FLAP_BAR:
                    block.append("FLAP (FineLAP %.3f < %.3f)" % (fl, FLAP_BAR))
            if not drawn:
                block.append("gate (%s)" % why)
            rec = {"clip": c.st, "act": "accept needed candidate", "pool": x["pool"], "label": x["label"], "start": a, "end": round(b, 2), "peak": r3(x.get("peak")),
                   "qwen_v4": (x.get("accept") or {}).get("V4"), "gold": g["label"], "gold_start": g["start"], "dasm_span": r3(ds), "finelap_span": r3(fl),
                   "gate_proxy": why, "blocked_by": block}
            if not block:
                keep.append((x["label"], a, b))
            if not block or all(bl.startswith("gate") for bl in block):
                added_fams.add(fam)                 # ONCE runs at stage 4, before the gate: a gate-silenced first rescue still counts
            log.append(rec)
        new[c.st] = keep
    return new, log


def o3_vetoes(clips, pics_of):
    new, log = {}, []
    for c in clips:
        keep = list(pics_of[c.st])
        for g in sorted(c.open_needed(keep), key=lambda g: g["start"]):
            ms = [m for m in ledger("misses", c.pt, c.st) if m["label"] == g["label"] and abs(m["start"] - g["start"]) < 0.01]
            if not ms:
                continue
            m = ms[0]; rb = m["removed_by"]; span = None; how = None
            rf = m["rescue_filter_dropped"]
            if rf:
                k = sorted(rf)[0]; r = rf[k][0]
                span, how = (r[0], float(r[1]), float(r[2])), "rescue filter %s (row conf %s)" % (k, r[3] if len(r) > 3 else "?")
            elif rb.startswith("removed by"):
                chain = m["rows_in_window_by_arm"]; last = [a for a in E.CHAIN if chain.get(a)]
                r = max(chain[last[-1]], key=lambda r: r[3])
                if r[3] >= FLOOR:
                    span, how = (r[0], float(r[1]), float(r[2])), "%s (row conf %.3f)" % (rb.split(" (")[0], r[3])
                else:
                    log.append({"clip": c.st, "act": "NOT restorable: below the 0.40 picture floor", "label": r[0], "start": r[1], "conf": r[3], "removed_by": rb, "gold": g["label"]})
                    continue
            elif rb.startswith("B0 veto"):
                # no B0r row exists: reconstruct the BEATs span (frames >= display bar 0.35) around the onset window
                fam = canonical(g["label"])
                if c.beats is not None:
                    fw, t, labs = c.beats
                    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
                    if cols:
                        sc = fw[:, cols].max(axis=1)
                        idx = [i for i in range(len(t)) if g["start"] - S.EARLY <= t[i] <= g["start"] + S.LATE and sc[i] >= 0.35]
                        if idx:
                            i0 = idx[0]; j = i0
                            while j + 1 < len(t) and sc[j + 1] >= 0.175:
                                j += 1
                            span, how = (g["label"], float(t[i0]), float(t[j]) + (float(t[1] - t[0]) if len(t) > 1 else 1.0)), "B0 clip veto (BEATs span reconstructed, peak %.3f)" % float(sc[i0:j + 1].max())
            if span is None:
                continue
            drawn, why = c.gate_proxy(span[0], span[1])
            rec = {"clip": c.st, "act": "restore veto-removed", "label": span[0], "start": round(span[1], 2), "end": round(span[2], 2), "how": how, "gate_proxy": why, "gold": g["label"], "gold_start": g["start"]}
            if drawn:
                keep.append(span)
            else:
                rec["act"] = "restore veto-removed: shipped gate silences it"
            log.append(rec)
        new[c.st] = keep
    return new, log


def o4_timing(clips, pics_of):
    new, log = {}, []
    for c in clips:
        pics = sorted(pics_of[c.st], key=lambda p: p[1]); cl = c.classes(pics); keep = []
        gs, m = c.matched(pics)
        for p, k in zip(pics, cl):
            if k[3] != "cross":
                keep.append(p); continue
            own = [g for g in gs if S.same_family(p[0], g["label"])]
            if not own:
                keep.append(p); continue
            opn = [g for g in own if scored_needed(g) and gs.index(g) not in m]
            g = min(opn or own, key=lambda g: abs(p[1] - g["start"]))
            q = (p[0], float(g["start"]) + 1e-3, float(g["start"]) + 1e-3 + (p[2] - p[1]))      # +1 ms: tells a moved picture from an original one at the same start (log only)
            keep.append(q)
            log.append({"clip": c.st, "act": "move to gold onset", "label": p[0], "from": round(p[1], 2), "to": g["start"], "gold": g["label"], "gold_needed": g["needed"], "gold_visible": g["visible"], "gold_importance": g["importance"]})
        new[c.st] = keep
        # log the new class of every moved picture
        cl2 = list(c.classes(keep)); used = set()
        for rec in log:
            if rec["clip"] == c.st and "new_class" not in rec:
                j = next((i for i, x in enumerate(cl2) if i not in used and x[0] == rec["label"] and abs(x[1] - rec["to"] - 1e-3) < 1e-6), None)
                rec["new_class"] = cl2[j][3] if j is not None else None
                if j is not None:
                    used.add(j)
    return new, log


def o5_family(clips, pics_of):
    new, log = {}, []
    for c in clips:
        pics = sorted(pics_of[c.st], key=lambda p: p[1]); cl = c.classes(pics); keep = []
        gs, m = c.matched(pics)
        for p, k in zip(pics, cl):
            if k[3] != "cross":
                keep.append(p); continue
            at = [g for g in gs if in_win(p[1], g["start"])]
            if not at:
                keep.append(p)
                log.append({"clip": c.st, "act": "cannot relabel: no gold onset window holds the picture start", "label": p[0], "start": round(p[1], 2)})
                continue
            opn = [g for g in at if scored_needed(g) and gs.index(g) not in m]
            g = min(opn or at, key=lambda g: abs(p[1] - g["start"]))
            keep.append((g["label"], p[1], p[2]))
            log.append({"clip": c.st, "act": "relabel", "label": p[0], "start": round(p[1], 2), "to": g["label"], "gold_needed": g["needed"], "gold_visible": g["visible"], "gold_importance": g["importance"]})
        new[c.st] = keep
        cl2 = c.classes(keep)
        for rec in log:
            if rec["clip"] == c.st and rec["act"] == "relabel" and "new_class" not in rec:
                rec["new_class"] = next((x[3] for x in cl2 if x[0] == rec["to"] and abs(x[1] - rec["start"]) < 1e-6), None)
    return new, log


def o54_joint(clips, pics_of):
    """combined run only: a cross picture that overlaps an OPEN needed gold of another family in time is relabeled AND
    moved to that gold's onset (family + timing together; neither oracle alone reaches it)"""
    new, log = {}, []
    for c in clips:
        pics = sorted(pics_of[c.st], key=lambda p: p[1]); cl = c.classes(pics); keep = []
        gs, m = c.matched(pics); used = set()
        for p, k in zip(pics, cl):
            if k[3] != "cross":
                keep.append(p); continue
            opn = [g for g in gs if scored_needed(g) and gs.index(g) not in m and gs.index(g) not in used
                   and (g["start"] - S.EARLY <= p[1] <= g["end"] or in_win(p[1], g["start"]))]
            if not opn:
                keep.append(p); continue
            g = min(opn, key=lambda g: abs(p[1] - g["start"])); used.add(gs.index(g))
            keep.append((g["label"], float(g["start"]), float(g["start"]) + (p[2] - p[1])))
            log.append({"clip": c.st, "act": "relabel + move", "label": p[0], "from": round(p[1], 2), "to_label": g["label"], "to": g["start"]})
        new[c.st] = keep
    return new, log


# ----------------------------------------------------------------------------- O6 / O7 (descriptive)
def o6_recall(clips):
    out = {"unheard": [], "faint": [], "bars": BARS}
    for c in clips:
        for m in ledger("misses", c.pt, c.st):
            rb = m["removed_by"]; on = m["start"]; fam = m["family"]
            flp = c.flap(m["label"], on - S.EARLY, on + S.LATE)
            rec = {"part": c.pt, "clip": c.st, "label": m["label"], "start": on, "importance": m["importance"], "removed_by": rb.split(" (")[0],
                   "beats": m["beats_peak_window"], "flexsed": m["flex_peak_window"], "dasm": m["dasm_peak_window"], "finelap": r3(flp),
                   "flexsed_queried": m["flex_peak_window"] is not None, "finelap_queried": flp is not None}
            if rb.startswith("unheard"):
                below = all((v is None or v < BARS[k]) for k, v in (("beats", rec["beats"]), ("flexsed", rec["flexsed"]), ("dasm", rec["dasm"]), ("finelap", rec["finelap"])))
                rec["all_models_under_lowest_bar"] = below
                out["unheard"].append(rec)
            elif rb.startswith(("below LO", "below display bar", "B0 veto", "listener candidate mistimed")) or "< floor anyway" in rb:
                out["faint"].append(rec)
    return out


def o7_doubt(clips):
    """doubtful gold calls, from the annotator's own flags and the prereg readings (round 31 SUBJ / gate group, round 33)"""
    gold_files = {"dev": G / "annotations" / "gold_AG.json", "dev2": G / "annotations" / "gold_AG.json"}
    raw = {}
    for pt, f in gold_files.items():
        d = json.loads(f.read_text(encoding="utf-8"))
        for cc in d["clips"]:
            if isinstance(cc, dict):
                raw[Path(str(cc.get("clip", ""))).stem] = cc
    # (a) annotator flags on the DEV clips: clip 'unsure', clip 'note', per-sound 'masked' on a needed sound
    flags = []
    for c in clips:
        cc = raw.get(c.st, {})
        if cc.get("unsure"):
            flags.append({"clip": c.st, "flag": "clip marked unsure", "note": (cc.get("note") or "")[:120]})
        for s in cc.get("sounds", []):
            if s.get("masked") and not (s.get("visible") or s.get("obvious")) and int(s.get("importance") or 2) >= 2:
                flags.append({"clip": c.st, "flag": "needed sound marked masked (hard to hear)", "label": s.get("family") or s.get("label"), "start": s.get("start")})
    # (b) free-text labels resolved through ALIASES (the label a picture must carry was chosen by us, not the annotator)
    alias = []
    for c in clips:
        cc = raw.get(c.st, {})
        for s in cc.get("sounds", []):
            t = (s.get("family") or s.get("label") or "").strip().lower()
            if t in S.ALIASES and not (s.get("visible") or s.get("obvious")):
                alias.append({"clip": c.st, "text": t, "resolved": S.ALIASES[t], "start": s.get("start")})
    # (c) prereg readings: visible calls where the sound IS the scene / a same-family thing is on screen (round 31 gate group), and
    # gate-silenced needed sounds with a same-family thing on screen ("not this one")
    vis_wrong = [w for w in LEDGER["wrong"] if w["class"] == "visible"]
    gate_sil = [m for m in LEDGER["misses"] if m["removed_by"] == "gate-silenced" or (m["removed_by"].startswith("stage-4 row in window but no spec") and not m["rescue_filter_dropped"])]
    return {"annotator_flags": flags, "alias_labels": alias,
            "visible_wrong_pictures": [{"clip": w["clip"], "label": w["label"], "start": w["start"], "gold_at": [(g["label"], g["visible"], g["obvious"], g["importance"]) for g in w["gold_at"]]} for w in vis_wrong],
            "gate_silenced_needed": [{"clip": m["clip"], "label": m["label"], "start": m["start"]} for m in gate_sil],
            "reading": "round 31 SUBJ/gate-group and round 33: the 9 visible pictures are scene sounds (rainstorm thunder x3, church bell, "
                       "fire alarm, protest air horn, starter's bang, bath water, kids' laughter) whose 'visible' call is a judgement; the 5 "
                       "gate-silenced needed sounds (macaws, robin, church tower, boxer dog x2) have a same-family thing on screen that the "
                       "annotator says is not the source. Both groups are one tick away from the other class."}


# ----------------------------------------------------------------------------- main
def main():
    P = B.parts()
    clips = [Clip(pt, st, gold, pics) for pt, st, gold, pics in P]
    caches = {pt: _cache_items(str(G / f"{CG.PARTS[pt]['lis']}_listener_v.json")) for pt in CG.PARTS}
    base_pics = {c.st: list(c.pics) for c in clips}
    X0 = score(clips, base_pics); b0 = brief(X0)
    print("BASE SHIP7:", fmt(X0))
    assert (b0["hits"], b0["wrong"], b0["cost"]) == BASE, b0
    # sanity: the needed-class rule on DEV P2/PV items reproduces the file's own gold_class
    n_file = sum(1 for x in caches["dev"] if x.get("pool") in ("P2", "PV") and x.get("gold") == "hit_needed")
    n_rule = 0
    for c in clips:
        if c.pt != "dev":
            continue
        for x in caches["dev"]:
            if x.get("clip") == c.st and x.get("pool") in ("P2", "PV") and any(scored_needed(g) and (S.same_family(x.get("family") or x["label"], g["label"]) or S.same_family(x["label"], g["label"])) and in_win(float(x.get("run_start", x["start"])), g["start"]) for g in c.gold):
                n_rule += 1
    print(f"needed-class rule on DEV P2/PV items: {n_rule} vs the file's hit_needed {n_file}")

    res = {"base": b0, "base_parts": X0, "needed_class_check": {"rule": n_rule, "file_hit_needed": n_file}, "alone": {}, "logs": {}}
    oracles = [("O1_gate", lambda p: o1_gate(clips, p)), ("O2_listener", lambda p: o2_listener(clips, p, caches)),
               ("O2_listener_filters_lifted", lambda p: o2_listener(clips, p, caches, lift_filters=True)),
               ("O3_vetoes", lambda p: o3_vetoes(clips, p)), ("O4_timing", lambda p: o4_timing(clips, p)), ("O5_family", lambda p: o5_family(clips, p)),
               ("O54_family_plus_timing", lambda p: o54_joint(clips, p))]
    for name, fn in oracles:
        pics, log = fn(base_pics)
        X = score(clips, pics); b = brief(X)
        b["d_cost"] = round(b["cost"] - b0["cost"], 3); b["d_hits"] = b["hits"] - b0["hits"]; b["d_wrong"] = b["wrong"] - b0["wrong"]
        res["alone"][name] = b; res["logs"][name] = log
        print(f"{name:28s} {fmt(X)}  dcost {b['d_cost']:+.3f}  ({len(log)} log lines)")
    # combined, in sequence: timing -> family -> family+timing joint -> gate -> vetoes -> listener (timing first: a cross picture
    # with an own-family gold is a timing error, not a family error)
    seq = [("O4_timing", lambda p: o4_timing(clips, p)), ("O5_family", lambda p: o5_family(clips, p)), ("O54_family_plus_timing", lambda p: o54_joint(clips, p)),
           ("O1_gate", lambda p: o1_gate(clips, p)), ("O3_vetoes", lambda p: o3_vetoes(clips, p)),
           ("O2_listener (filters perfect after O3)", lambda p: o2_listener(clips, p, caches, lift_filters=True))]
    cur = base_pics; prev = b0; water = [{"step": "shipped SHIP7", **b0}]
    res["logs"]["combined"] = {}
    for name, fn in seq:
        cur, log = fn(cur)
        X = score(clips, cur); b = brief(X)
        b["marginal_cost"] = round(b["cost"] - prev["cost"], 3); b["marginal_hits"] = b["hits"] - prev["hits"]; b["marginal_wrong"] = b["wrong"] - prev["wrong"]
        water.append({"step": name, **b}); prev = b; res["logs"]["combined"][name] = log
        print(f"  combined + {name:26s} {fmt(X)}  marginal {b['marginal_cost']:+.3f}")
    # residual after every oracle
    residual = {"wrong": [], "misses": []}
    for c in clips:
        for x in c.classes(cur[c.st]):
            if x[3] in ("visible", "cross", "phantom"):
                residual["wrong"].append({"part": c.pt, "clip": c.st, "label": x[0], "start": round(x[1], 2), "class": x[3]})
        for g in c.open_needed(cur[c.st]):
            ms = [m for m in ledger("misses", c.pt, c.st) if m["label"] == g["label"] and abs(m["start"] - g["start"]) < 0.01]
            residual["misses"].append({"part": c.pt, "clip": c.st, "label": g["label"], "start": g["start"], "importance": g["importance"],
                                       "removed_by": ms[0]["removed_by"] if ms else "(not in the SHIP7 ledger)"})
    res["waterfall"] = water; res["residual"] = residual
    res["O6_recall_ceiling"] = o6_recall(clips)
    res["O7_annotation_doubt"] = o7_doubt(clips)
    res["cost_units"] = {"per_wrong": round(2 / N_CLIPS, 4), "per_miss": round(4 / N_CLIPS, 4), "clips": N_CLIPS}
    ft = G / "final_test_ship7.json"
    if ft.exists():
        d = json.loads(ft.read_text(encoding="utf-8"))
        res["TEST"] = {"per_item_available": False, "note": "final_test_ship7.json holds only aggregate rows (rows / parts); TEST gold was not read",
                       "reported_rows": d.get("rows")}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("\nRESIDUAL wrong", Counter(w["class"] for w in residual["wrong"]), "misses", Counter(m["removed_by"].split(" (")[0] for m in residual["misses"]))
    for w in residual["wrong"]:
        print("   wrong", w)
    for m in residual["misses"]:
        print("   miss ", m)
    u = res["O6_recall_ceiling"]
    print(f"\nO6 unheard {len(u['unheard'])} (all under lowest bar: {sum(1 for x in u['unheard'] if x['all_models_under_lowest_bar'])}), faint {len(u['faint'])}")
    print(OUT)


if __name__ == "__main__":
    main()

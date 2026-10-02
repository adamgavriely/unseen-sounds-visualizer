"""Decision Inspector export (docs/inspector2, data contract in its README): the per-clip decision trails of the shipped arm
D' on merged DEV and merged TEST -> data.js, with a parity check against the frozen run.

Compatible with docs/inspector2/export.py (branch claude/project-thread-4rwrnp), with these differences:
  * stage 6 is not in trail.json in the harness (the scorer draws the pictures), so the export resets src/trail.py, reads
    the pictures with S.load_pictures under the arm's display flags (exactly as merged_dev.py / test_vs_ship8.py do) and
    appends those display records; the full trail is written next to trail.json as trail_full.json;
  * candidates are joined, not overwritten: many firings become one stage-5 sound (family merge), and each candidate
    inherits the later records of the sound it joined;
  * totals come from score_per_sound.score_clip + dev_candidates_check.metrics (the reported numbers); duplicate and
    don't-care pictures are not wrong (verdict "hit", class kept), as in the scorer;
  * --expect checks hits / needed / wrong (v / c / p) / cost per set, and every clip's pictures are compared with the
    frozen arm's (--frozen); the parity result is written to <out>_parity.json, and the per-clip picture signatures used as
    video keys to <out>_media_sigs.json.

    # cluster, from ~/MscProj_tg (as merged_dev.py and test_vs_ship8.py), after the _trail arm ran on all four parts:
    python benchmark/gold/inspector_trail_export.py --expect DEV=29/58/15/6/7/2/2.056 --expect TEST=24/65/24/4/15/5/2.409

Decides nothing; reads gold only to classify (merged TEST is a reporting re-run of the frozen arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S              # noqa: E402

_REAL_LOAD_GOLD = S.load_gold                                 # before any harness module installs its gold stub
GOLD_FILE = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
GOLD_ALL = _REAL_LOAD_GOLD([GOLD_FILE]) if __name__ == "__main__" else {}
from benchmark.gold.inspector_data import classify          # noqa: E402
from src import trail as TRAIL_LOG                           # noqa: E402

# the step catalogue of docs/inspector2/sample_data.py (branch claude/project-thread-4rwrnp), pipeline order
LISTEN = "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
STEPS = [
 # stage 4: hearing
 ("beats_extract", 4, "BEATs detection", "BEATs scores 2-s windows; a sound type must peak above the bar for at least 0.3 s.", "peak ≥ 0.175, length ≥ 0.3 s", "BEATs", None),
 ("flexsed_extract", 4, "FlexSED detection", "FlexSED scores 215 sound types frame by frame; a run above the bar becomes a span.", "peak ≥ 0.8, length ≥ 0.3 s", "FlexSED", None),
 ("twin_union", 4, "Twin join", "A FlexSED span and a BEATs span of the same type within 1 s become one span with the earlier start.", "same family, ≤ 1 s apart", None, None),
 ("mirror_veto", 4, "Mirror veto", "Drops a BEATs-only span when FlexSED clearly hears a different sound at that moment and not this one.", "other type ≥ 0.7 and own type < 0.4", "FlexSED", None),
 ("mirror_keep", 4, "Mirror keep (listener)", "A mirror-vetoed span comes back if the listener confirms it and an open list names it.", "Qwen V1+V2 accept, and a V4 list names it", "Qwen3-Omni", "Is the sound of {family} present in this recording? Answer yes or no."),
 ("masked_weak", 4, "Masked weak veto", "A weak BEATs-only span under speech or music is dropped unless a listener confirms it.", "conf < 0.5 and speech/music ≥ 0.3", "Qwen3-Omni / Audio Flamingo", None),
 ("dasm_clip_veto", 4, "DASM clip veto", "Drops a sound type DASM never hears anywhere in the clip, unless a listener keeps it.", "DASM clip max ≥ 0.084", "DASM", None),
 ("dasm_local_veto", 4, "Two witnesses", "A span needs DASM around it, or both listeners must name it.", "DASM max (span ± 0.5 s) ≥ 0.35, or Qwen and Audio Flamingo both name it", "DASM + Qwen3-Omni + Audio Flamingo", LISTEN),
 ("scene_margin", 4, "Scene check", "If only one listener names the sound, the vision model judges if it is credible in this scene (read from yes/no probabilities).", "yes-margin minus twin-question margin > 0, majority of stretches", "Qwen3.8-27B",
  "Could the sound of {label} plausibly be heard in this scene? Answer yes or no.  (twin: Could the sound of {label} NOT plausibly be heard in this scene?)"),
 ("k4a_inventory", 4, "Listener agreement", "Drops a picture that neither listener's open list names, unless DASM hears it clearly.", "a list names it, or DASM ≥ 0.575", "Qwen3-Omni + Audio Flamingo + DASM", LISTEN),
 ("band_twin_pull", 4, "Earlier start", "Moves a late start back to an earlier FlexSED run of the same sound.", "run ≥ 0.5 ending ≤ 1 s before", "FlexSED", None),
 ("continuation_veto", 4, "Continuation veto", "Drops a span that is only a later piece of a sound already going on.", "FlexSED run ≥ 0.5 began ≥ 1.5 s earlier and is still going", "FlexSED", None),
 ("flexsed_cross_veto", 4, "FlexSED clip veto", "Drops a sound type FlexSED never hears in the whole clip.", "FlexSED clip max ≥ 0.3", "FlexSED", None),
 ("panns_clip_veto", 4, "PANNs clip veto", "Drops a FlexSED-only span PANNs never hears in the clip, unless the listeners keep it.", "PANNs clip max ≥ 0.05", "PANNs", None),
 ("listener_keep", 4, "Listener keep", "A span the PANNs veto removed comes back if the listeners accept it.", "peak ≥ 0.6: Qwen names it; below: both name it", "Qwen3-Omni + Audio Flamingo", LISTEN),
 ("band_rescue", 4, "Listener rescue (weak FlexSED)", "A sound FlexSED heard just below its bar is added if the listeners confirm it.", "FlexSED peak 0.5–0.8; peak ≥ 0.6: Qwen names it; below: both", "Qwen3-Omni + Audio Flamingo", LISTEN),
 ("dasm_rescue", 4, "DASM rescue", "Adds a sound only DASM found when both listeners name it.", "both lists name it; type not already found", "DASM + Qwen3-Omni + Audio Flamingo", LISTEN),
 ("onset_refine", 4, "Start refinement", "Sharpens a BEATs start by masking the window's opening; can only move later.", "evidence drop ≥ 10 %", "BEATs", None),
 ("finelap_veto", 4, "FineLAP check", "A rescued sound is dropped if FineLAP does not support it.", "FineLAP ≥ 0.329", "FineLAP", None),
 ("dasm_vote", 4, "DASM vote on rescues", "A rescued sound is kept only if DASM also hears it around it.", "DASM ≥ 0.575 (span ± 0.5 s)", "DASM", None),
 ("rescue_once", 4, "One rescue per type", "Only the earliest rescued span of a sound type is kept.", "earliest per family", None, None),
 # stage 5: is it needed
 ("label_filter", 5, "Drawable sound type", "Speech, music, textures, wind and generic names are not drawn.", "depictable list", None, None),
 ("family_merge", 5, "Group by sound type", "Spans of one sound type within 1 s join; weak spans of a type that also has strong ones are dropped.", "join gap 1.0 s; weak = conf < 0.35", None, None),
 ("display_bar", 5, "Display bar", "A sound below the confidence bar is planned but not shown.", "conf ≥ 0.35", None, None),
 ("speech_rescue", 5, "Talked-about rescue", "A weak sound people react to in speech is shown.", "both letter orders say 'reacting'", "Qwen3.8-27B (text)", "A sound of {label} was heard in a video. Around that moment, someone said: \"{speech}\". Are they (a) reacting to that sound or talking about it, or (b) not referring to that sound? Answer with the letter only."),
 ("gate", 5, "Visibility gate", "The vision model judges, per ≤ 5-s stretch, if the sound's source is visible. The sound is silenced only if every stretch is seen.", "majority of 3 questions per stretch", "Qwen3.8-27B",
  "Q-name: These frames are from the moment a sound of {label} was heard. Name the thing in these frames that is making that sound… If nothing … is visible, answer exactly: nothing.\nQ-ab: (a) you can SEE {label} happening on screen … (b) {label} is not visibly happening in these frames. Answer with the letter only. (both orders)\nQ-desc: Describe what is happening in these frames in one sentence."),
 ("family_rule", 5, "Family rule", "A sound is silenced when a more specific related sound at the same time was silenced as visible.", "visible specific silences general, never the reverse", None, None),
 ("disambiguate", 5, "Same event, two names", "Two labels on one event: the frames pick one.", "both orders agree", "Qwen3.8-27B", None),
 ("dedup", 5, "Duplicate pictures", "Sounds that would give the same picture are merged.", "SigLIP similarity ≥ 0.80", "SigLIP", None),
 # stage 6: display
 ("depict_event", 6, "Event on screen", "Drops a picture whose event is visibly happening and whose visible maker could make that sound.", "event yes/no both ways, and look-alike yes twice", "Qwen3.8-27B", None),
 ("display_join", 6, "Repeat merge", "Two pictures of the same sound closer than the gap become one.", "gap ≤ 2.5 s", None, None),
 ("group", 6, "Smart grouping", "Two same-type pictures ≤ 8 s apart merge if the second is the same continuing sound.", "both orders answer 'same'", "Qwen3-Omni",
  "You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end the same continuing sound as at the start, or a new, separate event? Answer with exactly one word."),
 ("max_slots", 6, "At most 3 at once", "More than 3 pictures at the same moment: the weakest waits.", "≤ 3", None, None),
 ("scorer", 7, "Scoring", "A picture is a hit if it is the right sound type and starts 0.5 s before to 1.0 s after the needed sound.", "−0.5 … +1.0 s", None, None),
 ("never_heard", 4, "Never heard", "No detector produced any span of this sound type near its start.", "—", None, None),
]


# display / bookkeeping steps the pipeline logs beyond the sample catalogue (stage 5 picture subject; logging errors)
STEPS_EXTRA = [
 ("depiction", 5, "Picture subject", "What to draw: the heard sound, qualified by the scene (never drops a sound).", "—", "Qwen3.8-27B", None),
 ("trail_error", 9, "Logging error", "A logging hook failed here (the decision itself was not affected).", "—", None, None),
]
AFTER = {"depiction": "disambiguate"}

DROP_LIKE = {"drop"}
EDGE_STEPS = {"family_merge"}                    # a relabel into a stage-5 sound: always a join, never a continuation
EXTRA_TO_STEPS = {"twin_union", "dedup", "group"}  # merges with a target: the candidate lives on inside the target


def steps_catalogue():
    rows = list(STEPS)
    for x in STEPS_EXTRA:
        after = AFTER.get(x[0])
        i = next((k for k, r in enumerate(rows) if r[0] == after), len(rows) - 1) + 1 if after else len(rows)
        rows.insert(i, x)
    return [dict(id=a, stage=b, name=c, plain=d, bar=e, model=f, question=g) for a, b, c, d, e, f, g in rows]


def key(r):
    return (r["label"], round(float(r["start"]), 2), round(float(r["end"]), 2))


def is_burst(r):
    ex = r.get("extra") or {}
    return "burst" in ex or "picture" in ex


def build_cands(trail):
    """flat records -> candidate nodes. A record's span key names a node; a `to` link either continues the node under the
    new key (a move / rescue / pass onto an unused key) or joins it into another node (a family merge, a merge, or a key
    already in use), after which the candidate inherits the target's later records."""
    node_of, nodes = {}, []

    def new(r, hidden=False):
        n = {"id": f"c{len(nodes) + 1}", "label": r["label"], "start": r["start"], "end": r["end"],
             "origin": (r.get("extra") or {}).get("origin", ""), "recs": [], "edge": None, "hidden": hidden}
        nodes.append(n)
        return n
    for seq, r in enumerate(trail):
        if not r.get("label") and r.get("step") == "trail_error":
            continue
        k = key(r)
        n = node_of.get(k)
        if n is None:
            n = new(r)
            node_of[k] = n
        n["recs"].append((seq, r))
        to = r.get("to")
        if not to:
            continue
        k2 = key(to)
        if k2 == k:
            continue
        tgt = node_of.get(k2)
        if tgt is None and r["step"] not in EDGE_STEPS and not (r["res"] == "merge"):
            node_of[k2] = n                              # continuation under the new key
            if r["step"] not in ("family_merge", "dedup"):
                n.update(label=to["label"], start=to["start"], end=to["end"])
            continue
        if tgt is None:
            tgt = new(to, hidden=True)
            node_of[k2] = tgt
        if tgt is not n:
            n["edge"] = (tgt, seq)
    return nodes


def chain(n, after=-1, depth=0):
    """the records that apply to node n after `after`: its own, then (if it joined another node) the target's later ones"""
    out = [(s, r) for s, r in n["recs"] if s > after]
    if n["edge"] is not None and depth < 20:
        tgt, s0 = n["edge"]
        out += chain(tgt, s0, depth + 1)
    return out


def fate_of(recs):
    alive, at, last = True, None, None
    for _s, r in recs:
        if is_burst(r):
            continue
        if r["res"] == "drop" or (r["res"] == "merge" and not r.get("to")):
            alive, at, last = False, r["step"], r
        elif r["res"] == "rescue":
            alive, at, last = True, None, None
    return ("drawn" if alive else "dropped"), at, last


def ask_text(a):
    return f"{a.get('who', 'model')}: asked “{a.get('q', '')}” → answered “{a.get('a', '')}”" + (
        f" ({a['vote']})" if a.get("vote") else "")


def why_of(r, sidx):
    if r is None:
        return ""
    name = sidx.get(r["step"], {}).get("name", r["step"])
    bits = [name + (": " + r["value"] if r.get("value") else "")]
    if r.get("bar"):
        bits.append("bar " + r["bar"])
    if r.get("note"):
        bits.append(r["note"])
    out = "; ".join(bits)
    asks = r.get("asks") or []
    if asks:
        out += " | " + " | ".join(ask_text(a) for a in asks[:6]) + (f" | (+{len(asks) - 6} more)" if len(asks) > 6 else "")
    return out


def overlaps(c, a, b):
    return float(c["start"]) <= b and float(c["end"]) >= a


def export_clip(stem, split, trail, pics, gold, sidx, order, video):
    nodes = build_cands(trail)
    cands = []
    for n in nodes:
        if n["hidden"]:
            continue
        recs = chain(n)
        fate, at, last = fate_of(recs)
        c = {"id": n["id"], "label": n["label"], "start": n["start"], "end": n["end"], "origin": n["origin"], "fate": fate,
             "trail": [{x: r[x] for x in ("step", "res", "value", "bar", "note", "asks") if x in r} |
                       ({"span": f"{r['label']} {r['start']:.2f}-{r['end']:.2f}"} if key(r) != (n["label"], round(n["start"], 2), round(n["end"], 2)) else {}) |
                       ({"burst": (r.get("extra") or {}).get("burst") or (r.get("extra") or {}).get("picture")} if is_burst(r) else {})
                       for _s, r in recs]}
        if at:
            c["at"], c["why"] = at, why_of(last, sidx)
        c["_recs"] = recs
        cands.append(c)
    res, pout = classify(gold, pics)
    g_out, stats = [], defaultdict(int)
    for i, (g, r) in enumerate(zip(gold, res)):
        gid = f"g{i + 1}"
        lo, hi = g["start"] - S.EARLY, max(g["end"], g["start"] + S.LATE)
        near = [c for c in cands if S.same_family(c["label"], g["label"]) and overlaps(c, lo, hi)]
        row = {"id": gid, "label": g["label"], "start": g["start"], "end": g["end"], "needed": g["needed"],
               "visible": g["visible"], "importance": g.get("importance"), "cands": [c["id"] for c in near]}
        oc = r["outcome"]
        row["outcome"] = "hit" if oc == "hit" else "miss" if oc == "miss" else oc
        if oc in ("hit", "miss"):
            stats["needed"] += 1
            stats["hits" if oc == "hit" else "misses"] += 1
        if oc == "miss":
            row["lost_at"], row["why"] = lost(g, near, pics, sidx, order)
            stats["miss_with_reason"] += int(row["lost_at"] not in (None, "never_heard", "scorer"))
            stats["miss_trail"] += int(bool(near))
        g_out.append(row)
    p_out = []
    for j, p in enumerate(pout):
        cls = p["class"]
        v = {"wrong: source visible or obvious": "visible", "wrong: a different sound": "cross",
             "wrong: no such sound": "phantom"}.get(cls, "hit")
        link = [c for c in cands if c["fate"] == "drawn" and S.same_family(c["label"], p["label"])
                and overlaps(c, p["start"] - 1.0, p["end"])]
        c0 = min(link, key=lambda c: abs(float(c["start"]) - p["start"]), default=None)
        gi = (p.get("sound") or [None])[0]
        p_out.append({"id": f"p{j + 1}", "label": p["label"], "start": p["start"], "end": p["end"], "verdict": v, "class": cls,
                      "gold": f"g{gi + 1}" if gi is not None else None, "cand": c0["id"] if c0 else None,
                      "cands": [c["id"] for c in link]})
        if v != "hit":
            stats["wrong_with_trail"] += int(c0 is not None and len(c0["trail"]) > 1)
    for c in cands:
        c.pop("_recs", None)
    return {"clip": stem, "split": split, "video": video, "gold": g_out, "pictures": p_out, "cands": cands}, stats


def lost(g, near, pics, sidx, order):
    """the step where a missed needed sound was lost"""
    if not near:
        return "never_heard", "No detector produced a span of this sound type near its start."
    dead = [c for c in near if c["fate"] != "drawn"]
    if len(dead) == len(near):
        far = max(dead, key=lambda c: order.get(c.get("at"), -1))
        return far.get("at"), far.get("why", "")
    lo, hi = g["start"] - S.EARLY, g["start"] + S.LATE
    for c in near:                                       # alive: a display step may have hidden the burst at its onset
        if c["fate"] != "drawn":
            continue
        for _s, r in c["_recs"]:
            if not is_burst(r):
                continue
            ex = r.get("extra") or {}
            b = ex.get("burst") or ex.get("picture") or ""
            try:
                a0 = float(b.split("-")[0])
            except Exception:
                continue
            if r["step"] == "display_join" and ex.get("joined") == "yes" and lo - 1.0 <= a0 <= hi:
                return "display_join", (f"Its burst at {b} s was joined to the picture already on screen from {ex.get('into')} s "
                                        f"({r.get('value', '')}; {r.get('bar', '')}), so no new picture started at the sound.")
            if r["step"] == "group" and r["res"] == "merge" and lo <= a0 <= hi:
                return "group", why_of(r, sidx)
            if r["step"] == "max_slots" and lo <= a0 <= hi:
                return "max_slots", why_of(r, sidx)
    fam = [p for p in pics if S.same_family(p[0], g["label"])]
    if fam:
        near_p = min(fam, key=lambda p: abs(p[1] - g["start"]))
        return "scorer", (f"A {near_p[0]} picture was drawn at {near_p[1]:.2f}-{near_p[2]:.2f} s; the sound starts at "
                          f"{g['start']:.2f} s, so the picture is outside the -0.5 ... +1.0 s window (or it was matched to another "
                          f"sound of the family).")
    c = near[0]
    return "scorer", f"A candidate survived ({c['label']} {c['start']:.2f}-{c['end']:.2f} s) but no picture of this type was drawn."


# ============================================================================= the four parts of merged DEV / merged TEST
def parts(arm):
    """(split, part, arm folder parent, stems) in the order merged_dev.py / test_vs_ship8.py read them"""
    from benchmark.gold import dev_candidates_check as DCC
    subsets = S.subsets_of(GOLD_ALL)
    out = [("DEV", "dev", Path(os.environ.get("TRAIL_DEV_ROOT") or (DCC.WORK / "r13")), sorted(subsets["dev"]))]
    from benchmark.gold import tagger_prep as TP           # (installs its gold stub; GOLD_ALL is already read)
    out.append(("DEV", "dev2", TP.out("dev2"), [s for s in TP.stems_of("dev2") if s in GOLD_ALL]))
    st1 = sorted(x.strip() for x in (_ROOT / "benchmark" / "gold" / "test_stems.txt").read_text(encoding="utf-8").splitlines()
                 if x.strip())
    assert sorted(subsets["test_bench"]) == st1
    out.append(("TEST", "test", DCC.WORK / "r16final", st1))
    out.append(("TEST", "test2", TP.out("test2"), [s for s in TP.stems_of("test2") if s in GOLD_ALL]))
    return out


def run(arm, frozen, out, expect, media_prefix, splits):
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import round13_dev as R
    disp = {k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}
    dispf = {k: R.arm_cfg(frozen)[k] for k in R.DISPLAY_KEYS} if frozen else None
    steps = steps_catalogue()
    sidx = {s["id"]: s for s in steps}
    order = {s["id"]: i for i, s in enumerate(steps)}
    clips, rows, sets, parity, sigs = [], defaultdict(list), defaultdict(lambda: defaultdict(int)), defaultdict(list), {}
    for split, part, base, stems in parts(arm):
        if splits and split not in splits:
            continue
        root = base / f"{arm}_proposed"
        for st in stems:
            gold = GOLD_ALL[st]
            tf = root / st / "trail.json"
            trail = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else []
            TRAIL_LOG.reset()
            with R.flags(disp):
                pics = S.load_pictures(root, st, "proposed") or []
            s6 = TRAIL_LOG.snapshot()
            if tf.exists():
                (root / st / "trail_full.json").write_text(json.dumps(trail + s6, ensure_ascii=False, indent=1), encoding="utf-8")
            r = S.score_clip(gold, pics)
            rows[split].append(r)
            if dispf is not None:
                with R.flags(dispf):
                    pf = S.load_pictures(base / f"{frozen}_proposed", st, "proposed") or []
                if DCC.pics_sig(pf) != DCC.pics_sig(pics):
                    parity[split].append([st, DCC.pics_sig(pf), DCC.pics_sig(pics)])
            sig = DCC.pics_sig(pics)
            h = hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:10]
            sigs[f"{split}/{st}"] = {"sig": sig, "hash": h, "part": part}
            video = f"{media_prefix}/{split}/{st}.{h}.mp4"
            c, stt = export_clip(st, split, trail + s6, pics, gold, sidx, order, video)
            c["part"], c["has_trail"] = part, tf.exists()
            m = root / st / "media.json"
            c["dur"] = json.loads(m.read_text(encoding="utf-8")).get("duration") if m.exists() else None
            clips.append(c)
            for k_, v_ in stt.items():
                sets[split][k_] += v_
            sets[split]["clips"] += 1
            sets[split]["clips_without_trail"] += int(not tf.exists())
        print(f"[export] {split}/{part}: {len(stems)} clips from {root}", flush=True)
    out_sets = {}
    for split, rr in rows.items():
        mt = DCC.metrics(rr)
        s = dict(sets[split])
        s.update(hits=mt["hits"], misses=mt["misses"], needed=mt["hits"] + mt["misses"], wrong=mt["wrong"],
                 visible=mt["visible"], cross=mt["cross"], phantom=mt["phantom"], cost=round(mt["viewer_cost"], 3),
                 pictures_differ_from_frozen=len(parity[split]))
        out_sets[split] = s
        print(f"[export] {split}: {mt['hits']}/{mt['hits'] + mt['misses']}, wrong {mt['wrong']} ({mt['visible']}/{mt['cross']}/"
              f"{mt['phantom']}), cost {mt['viewer_cost']:.3f}; clips with pictures different from {frozen}: {len(parity[split])}",
              flush=True)
    ok = True
    for split, (h, n, w, v, c_, p, cost) in (expect or {}).items():
        s = out_sets.get(split, {})
        got = (s.get("hits"), s.get("needed"), s.get("wrong"), s.get("visible"), s.get("cross"), s.get("phantom"), s.get("cost"))
        want = (h, n, w, v, c_, p, cost)
        good = got[:6] == want[:6] and abs(float(got[6]) - float(cost)) < 1e-3 and not parity.get(split)
        ok &= good
        print(f"[parity] {split}: got {got} want {want} -> {'EXACT' if good else 'DIFFERENT'}", flush=True)
    data = {"meta": {"version": "D′", "arm": frozen or arm, "trail_arm": arm,
                     "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "sample": False,
                     "window": [-S.EARLY, S.LATE], "cost": "(4·miss + 2·wrong)/clips",
                     "clips_without_trail": sum(1 for c in clips if not c["has_trail"]),
                     "parity": {k: ("exact" if not parity.get(k) else f"{len(parity[k])} clips differ") for k in out_sets},
                     "note": "Trails from the DEV / TEST harness (benchmark/gold/round13_dev.py), stage 6 from the scorer's own "
                             "display step. Gate stretches whose verdict was reused from the stored gate run carry the votes "
                             "but not the raw answers."},
            "sets": out_sets, "steps": steps, "clips": clips}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("window.INSPECTOR2 = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    Path(out).with_name(Path(out).stem + "_parity.json").write_text(json.dumps(
        {"sets": out_sets, "differ": parity, "expect": expect, "exact": ok}, indent=1, default=str), encoding="utf-8")
    Path(out).with_name(Path(out).stem + "_media_sigs.json").write_text(json.dumps(sigs, indent=1), encoding="utf-8")
    print(f"[export] wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB); parity {'EXACT' if ok else 'NOT exact'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="SHIP8+MD3+WW5+SL_trail")
    ap.add_argument("--frozen", default="SHIP8+MD3+WW5+SL", help="the frozen arm whose pictures must be identical ('' = skip)")
    ap.add_argument("--out", default=str(_ROOT / "docs" / "inspector2" / "data.js"))
    ap.add_argument("--media-prefix", default="../inspector/media/bysig")
    ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--expect", action="append", default=[],
                    help="SPLIT=hits/needed/wrong/visible/cross/phantom/cost, e.g. DEV=29/58/15/6/7/2/2.056")
    a = ap.parse_args()
    exp = {}
    for e in a.expect:
        k, v = e.split("=", 1)
        x = v.split("/")
        exp[k] = tuple(int(y) for y in x[:6]) + (float(x[6]),)
    ok = run(a.arm, a.frozen or None, a.out, exp, a.media_prefix, a.splits)
    sys.exit(0 if ok or not exp else 2)


if __name__ == "__main__":
    main()

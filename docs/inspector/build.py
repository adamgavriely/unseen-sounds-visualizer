"""Build data.js for the inspector page from data.json.

The page (index.html) must open by double-click (file://), where the browser blocks fetch(); so the data is loaded as a
script: data.js = "window.INSPECTOR = {...}". This script copies data.json unchanged and adds a "derived" block per
clip with the facts that need the AudioSet family rule (benchmark/gold/score_per_sound.same_family), which the page
cannot compute in JavaScript:

  - why each missed needed sound (importance 2-3) was missed, per system
    (reasons in the order of benchmark/gold/dev_miss_table.py, plus one bookkeeping case)
  - which gate stretch each shown picture came from (and the gate votes there)
  - which augmentation (index) each of ours' pictures belongs to (for the picture thumbnail)
  - gate errors: a needed sound where the VLM voted "visible" at the onset

It also copies the existing-approach result (benchmark/gold/baseline_panns.json: PANNs alone + a picture for every
detection) into a "baseline" block for the Overview. data.json keeps the key "blind" for the pipeline without gate.

Decides nothing and changes no score. Re-run after data.json changes:

    python docs/inspector/build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / "data.json"
OUT = HERE / "data.js"
EARLY, LATE = 0.5, 1.0
NOTE = ""

try:
    sys.path.insert(0, str(ROOT))
    from benchmark.gold import score_per_sound as S
    same_family = S.same_family
    EARLY, LATE = S.EARLY, S.LATE
    from src.labels import canonical
except Exception as e:  # the page still works, but family matches are exact-name only
    NOTE = f"WARNING: could not import the scorer ({e!r}); family match = exact label only. Reasons may be wrong."
    print(NOTE)

    def same_family(a, b):
        return a == b

    def canonical(label):
        return label


def in_window(pic_start, onset):
    return onset - EARLY <= pic_start <= onset + LATE


def gate_of_picture(pic, gate):
    """index of the gate entry (stretch) a picture was drawn from, and whether the match is approximate"""
    for k, g in enumerate(gate):
        if g["label"] == pic["label"] and abs(g["stretch"][0] - pic["start"]) < 0.05:
            return k, False
    # display spans can merge or relabel stretches: fall back to a same-family stretch holding the start
    best = None
    for k, g in enumerate(gate):
        a, b = g["stretch"]
        if same_family(g["label"], pic["label"]) and a - 0.05 <= pic["start"] <= b + 0.05:
            d = abs(a - pic["start"])
            if best is None or d < best[0]:
                best = (d, k)
    return (best[1], True) if best else (None, True)


def aug_of_gate(g, augs):
    """augmentation index of a gate entry: same event (label + start). The thumbnail file is
    media/<SPLIT>/<clip>_pics/<index>.png with this index (position in augmentations.json)."""
    for i, a in enumerate(augs):
        if a["event_label"] == g["label"] and abs(float(a["start"]) - float(g["start"])) < 0.01:
            return i
    return None


def aug_of_picture(pic, gate, augs, gk):
    if gk is not None:
        i = aug_of_gate(gate[gk], augs)
        if i is not None:
            return i, False
    # fallback: a shown augmentation of the same family, nearest start
    cands = [(abs(float(a["start"]) - pic["start"]), i) for i, a in enumerate(augs)
             if a.get("augment") and same_family(a["event_label"], pic["label"])]
    return (min(cands)[1], True) if cands else (None, True)


def miss_reason(snd, own_pics, blind_pics, events, is_ours):
    """why a needed sound (scorer outcome: miss) got no picture; same order as dev_miss_table.py"""
    lab, on, end = snd["label"], snd["start"], snd["end"]
    fam = [p for p in own_pics if same_family(p["label"], lab)]
    if any(in_window(p["start"], on) for p in fam):
        p = min((p for p in fam if in_window(p["start"], on)), key=lambda p: abs(p["start"] - on))
        return {"reason": "taken by another sound", "pic_start": p["start"], "pic_label": p["label"]}
    bfam = [p for p in blind_pics if same_family(p["label"], lab)]
    if is_ours and any(in_window(p["start"], on) for p in bfam):
        p = min((p for p in bfam if in_window(p["start"], on)), key=lambda p: abs(p["start"] - on))
        return {"reason": "removed by the gate", "pic_start": p["start"], "pic_label": p["label"]}
    over = [(p, s) for s, ps in (("ours" if is_ours else "blind", fam), ("blind", bfam if is_ours else [])) for p in ps
            if p["end"] > on and p["start"] < end]
    if over:
        p, s = min(over, key=lambda x: abs(x[0]["start"] - on))
        return {"reason": "timing", "pic_start": p["start"], "pic_label": p["label"], "pic_system": s,
                "late": round(p["start"] - on, 2)}
    ev = [e for e in events if same_family(e["label"], lab) and e["end"] >= on - EARLY and e["start"] <= on + LATE]
    if ev:
        e = max(ev, key=lambda e: e.get("confidence") or 0)
        return {"reason": "detected, not drawn", "event_label": e["label"], "event_start": e["start"],
                "confidence": e.get("confidence")}
    return {"reason": "never detected"}


def position(t, snd):
    """where a picture starts against a sound: early / window / during (later, sound still playing) / after"""
    on, end = snd["start"], snd["end"]
    if t < on - EARLY:
        return "early"
    if t <= on + LATE:
        return "window"
    if t <= end:
        return "during"
    return "after"


def family_name(label):
    try:
        return canonical(label) or label
    except Exception:
        return label


def sensitivity(d):
    """Sensitivity row (not the scoring rule): a wrong picture of the same family that starts anywhere in
    [onset - 0.5, end] of a MISSED needed sound (importance 2-3) turns into a hit; one picture per sound,
    sounds in onset order, the picture nearest the onset first. Cost = (4 x misses + 2 x wrong) / clips."""
    out = {}
    groups = {"DEV": ["DEV"], "TEST": ["TEST"], "DEV+TEST": ["DEV", "TEST"], "sliceB": ["sliceB"]}
    for gname, splits in groups.items():
        out[gname] = {}
        for sysn in ("ours", "blind", "silence"):
            acc = {"onset": {"hits": 0, "misses": 0, "wrong": 0}, "during": {"hits": 0, "misses": 0, "wrong": 0}, "clips": 0}
            for c in d["clips"]:
                if c["split"] not in splits:
                    continue
                acc["clips"] += 1
                sd = c["systems"].get(sysn) or {"sounds": [], "pictures": []}
                gold = c["sounds"]
                outs = sd.get("sounds") or []
                pics = sd.get("pictures") or []
                if sysn == "silence":
                    h = 0
                    m = sum(1 for s in gold if s["needed"] and s["importance"] >= 2)
                    w = 0
                else:
                    h = sum(1 for o in outs if o and o["outcome"] == "hit")
                    m = sum(1 for o in outs if o and o["outcome"] == "miss")
                    w = sum(1 for p in pics if p["class"].startswith("wrong"))
                conv, used = 0, set()
                for i in sorted(range(len(outs)), key=lambda i: gold[i]["start"]):
                    if not (outs[i] and outs[i]["outcome"] == "miss"):
                        continue
                    s = gold[i]
                    cands = [j for j, p in enumerate(pics) if j not in used and p["class"].startswith("wrong")
                             and same_family(p["label"], s["label"]) and s["start"] - EARLY <= p["start"] <= max(s["end"], s["start"] + LATE)]
                    if cands:
                        j = min(cands, key=lambda j: abs(pics[j]["start"] - s["start"]))
                        used.add(j); conv += 1
                for k, v in (("onset", (h, m, w)), ("during", (h + conv, m - conv, w - conv))):
                    acc[k]["hits"] += v[0]; acc[k]["misses"] += v[1]; acc[k]["wrong"] += v[2]
            n = max(1, acc["clips"])
            for k in ("onset", "during"):
                a = acc[k]
                shown = a["hits"] + a["wrong"]
                a["P"] = a["hits"] / shown if shown else None
                a["R"] = a["hits"] / (a["hits"] + a["misses"]) if a["hits"] + a["misses"] else None
                a["F1"] = 2 * a["P"] * a["R"] / (a["P"] + a["R"]) if a["P"] and a["R"] else 0.0
                a["cost"] = (4 * a["misses"] + 2 * a["wrong"]) / n
            acc["clips"] = n
            out[gname][sysn] = acc
    return out


def kept_parts(pic, gate, augs):
    """Parts of a picture's time on screen where its detected sound was not playing: the display keeps a picture up
    at least MIN_DWELL (1.5 s) and joins repeats of the same sound closer than MERGE_GAP (2.0 s) into one picture
    (src/stage6_visual_augmentation._display_spans). The sound = the same-label gate stretches (one per burst) and
    augmentation spans inside the picture. Returns [[a, b, kind]] with kind 'dwell' (after the last burst) or 'join'
    (between two bursts), or None when no burst of that label was found."""
    a0, b0 = pic["start"], pic["end"]
    parts = [tuple(g["stretch"]) for g in gate if g["label"] == pic["label"]]
    parts += [(float(x["start"]), float(x["end"])) for x in augs if x.get("event_label") == pic["label"] and x.get("augment")]
    parts = sorted((max(a, a0), min(b, b0)) for a, b in parts if b > a0 - 1e-6 and a < b0 + 1e-6)
    if not parts:
        return None
    merged = []
    for a, b in parts:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, t = [], a0
    for a, b in merged:
        if a - t > 0.05:
            out.append([round(t, 3), round(a, 3), "join" if t > a0 else "before"])
        t = max(t, b)
    if b0 - t > 0.05:
        out.append([round(t, 3), round(b0, 3), "dwell"])
    return out


def near_gate(snd, gate):
    """gate entries (stretches) of the sound's family that overlap its onset window or its span"""
    lab, on, end = snd["label"], snd["start"], max(snd["end"], snd["start"] + LATE)
    return [k for k, g in enumerate(gate)
            if same_family(g["label"], lab) and g["stretch"][1] >= on - EARLY and g["stretch"][0] <= end]


KEPT_STATS = []


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    counts = {}
    for c in d["clips"]:
        ours = c["systems"].get("ours") or {}
        blind = c["systems"].get("blind") or {}
        gate = ours.get("gate") or []
        augs = ours.get("augmentations") or []
        events = ours.get("events") or []
        der = {"ours_pics": [], "blind_pics": [], "miss": {"ours": {}, "blind": {}}, "near_gate": {}}
        for p in ours.get("pictures") or []:
            gk, ga = gate_of_picture(p, gate)
            ai, aa = aug_of_picture(p, gate, augs, gk)
            kp = kept_parts(p, gate, augs)
            der["ours_pics"].append({"gate": gk, "gate_approx": ga, "aug": ai, "aug_approx": aa, "kept": kp})
            KEPT_STATS.append(kp)
        for p in blind.get("pictures") or []:
            gk, ga = gate_of_picture(p, gate)
            # was the same stretch also drawn by ours? (if not, the gate removed it)
            in_ours = any(q["label"] == p["label"] and abs(q["start"] - p["start"]) < 0.05 for q in ours.get("pictures") or [])
            kp = kept_parts(p, gate, augs)
            der["blind_pics"].append({"gate": gk, "gate_approx": ga, "in_ours": in_ours, "kept": kp})
            KEPT_STATS.append(kp)
        for i, s in enumerate(c["sounds"]):
            ng = near_gate(s, gate)
            if ng:
                der["near_gate"][str(i)] = ng
            for name, sysd, is_ours in (("ours", ours, True), ("blind", blind, False)):
                out = (sysd.get("sounds") or [None] * len(c["sounds"]))[i]
                if out and out.get("outcome") == "miss":
                    r = miss_reason(s, sysd.get("pictures") or [], blind.get("pictures") or [], events, is_ours)
                    if r["reason"] == "removed by the gate":
                        r["gate"] = ng
                    # every same-family picture (both systems) and detector event, placed against the sound
                    r["fam_pics"] = [{"sys": sn, "j": j, "label": p["label"], "start": p["start"], "end": p["end"],
                                      "class": p["class"], "pos": position(p["start"], s)}
                                     for sn, sd in (("ours", ours), ("blind", blind))
                                     for j, p in enumerate(sd.get("pictures") or []) if same_family(p["label"], s["label"])]
                    r["fam_events"] = [{"label": e["label"], "start": e["start"], "end": e["end"], "confidence": e.get("confidence")}
                                       for e in events if same_family(e["label"], s["label"])]
                    der["miss"][name][str(i)] = r
                    key = (c["split"], name, r["reason"])
                    counts[key] = counts.get(key, 0) + 1
        der["family"] = [family_name(s["label"]) for s in c["sounds"]]
        c["derived"] = der
    n_none = sum(1 for k in KEPT_STATS if k is None)
    allp = [x for k in KEPT_STATS if k for x in k]
    kinds = {}
    for x in allp:
        kinds[x[2]] = kinds.get(x[2], 0) + 1
    too_long = [x for x in allp if (x[2] == "dwell" and x[1] - x[0] > 1.5 + 0.05) or (x[2] == "join" and x[1] - x[0] > 2.0 + 0.05)]
    print(f"kept-on-screen parts: {len(KEPT_STATS)} pictures, {n_none} with no burst found, parts by kind {kinds}, "
          f"{len(too_long)} longer than the rule allows (dwell > 1.5 s or join > 2.0 s)")
    d["sensitivity"] = sensitivity(d)
    print("sensitivity (onset rule -> 'any time while the sound plays'):")
    for sp, v in d["sensitivity"].items():
        for sysn, x in v.items():
            print(f"   {sp:7s} {sysn:7s} hits {x['onset']['hits']}->{x['during']['hits']} of {x['onset']['hits'] + x['onset']['misses']}, "
                  f"wrong {x['onset']['wrong']}->{x['during']['wrong']}, cost {x['onset']['cost']:.2f}->{x['during']['cost']:.2f}")
    d["build_note"] = NOTE
    d["window"] = [-EARLY, LATE]
    # existing approaches (one detector alone + a picture for every detection, threshold chosen on DEV), scored on TEST.
    # Both files share one script, so their keys are named panns_* / ours_minus_panns_* even for PretrainedSED.
    d["baselines"] = {}
    for key, fname in (("psed", "baseline_psed.json"), ("panns", "baseline_panns.json")):
        bp = ROOT / "benchmark" / "gold" / fname
        try:
            b = json.loads(bp.read_text(encoding="utf-8"))
            t = b["test"]
            d["baselines"][key] = {"threshold": b.get("threshold"), "base": t["panns_every_detection"], "ours": t["ours"],
                                   "silence": t.get("silence"),
                                   "diff": {m: t.get("ours_minus_panns_" + m) for m in ("F1", "P", "R", "viewer_cost")}}
            print(f"baseline {key} (thr {b.get('threshold')}): TEST F1 {t['panns_every_detection']['F1']:.3f} vs ours {t['ours']['F1']:.3f}")
        except Exception as e:  # the page still works without this block
            print(f"WARNING: no baseline {key} ({e!r})")
    # audio-to-image comparison (made by another script): wrap a2i.json into a2i.js if it is there
    a2i = HERE / "a2i.json"
    if a2i.exists():
        try:
            a = json.loads(a2i.read_text(encoding="utf-8"))
            (HERE / "a2i.js").write_text("window.A2I = " + json.dumps(a, separators=(",", ":")) + ";\n", encoding="utf-8")
            print("->", HERE / "a2i.js", f"({len(a.get('clips') or [])} clips)")
        except Exception as e:
            print(f"WARNING: a2i.json not read ({e!r}); a2i.js not written")
    else:
        (HERE / "a2i.js").write_text("window.A2I = null;\n", encoding="utf-8")
        print("a2i.json not there yet: the Audio-to-image tab will say 'not ready yet'")
    OUT.write_text("window.INSPECTOR = " + json.dumps(d, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("miss reasons (split, system, reason): count")
    for k in sorted(counts):
        print("  ", k, counts[k])
    print("->", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

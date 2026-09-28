"""DEV check of detector round 9's J2 (docs/prereg_round9_contrast.md, amendment 1 B): which shown pictures rest only on
stage-4 spans that J2 would drop, and how the official per-sound score moves when they are removed. DEV only.

  J2 = a BEATs-only span is dropped if its family's mean BEATs score in the span minus the mean over the +-3 s flanks < 0.1;
       a FlexSED-only span likewise with FlexSED scores and margin 0.2; spans with no flank frames are kept.

    python benchmark/gold/j2_dev_check.py beats     # GPU: BEATs frame scores on each DEV clip's audio.wav
    python benchmark/gold/j2_dev_check.py score     # CPU: gate D0, the J2 decisions, before / after scores
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from src.labels import canonical, is_descendant
from src.stage4_audio_event_detection import _extract_events

WORK = _ROOT / "data" / "work"
TAG = "dev_monocap_v31"
SYSTEMS = ("proposed", "blind_a2i")
BEATS_DIR = WORK / "j2_dev_beats"
FLEX_DIR = WORK / "flexsed_cache"
OUT = _ROOT / "benchmark" / "gold" / "j2_dev_check.json"
J_MARGIN, I7_MARGIN, FLANK, TOL = 0.1, 0.2, 3.0, 0.01


def same(a, b):
    return a == b or canonical(a) == canonical(b) or is_descendant(a, b) or is_descendant(b, a)


def dev_stems():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    return gold, sorted(S.subsets_of(gold)["dev"])


def beats():
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    _g, stems = dev_stems()
    BEATS_DIR.mkdir(parents=True, exist_ok=True)
    for st in stems:
        dst = BEATS_DIR / f"{st}.npz"
        wav = WORK / f"protocol_proposed_{TAG}" / st / "audio.wav"
        if dst.exists() or not wav.exists():
            continue
        fw, t, labs = infer_beats(wav, "cuda")
        np.savez_compressed(dst, fw=fw.astype(np.float32), times=np.asarray(t, np.float64), labels=np.array(labs))
        print(f"[beats] {st} {fw.shape}", flush=True)


# ----------------------------------------------------------------------------- one clip
def sp(x):
    return (x["label"], float(x["start"]), float(x["end"]))


def close(a, b):
    return a[0] == b[0] and abs(a[1] - b[1]) <= TOL and abs(a[2] - b[2]) <= TOL


def twin_rule(extract, flex, weak):
    """stage 4's union: a FlexSED span with a same-family BEATs span within 1 s pulls its start earlier; else it is new"""
    ev = [dict(label=x["label"], start=float(x["start"]), end=float(x["end"]), conf=float(x["conf"]), origin="beats", twin=False)
          for x in extract]
    fresh = []
    for f in flex:
        tw = [y for y in ev if canonical(y["label"]) == canonical(f["label"]) and y["start"] - 1.0 <= f["end"] and f["start"] - 1.0 <= y["end"]]
        if weak == "ignore":
            tw = [y for y in tw if y["conf"] >= float(getattr(config, "DISPLAY_THRESHOLD", 0.35)) - 5e-4]
        if tw:
            for y in tw:
                y["start"] = min(y["start"], float(f["start"])); y["twin"] = True
        else:
            fresh.append(dict(label=f["label"], start=float(f["start"]), end=float(f["end"]), conf=float(f["conf"]), origin="flex", twin=False))
    return ev + fresh


def contrast(fr, label, a, b):
    fw, ts, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(label)]
    if not cols:
        return None
    s = fw[:, cols].max(axis=1)
    ins = (ts >= a) & (ts < b)
    fl = ((ts >= a - FLANK) & (ts < a)) | ((ts >= b) & (ts < b + FLANK))
    if not fl.any() or not ins.any():
        return None
    return float(s[ins].mean() - s[fl].mean())


def clip_decisions(st, sysn, log):
    """{index in refine list: dropped?}, refine list; None when the clip cannot be checked (gate D0)"""
    d = WORK / f"protocol_{sysn}_{TAG}" / st
    tp = d / "onset_trace.json"
    if not tp.exists():
        log["no_trace"].append(st); return None
    tr = json.loads(tp.read_text(encoding="utf-8"))
    step = lambda n: [x for x in tr if x["step"] == n]
    extract, flex, union, veto, refine = (step(n) for n in ("extract", "flexsed_raw", "union", "veto", "refine"))
    bp = BEATS_DIR / f"{st}.npz"
    if not bp.exists():
        log["d0_fail"].append((st, "no BEATs cache")); return None
    z = np.load(bp)
    B = (z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]])
    mine = _extract_events(B[0], B[1], B[2], float(config.AED_THRESHOLD), None, config.AED_MIN_DUR,
                           low=float(config.AED_THRESHOLD) * float(getattr(config, "AED_HYSTERESIS", 1.0)))
    a = sorted((e.label, e.start, e.end) for e in mine); b = sorted(sp(x) for x in extract)
    if len(a) != len(b) or not all(close(x, y) for x, y in zip(a, b)):
        log["d0_fail"].append((st, f"extract {len(a)} vs trace {len(b)}")); return None
    un = None
    for weak in ("absorb", "ignore"):
        u = twin_rule(extract, flex, weak)
        if sorted((x["label"], x["start"], x["end"]) for x in u) and len(u) == len(union) and all(
                close(x, y) for x, y in zip(sorted((x["label"], x["start"], x["end"]) for x in u), sorted(sp(x) for x in union))):
            un = u; log["weak_mode"][weak] = log["weak_mode"].get(weak, 0) + 1; break
    if un is None:
        log["d0_fail"].append((st, "union not rebuilt")); return None
    fp = FLEX_DIR / f"{st}.npz"
    F = None
    if fp.exists():
        zf = np.load(fp)
        fw = zf["fw"].astype(np.float32).T
        F = (fw, np.arange(fw.shape[0], dtype=np.float64) / float(zf["fps"]), [str(x) for x in zf["labels"]])
    assert len(veto) == len(refine), st
    drop = {}
    for i, (v, r) in enumerate(zip(veto, refine)):
        assert v["label"] == r["label"] and abs(v["end"] - r["end"]) <= TOL, (st, v, r)
        o = [x for x in un if close((x["label"], x["start"], x["end"]), sp(v))]
        if not o:
            log["veto_unmatched"] += 1; drop[i] = False; continue
        o = o[0]
        c, kind = None, None
        if o["origin"] == "beats" and not o["twin"]:
            c, kind, m = contrast(B, v["label"], v["start"], v["end"]), "beats_only", J_MARGIN
        elif o["origin"] == "flex" and F is not None:
            c, kind, m = contrast(F, v["label"], v["start"], v["end"]), "flex_only", I7_MARGIN
        drop[i] = bool(c is not None and c < m)
        if kind:
            log[kind] += 1; log[kind + "_dropped"] += drop[i]
    return drop, refine


def variant(root, stems, sysn, log):
    tmp = Path(tempfile.mkdtemp(prefix="j2_"))
    changes = []
    for st in stems:
        d = root / st
        if not (d / "augmentations.json").exists():
            continue
        (tmp / st).mkdir()
        if (d / "media.json").exists():
            shutil.copy(d / "media.json", tmp / st / "media.json")
        specs = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
        anypng = next((str(p) for p in (d / "augmentations").glob("*.png")), str(_ROOT / "README.md"))
        dec = clip_decisions(st, sysn, log)
        for s in specs:
            if s.get("image_path") and not Path(s["image_path"]).exists():
                c = d / "augmentations" / Path(s["image_path"]).name
                s["image_path"] = str(c) if c.exists() else anypng
            if dec is None or not s.get("augment"):
                continue
            drop, refine = dec
            keep = []
            for span in s.get("spans") or [[s["start"], s["end"]]]:
                ev = [i for i, r in enumerate(refine) if same(r["label"], s["event_label"])
                      and min(span[1], r["end"]) - max(span[0], r["start"]) > 0]
                if not ev:
                    log["span_unmapped"] += 1; keep.append(span); continue
                if all(drop[i] for i in ev):
                    changes.append((st, s["event_label"], [round(x, 2) for x in span]))
                else:
                    keep.append(span)
            if not keep:
                s["augment"] = False; log["pictures_removed"] += 1
            s["spans"] = keep
        (tmp / st / "augmentations.json").write_text(json.dumps(specs), encoding="utf-8")
    return tmp, changes


def score():
    gold, stems = dev_stems()
    res = {}
    for sysn in SYSTEMS:
        root = WORK / f"protocol_{sysn}_{TAG}"
        log = {"no_trace": [], "d0_fail": [], "weak_mode": {}, "veto_unmatched": 0, "beats_only": 0, "beats_only_dropped": 0,
               "flex_only": 0, "flex_only_dropped": 0, "span_unmapped": 0, "pictures_removed": 0}
        tmp, changes = variant(root, stems, sysn, log)
        rows = {}
        for name, r in (("scored", root), ("J2", tmp)):
            a = S.aggregate([S.score_clip(gold[st], S.load_pictures(r, st, sysn) or []) for st in stems])
            rows[name] = {k: a[k] for k in ("hits", "misses", "visible", "cross", "phantom", "dup", "F1", "viewer_cost", "coverage")}
            rows[name]["wrong"] = a["visible"] + a["cross"] + a["phantom"]
            x = rows[name]
            print(f"DEV {sysn:9s} {name:6s} hits {x['hits']}/{x['hits'] + x['misses']} wrong {x['wrong']} (visible {x['visible']}, "
                  f"cross {x['cross']}, phantom {x['phantom']}) dup {x['dup']} F1 {x['F1']:.3f} cost {x['viewer_cost']:.2f} "
                  f"cov {x['coverage'] if x['coverage'] is None else round(x['coverage'], 3)}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {sysn}: removed spans {changes}\n  log {log}", flush=True)
        res[sysn] = {"rows": rows, "removed_spans": changes, "log": log}
    p, q = res["proposed"]["rows"]["scored"], res["proposed"]["rows"]["J2"]
    res["dev_rule"] = {"hits_not_down": q["hits"] >= p["hits"], "wrong_down": q["wrong"] < p["wrong"]}
    res["dev_rule"]["pass"] = bool(res["dev_rule"]["hits_not_down"] and res["dev_rule"]["wrong_down"])
    print(f"[dev rule] {res['dev_rule']}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("beats", "score"))
    a = ap.parse_args()
    beats() if a.step == "beats" else score()


if __name__ == "__main__":
    main()

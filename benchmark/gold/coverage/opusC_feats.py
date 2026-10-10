"""opusC (10 Oct): every v1.6 picture on the 158 clips with its marginal effect (hits lost / wrongs removed if it alone is
dropped) and the cached audio + decision signals around it: stage-5 spec, gate votes, trail values of the drawn stage-4
candidate(s) it came from (docs/decision_trail/data.js), and the BEATs / FlexSED / DASM family curves at its start
(benchmark/gold/v14/detector_curves.json).  -> opusC_feats.json
    python benchmark/gold/coverage/opusC_feats.py"""
import json, re, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, stems, IN, GOLD
from src.stage6_visual_augmentation import v16

HERE = Path(__file__).resolve().parent


def setup():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    config.ONSET_CURVES = str(IN / "detector_curves.json")
    return gold


def pictures_spec(specs, dur, stem):
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])]) for s in specs]
    placed, _ = _assign_rows(_display_spans(objs, dur, require_image=True, clip=stem))
    return [(lab, float(a), float(b), sp) for _, lab, a, b, sp in placed]


def wrongs(r):
    return r["visible"] + r["cross"] + r["phantom"]


def num(rx, s, k=1):
    m = re.search(rx, s or "")
    return float(m.group(k)) if m else None


def trail_feats(cands):
    """flatten the drawn candidates' trail values (first match over candidates)"""
    f = {"origin": ",".join(sorted({c["origin"] for c in cands})), "n_cand": len(cands)}
    for c in cands:
        for s in c["trail"]:
            v, st = s.get("value") or "", s["step"]
            if st == "beats_extract":
                f.setdefault("beats_peak", num(r"peak ([\d.]+)", v))
            elif st == "flexsed_extract":
                f.setdefault("flex_peak", num(r"peak ([\d.]+)", v))
            elif st == "mirror_veto" and "top here" in v:
                f.setdefault("mirror_top", num(r"top here: .*? ([\d.]+);", v)); f.setdefault("mirror_own", num(r"own family .*? ([\d.]+)", v))
                f.setdefault("mirror_top_lab", re.search(r"top here: ([^(]+)", v).group(1).strip())
            elif st == "dasm_clip_veto":
                f.setdefault("dasm_clip", num(r"([\d.]+)$", v))
            elif st == "dasm_local_veto":
                f.setdefault("dasm_local", num(r"0\.5 s ([\d.]+)", v))
            elif st == "flexsed_cross_veto":
                f.setdefault("flex_clip", num(r"([\d.]+)$", v))
            elif st == "masked_weak":
                f.setdefault("sm_max", num(r"Speech/Music max in span ([\d.]+)", v))
            for a in s.get("asks") or []:
                w, ans, vote = a.get("who", ""), a.get("a", ""), a.get("vote", "")
                if "V1 (multiple choice)" in w:
                    f.setdefault("v1_p", num(r"p\(X\) ([\d.e-]+)", ans))
                elif "V2 (yes/no logit" in w:
                    f.setdefault("v2_run", num(r"s_run (-?[\d.]+)", ans)); f.setdefault("v2_ctrl", num(r"s_ctrl (-?[\d.]+)", ans))
                elif "Qwen3-Omni-30B-A3B V4" in w and "not asked" not in ans:
                    lines = [x.strip() for x in ans.split("\n") if x.strip()]
                    f.setdefault("q4_names", vote.startswith("names") or vote.startswith("accepts"))
                    f.setdefault("q4_n", len(set(lines))); f.setdefault("q4_first", lines[0] if lines else "")
                elif "Audio Flamingo Next V4" in w and "not asked" not in ans:
                    f.setdefault("af4_names", vote.startswith("accepts") or vote.startswith("names"))
                    f.setdefault("af4_ans", ans[:80])
    return f


def curve_feats(st, lab, a, b):
    stored = v16._offline(st) or {}
    c = v16.curves(lab, st, stored=stored)
    f = {}
    for det in v16.DETECTORS:
        sc, rise = v16.onset(c[det], a)
        f[f"{det}_on"], f[f"{det}_rise"] = sc, rise
        if c[det] is not None:
            t, v = c[det]
            w = (t >= a) & (t <= b)
            f[f"{det}_inmax"] = float(v[w].max()) if w.any() else None
            f[f"{det}_inmed"] = float(np.median(v[w])) if w.any() else None
            f[f"{det}_clipmax"] = float(v.max())
            pre = (t >= a - 3) & (t <= a - 0.5)
            f[f"{det}_premax"] = float(v[pre].max()) if pre.any() else None
    return f


def main():
    gold = setup()
    st5 = json.loads((IN / "stage5_specs.json").read_text(encoding="utf-8"))
    dur = json.loads((IN / "durations.json").read_text(encoding="utf-8"))
    t = open(_ROOT / "docs/decision_trail/data.js", encoding="utf-8").read()
    trail = {c["clip"]: c for c in json.loads(t[t.index("=") + 1:].strip().rstrip(";"))["clips"]}
    dev = set(stems("dev"))
    rows = []
    for st in stems("dev") + stems("test"):
        r5 = st5[st]
        P = decide(r5["P"], r5["B"], r5["gate"])
        pics = pictures_spec(P, float(dur[st] or 10), st)
        base = S.score_clip(gold[st], [p[:3] for p in pics])
        for i, (lab, a, b, sp) in enumerate(pics):
            rest = [p[:3] for j, p in enumerate(pics) if j != i]
            r = S.score_clip(gold[st], rest)
            alone = S.score_clip(gold[st], [(lab, a, b)])
            spec = next((s for s in P if s["event_label"] == lab and abs(s["start"] - sp.start) < 0.011 and abs(s["end"] - sp.end) < 0.011), {})
            g = [x for x in r5["gate"] if x["label"] == lab and abs(x["start"] - sp.start) < 0.011 and abs(x["end"] - sp.end) < 0.011]
            stv = g[0]["stretches"] if g else []
            cands = [c for c in trail[st]["cands"] if (c["fate"] == "drawn" or any(x["step"] == "gate" for x in c["trail"]))
                     and S.same_family(c["label"], lab) and c["start"] - 1.0 <= a <= c["end"] + 0.5]
            near = sorted(cands, key=lambda c: (c["fate"] != "drawn", abs(c["start"] - a)))[:1]
            orig = next((s0 for s0 in r5["P"] if s0["event_label"] == lab and abs(s0["start"] - sp.start) < 0.011 and abs(s0["end"] - sp.end) < 0.011), {})
            other = [x for x in pics if x[0] != lab and not S.same_family(x[0], lab) and x[2] > a and x[1] < b]
            playing = [gg["label"] for gg in gold[st] if S.in_window(a, gg["start"], S.EARLY, S.LATE) or gg["start"] <= a <= gg["end"]]
            row = {"clip": st, "split": "dev" if st in dev else "test", "label": lab, "start": a, "end": b, "dur": b - a,
                   "d_hit": base["hit"] - r["hit"], "d_wrong": wrongs(base) - wrongs(r),
                   "cls": "hit" if base["hit"] - r["hit"] > 0 else ("visible" if alone["visible"] else "cross" if alone["cross"] else "phantom" if alone["phantom"] else "other") if wrongs(base) - wrongs(r) > 0 else "neutral",
                   "conf": float(spec.get("confidence", sp.confidence)), "reason": spec.get("reason", ""), "rescued": spec.get("rescued"),
                   "arbiter": spec.get("arbiter"), "n_spans": len(spec.get("spans") or []), "detail": spec.get("detail", ""),
                   "spec_start": sp.start, "spec_end": sp.end, "playing": playing,
                   "gate_n": len(stv), "gate_seen": sum(bool(v["seen"]) for v in stv), "gate_name": sum(bool(v.get("name")) for v in stv),
                   "gate_ab": sum(bool(v.get("ab")) for v in stv), "gate_desc": sum(bool(v.get("desc")) for v in stv),
                   "gate_named": [re.search(r"named '([^']*)'", v.get("raw", "")).group(1) if re.search(r"named '([^']*)'", v.get("raw", "")) else "" for v in stv],
                   "overlap_other": [x[0] for x in other], "n_pics_clip": len(pics),
                   "ab_override": (not orig.get("augment")) and str(orig.get("reason", "")).startswith("source visible on screen")}
            row.update(trail_feats(near)); row["trail_cands"] = [c["id"] for c in near]
            row.update(curve_feats(st, lab, a, b))
            rows.append(row)
    (HERE / "opusC_feats.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    from collections import Counter
    print(Counter(r["cls"] for r in rows), sum(r["d_hit"] for r in rows), sum(r["d_wrong"] for r in rows))


if __name__ == "__main__":
    main()

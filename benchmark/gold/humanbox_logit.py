"""Round 50L HUMAN-BOX-LOGIT (docs/prereg_round13_detector_push.md "Round 50L"): Round 50's cached phrase + box, crop rebuilt
exactly, crop question read as m = s(Q) - s(notQ) (logit_gate.Reader, one forward pass each, no generation).

    python benchmark/gold/humanbox_logit.py run      # GPU, ~/MscProj -> gate_gold/humanbox_logit_Qwen38-27B/
    python benchmark/gold/humanbox_logit.py score    # CPU, ~/MscProj -> benchmark/gold/humanbox_logit_gold.json (exit 0 = step 1 PASS)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

CROP_Q = ("This is a close-up cut from a video frame. Is this {phrase} making the {label} sound right now? "
          "Answer yes or no.")
CROP_NQ = ("This is a close-up cut from a video frame. Is this {phrase} NOT making the {label} sound right now? "
           "Answer yes or no.")
NAMED = "bell_miami"
NEG = float("-inf")


def _paths():
    from benchmark.gold import gate_gold as G
    return G, G.OUT_DIR / "humanbox_Qwen38-27B", G.OUT_DIR / "humanbox_logit_Qwen38-27B"


def run():
    import config
    from src.stage2_video_understanding import _sample_frames_at
    from benchmark.gold.som_gate import crops
    from benchmark.gold.humanbox_gate import CROP_Q as Q50
    from benchmark.gold.logit_gate import Reader
    assert Q50 == CROP_Q, "CROP_Q must be Round 50's verbatim"
    config.use_v4("5")
    G, SRC, OUT = _paths()
    OUT.mkdir(parents=True, exist_ok=True)
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    R = None
    for f in sorted(SRC.glob("*.json")):
        if (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        for s in d["sounds"]:
            for st in s["stretches"]:
                hb = st.get("hbox") or {}
                st["hlog"] = None
                if hb.get("status") != "box" or not hb.get("crop_file") or p is None:
                    continue
                times = [t for _, t in (st.get("human") or {}).get("frames", [])]
                frames = _sample_frames_at(p, times)
                img = frames[hb["frame"] - 1]
                assert list(img.size) == hb["size"], (f.stem, img.size, hb["size"])
                cr = crops(img, [hb["bbox_px"]])
                if not cr:
                    continue
                R = R or Reader()
                r = R.pair([CROP_Q.format(phrase=hb["cand"], label=s["label"]), CROP_NQ.format(phrase=hb["cand"], label=s["label"])], [cr[0]])
                st["hlog"] = r
                print(f.stem, s["label"], st["start"], repr(hb["cand"]), "text", hb.get("crop_yes"), "m", round(r["d"], 3), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def m_of(st):
    return st["hlog"]["d"] if st.get("hlog") else None


def agg(sts):
    v = sorted([(m_of(st) if m_of(st) is not None else NEG) for st in sts], reverse=True)
    return v[len(v) // 2]                       # the floor(n/2)+1-th largest: A > t <=> more than half the stretches m > t


def score():
    from benchmark.gold.box_gate import gold_index, majority
    from benchmark.gold import human_gate as H
    G, SRC, OUT = _paths()
    gold = gold_index()
    files = sorted(OUT.glob("*.json"))
    assert len(files) == 49, len(files)
    sounds, crops_m = [], ([], [])
    n_yn = n_reads = 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            sts = s["stretches"]
            ms = [m_of(st) for st in sts]
            for st in sts:
                if st.get("hlog"):
                    n_yn += sum(st["hlog"]["yn_argmax"]); n_reads += 2
                    crops_m[0 if g["seen"] else 1].append(st["hlog"]["d"])
            sounds.append({"clip": f.stem, "label": s["label"], "start": s["start"], "seen": bool(g["seen"]),
                           "shipped_silenced": all(majority(st) for st in sts), "A": agg(sts), "m": ms,
                           "mean_crop": statistics.fmean([x for x in ms if x is not None]) if any(x is not None for x in ms) else None,
                           "phrases": [(st.get("hbox") or {}).get("cand") for st in sts],
                           "text_crop": [(st.get("hbox") or {}).get("crop_yes") for st in sts],
                           "votes": [[st.get("name"), st.get("ab"), st.get("desc")] for st in sts], "_sts": sts})
    base = {"seen": sum(x["seen"] for x in sounds), "seen_sil": sum(x["seen"] and x["shipped_silenced"] for x in sounds),
            "needed": sum(not x["seen"] for x in sounds), "needed_kept": sum(not x["seen"] and not x["shipped_silenced"] for x in sounds)}
    assert base == H.BASE, base
    A = auroc([x["A"] for x in sounds if x["seen"]], [x["A"] for x in sounds if not x["seen"]])
    mc = [x for x in sounds if x["mean_crop"] is not None]
    A_mean = auroc([x["mean_crop"] for x in mc if x["seen"]], [x["mean_crop"] for x in mc if not x["seen"]]) if mc else None
    A_st = auroc(*crops_m) if crops_m[0] and crops_m[1] else None
    stop = A < 0.65
    print(f"base {base}; crops {len(crops_m[0]) + len(crops_m[1])} (seen {len(crops_m[0])} / needed {len(crops_m[1])}); "
          f"yes/no argmax {n_yn}/{n_reads}")
    print(f"STEP 0: AUROC of A {A:.3f} -> {'STOP' if stop else 'GO'} | report only: mean-over-crops AUROC "
          f"{A_mean if A_mean is None else round(A_mean, 3)} on {len(mc)} sounds, per-crop AUROC {A_st if A_st is None else round(A_st, 3)}")

    def vote(st, t):
        return st.get("hlog") is not None and st["hlog"]["d"] > t

    def seen_b(st, t):
        vs = [st.get("name"), st.get("ab"), st.get("desc"), vote(st, t)]
        tr, fa = sum(v is True for v in vs), sum(v is False for v in vs)
        return tr > fa if tr != fa else majority(st)

    def tally(t, var):
        sil = kept = 0
        for x in sounds:
            n = len(x["_sts"])
            fn = (lambda st: vote(st, t)) if var == "a" else (lambda st: seen_b(st, t))
            pred = sum(bool(fn(st)) for st in x["_sts"]) * 2 > n
            sil += x["seen"] and pred
            kept += (not x["seen"]) and (not pred)
        return sil, kept

    vals = sorted({v for v in crops_m[0] + crops_m[1]})
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [float("inf"), (vals[0] - 1.0) if vals else 0.0]
    best = {}
    for var in "ab":
        curve = []
        for t in cands:
            sil, kept = tally(t, var)
            curve.append({"t": t, "silenced": sil, "kept": kept,
                          "bar": (sil >= 19 and kept >= 32) or (sil >= 15 and kept >= 35)})
        ok = [c for c in curve if c["bar"]]
        best[var] = max(ok, key=lambda c: (c["silenced"], c["kept"], c["t"])) if ok else None
        top = max(curve, key=lambda c: (c["silenced"] - (38 - c["kept"]), c["kept"]))
        print(f"step 1 ({var}) {'REPLACE' if var == 'a' else '4TH VOTE'}: best under bar {best[var]} | best net (report) {top}")
    pas = [v for v in "ab" if best[v]]
    chosen = max(pas, key=lambda v: (best[v]["silenced"], best[v]["kept"], v == "b")) if (pas and not stop) else None
    step1 = "not run (step 0 STOP)" if stop else ("PASS" if chosen else "FAIL")
    print("STEP 1:", step1, "| variant", chosen, "| t", best[chosen]["t"] if chosen else None)
    for x in sounds:
        if x["clip"] == NAMED:
            print("named", NAMED, x["label"], x["start"], "seen" if x["seen"] else "NEEDED", "phrases", x["phrases"],
                  "m", [None if v is None else round(v, 3) for v in x["m"]], "A", x["A"], "text crop", x["text_crop"])
    ranked = sorted([[x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED", x["A"], x["m"]] for x in sounds
                     if x["A"] != NEG], key=lambda r: -r[4])
    for r in ranked:
        print("   A", r)
    for x in sounds:
        x.pop("_sts")
    out = _ROOT / "benchmark" / "gold" / "humanbox_logit_gold.json"
    out.write_text(json.dumps({"base": base, "auroc_A": A, "auroc_mean_crop": A_mean, "auroc_per_crop": A_st, "stop": stop,
                               "best": best, "chosen": chosen, "step1": step1, "yn_argmax": [n_yn, n_reads],
                               "ranked": ranked, "named": [x for x in sounds if x["clip"] == NAMED], "sounds": sounds,
                               "q": [CROP_Q, CROP_NQ]}, indent=1, default=lambda v: None if v == NEG else float(v)), encoding="utf-8")
    print(out)
    sys.exit(0 if chosen else 3)


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()

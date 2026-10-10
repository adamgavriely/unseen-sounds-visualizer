"""Opus F (11 Oct, logging only): features of every dropped stage-4 candidate (decision trail, docs/decision_trail/data.js)
and of every stage-5 sound that the plan without the on-screen check (B) draws but v1.7 does not, labelled against the
v1.7 misses. Shared by opusF_rules.py.
    python benchmark/gold/coverage/opusF_feats.py   -> prints the 46-style miss table for v1.7"""
import json, re, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import stems, GOLD
from benchmark.gold.coverage.screen_reason import base_pics

NUM = re.compile(r"(-?\d+\.\d+|-?\d+)")


def _f(txt, key):
    if not txt or key not in txt:
        return None
    m = NUM.search(txt[txt.index(key) + len(key):])
    return float(m.group(1)) if m else None


def cand_feats(c):
    f = {"label": c["label"], "start": c["start"], "end": c["end"], "origin": c["origin"], "at": c.get("at"), "fate": c["fate"],
         "peak": None, "len": c["end"] - c["start"], "qwen": 0, "af": 0, "dasm": None, "dasm_clip": None, "flex_clip": None,
         "finelap": None, "sm": None, "steps": [t["step"] for t in c["trail"]]}
    for t in c["trail"]:
        v = t.get("value") or ""
        st = t["step"]
        if st in ("beats_extract", "flexsed_extract"):
            f["peak"] = _f(v, "peak")
        elif st in ("band_rescue", "dasm_rescue"):
            f["peak"] = _f(v, "run peak")
        elif st == "dasm_local_veto":
            f["dasm"] = _f(v, "+- 0.5 s") if "n/a" not in v else None
        elif st == "dasm_vote":
            f["dasm"] = _f(v, "DASM max")
        elif st == "dasm_clip_veto":
            f["dasm_clip"] = None if "n/a" in v else _f(v, c["label"]) if c["label"] in v else _f(v, "max of")
        elif st == "flexsed_cross_veto":
            f["flex_clip"] = None if "no query" in v else _f(v, "max of " + c["label"]) if ("max of " + c["label"]) in v else None
        elif st == "finelap_veto":
            f["finelap"] = _f(v, "FineLAP max")
        elif st == "masked_weak":
            f["sm"] = _f(v, "Speech/Music max in span")
        for a in t.get("asks", []):
            vote = a.get("vote", "")
            if vote.startswith("names") or vote.startswith("accepts"):
                if a["who"].startswith("Qwen3-Omni"):
                    f["qwen"] = 1
                elif a["who"].startswith("Audio Flamingo"):
                    f["af"] = 1
    if c["origin"] == "dasm" and f["dasm"] is None:
        f["dasm"] = f["peak"]
    return f


def load():
    gold = S.load_gold([GOLD]); pics = base_pics()
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = {c["clip"]: c for c in json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["clips"]}
    allc = stems("dev") + stems("test")
    miss, cands = {}, {}
    for st in allc:
        miss[st] = [g for g in gold[st] if g["needed"] and g["importance"] >= 2 and not any(
            S.same_family(l, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE) for l, a, b in pics[st])]
        out = []
        for c in dj[st]["cands"]:
            if c["fate"] != "dropped":
                continue
            f = cand_feats(c); f["clip"] = st
            f["real"] = any(S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE) for g in miss[st])
            out.append(f)
        cands[st] = out
    return gold, pics, miss, cands, allc


def main():
    gold, pics, miss, cands, allc = load()
    rows = [S.score_clip(gold[st], pics[st]) for st in allc]
    a = S.aggregate(rows); print("v1.7:", a["hits"], a["needed"], a["visible"] + a["cross"] + a["phantom"], round(a["viewer_cost"], 3))
    nm = sum(len(v) for v in miss.values()); nc = 0
    for st in allc:
        for g in miss[st]:
            cs = [f for f in cands[st] if S.same_family(f["label"], g["label"]) and S.in_window(f["start"], g["start"], S.EARLY, S.LATE)]
            if not cs:
                continue
            nc += 1
            b = max(cs, key=lambda f: len(f["steps"]))
            print(f"{st[:28]:28s} {g['label'][:18]:18s} {b['at']:18s} {b['origin']:12s} pk {b['peak']} q{b['qwen']} af{b['af']} dasm {b['dasm']} dclip {b['dasm_clip']} flex {b['flex_clip']} fl {b['finelap']} n={len(cs)}")
    print("misses", nm, "with in-window candidate", nc)


if __name__ == "__main__":
    main()

"""Round 61c v-think, stopped early (Adam 21:49): per-picture diff of the thinking-on answers (explain_pics_think/, partial n)
vs the short 61b answers (explain_pics/), and for every changed picture whether its stage-4 span(s) are "unsure": DASM max over
[pre_start - 0.5, end + 0.5] < 0.35 and not both ears (Qwen V4 and AF V4), as weakwitness_dev.step0 but for any class.
CPU, msproj, from ~/MscProj_tg:  python benchmark/gold/explain_think_diff.py  -> benchmark/gold/explain_think_diff.json
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")
G = _ROOT / "benchmark" / "gold"


def verdicts(r):
    q3 = r.get("q3") or [None, None]
    return {"q1": r.get("q1"), "q1_unlikely": bool(r.get("q1_unlikely")),
            "q2": r.get("q2_thing") if r.get("q1_unlikely") else None, "q3": q3, "drop_61b": bool(r.get("silenced"))}


def main():
    T = {f.name: json.loads(f.read_text(encoding="utf-8")) for f in sorted((G / "explain_pics_think").glob("*.json"))}
    M = {n: json.loads((G / "explain_pics" / n).read_text(encoding="utf-8")) for n in T}
    pairs = collections.Counter(str(r["q1"]) for r in T.values())
    none = sum(1 for r in T.values() if not r["q1"] or None in r["q1"])
    reprompts = sum(x is not None for r in T.values() for x in (r.get("q1_reprompt") or []))
    cut = sum(1 for r in T.values() for x in r["q1_replies"] if len(x.split()) > 3)   # last line not a bare verdict
    changed = []
    for n, t in T.items():
        a, b = verdicts(M[n]), verdicts(t)
        diff = [k for k in ("q1_unlikely", "q2", "drop_61b") if a[k] != b[k]]
        if diff:
            changed.append({"key": n[:-5], "part": t["part"], "clip": t["clip"], "label": t["label"], "start": t["start"],
                            "diff": diff, "short": a, "think": b, "think_q1_replies": t["q1_replies"],
                            "think_q1_reprompt": t.get("q1_reprompt"), "think_q2": [t.get("q2_reply"), t.get("q2_reprompt")],
                            "think_q3": [t.get("q3_replies"), t.get("q3_reprompt")]})
    # unsure flag for the changed pictures (weakwitness_dev.step0 logic, any class)
    import config
    from src.labels import canonical
    from src.stage4_audio_event_detection import _v4_names_qwen, _af_p1_accepts
    from src.types import AudioEvent
    from benchmark.gold import weakwitness_dev as W
    from benchmark.gold import round13_dev as R
    arm = "SHIP8+MD3"
    P = W.parts([arm])
    for c in changed:
        gold, pics, cfgs, s4 = P[c["part"]]
        cfg = cfgs[arm]
        rows_all = json.loads(Path(s4).read_text(encoding="utf-8"))["arms"][f"{arm}|{W.SYS}"]
        rr = rows_all.get(c["clip"], [])
        rr = rr.get("rows", rr) if isinstance(rr, dict) else rr
        f = Path(cfg["LISTENER_DASM_DIR"]) / f"{c['clip']}.npz"
        z = np.load(f, allow_pickle=True) if f.exists() else None
        fam, a = canonical(c["label"]), float(c["start"])
        rows = []
        for r in [r for r in rr if canonical(r["label"]) == fam and r["start"] - 1.0 <= a <= r["end"] + 0.5]:
            ps, en = float(r.get("pre_start", r["start"])), float(r["end"])
            dm = None
            if z is not None:
                cols = [i for i, l in enumerate([str(x) for x in z["labels"]]) if canonical(l) == fam]
                m = (z["times"] >= ps - 0.5) & (z["times"] <= en + 0.5)
                if cols and m.any():
                    dm = round(float(z["fw"][m][:, cols].max()), 3)
            with R.flags(cfg):
                config._CURRENT_CLIP = c["clip"]
                e = AudioEvent(r["label"], ps, en, float(r["conf"]))
                q, af = bool(_v4_names_qwen(e)), bool(_af_p1_accepts(e))
            rows.append({"pre_start": round(ps, 2), "end": round(en, 2), "dasm_win": dm, "qwen_v4": q, "af_v4": af,
                         "ears": int(q) + int(af), "unsure": dm is not None and dm < 0.35 and not (q and af)})
        c["rows"] = rows
        c["unsure"] = any(x["unsure"] for x in rows) if rows else None
    for c in changed:
        print(c["part"], c["clip"], c["label"], c["start"], c["diff"], "| short", c["short"]["q1"], c["short"]["q2"],
              c["short"]["drop_61b"], "| think", c["think"]["q1"], c["think"]["q2"], c["think"]["q3"], c["think"]["drop_61b"],
              "| unsure", c["unsure"], [(x["dasm_win"], x["ears"]) for x in c["rows"]])
    s = {"n_think": len(T), "q1_pairs": dict(pairs), "q1_pairs_with_none": none, "q1_reprompts_used": reprompts,
         "q1_replies_not_bare_verdict": cut, "n_changed": len(changed), "changed": changed}
    print({k: v for k, v in s.items() if k != "changed"})
    (G / "explain_think_diff.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Round 31 PTC screen (docs/prereg_round13_detector_push.md, "Round 31 PTC"): on merged-DEV P2/PV candidates with a long
listener cut, how many needed-class candidates does the peak-tight cut add under the TIER rule (each ear's flag = original OR
tight), and how many other-class. GO iff needed added >= 2 and other added <= 2 x needed. Reported beside: per-leg additions
(Qwen-only, AF-only) and, of the needed additions, how many have DASM >= 0.575 within +-0.5 s (F8's ceiling).

    python benchmark/gold/ptc_screen.py            # from ~/MscProj_tg on the cluster (CPU)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_listener as L
from src.labels import canonical
from src.stage4_audio_event_detection import _tier

G = _ROOT / "benchmark" / "gold"
H = Path.home()
PARTS = {"dev": (H / "MscProj_r13" / "benchmark" / "gold" / "dev_listener_v.json", G / "dev_listener_afn.json",
                 G / "dev_listener_ptc.json", G / "annotations" / "gold_AG.json",
                 H / "MscProj_r13" / "data" / "work" / "devcand" / "dasm_cache"),
         "dev2": (G / "dev2_listener_v.json", G / "dev2_listener_afn.json", G / "dev2_listener_ptc.json",
                  G / "annotations" / "tagger_AG.json", H / "MscProj" / "data" / "work" / "dasm_dev2")}
DASM_BAR = float(getattr(config, "LISTENER_DASM_BAR", 0.575))


def key(x):
    return (x["clip"], x["pool"], x["label"], round(x["start"], 2), round(x["end"], 2))


def dasm_ok(ddir, x):
    f = ddir / f"{x['clip']}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    fw, t, labs = z["fw"], z["times"], [str(v) for v in z["labels"]]
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(x["label"])]
    m = (t >= x["start"] - 0.5) & (t <= x["end"] + 0.5)
    if not cols or not m.any():
        return False
    return bool(float(fw[m][:, cols].max()) >= DASM_BAR)


def main():
    tot = {"needed_added": 0, "other_added": 0, "needed_added_dasm_ok": 0, "qwen_only_needed": 0, "qwen_only_other": 0,
           "af_only_needed": 0, "af_only_other": 0, "long_cut_items": 0, "needed_class_long": 0}
    lines = []
    for part, (vp, ap, pp, gp, dd) in PARTS.items():
        v = {key(x): x for x in json.loads(vp.read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")}
        af = {key(x): x for x in json.loads(ap.read_text(encoding="utf-8"))["items"]}
        p = {key(x): x for x in json.loads(pp.read_text(encoding="utf-8"))["items"]}
        gold = S.load_gold([gp])
        for k, x in p.items():
            if k not in v or x["clip"] not in gold:
                continue
            peak = float(v[k].get("peak") or 0.0)
            q0 = bool(v[k]["accept"].get("V4")); a0 = bool((af.get(k, {}).get("accept") or {}).get("V4"))
            q1 = q0 or bool(x["accept"].get("PTC_V4")); a1 = a0 or bool(x["accept"].get("PTC_AF"))
            t0 = _tier({"V4": q0, "AF_V4": a0}, peak); t1 = _tier({"V4": q1, "AF_V4": a1}, peak)
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            needed = cls == "hit_needed"
            tot["long_cut_items"] += 1; tot["needed_class_long"] += needed
            if q1 and not q0:
                tot["qwen_only_needed" if needed else "qwen_only_other"] += 1
            if a1 and not a0:
                tot["af_only_needed" if needed else "af_only_other"] += 1
            if t1 and not t0:
                d = dasm_ok(dd, x)
                tot["needed_added" if needed else "other_added"] += 1
                if needed and d:
                    tot["needed_added_dasm_ok"] += 1
                lines.append(f"[{part}] {x['clip']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {peak:.2f} "
                             f"win {x['ptc_win']} [{cls}] qwen {q0}->{q1} af {a0}->{a1} dasm_ok {d} | "
                             f"Q: {x.get('ptc_v4_text', '')[:60]!r} | AF: {x.get('ptc_afn_text', '')[:60]!r}")
    go = tot["needed_added"] >= 2 and tot["other_added"] <= 2 * tot["needed_added"]
    for s in lines:
        print(s)
    print("PTC screen:", json.dumps(tot), "GO" if go else "STOP")
    (G / "ptc_screen.json").write_text(json.dumps({**tot, "go": go, "added": lines}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

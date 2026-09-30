"""Round 31 H2 screen (docs/prereg_round13_detector_push.md, "Round 31 H2"): on merged-DEV P2/PV candidates with FlexSED peak
>= TIER_SPLIT (the Qwen-only tier), how many needed-class candidates are added when the high tier also accepts
"AF V4 yes AND Kimi V4 yes" (two independent ears agree while Qwen is silent), and how many other-class. Same GO bar as
v4d_screen: needed added >= 2 and other added <= 2 x needed. Reported beside: AF-alone additions (K3's rule) and, of the
needed additions, how many have DASM >= 0.575 within +-0.5 s (F8's ceiling). CPU, existing caches only.

    python benchmark/gold/h2_screen.py            # from ~/MscProj_tg on the cluster
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_listener as L
from benchmark.gold.ptc_screen import PARTS, key, dasm_ok

G = _ROOT / "benchmark" / "gold"
KIMI = {"dev": next(q for q in (G / "dev_listener_kimi.json", Path.home() / "MscProj_r13" / "benchmark" / "gold" / "dev_listener_kimi.json",
                                Path.home() / "MscProj" / "benchmark" / "gold" / "dev_listener_kimi.json") if q.exists()),
        "dev2": G / "dev2_listener_kimi.json"}
SPLIT = float(getattr(config, "TIER_SPLIT", 0.6))


def main():
    tot = {"needed_added": 0, "other_added": 0, "needed_added_dasm_ok": 0, "af_alone_needed": 0, "af_alone_other": 0,
           "high_tier_items": 0}
    lines = []
    for part, (vp, ap, _pp, gp, dd) in PARTS.items():
        v = {key(x): x for x in json.loads(vp.read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")}
        af = {key(x): x for x in json.loads(ap.read_text(encoding="utf-8"))["items"]}
        km = {key(x): x for x in json.loads(KIMI[part].read_text(encoding="utf-8"))["items"]}
        gold = S.load_gold([gp])
        for k, x in v.items():
            peak = float(x.get("peak") or 0.0)
            if peak < SPLIT or x["clip"] not in gold:
                continue
            tot["high_tier_items"] += 1
            q = bool(x["accept"].get("V4"))
            a = bool((af.get(k, {}).get("accept") or {}).get("V4"))
            ki = bool((km.get(k, {}).get("accept") or {}).get("V4"))
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            needed = cls == "hit_needed"
            if a and not q:
                tot["af_alone_needed" if needed else "af_alone_other"] += 1
            if a and ki and not q:
                d = dasm_ok(dd, x)
                tot["needed_added" if needed else "other_added"] += 1
                if needed and d:
                    tot["needed_added_dasm_ok"] += 1
                lines.append(f"[{part}] {x['clip']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {peak:.2f} [{cls}] "
                             f"dasm_ok {d} | AF: {af[k].get('afn_v4_text', '')[:50]!r} | Kimi: {km[k].get('kimi_v4_text', '')[:50]!r}")
    go = tot["needed_added"] >= 2 and tot["other_added"] <= 2 * tot["needed_added"]
    for s in lines:
        print(s)
    print("H2 screen:", json.dumps(tot), "GO" if go else "STOP")
    (G / "h2_screen.json").write_text(json.dumps({**tot, "go": go, "added": lines}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

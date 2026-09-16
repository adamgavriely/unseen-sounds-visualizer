"""Cost to the viewer, next to the benefit: how long the panel is on, and how much of that
is wasted (on clips where no picture was due).

Computed from the cached gate decisions of the test clips (benchmark/gate_votes/test, the
same votes the shipping gate uses; agrees with the v3 protocol run on 97 of 100 clips on
whether anything was shown). Per system the panel is on for the union of the spans of the
sounds it shows, each held at least the panel's minimum dwell (1.5 s):
  gated    the sounds the gate lets through
  blind    every detected sound (no gate)
  caption  the same timeline as blind (a text line instead of a picture)
Reported per scenario group: mean panel-on fraction of the clip, and mean wasted seconds
(panel-on seconds on seen / no-ambient clips, where the label says nothing was due).

    python scripts/cost_metrics.py            -> benchmark/cost_metrics_v3.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gate_dev_sweep import load, decide, min_confidence, SHOULD_SHOW

OUT = _ROOT / "benchmark" / "cost_metrics_v3.json"
DWELL = 1.5


def union_seconds(spans, duration: float) -> float:
    iv = []
    for a, b in spans:
        b = max(b, a + DWELL)
        iv.append((max(0.0, a), min(duration, b)))
    iv.sort()
    total = 0.0; cur = None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur: total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: total += cur[1] - cur[0]
    return total


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    g = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    rows = []
    for rec in load("test"):
        dur = float(rec["duration"]) or 1.0
        live = [s for s in rec["sounds"] if s["confidence"] >= min_confidence(s["label"], g["bar"])]
        d = decide(rec, g["bar"], g["rule"], g["kinds"])
        shown = [s for s in live if s["label"] in d["shown"]]
        on = {"gated": union_seconds([(s["start"], s["end"]) for s in shown], dur),
              "blind": union_seconds([(s["start"], s["end"]) for s in live], dur)}
        on["caption"] = on["blind"]
        rows.append({"clip": rec["clip"], "tag": rec["tag"], "duration": dur, "due": rec["tag"] in SHOULD_SHOW,
                     "panel_on_s": on, "panel_on_frac": {k: v / dur for k, v in on.items()},
                     "n_shown": {"gated": len(shown), "blind": len(live)}})
    groups = {"picture due": [r for r in rows if r["due"]], "no picture due": [r for r in rows if not r["due"]],
              "all": rows}
    summary = {"dwell_s": DWELL, "n": len(rows), "groups": {}}
    print(f"cost to the viewer (test, {len(rows)} clips; panel-on = union of shown spans, dwell >= {DWELL} s)")
    for gname, rs in groups.items():
        summary["groups"][gname] = {"n": len(rs)}
        for sysname in ("gated", "blind"):
            frac = [r["panel_on_frac"][sysname] for r in rs]
            secs = [r["panel_on_s"][sysname] for r in rs]
            lo, hi = boot_ci(frac)
            summary["groups"][gname][sysname] = {"panel_on_frac_mean": float(np.mean(frac)), "ci": [lo, hi],
                                                 "panel_on_s_mean": float(np.mean(secs)),
                                                 "clips_with_panel": int(sum(f > 0 for f in frac))}
            print(f"  {gname:15s} n={len(rs):3d}  {sysname:6s} panel on {np.mean(frac):5.1%} of the clip "
                  f"[{lo:.1%}, {hi:.1%}]  = {np.mean(secs):4.1f} s/clip; panel shown on {sum(f > 0 for f in frac)} clips")
    nd = groups["no picture due"]
    wasted = {s: float(sum(r["panel_on_s"][s] for r in nd)) for s in ("gated", "blind")}
    summary["wasted_seconds_total"] = wasted
    summary["wasted_seconds_per_clip"] = {s: wasted[s] / len(nd) for s in wasted}
    print(f"  wasted seconds (panel on where nothing was due, {len(nd)} clips): gated {wasted['gated']:.0f} s "
          f"({wasted['gated']/len(nd):.1f} s/clip), blind {wasted['blind']:.0f} s ({wasted['blind']/len(nd):.1f} s/clip)")
    summary["rows"] = rows
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

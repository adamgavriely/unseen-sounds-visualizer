"""Attempt N+1 bar (docs/prereg_v4.md, "span-rate-matched PSED"): on the calibration set, the
loosest PSED bar (0.01 grid) whose FIRED spans per minute (all detections, true or false --
the quantity that becomes pictures) does not exceed BEATs' at its shipped bar 0.35. CPU, on
the cached windows; writes benchmark/psed_span_match.json."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from benchmark import audioset_detector_eval as E
from src.labels import is_salient_nonspeech, is_music

E.use_set("calib")
def rate(which, bar):
    n = 0; minutes = 0.0
    for c in E.clips():
        p = E.WIN / which / f"{c['id']}.npz"
        if not p.exists(): continue
        fw, t, lab = E._load(p)
        n += sum(1 for e in E._spans(fw, t, lab, bar) if is_salient_nonspeech(e.label) and not is_music(e.label))
        minutes += c["duration"] / 60.0
    return n / minutes, n, minutes
b_rate, bn, mins = rate("beats", 0.35)
grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]
chosen = None; table = {}
for bar in grid:
    r, n, _ = rate("psed", bar); table[bar] = r
    if chosen is None and r <= b_rate: chosen = bar
print(f"BEATs@0.35: {bn} fired spans in {mins:.1f} min = {b_rate:.2f}/min")
print(f"PSED fired spans/min: 0.15 -> {table[0.15]:.2f}, 0.20 -> {table[0.20]:.2f}, 0.28 -> {table[0.28]:.2f}, 0.35 -> {table[0.35]:.2f}")
print(f"span-rate-matched PSED bar: {chosen} ({table[chosen]:.2f}/min)")
json.dump({"beats_bar": 0.35, "beats_spans_per_min": b_rate, "psed_bar_matched": chosen, "psed_spans_per_min": {str(k): v for k, v in table.items()}},
          open("benchmark/psed_span_match.json", "w"), indent=1)

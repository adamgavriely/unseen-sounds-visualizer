"""Step 2 (PREREG_step2_gate_mirror.md): the time-local mirror veto arms, run through the scoring harness. Cluster.

Registers SHIP8+MD3+WW5+SL+TLS (TL-span) and +TLO (TL-onset) = the frozen arm with config.MIRROR_LOCAL set, replaces
stage 4's _mirror_veto by the time-local rule when MIRROR_LOCAL is set (the shipped rule otherwise), then hands over to
the harness CLI. Run from the checkout the harness runs in (old names: round13_dev / tagger_prep):

    python benchmark/gold/coverage/step2_arms.py dev  stage4 --arms SHIP8+MD3+WW5+SL+TLS SHIP8+MD3+WW5+SL+TLO
    python benchmark/gold/coverage/step2_arms.py dev2 --split dev2 stage4 --arms ...
"""
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(os.getcwd())
sys.path.insert(0, str(_ROOT))
import config
from src import stage4_audio_event_detection as S4

BASE_ARM = "SHIP8+MD3+WW5+SL"
NEW = {BASE_ARM + "+TLS": "span", BASE_ARM + "+TLO": "onset"}
ONSET_S, SHARE = 1.0, 0.5
_ORIG = S4._mirror_veto


def _mirror_veto_local(events, keep_ids, ffw, ftimes, flabels, bar, own_max):
    mode = getattr(config, "MIRROR_LOCAL", None)
    if not mode:
        return _ORIG(events, keep_ids, ffw, ftimes, flabels, bar, own_max)
    from src.labels import canonical
    fams = [canonical(l) for l in flabels]
    ft = np.asarray(ftimes)
    F = np.asarray(ffw)
    out, dropped = [], []
    for e in events:
        if id(e) in keep_ids:
            out.append(e); continue
        sk = getattr(config, "STRONG_BEATS_KEEP", None)
        if sk is not None and float(e.confidence) >= float(sk):
            out.append(e); continue
        own = [i for i, f in enumerate(fams) if f == canonical(e.label)]
        if not own:
            out.append(e); continue
        hi = e.end if mode == "span" else min(e.end, e.start + ONSET_S)
        m = (ft >= e.start) & (ft < hi)
        if not m.any():
            m = np.zeros(len(ft), bool); m[int(np.argmin(np.abs(ft - 0.5 * (e.start + hi))))] = True
        sub = F[m]
        top = sub.argmax(axis=1)
        other = np.array([fams[t] != canonical(e.label) for t in top])
        mirrored = other & (sub[np.arange(len(sub)), top] >= bar) & (sub[:, own].max(axis=1) < own_max)
        share = float(mirrored.mean())
        val = f"time-local ({mode}): {int(mirrored.sum())} of {len(mirrored)} frames mirrored ({share:.2f})"
        rule = f"drop if >= {SHARE:.0%} of frames have another family >= {bar} and own < {own_max} at the same frame"
        try:
            S4._tr(lambda: S4._T.decide("mirror_veto", e, "drop" if share >= SHARE else "pass", value=val, bar=rule))
        except Exception:
            pass
        (dropped if share >= SHARE else out).append(e)
    return out, dropped


S4._mirror_veto = _mirror_veto_local

try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R
for name, mode in NEW.items():
    R.ARMS[name] = {**R.ARMS[BASE_ARM], "MIRROR_LOCAL": mode}

part = sys.argv.pop(1)
if part == "dev":
    sys.argv[0] = "round13_dev"
    R.main()
else:
    os.environ["TG_ARMS"] = " ".join(list(NEW) + os.environ.get("TG_ARMS", "").split())
    try:
        from benchmark.gold import clip_prep as T
    except ImportError:
        from benchmark.gold import tagger_prep as T
    sys.argv[0] = "tagger_prep"
    T.main()

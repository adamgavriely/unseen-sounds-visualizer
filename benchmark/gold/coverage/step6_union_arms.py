"""Step 6 B1 (PREREG_step6_detector_retry_and_extra_ear.md): ATST-F as an extra ear alongside BEATs, through the harness.

Arms = the frozen arm + ATST_EAR (cached frame probabilities of an ATST-F ear, ears/<name>/dev/<stem>.npz) at ATST_BAR.
After stage 4's fuse_flexsed (every BEATs / FlexSED veto done), each ATST-F span of a depictable family (prob >= bar,
gaps < 0.5 s bridged, length >= 0.3 s) that no surviving event of the same family overlaps joins as a frame-level
candidate (confidence = peak mapped to 0.35 at the bar). It then passes the frozen DASM rules as written for a span
no listener was asked about: the clip veto (family's DASM clip max >= DASM_CLIP_VETO) and the two-witness rule (DASM max
in span +- 0.5 s >= DASM_LOCAL_VETO). Everything after stage 4 is the frozen system's (stage 5 asks the gate live for
new stretches, memoised).

    cd ~/MscProj_r13; python ~/MscProj_tg/benchmark/gold/coverage/step6_union_arms.py dev stage4 --arms <arms>
    cd ~/MscProj_tg;  python benchmark/gold/coverage/step6_union_arms.py dev2 stage4 --split dev2 --arms <arms>
"""
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(os.getcwd())
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_salient_nonspeech, is_music
from src.types import AudioEvent

EARS = Path.home() / "MscProj_tg" / "scratch_cov" / "ears"
BASE_ARM = "SHIP8+MD3+WW5+SL"
NEW = {BASE_ARM + "+UNO3": ("atstf_orig", 0.3), BASE_ARM + "+UNO5": ("atstf_orig", 0.5),
       BASE_ARM + "+UNF3": ("atstf_ft_s0", 0.3), BASE_ARM + "+UNF5": ("atstf_ft_s0", 0.5)}

try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R
_FUSE = R.fuse_flexsed
STATS = {"added": 0, "overlap": 0, "dasm_clip": 0, "dasm_local": 0}


def drawable(lab):
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return bool(is_salient_nonspeech(lab)) and not is_music(lab) and canonical(lab) not in ("Speech", "Music")
    finally:
        config.LABEL_FILTER = old


def runs(p, t, bar):
    """frames >= bar; gaps < 0.5 s between them bridged; runs >= 0.3 s kept: (start, end, peak)"""
    on = p >= bar
    idx = np.flatnonzero(on)
    if not len(idx):
        return []
    out, a = [], idx[0]
    for k in range(1, len(idx) + 1):
        if k == len(idx) or t[idx[k]] - t[idx[k - 1]] >= 0.5:
            b = idx[k - 1]
            if t[b] - t[a] >= 0.3:
                out.append((float(t[a]), float(t[b]), float(p[a:b + 1].max())))
            if k < len(idx):
                a = idx[k]
    return out


def fuse_plus(*args, **kw):
    events, flex_ids, ffw = _FUSE(*args, **kw)
    ear, bar = getattr(config, "ATST_EAR", None), getattr(config, "ATST_BAR", None)
    clip = getattr(config, "_CURRENT_CLIP", None)
    if not ear or clip is None:
        return events, flex_ids, ffw
    z = np.load(EARS / ear / "dev" / f"{clip}.npz")
    fw, t, labs = z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
    dz = None
    d = getattr(config, "LISTENER_DASM_DIR", None)
    if d and (Path(d) / f"{clip}.npz").exists():
        dz = np.load(Path(d) / f"{clip}.npz", allow_pickle=True)
        dl, dfw, dt = [str(x) for x in dz["labels"]], dz["fw"], dz["times"]
    for i, lab in enumerate(labs):
        if not drawable(lab) or fw[:, i].max() < bar:
            continue
        for a, b, pk in runs(fw[:, i], t, bar):
            if any(canonical(e.label) == canonical(lab) and e.end > a and e.start < b for e in events):
                STATS["overlap"] += 1; continue
            if dz is not None:
                cols = [k for k, l in enumerate(dl) if canonical(l) == canonical(lab)]
                if cols:
                    if getattr(config, "DASM_CLIP_VETO", None) and float(dfw[:, cols].max()) < float(config.DASM_CLIP_VETO):
                        STATS["dasm_clip"] += 1; continue
                    m = (dt >= a - 0.5) & (dt <= b + 0.5)
                    if getattr(config, "DASM_LOCAL_VETO", None) and m.any() and float(dfw[m][:, cols].max()) < float(config.DASM_LOCAL_VETO):
                        STATS["dasm_local"] += 1; continue
            conf = 0.35 + 0.65 * (pk - bar) / (1 - bar)
            e = AudioEvent(lab, a, b, float(conf))
            events.append(e); flex_ids.add(id(e)); STATS["added"] += 1
    print(f"       [step6] ATST-F {ear} bar {bar}: {STATS}", flush=True)
    return events, flex_ids, ffw


R.fuse_flexsed = fuse_plus
for name, (ear, bar) in NEW.items():
    R.ARMS[name] = {**R.ARMS[BASE_ARM], "ATST_EAR": ear, "ATST_BAR": bar}

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

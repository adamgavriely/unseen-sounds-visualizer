"""Round 31 FLAP-F8 screen (docs/prereg_round13_detector_push.md, "Round 31 FLAP-F8"): among merged-DEV P2/PV candidates
the TIER listener rule accepts, F8 (DASM family max over span +-0.5 s >= 0.575) drops some. FLAP-F8 passes iff DASM >= 0.575
OR FineLAP family max over the span >= 0.329. GO iff needed restored >= 2 and other restored <= 2x needed.
Reported: FLAP-R (FineLAP replaces DASM): restored and newly dropped.

    python benchmark/gold/finelap_f8_screen.py      # from ~/MscProj_tg on the cluster (CPU; needs data/work/finelap_cache)
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
from benchmark.gold.finelap_screen import PARTS, CACHE, items, item_score, key
from src.labels import canonical
from src.stage4_audio_event_detection import _tier

G = _ROOT / "benchmark" / "gold"
BAR_FL, BAR_D, PAD = 0.329, 0.575, 0.5


def dasm_dir(part):
    d = Path(config.set_listener_split(part)["LISTENER_DASM_DIR"])
    return d if d.exists() else _ROOT / "data" / "work" / "devcand" / "dasm_cache"


def dasm_val(ddir, x):
    """None = no DASM file (F8 keeps); else the filter_rescued value (0.0 with no column / no frame)"""
    f = ddir / f"{x['clip']}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    fw, t, labs = z["fw"], np.asarray(z["times"]), [str(v) for v in z["labels"]]
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(x["label"])]
    m = (t >= x["start"] - PAD) & (t <= x["end"] + PAD)
    return float(np.asarray(fw)[m][:, cols].max()) if cols and m.any() else 0.0


def main():
    out = {"dasm_dirs": {}, "accepted": 0, "f8_dropped": 0, "no_dasm": 0,
           "F8OR": {"needed": [], "other": []}, "R_restored": {"needed": [], "other": []},
           "R_newly_dropped": {"needed": [], "other": []}}
    for part in PARTS:
        dd = dasm_dir(part); out["dasm_dirs"][part] = str(dd)
        gold = S.load_gold([PARTS[part]["gold"]])
        af = {key(x): x for x in json.loads(PARTS[part]["af"].read_text(encoding="utf-8"))["items"]}
        _, cand = items(part)
        for x in cand:
            if x["clip"] not in gold:
                continue
            peak = float(x.get("peak") or 0.0)
            q = bool(x["accept"].get("V4")); a = bool((af.get(key(x), {}).get("accept") or {}).get("V4"))
            if not _tier({"V4": q, "AF_V4": a}, peak):
                continue
            out["accepted"] += 1
            d = dasm_val(dd, x)
            if d is None:
                out["no_dasm"] += 1
                continue
            f = CACHE / f"{x['clip']}.npz"
            fl = item_score(np.load(f, allow_pickle=True), x) if f.exists() else None
            fl = -1.0 if fl is None else fl
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            c = "needed" if cls == "hit_needed" else "other"
            ln = (f"[{part}] {x['clip']} {x['pool']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {peak:.2f} "
                  f"Q {int(q)} AF {int(a)} DASM {d:.3f} FL {fl:.3f} [{cls}]")
            f8 = d >= BAR_D
            fok = fl >= BAR_FL
            if not f8:
                out["f8_dropped"] += 1
                if fok:
                    out["F8OR"][c].append(ln); out["R_restored"][c].append(ln)
            elif not fok:
                out["R_newly_dropped"][c].append(ln)
    n, o = len(out["F8OR"]["needed"]), len(out["F8OR"]["other"])
    out["go"] = n >= 2 and o <= 2 * n
    print(f"TIER-accepted {out['accepted']}, F8 drops {out['f8_dropped']}, no DASM file {out['no_dasm']}, dirs {out['dasm_dirs']}")
    for k in ("F8OR", "R_newly_dropped"):
        print(f"{k}: needed {len(out[k]['needed'])}, other {len(out[k]['other'])}")
        for s in out[k]["needed"] + out[k]["other"]:
            print("   ", s)
    print("FLAP-F8:", "GO" if out["go"] else "STOP")
    (G / "finelap_f8_screen.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

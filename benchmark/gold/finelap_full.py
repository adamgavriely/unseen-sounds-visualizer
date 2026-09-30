"""Round 32 (docs/prereg_round13_detector_push.md, "Round 32"): FineLAP frame scores for EVERY family the shipped vetoes ask
about (cache 2), the same scores in the DASM npz layout (so LISTENER_DASM_DIR can point at FineLAP), and the P1-calibrated
clip bar for the DV seat.

    # from ~/MscProj_tg on the cluster
    python benchmark/gold/finelap_full.py run     # GPU, ~/venv_flap: data/work/finelap_cache2/<clip>.npz
    python benchmark/gold/finelap_full.py build   # CPU, msproj: data/work/finelap_as_dasm/<clip>.npz + cache-1 equality check
    python benchmark/gold/finelap_full.py calib   # CPU, msproj: span bar (FLAP rule, must reproduce 0.329) and clip bar
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import canonical
from benchmark.gold.finelap_screen import PARTS, CACHE, SR, WIN_S, items, item_score, _load, frame_scores

G = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
CACHE2 = WORK / "finelap_cache2"
AS_DASM = WORK / "finelap_as_dasm"
STAGE4 = {"dev": WORK / "r13" / "stage4.json", "dev2": WORK / "r13dev2" / "stage4.json"}
P4 = {"dev": G / "dev_listener_p4.json", "dev2": G / "dev2_listener_p4.json"}
ARMS_ASKED = ("SHIP2+KV4|proposed", "SHIP3+DV|proposed", "SHIP6|proposed")
OUT = G / "finelap_full.json"


def queries(part):
    """clip -> sorted families: cache-1 queries + stage-4 row families of the asked arms + SHIP6 F8/ONCE-dropped + P4 items"""
    want = {}
    p1, cand = items(part)
    for x in p1 + cand:
        want.setdefault(x["clip"], set()).add(canonical(x["label"]))
    s4 = json.loads(STAGE4[part].read_text(encoding="utf-8"))
    for a in ARMS_ASKED:
        for st, rows in s4["arms"].get(a, {}).items():
            for r in rows:
                want.setdefault(st, set()).add(canonical(r["label"]))
    for k, v in s4.get("r14_dropped", {}).items():
        if k.startswith("SHIP6|proposed|"):
            st = k.split("|", 2)[2]
            for f in ("F8", "ONCE"):
                for row in v.get(f, []):
                    want.setdefault(st, set()).add(canonical(row[0]))
    if P4[part].exists():
        d = json.loads(P4[part].read_text(encoding="utf-8"))
        for x in (d["items"] if isinstance(d, dict) else d):
            want.setdefault(x["clip"], set()).add(canonical(x["family"]))
    return {k: sorted(v) for k, v in want.items()}


def run():
    import torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(dev).eval()
    CACHE2.mkdir(parents=True, exist_ok=True)
    for part, c in PARTS.items():
        q = queries(part)
        for clip, ph in sorted(q.items()):
            wp = c["wav"] / f"{clip}.wav"
            if not wp.exists():
                print(f"[run] {part} {clip}: no wav", flush=True)
                continue
            fs, fe, sc = frame_scores(model, _load(wp), ph, dev)
            np.savez(CACHE2 / f"{clip}.npz", fs=fs, fe=fe, scores=sc, labels=np.array(ph))
            print(f"[run] {part} {clip}: {len(ph)} queries, {len(fs)} frames", flush=True)


def build():
    AS_DASM.mkdir(parents=True, exist_ok=True)
    eq, n = [], 0
    for f in sorted(CACHE2.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        labs = [str(x) for x in z["labels"]]
        times = 0.5 * (np.asarray(z["fs"], float) + np.asarray(z["fe"], float))
        np.savez(AS_DASM / f.name, fw=np.asarray(z["scores"], np.float32), times=times, labels=np.array(labs))
        f1 = CACHE / f.name
        if f1.exists():
            z1 = np.load(f1, allow_pickle=True)
            l1 = [str(x) for x in z1["labels"]]
            same = len(z1["fs"]) == len(z["fs"]) and all(
                np.allclose(z1["scores"][:, l1.index(l)], z["scores"][:, labs.index(l)], atol=1e-4) for l in l1 if l in labs)
            eq.append(same); n += 1
            if not same:
                print(f"[build] {f.stem}: cache-1 scores differ", flush=True)
    print(f"[build] {len(list(AS_DASM.glob('*.npz')))} clips written; cache-1 equality {sum(eq)}/{n}")


def calib():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import dev_listener as L
    span, clip, per = [], [], {}
    for part, c in PARTS.items():
        gold = S.load_gold([c["gold"]])
        p1, _ = items(part)
        sp, cl = [], []
        for x in p1:
            if x["clip"] not in gold or L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"]) != "hit_needed":
                continue
            f = CACHE2 / f"{x['clip']}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            s = item_score(z, x)
            labs = [str(v) for v in z["labels"]]
            q = canonical(x["label"])
            if s is None or q not in labs:
                continue
            sp.append(s); cl.append(float(z["scores"][:, labs.index(q)].max()))
        per[part] = {"n": len(sp)}
        span += sp; clip += cl
    span.sort(); clip.sort()
    bs = span[int(math.floor(0.1 * len(span)))]
    bc = clip[int(math.floor(0.1 * len(clip)))]
    out = {"n": len(span), "per_split": per, "span_bar": bs, "clip_bar": bc,
           "span_kept": int(sum(v >= bs for v in span)), "clip_kept": int(sum(v >= bc for v in clip))}
    print(f"calibration on P1 hit_needed: n {len(span)}; span bar {bs:.4f} (FLAP 0.3292), clip bar {bc:.4f}; "
          f"kept {out['span_kept']} / {out['clip_kept']} of {len(span)}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"run": run, "build": build, "calib": calib}[sys.argv[1]]()

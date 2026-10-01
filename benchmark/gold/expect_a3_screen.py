"""Round 40d EXPECT-A3 (docs/prereg_round13_detector_push.md "Round 40d EXPECT-A3"): the Round 40c (clip, family) pairs that were
not already drawn, placed and confirmed by FineLAP (family frame score reaches FINELAP_VETO 0.329; onset = start of the run >= bar
with the highest maximum), then the shipped gate (40b cached answer re-used iff |onset - cached| <= 0.5 s), 2-s picture.

    python benchmark/gold/expect_a3_screen.py flap     # GPU, ~/venv_flap: FineLAP frame scores -> expect_a3/flap/<clip>.npz
    TG_ARMS=SHIP8 python benchmark/gold/expect_a3_screen.py cands    # CPU msproj -> expect_a3/cands.json (+ reusable gate files)
    TG_ARMS=SHIP8 python benchmark/gold/expect_a3_screen.py gate     # GPU msproj: the unmatched onsets (expect_screen.cmd_gate)
    TG_ARMS=SHIP8 python benchmark/gold/expect_a3_screen.py score    # CPU msproj -> expect_a3_screen.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

SRC40B = _ROOT / "benchmark" / "gold" / "expect_a"             # Round 40b gate cache
SRC40C = _ROOT / "benchmark" / "gold" / "expect_a2" / "cands.json"
DIR = _ROOT / "benchmark" / "gold" / "expect_a3"
OUT = _ROOT / "benchmark" / "gold" / "expect_a3_screen.json"
BAR, GAP, FRAME, REUSE = 0.329, 0.24, 0.16, 0.5
WAV = {"dev": _ROOT / "data" / "work" / "devcand" / "wav16", "dev2": _ROOT / "data" / "work" / "r13dev2" / "wav16"}


def pairs():
    """[(part, clip, family)] = the 40c rows that were not already drawn"""
    C = json.loads(SRC40C.read_text(encoding="utf-8"))
    return [(r["part"], r["clip"], r["family"]) for r in C["rows"] if r["outcome"] in ("candidate", "no run at weak bars")]


# ---------------------------------------------------------------- flap (GPU, venv_flap; no project imports beyond finelap_screen)
def cmd_flap():
    import torch
    from transformers import AutoModel
    from benchmark.gold import finelap_screen as FS
    d = DIR / "flap"; d.mkdir(parents=True, exist_ok=True)
    want = {}
    for pt, st, F in pairs():
        want.setdefault((pt, st), set()).add(F)
    todo = {k: v for k, v in want.items() if not (d / f"{k[1]}.npz").exists()}
    print(f"{len(want)} clips, {sum(len(v) for v in want.values())} (clip, family) pairs, {len(todo)} clips to score", flush=True)
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(dev).eval()
    for (pt, st), fams in sorted(todo.items()):
        ph = sorted(fams)
        fs, fe, sc = FS.frame_scores(model, FS._load(WAV[pt] / f"{st}.wav"), ph, dev)
        np.savez(d / f"{st}.npz", fs=fs, fe=fe, scores=sc, labels=np.array(ph))
        print(f"[flap] {pt} {st}: {ph} -> {len(fs)} frames, max {sc.max(axis=0).round(3).tolist()}", flush=True)


# ---------------------------------------------------------------- cands (CPU)
def grid_scores(z, fam):
    """(times on a 0.16-s grid, score = max over the windows covering each grid cell) or None"""
    labs = [str(v) for v in z["labels"]]
    if fam not in labs:
        return None
    c = labs.index(fam)
    fs, fe, sc = np.asarray(z["fs"], float), np.asarray(z["fe"], float), np.asarray(z["scores"], float)[:, c]
    n = int(np.ceil(fe.max() / FRAME - 1e-9))
    ts = np.arange(n) * FRAME
    g = np.full(n, -np.inf)
    for a, b, s in zip(fs, fe, sc):
        i0, i1 = int(round(a / FRAME)), max(int(round(a / FRAME)) + 1, int(np.ceil(b / FRAME - 1e-9)))
        g[i0:i1] = np.maximum(g[i0:i1], s)
    return ts, g


def cmd_cands():
    from benchmark.gold import expect_screen as E
    from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP
    assert abs(LISTEN_RUN_GAP - GAP) < 1e-9
    (DIR / "gate").mkdir(parents=True, exist_ok=True)
    rows = []; tally = {"pairs": 0, "below_bar": 0, "candidate": 0, "gate_reused": 0, "gate_to_ask": 0}
    for pt, st, F in pairs():
        tally["pairs"] += 1
        z = np.load(DIR / "flap" / f"{st}.npz")
        ts, g = grid_scores(z, F)
        row = {"part": pt, "clip": st, "family": F, "omni_match": True, "flap_max": round(float(g.max()), 3)}
        rr, dt = _runs(g, ts, BAR, GAP)
        if not rr:
            tally["below_bar"] += 1; row["outcome"] = "below FineLAP bar"
        else:
            best = max(rr, key=lambda ij: (float(g[ij[0]:ij[1]].max()), -ij[0]))
            onset = float(ts[best[0]])
            row.update({"outcome": "candidate", "onset": round(onset, 2), "from": "flap",
                        "run": [round(onset, 2), round(float(ts[best[1] - 1]) + dt, 2)], "run_max": round(float(g[best[0]:best[1]].max()), 3),
                        "n_runs": len(rr)})
            tally["candidate"] += 1
            cached = [f for f in (SRC40B / "gate").glob(f"{st}_*.json")]
            hit = None
            for f in cached:
                c = json.loads(f.read_text(encoding="utf-8"))
                if c["family"] == F and abs(float(c["onset"]) - onset) <= REUSE:
                    hit = c; break
            if hit is not None:
                rec = {**hit, **row}                                   # the cached votes, this round's onset / run fields
                rec["gate_reused_from"] = hit["onset"]
                (DIR / "gate" / f"{E.gate_key(row)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
                tally["gate_reused"] += 1; row["gate"] = "reused"
            else:
                tally["gate_to_ask"] += 1; row["gate"] = "ask"
        rows.append(row)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "caches_missing": {}, "rows": rows}, indent=1), encoding="utf-8")
    print(tally)
    for r in rows:
        print(f"   {r['part']:4s} {r['clip']} {r['family']} max {r['flap_max']} -> {r['outcome']} "
              f"{r.get('onset', '')} run {r.get('run', '')} gate {r.get('gate', '')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "flap":
        cmd_flap()
    elif cmd == "cands":
        cmd_cands()
    else:
        from benchmark.gold import expect_screen as E
        E.DIR, E.OUT = DIR, OUT
        {"gate": E.cmd_gate, "score": E.cmd_score}[cmd]()

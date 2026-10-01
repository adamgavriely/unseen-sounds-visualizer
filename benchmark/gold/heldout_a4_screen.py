"""Round 42 HELDOUT-A4 (docs/prereg_round13_detector_push.md "Round 42 HELDOUT-A4"): the EXPECT-A4 audio chain, frozen as in the
Round 40e TEST read but WITHOUT the visibility gate, on the 415 held-out AudioSet-Strong clips (benchmark/gold/audioset_heldout.json);
every kept detection scored against the strong labels (same_family, onset window [-0.5, +1.0]). DASM frame scores = the Round 6
held-out cache (not recomputed). Strong labels are read ONLY in `score`. Nothing in src/ or config.py is edited.

    python benchmark/gold/heldout_a4_screen.py wav       # CPU msproj (ffmpeg): mp4 -> heldout_a4/wav16/<id>.wav
    python benchmark/gold/heldout_a4_screen.py listen    # GPU msproj: Qwen3-Omni list -> heldout_a4/listen/
    python benchmark/gold/heldout_a4_screen.py cands     # CPU: map, first two -> heldout_a4/pairs.json
    python benchmark/gold/heldout_a4_screen.py flap      # GPU venv_flap: FineLAP -> heldout_a4/flap/<id>.npz
    python benchmark/gold/heldout_a4_screen.py dasm      # CPU: FineLAP onset + DASM check -> heldout_a4/cands.json
    python benchmark/gold/heldout_a4_screen.py score     # CPU: strong labels read here only -> heldout_a4_screen.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
MP4 = _ROOT / "data" / "input" / "audioset_heldout"
DIR = _ROOT / "benchmark" / "gold" / "heldout_a4"
OUT = _ROOT / "benchmark" / "gold" / "heldout_a4_screen.json"
WAV = DIR / "wav16"
HOME = Path.home()
FLEX_HELD = Path(os.environ.get("FLEX_HELD", str(_ROOT / "data" / "work" / "flexsed_heldout")))
WINDOWS = Path(os.environ.get("HELDOUT_WINDOWS", str(HOME / "MscProj" / "benchmark" / "audioset_heldout_windows")))
BEATS_HELD, DASM_HELD = WINDOWS / "beats", WINDOWS / "dasm_cache"
FLAP_BAR, DASM_BAR, HALF, GAP = 0.329, 0.575, 0.5, 0.24
EARLY, LATE = 0.5, 1.0
MIN_DET = 3


def ids():
    """clip ids only (events are never touched here)"""
    H = json.loads(HELDOUT.read_text(encoding="utf-8"))
    out = [c["id"] for c in H["clips"]]
    assert len(out) == 415, len(out)
    return out


def check_caches(cl):
    miss = {"mp4": [c for c in cl if not (MP4 / f"{c}.mp4").exists()],
            "flexsed": [c for c in cl if not (FLEX_HELD / f"{c}.npz").exists()],
            "beats": [c for c in cl if not (BEATS_HELD / f"{c}.npz").exists()],
            "dasm": [c for c in cl if not (DASM_HELD / f"{c}.npz").exists()]}
    assert not any(miss.values()), {k: v[:3] for k, v in miss.items() if v}
    n_dasm = len(list(DASM_HELD.glob("*.npz")))
    print(f"{len(cl)} clips; every clip has mp4 + FlexSED + BEATs + DASM cache; DASM folder holds {n_dasm} files "
          f"(stray {sorted(p.stem for p in DASM_HELD.glob('*.npz') if p.stem not in set(cl))})", flush=True)


# ---------------------------------------------------------------- wav (CPU, ffmpeg of the msproj env)
def cmd_wav():
    cl = ids(); check_caches(cl)
    WAV.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in cl:
        w = WAV / f"{c}.wav"
        if w.exists():
            continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(MP4 / f"{c}.mp4"), "-ac", "1", "-ar", "16000", str(w)], check=True)
        n += 1
    print(f"wav16: {n} written, {sum((WAV / f'{c}.wav').exists() for c in cl)} present", flush=True)


# ---------------------------------------------------------------- listen (GPU msproj)
def cmd_listen():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import expect_a_screen as A
    d = DIR / "listen"; d.mkdir(parents=True, exist_ok=True)
    todo = [c for c in ids() if not (d / f"{c}.json").exists()]
    print(f"{len(todo)} clips to listen", flush=True)
    if not todo:
        return
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    for c in todo:
        w, sr = sf.read(str(WAV / f"{c}.wav"), dtype="float32")
        assert sr == LV.SR, (c, sr)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": A.LIST_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=A.MAX_NEW, do_sample=False, repetition_penalty=A.REP_PEN)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec = {"clip": c, "seconds": round(len(w) / sr, 2), "prompt": A.LIST_Q, "max_new": A.MAX_NEW,
               "repetition_penalty": A.REP_PEN, "text": txt, "items": A.items_of(txt)}
        (d / f"{c}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{c} -> {txt[:100]!r}", flush=True)


# ---------------------------------------------------------------- cands (CPU): map, first two (no SHIP8 pictures here: no "already drawn")
def cmd_cands():
    from benchmark.gold import expect_a_screen as A
    rows = []; tally = {"clips": 0, "items": 0, "named": 0, "kept2": 0, "clips_with_pair": 0}
    for c in ids():
        L = json.loads((DIR / "listen" / f"{c}.json").read_text(encoding="utf-8"))
        items = A.items_of(L["text"]); tally["items"] += len(items); tally["clips"] += 1
        fams, how = [], {}
        for it in items:
            f = A.map_item(it)
            if f and f not in fams:
                fams.append(f); how[f] = f"map:{it}"
        tally["named"] += len(fams)
        fams = fams[:2]; tally["kept2"] += len(fams)
        tally["clips_with_pair"] += bool(fams)
        for F in fams:
            rows.append({"clip": c, "family": F, "how": how[F], "outcome": "pair"})
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "pairs.json").write_text(json.dumps({"tally": tally, "rows": rows}, indent=1), encoding="utf-8")
    print(tally)


# ---------------------------------------------------------------- flap (GPU venv_flap)
def cmd_flap():
    import torch
    from transformers import AutoModel
    from benchmark.gold import finelap_screen as FS
    d = DIR / "flap"; d.mkdir(parents=True, exist_ok=True)
    rows = json.loads((DIR / "pairs.json").read_text(encoding="utf-8"))["rows"]
    want = defaultdict(set)
    for r in rows:
        want[r["clip"]].add(r["family"])
    todo = {k: v for k, v in want.items() if not (d / f"{k}.npz").exists()}
    print(f"{len(want)} clips, {len(rows)} pairs, {len(todo)} clips to score", flush=True)
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(dev).eval()
    for c, fams in sorted(todo.items()):
        ph = sorted(fams)
        fs, fe, sc = FS.frame_scores(model, FS._load(WAV / f"{c}.wav"), ph, dev)
        np.savez(d / f"{c}.npz", fs=fs, fe=fe, scores=sc, labels=np.array(ph))
        print(f"[flap] {c}: {ph} -> max {sc.max(axis=0).round(3).tolist()}", flush=True)


# ---------------------------------------------------------------- dasm (CPU): FineLAP onset, DASM confirmation (Round 6 cache)
def cmd_dasm():
    from benchmark.gold import expect_a3_screen as A3
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP
    assert abs(LISTEN_RUN_GAP - GAP) < 1e-9 and A3.BAR == FLAP_BAR and A4.BAR == DASM_BAR
    rows = json.loads((DIR / "pairs.json").read_text(encoding="utf-8"))["rows"]
    out = []; tally = {"pairs": len(rows), "below_flap": 0, "no_dasm_column": 0, "below_dasm": 0, "kept": 0}
    frs = {}
    for r in rows:
        c, F = r["clip"], r["family"]
        z = np.load(DIR / "flap" / f"{c}.npz")
        ts, g = A3.grid_scores(z, F)
        r["flap_max"] = round(float(g.max()), 3)
        rr, dt = _runs(g, ts, FLAP_BAR, GAP)
        if not rr:
            tally["below_flap"] += 1; r["outcome"] = "below FineLAP bar"; out.append(r); continue
        best = max(rr, key=lambda ij: (float(g[ij[0]:ij[1]].max()), -ij[0]))
        onset = float(ts[best[0]])
        r.update({"onset": round(onset, 2), "run": [round(onset, 2), round(float(ts[best[1] - 1]) + dt, 2)]})
        if c not in frs:
            frs[c] = DCC.load_fr(DASM_HELD / f"{c}.npz")
        v = A4.dasm_max(frs[c], F, onset - HALF, onset + HALF)
        r["dasm_onset"] = None if v is None else round(v, 3)
        if v is None:
            tally["no_dasm_column"] += 1; r["outcome"] = "no DASM column"
        elif v < DASM_BAR:
            tally["below_dasm"] += 1; r["outcome"] = "below DASM bar"
        else:
            tally["kept"] += 1; r["outcome"] = "kept"
        out.append(r)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "rows": out}, indent=1), encoding="utf-8")
    print(tally)


# ---------------------------------------------------------------- score (CPU): strong labels read here only
def classify(fam, onset, events, same_family):
    """'correct' | 'wrong time' | 'family absent' against the clip's strong events"""
    sf = [ev for ev in events if same_family(fam, ev["label"])]
    if any(ev["start"] - EARLY <= onset <= ev["start"] + LATE for ev in sf):
        return "correct"
    return "wrong time" if sf else "family absent"


def per_family(dets):
    """{family: {n, correct, wrong_time, absent, precision}} over detections [(family, class)]"""
    pf = defaultdict(Counter)
    for fam, k in dets:
        pf[fam][k] += 1; pf[fam]["n"] += 1
    out = {}
    for fam, c in sorted(pf.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        out[fam] = {"n": c["n"], "correct": c["correct"], "wrong_time": c["wrong time"], "absent": c["family absent"],
                    "precision": round(c["correct"] / c["n"], 3)}
    return out


def summ(dets):
    n = len(dets); c = Counter(k for _f, k in dets)
    return {"n": n, "correct": c["correct"], "wrong_time": c["wrong time"], "absent": c["family absent"],
            "precision": round(c["correct"] / n, 3) if n else None}


def cmd_score():
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import prior415_screen as P
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    H = json.loads(HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in H["clips"]}
    C = json.loads((DIR / "cands.json").read_text(encoding="utf-8"))
    kept, placed, rows = [], [], []
    for r in C["rows"]:
        if "onset" not in r:
            continue
        k = classify(r["family"], r["onset"], ev_of[r["clip"]], same_family)
        r["class"] = k
        placed.append((r["family"], k))
        if r["outcome"] == "kept":
            kept.append((r["family"], k))
        rows.append(r)
    K, Pl = summ(kept), summ(placed)
    pfk = per_family(kept)
    judged = {f: x for f, x in pfk.items() if x["n"] >= MIN_DET}
    thin = {f: x for f, x in pfk.items() if x["n"] < MIN_DET}
    print(f"HELDOUT-A4 kept (after DASM, no gate): {K}")
    print(f"  FineLAP-placed before DASM (secondary): {Pl}")
    for f, x in judged.items():
        print(f"   {f:30s} n {x['n']:3d} correct {x['correct']:3d} wrong-time {x['wrong_time']:3d} absent {x['absent']:3d} p {x['precision']}")
    print(f"  thin (< {MIN_DET}): " + ", ".join(f"{f} {x['correct']}/{x['n']}" for f, x in thin.items()))
    # the raw detectors, like-for-like (this round's rule, depictable families only)
    P.shipped_flags()
    dep = set(E.FAMILIES)
    det = {}
    for model, fn in P.RUNS.items():
        d = P.DIRS[model] if model == "flexsed" else BEATS_HELD
        ds = []
        for c in ids():
            fw, t, labs = DCC.load_fr(d / f"{c}.npz")
            for lab, a, b in fn(fw, t, labs):
                f = canonical(lab)
                if f in dep:
                    ds.append((f, classify(f, a, ev_of[c], same_family)))
        pf = per_family(ds)
        det[model] = {"summary": summ(ds), "per_family": {f: x for f, x in pf.items() if x["n"] >= MIN_DET}}
        print(f"  raw {model} (depictable families, same rule): {det[model]['summary']}")
    L = json.loads(P.LISTS.read_text(encoding="utf-8"))
    strict = {}
    for model, m in L["models"].items():
        runs = sum(x["runs"] for x in m["families"].values()); cor = sum(x["correct"] for x in m["families"].values())
        strict[model] = {"runs": runs, "correct": cor, "precision": round(cor / runs, 3) if runs else None}
    print(f"  PRIOR415 strict (canonical equality, families >= 10 instances): {strict}")
    res = {"round": "Round 42 HELDOUT-A4", "clips": len(ids()), "rule": "correct iff a same_family strong event starts in [onset-0.5, onset+1.0]",
           "funnel": {**json.loads((DIR / "pairs.json").read_text(encoding="utf-8"))["tally"], **C["tally"]},
           "kept": K, "placed_before_dasm": Pl, "per_family_kept": judged, "thin_kept": thin, "per_family_placed": per_family(placed),
           "raw_detectors_same_rule": det, "prior415_strict": strict, "detections": rows}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"wav": cmd_wav, "listen": cmd_listen, "cands": cmd_cands, "flap": cmd_flap, "dasm": cmd_dasm, "score": cmd_score}[sys.argv[1]]()

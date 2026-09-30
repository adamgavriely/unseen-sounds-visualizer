"""Round 31 FLAP screen (docs/prereg_round13_detector_push.md, "Round 31 FLAP"): FineLAP (ACL 2026) frame-level text-audio
scores as a vote on the listener-rescue candidates (P2 / PV) of merged DEV. The bar is calibrated only on P1 hit_needed
items (FineLAP >= bar keeps 90 %). (a) new accept path: TIER false AND FineLAP >= bar AND (Qwen V4 OR AF V4);
GO iff needed added >= 2 and other added <= needed added. (b) veto: TIER true AND FineLAP < bar; GO iff other removed
>= 3 x needed lost (and >= 1).

    # from ~/MscProj_tg on the cluster
    python benchmark/gold/finelap_screen.py run     # GPU, ~/venv_flap (transformers 4.51.3), HF offline
    python benchmark/gold/finelap_screen.py score   # CPU, msproj env
    python benchmark/gold/finelap_screen.py run test test2   # TEST frame caches -> data/work/finelap_test{,2}/
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

G = _ROOT / "benchmark" / "gold"
H = Path.home()
R13 = H / "MscProj_r13"
PARTS = {"dev": {"p1": R13 / "benchmark" / "gold" / "dev_listener.json", "v": R13 / "benchmark" / "gold" / "dev_listener_v.json",
                 "af": R13 / "benchmark" / "gold" / "dev_listener_afn.json", "gold": G / "annotations" / "gold_AG.json",
                 "wav": R13 / "data" / "work" / "devcand" / "wav16"},
         "dev2": {"p1": G / "dev2_listener.json", "v": G / "dev2_listener_v.json", "af": G / "dev2_listener_afn.json",
                  "gold": G / "annotations" / "tagger_AG.json", "wav": _ROOT / "data" / "work" / "r13dev2" / "wav16"}}
# TEST caches for the SHIP6+FLAP TEST read (no TEST gold is read): same windows, same families-per-clip rule
TEST_PARTS = {"test": {"p1": G / "test_listener.json", "v": G / "test_listener_v.json",
                       "wav": _ROOT / "data" / "work" / "r13test" / "wav16", "out": _ROOT / "data" / "work" / "finelap_test"},
              "test2": {"p1": G / "test2_listener.json", "v": G / "test2_listener_v.json",
                        "wav": _ROOT / "data" / "work" / "r13test2" / "wav16", "out": _ROOT / "data" / "work" / "finelap_test2"}}
CACHE = _ROOT / "data" / "work" / "finelap_cache"
OUT = G / "finelap_screen.json"
SR, WIN_FR, HOP_S, FRAME_S = 16000, 1024, 5.12, 0.16
WIN_S = WIN_FR * 0.01


def key(x):
    return (x["clip"], x["pool"], x["label"], round(x["start"], 2), round(x["end"], 2))


def items(part):
    c = PARTS[part] if part in PARTS else TEST_PARTS[part]
    p1 = [x for x in json.loads(c["p1"].read_text(encoding="utf-8"))["items"] if x.get("pool") == "P1"]
    cand = [x for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
    return p1, cand


# ---------------------------------------------------------------- run (GPU)
def _mel(wav):
    """FineLAP's load_audio fbank block, on an in-memory 16 kHz mono slice"""
    import torch
    import torch.nn.functional as F
    import torchaudio
    wav = wav - wav.mean()
    mel = torchaudio.compliance.kaldi.fbank(wav.unsqueeze(0), htk_compat=True, sample_frequency=16000, use_energy=False,
                                            window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    mel = F.pad(mel, (0, 0, 0, WIN_FR - mel.shape[0])) if mel.shape[0] < WIN_FR else mel[:WIN_FR]
    return (mel - (-4.268)) / (4.569 * 2)


def _load(path):
    import soundfile as sf
    import torch
    import torchaudio
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(w.mean(axis=1))
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w


def frame_scores(model, wav, phrases, device):
    """-> fs, fe [F], scores [F, Q]: every frame of every 10.24-s window (hop 5.12 s, last aligned to the end), padding dropped"""
    import torch
    dur = wav.shape[0] / SR
    offs = [0.0] if dur <= WIN_S else sorted({*(k * HOP_S for k in range(int(math.floor((dur - WIN_S) / HOP_S)) + 1)),
                                                 dur - WIN_S})
    mels, lens = [], []
    for o in offs:
        a = int(round(o * SR))
        sl = wav[a:a + int(WIN_S * SR)]
        mels.append(_mel(sl)); lens.append(sl.shape[0] / SR)
    batch = torch.stack(mels).unsqueeze(1).to(device)
    model.load_audio = lambda x, device=None: x          # feed the mel batch straight in
    s = model.get_frame_level_score(batch, phrases, device=device)   # (B, Q, T)
    s = s.float().cpu().numpy()
    fs, fe, sc = [], [], []
    for b, (o, ln) in enumerate(zip(offs, lens)):
        for i in range(s.shape[2]):
            if i * FRAME_S >= ln:
                break
            fs.append(o + i * FRAME_S); fe.append(o + min((i + 1) * FRAME_S, ln)); sc.append(s[b, :, i])
    return np.array(fs), np.array(fe), np.stack(sc)


def run(splits=None):
    import torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(dev).eval()
    # self-check: the in-memory mel equals FineLAP's own load_audio on a real file
    smoke = H / "FineLAP" / "resources" / "1.wav"
    if smoke.exists():
        ref = type(model).load_audio(model, [str(smoke)], device=dev)[0, 0].cpu()
        mine = _mel(_load(smoke)[: int(WIN_S * SR)])
        print(f"[check] mel max abs diff vs load_audio: {float((ref - mine).abs().max()):.2e}", flush=True)
    todo = {k: {**v, "out": CACHE} for k, v in PARTS.items()} if not splits else {k: TEST_PARTS[k] for k in splits}
    for part, c in todo.items():
        c["out"].mkdir(parents=True, exist_ok=True)
        p1, cand = items(part)
        want = {}
        for x in p1 + cand:
            want.setdefault(x["clip"], set()).add(canonical(x["label"]))
        for clip, ph in sorted(want.items()):
            ph = sorted(ph)
            wp = c["wav"] / f"{clip}.wav"
            if not wp.exists():
                print(f"[run] {part} {clip}: no wav", flush=True)
                continue
            fs, fe, sc = frame_scores(model, _load(wp), ph, dev)
            np.savez(c["out"] / f"{clip}.npz", fs=fs, fe=fe, scores=sc, labels=np.array(ph))
            print(f"[run] {part} {clip}: {len(ph)} queries, {len(fs)} frames", flush=True)


# ---------------------------------------------------------------- score (CPU)
def item_score(z, x):
    labs = [str(v) for v in z["labels"]]
    q = canonical(x["label"])
    if q not in labs:
        return None
    fs, fe, col = z["fs"], z["fe"], z["scores"][:, labs.index(q)]
    m = (fe > x["start"]) & (fs < x["end"])
    if not m.any():
        mid = 0.5 * (x["start"] + x["end"])
        m = (fs <= mid) & (fe > mid)
    return float(col[m].max()) if m.any() else None


def score():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import dev_listener as L
    from src.stage4_audio_event_detection import _tier
    zs = {}

    def fl(x):
        if x["clip"] not in zs:
            f = CACHE / f"{x['clip']}.npz"
            zs[x["clip"]] = np.load(f, allow_pickle=True) if f.exists() else None
        return None if zs[x["clip"]] is None else item_score(zs[x["clip"]], x)

    # calibration on P1 hit_needed
    cal, per = [], {}
    rows = {}
    for part, c in PARTS.items():
        gold = S.load_gold([c["gold"]])
        p1, cand = items(part)
        sc = [fl(x) for x in p1 if x["clip"] in gold
              and L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"]) == "hit_needed"]
        sc = sorted(s for s in sc if s is not None)
        per[part] = {"n": len(sc), "bar": sc[int(math.floor(0.1 * len(sc)))] if sc else None}
        cal += sc
        rows[part] = (gold, cand)
    cal.sort()
    bar = cal[int(math.floor(0.1 * len(cal)))]
    print(f"calibration: n {len(cal)} bar {bar:.4f} per split {per}")
    a_n, a_o, b_n, b_o, lines_a, lines_b = 0, 0, 0, 0, [], []
    n_cand, n_miss, stats = 0, 0, {"needed": [], "other": []}
    for part, (gold, cand) in rows.items():
        af = {key(x): x for x in json.loads(PARTS[part]["af"].read_text(encoding="utf-8"))["items"]}
        for x in cand:
            if x["clip"] not in gold:
                continue
            s = fl(x)
            if s is None:
                n_miss += 1
                continue
            n_cand += 1
            peak = float(x.get("peak") or 0.0)
            q = bool(x["accept"].get("V4")); a = bool((af.get(key(x), {}).get("accept") or {}).get("V4"))
            t = _tier({"V4": q, "AF_V4": a}, peak)
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            needed = cls == "hit_needed"
            stats["needed" if needed else "other"].append(s)
            ln = (f"[{part}] {x['clip']} {x['pool']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {peak:.2f} "
                  f"Q {int(q)} AF {int(a)} FL {s:.3f} [{cls}]")
            if not t and s >= bar and (q or a):
                lines_a.append(ln); a_n += needed; a_o += not needed
            if t and s < bar:
                lines_b.append(ln); b_n += needed; b_o += not needed
    go_a = a_n >= 2 and a_o <= a_n
    go_b = b_o >= 1 and b_o >= 3 * b_n
    print(f"candidates scored {n_cand} (no score {n_miss}); FL >= bar: needed "
          f"{sum(v >= bar for v in stats['needed'])}/{len(stats['needed'])}, other "
          f"{sum(v >= bar for v in stats['other'])}/{len(stats['other'])}")
    print(f"(a) accept path: needed added {a_n}, other added {a_o} -> {'GO' if go_a else 'STOP'}")
    for s in lines_a:
        print("   ", s)
    print(f"(b) veto: needed lost {b_n}, other removed {b_o} -> {'GO' if go_b else 'STOP'}")
    for s in lines_b:
        print("   ", s)
    OUT.write_text(json.dumps({"bar": bar, "cal_n": len(cal), "cal_per_split": per, "candidates": n_cand,
                               "no_score": n_miss,
                               "cand_above_bar": {k: [int(sum(v >= bar for v in vs)), len(vs)] for k, vs in stats.items()},
                               "a": {"needed_added": a_n, "other_added": a_o, "go": go_a, "items": lines_a},
                               "b": {"needed_lost": b_n, "other_removed": b_o, "go": go_b, "items": lines_b}},
                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:] or None)
    else:
        score()

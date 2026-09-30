"""Round 33 SSL-SaN (docs/prereg_round13_detector_push.md): audio-visual sound-source localisation as a third vote for
the stage-5 visibility gate. SSL-SaN (BMVC 2025, xavijuanola/SSL_SaN, official weights) is trained with silence / noise /
off-screen negatives, so its audio-image cosine map should sit near zero when the sound's source is not on screen.

For every cached Qwen3.8-27B gate stretch (benchmark/gold/gate_gold/Qwen38-27B, all 139 gold clips) the SAME six frames
(stretch +-1 s, as gate_gold.run_vlm) and the stretch audio (+-1 s, 16 kHz mono, repeated to 10 s as the official
dataloader does) go through the official encoders; the stretch's score = max over the 6 frames of the max of the raw
14x14 cosine map. Writes gate_gold/sslsan/<stem>.json (the cached votes + sslsan_peak per stretch).

    python benchmark/gold/gate_sslsan.py run [--device cuda] [--limit N]
    python benchmark/gold/gate_sslsan.py score

Rule (fixed before any number): the sound-level score is the MIN over its stretches (the gate silences only when every
stretch is seen); the threshold t is the value with the best balanced accuracy on the NON-judge gold clips (test85 part,
highest t on ties), then on the DEV judge clips (a) tie-break: a stretch with yes == no among name / a-b / description is
decided by sslsan_peak >= t; (b) fourth vote: yes > no over the four votes, a 2-2 tie decided by SSL-SaN (so SSL-SaN
decides every non-unanimous VLM stretch, unanimous ones stand).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import gate_gold as G
from benchmark.gold import detector_dry as DD

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "sslsan"
SUMMARY = G.OUT_DIR / "sslsan_summary.json"
REPO = _ROOT / "third_party" / "SSL_SaN" / "evaluate"
CKPT = REPO / "checkpoints" / "sslsan.pth.tar"
SR = 16000
AUDIO_LEN = 10  # s; the official dataloader repeats shorter audio to 10 s


def load_model(device: str):
    import types
    import torch
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))       # model_ssltie imports it at module level
    sys.path.append(str(REPO))                                       # after the project: 'models' / 'networks' names
    from models.model_ssltie import AVENet_ssltie
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # the fields AVENet_ssltie / base_models read, with the values of utils_dir/opts_ssltie.SSLTIE_args (no easydict)
    args = types.SimpleNamespace(heatmap_size=14, epsilon=0.65, epsilon2=0.4, tri_map=True, Neg=True, out_channels=512,
                                 pth_name=str(CKPT) + ("_learnable" if "epsilon" in sd else ""))
    model = AVENet_ssltie(args)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("checkpoint keys", len(sd), "missing", missing, "unexpected", unexpected, flush=True)
    assert not [k for k in missing if k.startswith(("imgnet", "audnet"))], "encoder weights missing"
    return model.to(device).eval()


def _img_tf():
    from torchvision import transforms
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(224),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def spectrogram(wave):
    """the official test-time audio path: repeat to 10 s, clip, mel (512 / 239 / 257, normalized), dB"""
    import numpy as np
    import torch
    import torchaudio.transforms as T
    x = torch.from_numpy(np.asarray(wave, dtype="float32")).unsqueeze(0)
    if x.shape[1] < SR * AUDIO_LEN:
        x = x.repeat(1, int(SR * AUDIO_LEN / max(1, x.shape[1])) + 1)
    x = x[:, :SR * AUDIO_LEN].clamp(-1.0, 1.0)
    spec = T.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=239, n_mels=257, normalized=True)(x)
    return T.AmplitudeToDB()(spec).unsqueeze(0)        # (1, 1, 257, T)


def cosine_maps(model, frames, spec, device):
    """raw 14x14 audio-image cosine map per frame (the model's A before any sigmoid / normalisation)"""
    import torch
    import torch.nn.functional as F
    tf = _img_tf()
    with torch.no_grad():
        img = torch.stack([tf(f) for f in frames]).to(device)
        img = F.normalize(model.imgnet(img), dim=1)                        # (B, 512, 14, 14)
        aud = model.audnet(spec.to(device))
        aud = F.normalize(model.avgpool(aud).view(1, -1), dim=1)[0]        # (512,)  avgpool is AdaptiveMaxPool2d
        return torch.einsum("nchw,c->nhw", img, aud).cpu()                 # (B, 14, 14)


def run(device: str = "cuda", limit: int = 0, stems=None):
    import numpy as np
    import soundfile as sf
    from src.stage2_video_understanding import _sample_frames_at
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    zeros = spectrogram(np.zeros(SR * AUDIO_LEN, dtype="float32"))
    done = 0
    for f in sorted(SRC.glob("*.json")):
        if stems and f.stem not in stems:
            continue
        if (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = DD.clip_path(d["clip"])
        if p is None:
            print("missing", d["clip"]); continue
        w, sr = sf.read(str(DD.wav_for(p)), dtype="float32")
        assert sr == SR, sr
        if w.ndim > 1:
            w = w.mean(1)
        for s in d["sounds"]:
            for st in s["stretches"]:
                a, b = float(st["start"]), float(st["end"])
                n = 6; lo, hi = a - 1.0, b + 1.0
                times = [max(0.0, lo + (hi - lo) * t / (n - 1)) for t in range(n)]
                frames = _sample_frames_at(p, times)
                if not frames:
                    st["sslsan_peak"] = None; continue
                seg = w[int(max(0.0, a - 1.0) * SR):int((b + 1.0) * SR)]
                if len(seg) < SR // 2:
                    st["sslsan_peak"] = None; continue
                A = cosine_maps(model, frames, spectrogram(seg), device)
                A0 = cosine_maps(model, frames, zeros, device)
                st["sslsan_frame_peaks"] = [round(float(x), 4) for x in A.flatten(1).max(1).values]
                st["sslsan_peak"] = round(float(A.max()), 4)
                st["sslsan_mean"] = round(float(A.mean()), 4)
                st["sslsan_silence_peak"] = round(float(A0.max()), 4)      # same frames, zero audio (report only)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f.stem, [(s["label"], [st.get("sslsan_peak") for st in s["stretches"]]) for s in d["sounds"]], flush=True)
        done += 1
        if limit and done >= limit:
            break


def _votes(st):
    v = [st.get("name"), st.get("ab"), st.get("desc")]
    return sum(1 for x in v if x is True), sum(1 for x in v if x is False)


def decide(stretches, rule: str, t: float) -> bool:
    """clip verdict: silent only if every stretch is seen. rules: majority (base), sslsan (alone), tiebreak (a), vote4 (b)"""
    for st in stretches:
        yes, no = _votes(st)
        pk = st.get("sslsan_peak")
        ss = (pk is not None) and pk >= t
        if rule == "majority":
            seen = yes > no
        elif rule == "sslsan":
            seen = ss
        elif rule == "tiebreak":
            seen = ss if yes == no else yes > no
        elif rule == "vote4":
            yes4, no4 = yes + int(ss), no + int(not ss)
            seen = ss if yes4 == no4 else yes4 > no4
        else:
            raise ValueError(rule)
        if not seen:
            return False
    return True


def _rates(rows, rule, t):
    seen_sil = n_seen = kept = n_needed = 0
    for s in rows:
        pred = decide(s["stretches"], rule, t)
        if s["seen"]:
            n_seen += 1; seen_sil += int(pred)
        else:
            n_needed += 1; kept += int(not pred)
    return seen_sil, n_seen, kept, n_needed


def score():
    judge = set(G.JUDGE100.read_text().split())
    dev, cal = [], []
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            if s["importance"] < 2:
                continue
            s["_stem"] = f.stem
            (dev if f.stem in judge else cal).append(s)
    # calibration on the non-judge clips: t with the best balanced accuracy (highest t on ties)
    scores = sorted({min((st.get("sslsan_peak") if st.get("sslsan_peak") is not None else -9.0) for st in s["stretches"])
                     for s in cal})
    best = None
    for t in scores + [scores[-1] + 1e-6]:
        a, na, b, nb = _rates(cal, "sslsan", t)
        bal = (a / na + b / nb) / 2
        if best is None or bal >= best[0]:
            best = (bal, t, a, na, b, nb)
    bal, t, a, na, b, nb = best
    print(f"calibration (non-judge, {len(cal)} sounds): t = {t:.4f}  seen silenced {a}/{na}  needed kept {b}/{nb}  balanced {bal:.3f}")
    out = {"t": t, "calibration": {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb, "balanced_acc": bal, "n": len(cal)},
           "dev": {}, "flips": {}}
    base = {id(s): decide(s["stretches"], "majority", t) for s in dev}
    for rule in ("majority", "sslsan", "tiebreak", "vote4"):
        a, na, b, nb = _rates(dev, rule, t)
        out["dev"][rule] = {"seen_silenced": a, "seen": na, "needed_kept": b, "needed": nb}
        flips = []
        for s in dev:
            pred = decide(s["stretches"], rule, t)
            if pred != base[id(s)]:
                flips.append({"clip": s["_stem"], "label": s["label"], "start": s["start"], "gold_seen": s["seen"],
                              "base_seen": base[id(s)], "new_seen": pred,
                              "peaks": [st.get("sslsan_peak") for st in s["stretches"]],
                              "votes": [_votes(st) for st in s["stretches"]]})
        out["flips"][rule] = flips
        print(f"DEV judge  {rule:9s} seen silenced {a}/{na}  needed kept {b}/{nb}  flips {len(flips)}")
        for x in flips:
            print(f"   {'good' if x['new_seen'] == x['gold_seen'] else 'BAD '} {x['clip']} {x['label']} {x['start']:.1f}s gold_seen={x['gold_seen']} "
                  f"{x['base_seen']}->{x['new_seen']} peaks={x['peaks']} votes={x['votes']}")
    go = any(out["dev"][r]["seen_silenced"] >= 19 and out["dev"][r]["needed_kept"] >= 31 for r in ("tiebreak", "vote4"))
    out["GO"] = go
    # report only: the silence reference on the same frames
    import statistics
    pk = [st["sslsan_peak"] for s in dev + cal for st in s["stretches"] if st.get("sslsan_peak") is not None]
    sp = [st["sslsan_silence_peak"] for s in dev + cal for st in s["stretches"] if st.get("sslsan_silence_peak") is not None]
    if pk and sp:
        out["peak_median"] = statistics.median(pk); out["silence_peak_median"] = statistics.median(sp)
        print(f"median stretch peak {out['peak_median']:.3f} vs the same frames with zero audio {out['silence_peak_median']:.3f}")
    print("GO" if go else "STOP")
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stems", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.device, a.limit, set(a.stems) if a.stems else None)
    else:
        score()


if __name__ == "__main__":
    main()

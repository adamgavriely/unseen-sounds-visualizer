"""Round 15 amendment N (docs/prereg_round13_detector_push.md): Qwen3-Omni audio-visual gate vote, DEV only.
For every (gold sound, stretch) of the cached Qwen3.8-27B gate run (benchmark/gold/gate_gold/Qwen38-27B), Qwen3-Omni gets
the same 6 frames and the stretch audio and answers whether the heard {label} sound is made by something visible.
Vote = logit(yes) - logit(no) > 0. Scored with som_gate.score's counts under rule N.

    python benchmark/gold/omni_gate.py run      # GPU
    python benchmark/gold/omni_gate.py score    # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gate_gold as G

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "omni_Qwen3-Omni"
WAV = _ROOT / "data" / "work" / "devcand" / "wav16"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000
Q = ("These are frames from a video, and this is its sound. Is the {label} sound you hear made by something you can see "
     "in these frames? Answer yes or no.")


def run():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from src.stage2_video_understanding import _sample_frames_at
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    yes_ids = sorted({first(w) for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({first(w) for w in ("no", "No", " no", " No")})
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        wp = WAV / f"{f.stem}.wav"
        if p is None or not wp.exists():
            print("missing", f.stem, p, wp.exists()); continue
        w, sr = sf.read(str(wp), dtype="float32")
        assert sr == SR and w.ndim == 1
        for s in d["sounds"]:
            for st in s["stretches"]:
                lo, hi = st["start"] - 1.0, st["end"] + 1.0
                times = [max(0.0, lo + (hi - lo) * t / 5) for t in range(6)]
                imgs = _sample_frames_at(p, times)
                a = w[int(max(0.0, lo) * SR):int(hi * SR)]
                content = [{"type": "image", "image": im} for im in imgs]
                content += [{"type": "audio", "audio": a}, {"type": "text", "text": Q.format(label=s["label"].lower())}]
                text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                tokenize=False)
                inp = proc(text=text, images=imgs, audio=[a], return_tensors="pt", padding=True, use_audio_in_video=False)
                inp = inp.to(model.thinker.device).to(torch.bfloat16)
                with torch.inference_mode():
                    lg = model.thinker(**inp).logits[0, -1].float()
                st["omni_yn"] = float(lg[yes_ids].max() - lg[no_ids].max())
                st["omni"] = st["omni_yn"] > 0
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f.stem, sum(st.get("omni") is True for s in d["sounds"] for st in s["stretches"]), "omni yes", flush=True)


def seen_stretch(st, rule):
    votes = [st.get("name"), st.get("ab"), st.get("desc")]
    yes = sum(v is True for v in votes); no = sum(v is False for v in votes)
    maj = yes > no
    if rule == "majority":
        return maj
    if rule == "N":
        return maj or (st.get("omni") is True and yes >= 1)
    if rule == "N-any":
        return maj or st.get("omni") is True
    raise ValueError(rule)


def score():
    files = sorted(OUT.glob("*.json"))
    res = {}
    for rule in ("majority", "N", "N-any"):
        c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0}
        flips = []
        for f in files:
            for s in json.loads(f.read_text(encoding="utf-8"))["sounds"]:
                if s["importance"] < 2:
                    continue
                pred = all(seen_stretch(st, rule) for st in s["stretches"])
                base = all(seen_stretch(st, "majority") for st in s["stretches"])
                if s["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if pred != base:
                    flips.append([f.stem, s["label"], s["start"], "seen" if s["seen"] else "NEEDED", "silenced" if pred else "kept"])
        res[rule] = {**c, "flips": flips}
        print(f"[{rule:8s}] clips {len(files)} | seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in flips:
            print("    ", x)
    m, b = res["N"], res["majority"]
    res["go"] = (m["seen_sil"] - b["seen_sil"] >= 3) and (b["needed_kept"] - m["needed_kept"] <= 1)
    print("amendment N screen:", "GO (full DEV arm)" if res["go"] else "STOP")
    (G.OUT_DIR / "omni_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "score"))
    a = ap.parse_args()
    run() if a.step == "run" else score()

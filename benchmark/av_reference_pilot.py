"""Pilot of the "with sound / without sound" evaluation reference (docs/prereg_av_reference.md).

One audio-visual model looks at a clip twice on the same frames: with the soundtrack and
muted. Both fill the same fixed list of events, each marked seen / heard / both. The
reference is the heard items of the with-sound run minus anything the muted run already
listed; speech and music items are dropped; an empty list is "nothing beyond the picture".

Pre-registered before this file existed (commit e63a192). Measured on the development
split only, against Adam's labels: balanced accuracy of the "is anything missing?"
decision (bar: model-derived reference 57.7% + 10 points = 67.7%) and the placebo rate
(clips where the MUTED run still claims to hear something; bar < 20%). Nothing here is
tuned after a run; the match threshold and prompt below were fixed before the first run.

Backends: minicpm (openbmb/MiniCPM-o-2_6, first choice: not a Qwen vision encoder) and
omni (Qwen/Qwen2.5-Omni-7B, fallback, reported with the shared-encoder caveat).

    python -m benchmark.av_reference_pilot --backend minicpm --limit 3      # smoke (GPU)
    python -m benchmark.av_reference_pilot --backend minicpm                # dev run (GPU)
    python -m benchmark.av_reference_pilot --backend minicpm --eval         # summary
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

OUT_DIR = _ROOT / "benchmark" / "av_reference_pilot"
SUMMARY = _ROOT / "benchmark" / "av_reference_pilot.json"
MODELS = {"minicpm": "openbmb/MiniCPM-o-2_6", "omni": "Qwen/Qwen2.5-Omni-7B"}
N_FRAMES = 8
MATCH_COS = 0.60          # MiniLM cosine at which a heard item counts as already seen (fixed before the run)
BAR = {"balanced_accuracy": 0.677, "placebo_rate": 0.20}
PROMPT = (
    "You are watching a short video clip. List the events happening in it, at most eight. "
    "For each event give a short phrase (2-6 words) and how you know about it: \"seen\" if you only "
    "see it, \"heard\" if you only hear it (its source is not visible), \"both\" if you see and hear it. "
    "Include sounds whose source is out of view. Do not include what people say, and do not include music. "
    "Answer with JSON only, in this exact form: "
    "{\"events\": [{\"what\": \"...\", \"evidence\": \"seen|heard|both\"}]}"
)
SPEECH_MUSIC = re.compile(r"\b(speech|speak|speaking|talk|talking|voice|voices|conversation|dialogue|narrat\w*|"
                          r"music|musical|song|singing|sings|melody|tune|instrument|guitar|piano|drums?|"
                          r"soundtrack|lyrics|rap|beat)\b", re.I)


# ----------------------------------------------------------------------------- media
def frames_and_audio(video: Path, td: Path, n: int = N_FRAMES):
    from PIL import Image
    import soundfile as sf
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(video)], capture_output=True, text=True).stdout.strip() or 0)
    times = [dur * (i + 0.5) / n for i in range(n)]
    frames = []
    for i, t in enumerate(times):
        fp = td / f"f{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=448:-2",
                        "-q:v", "3", str(fp), "-loglevel", "error"], check=True)
        frames.append(Image.open(fp).convert("RGB").copy())
    wav = td / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav),
                    "-loglevel", "error"], check=True)
    audio, sr = sf.read(str(wav), dtype="float32")
    return frames, audio, sr, dur


# ----------------------------------------------------------------------------- backends
class MiniCPM:
    name = "minicpm"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(MODELS["minicpm"], trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16, init_vision=True, init_audio=True,
                                               init_tts=False).eval().to(device)
        self.tok = AutoTokenizer.from_pretrained(MODELS["minicpm"], trust_remote_code=True)

    def ask(self, frames, audio, sr) -> str:
        content = list(frames) + ([audio] if audio is not None else []) + [PROMPT]
        return self.model.chat(msgs=[{"role": "user", "content": content}], tokenizer=self.tok, sampling=False,
                               max_new_tokens=400, omni_input=True, use_tts_template=False, generate_audio=False,
                               max_slice_nums=1, use_image_id=False)


class Omni:
    name = "omni"

    def __init__(self, device: str):
        import torch
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODELS["omni"], torch_dtype=torch.bfloat16, device_map=device).eval()
        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODELS["omni"])
        self.device = device

    def ask(self, frames, audio, sr) -> str:
        import inspect
        content = [{"type": "video"}] + ([{"type": "audio"}] if audio is not None else []) + [{"type": "text", "text": PROMPT}]
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        video = np.stack([np.asarray(f) for f in frames])
        kw = {"text": [text], "videos": [video], "return_tensors": "pt", "padding": True, "use_audio_in_video": False}
        if audio is not None:
            key = "audio" if "audio" in inspect.signature(self.proc.__call__).parameters else "audios"
            kw[key] = [audio]
        inputs = self.proc(**kw).to(self.device)
        if audio is not None:
            assert "input_features" in inputs, "audio was not encoded by the processor"
        out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def backend(name: str, device: str):
    return {"minicpm": MiniCPM, "omni": Omni}[name](device)


# ----------------------------------------------------------------------------- parsing
def parse_events(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    items = []
    if m:
        try:
            for e in json.loads(m.group(0)).get("events", []):
                w = str(e.get("what", "")).strip(); ev = str(e.get("evidence", "")).strip().lower()
                if w and ev in ("seen", "heard", "both"):
                    items.append({"what": w, "evidence": ev})
        except Exception:
            pass
    if not items:                                  # tolerate a loose "what - evidence" list
        for w, ev in re.findall(r"[\"']?what[\"']?\s*:\s*[\"']([^\"']+)[\"'].{0,40}?(seen|heard|both)", text, re.I | re.S):
            items.append({"what": w.strip(), "evidence": ev.lower()})
    return items[:8]


# ----------------------------------------------------------------------------- run
def clips(limit: int):
    from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip, load
    tags = {r["clip"]: r["tag"] for r in load("dev")}
    items = [(f.stem, _find_clip(f.stem), tags.get(f.stem)) for f in sorted((CACHE_DIR / "dev").glob("*.json"))]
    items = [x for x in items if x[1] is not None and x[2] is not None]
    return items[:limit] if limit else items


def run(name: str, limit: int):
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    todo = [x for x in clips(limit) if not (out / (x[0] + ".json")).exists()]
    print(f"[av] {name}: {len(todo)} clips to do", flush=True)
    if not todo:
        return
    bk = backend(name, config.DEVICE)
    with tempfile.TemporaryDirectory() as td:
        for i, (stem, video, tag) in enumerate(todo, 1):
            try:
                frames, audio, sr, dur = frames_and_audio(Path(video), Path(td))
                with_text = bk.ask(frames, audio, sr)
                mute_text = bk.ask(frames, None, sr)
            except Exception as e:
                print(f"  ! {stem}: {type(e).__name__}: {e}", flush=True); continue
            rec = {"clip": stem, "tag": tag, "duration": dur, "with_sound": parse_events(with_text),
                   "muted": parse_events(mute_text), "raw": {"with_sound": with_text[:1500], "muted": mute_text[:1500]}}
            (out / (stem + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[av] {i}/{len(todo)} {stem}: with={len(rec['with_sound'])} muted={len(rec['muted'])} "
                  f"heard={sum(e['evidence'] != 'seen' for e in rec['with_sound'])}", flush=True)


# ----------------------------------------------------------------------------- reference + eval
def build_reference(rec, embed):
    heard = [e for e in rec["with_sound"] if e["evidence"] in ("heard", "both") and not SPEECH_MUSIC.search(e["what"])]
    seen = [e["what"] for e in rec["muted"]]
    if heard and seen:
        H = embed([e["what"] for e in heard]); S = embed(seen)
        sims = (H @ S.T)
        kept = [e for e, row in zip(heard, sims) if row.max() < MATCH_COS]
    else:
        kept = heard
    placebo = any(e["evidence"] in ("heard", "both") for e in rec["muted"])
    return kept, placebo


def evaluate(name: str):
    from benchmark.gate_dev_sweep import SHOULD_SHOW
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embed = lambda xs: m.encode(xs, normalize_embeddings=True)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT_DIR / name).glob("*.json"))]
    tp = fn = tn = fp = 0; placebo_n = 0; per_tag = {}
    rows = []
    for r in recs:
        kept, placebo = build_reference(r, embed)
        missing = bool(kept); due = r["tag"] in SHOULD_SHOW
        placebo_n += placebo
        if due: tp += missing; fn += (not missing)
        else: fp += missing; tn += (not missing)
        t = per_tag.setdefault(r["tag"], [0, 0]); t[0] += (missing == due); t[1] += 1
        sentence = ("A hearing viewer would also notice: " + "; ".join(e["what"] for e in kept) + "."
                    if kept else "nothing beyond the picture")
        rows.append({"clip": r["clip"], "tag": r["tag"], "reference": sentence, "items": [e["what"] for e in kept],
                     "placebo": placebo})
    n = len(recs)
    r_due = tp / max(1, tp + fn); r_not = tn / max(1, tn + fp)
    bal = (r_due + r_not) / 2
    po = (tp + tn) / max(1, n); pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(1, n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    placebo_rate = placebo_n / max(1, n)
    passed = bal >= BAR["balanced_accuracy"] and placebo_rate < BAR["placebo_rate"]
    summary = {"when": datetime.now().isoformat(timespec="minutes"), "backend": name, "model": MODELS[name], "clips": n,
               "due_recall": [tp, tp + fn], "not_due_recall": [tn, tn + fp], "balanced_accuracy": bal, "raw_agreement": po,
               "kappa": kappa, "placebo_rate": placebo_rate, "per_tag_agreement": per_tag, "bar": BAR, "passed": passed,
               "comparators": {"model_derived_dev": {"balanced_accuracy": 0.577, "kappa": 0.045},
                               "independent_test": {"raw_agreement": 0.55}},
               "references": rows}
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[eval] {name} on {n} dev clips: picture-due recall {tp}/{tp+fn} ({r_due:.0%}), nothing-due recall "
          f"{tn}/{tn+fp} ({r_not:.0%}) -> balanced {bal:.1%} (bar >= 67.7%; model-derived 57.7%) | kappa {kappa:.2f} | "
          f"placebo {placebo_n}/{n} ({placebo_rate:.0%}, bar < 20%) | per tag {per_tag} | "
          f"{'PASSED' if passed else 'FAILED'} -> {SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(MODELS), default="minicpm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if not a.eval:
        run(a.backend, a.limit)
    else:
        evaluate(a.backend)


if __name__ == "__main__":
    main()

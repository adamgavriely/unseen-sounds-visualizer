"""Smoke test of the SOTA candidates on one clip each (scope v2, point 1):

  detector   FLAM (openflam, frame-wise open-vocabulary): score the clip's audio against the
             benchmark's sound names, print the top frames per query
  vlm        Qwen3.8-27B (native vision-language, transformers >= 5.8): the visibility
             question on the ambulance clip's frames, in the same words the pipeline uses
  images     Z-Image-Turbo (diffusers): one picture from the pipeline's own phrase

Each part is independent and reports load time, peak VRAM and its answer; a failure in one
does not stop the others.

    python scripts/sota_smoke.py --parts detector vlm images --clip data/input/benchmark/mixed/ly_ambulance_(siren)_-yPSgCn.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

QUERIES = ["siren", "car horn", "dog barking", "glass breaking", "rain", "thunder", "crowd of people", "footsteps",
           "helicopter", "bird chirping", "alarm", "engine", "applause", "gunshot", "wind"]


def vram():
    try:
        import torch
        return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak"
    except Exception:
        return "n/a"


def wav_of(video: Path, td: Path) -> Path:
    wav = td / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def frames_of(video: Path, td: Path, n: int = 6):
    from PIL import Image
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                               capture_output=True, text=True).stdout.strip())
    out = []
    for i in range(n):
        t = dur * (i + 0.5) / n; fp = td / f"f{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2", str(fp)], check=True)
        out.append(Image.open(fp).convert("RGB").copy())
    return out


def part_detector(video: Path, td: Path):
    """FLAM: 10-second windows at 48 kHz; local similarity per frame for each text query."""
    import torch, numpy as np, librosa
    import openflam
    t0 = time.time()
    model = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path=str(Path.home() / ".cache" / "openflam")).to("cuda").eval()
    print(f"[detector] FLAM loaded in {time.time() - t0:.0f}s, {vram()}", flush=True)
    audio, sr = librosa.load(str(wav_of(video, td)), sr=48000, mono=True)
    win = 480000
    chunks = [audio[i:i + win] for i in range(0, max(1, len(audio)), win)]
    chunks = [np.pad(c, (0, win - len(c))) for c in chunks if len(c) > sr]
    rows = {q: [] for q in QUERIES}
    t0 = time.time()
    with torch.inference_mode():
        for c in chunks:
            x = torch.from_numpy(c).float().unsqueeze(0).to("cuda")
            sim = model.get_local_similarity(x.repeat(len(QUERIES), 1), QUERIES, method="unbiased")   # [queries, frames]
            sim = torch.sigmoid(sim) if sim.min() < 0 or sim.max() > 1 else sim
            for q, r in zip(QUERIES, sim.float().cpu().numpy()):
                rows[q].append(r)
    print(f"[detector] {len(chunks)} x 10 s windows scored in {time.time() - t0:.1f}s, {vram()}")
    for q in QUERIES:
        r = np.concatenate(rows[q]); hop = 10.0 * len(chunks) / len(r)
        i = int(r.argmax()); print(f"[detector]   {q:18s} max {r.max():.2f} at {i * hop:5.1f}s  mean {r.mean():.2f}")


def part_vlm(video: Path, td: Path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    name = "Qwen/Qwen3.8-27B"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[vlm] {name} loaded in {time.time() - t0:.0f}s, {vram()}", flush=True)
    frames = frames_of(video, td)
    q = ("These frames span a few seconds of a video in which the sound of a SIREN is heard. "
         "What in these frames, if anything, is visibly making that sound right now? Answer with the name of the thing, or 'nothing'.")
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], images=frames, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[vlm] answer in {time.time() - t0:.1f}s: {ans.strip()!r}  {vram()}")


def part_images(video: Path, td: Path):
    import torch
    from diffusers import DiffusionPipeline
    name = "Tongyi-MAI/Z-Image-Turbo"
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda")
    print(f"[images] {name} loaded in {time.time() - t0:.0f}s, {vram()}", flush=True)
    prompt = "ambulance siren blaring, a single clear picture of the event, plain white background, no text"
    t0 = time.time()
    img = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=0.0, height=768, width=768).images[0]
    out = _ROOT / "data" / "output" / "sota_smoke_zimage.png"; out.parent.mkdir(parents=True, exist_ok=True); img.save(out)
    print(f"[images] one picture in {time.time() - t0:.1f}s -> {out}  {vram()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["detector", "vlm", "images"], choices=["detector", "vlm", "images"])
    ap.add_argument("--clip", default=str(_ROOT / "data" / "input" / "benchmark" / "mixed" / "ly_ambulance_(siren)_-yPSgCn.mp4"))
    a = ap.parse_args()
    video = Path(a.clip)
    with tempfile.TemporaryDirectory() as td:
        for part in a.parts:
            print(f"=== {part} ===", flush=True)
            try:
                {"detector": part_detector, "vlm": part_vlm, "images": part_images}[part](video, Path(td))
            except Exception:
                print(f"[{part}] FAILED:\n" + traceback.format_exc()[-1500:], flush=True)
            try:
                import torch, gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


if __name__ == "__main__":
    main()

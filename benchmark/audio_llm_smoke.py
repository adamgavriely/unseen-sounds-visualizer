"""Smoke test: can an audio language model act as a second opinion on BEATs' detections?

The family not yet tried (Fable, 2026-09-15): every earlier attempt filtered one
closed-vocabulary tagger's scores. An audio LLM hears with an open vocabulary and carries
world knowledge, so it can (a) veto a tag it does not hear -- a whale at a Christmas
market -- and (b) name an ambient sound BEATs buried under speech. Before any full run,
a two-hour smoke test on 20 dev sounds drawn before running (audio_llm_smoke_set.json):
10 phantoms and 10 currently shown sounds on unseen clips.

Question form: a forced choice over BEATs' own candidates for the clip, so the model's
readiness to answer "none" to an open question does not decide it, plus one open line
for sounds it hears that BEATs did not name.

Declared pass rule (Fable): vetoes >= 5 of the 10 phantoms AND keeps >= 8 of the 10 real
sounds. Pass -> a full dev run with the same question and a declared bar (>= 40/77
phantoms removed, <= 2/19 unseen clips lost); fail -> drop, report, stop.

    python -m benchmark.audio_llm_smoke            # GPU
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
SMOKE = _ROOT / "benchmark" / "audio_llm_smoke_set.json"
OUT = _ROOT / "benchmark" / "audio_llm_smoke.json"
MAX_SEC = 30.0
LETTERS = "ABCDEFGHIJKLMNOP"


def _wav(video: Path, td: Path) -> Path:
    wav = td / (video.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    "-t", str(MAX_SEC), str(wav), "-loglevel", "error"], check=True)
    return wav


def ask(proc, mdl, audio, sr, candidates):
    import torch
    from src.stage7_evaluation.independent_reference import _with_audio
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(candidates))
    prompt = ("This audio is the soundtrack of a video. Which of the following sounds can you "
              "actually hear in it? Judge by the audio only.\n" + opts +
              "\nAnswer with the letters of the sounds you hear, separated by commas, or 'none'. "
              "Then, on a second line starting with 'Other:', list any other clear non-speech "
              "sounds you hear, or 'none'.")
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": ""},
                                         {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = _with_audio(proc, [audio], sr, text=text).to(mdl.device)
    with torch.no_grad():
        gen = mdl.generate(**inputs, max_new_tokens=64, do_sample=False)
    reply = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    first = reply.split("\n")[0]
    heard = {LETTERS[i] for i in range(len(candidates))
             if re.search(rf"(?<![A-Za-z]){LETTERS[i]}(?![A-Za-z])", first)}
    other = ""
    m = re.search(r"other:\s*(.*)", reply, re.I | re.S)
    if m:
        other = m.group(1).strip()
    return reply, {candidates[i] for i in range(len(candidates)) if LETTERS[i] in heard}, other


def main():
    import librosa
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    from benchmark.gate_dev_sweep import load, decide, _find_clip
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    targets = {}
    for kind in ("phantoms", "real"):
        for clip, label, conf in smoke[kind]:
            targets.setdefault(clip, {})[label] = kind
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    shown_by = {}
    for rec in load("dev"):
        if rec["clip"] in targets:
            shown_by[rec["clip"]] = decide(rec, gate["bar"], gate["rule"], gate["kinds"])["shown"]
    proc = AutoProcessor.from_pretrained(MODEL)
    mdl = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    sr = proc.feature_extractor.sampling_rate
    results = []
    with tempfile.TemporaryDirectory() as td:
        for clip, want in targets.items():
            video = _find_clip(clip)
            if video is None:
                continue
            audio, _ = librosa.load(str(_wav(video, Path(td))), sr=sr, mono=True)
            cands = sorted(set(shown_by.get(clip, [])) | set(want))
            reply, heard, other = ask(proc, mdl, audio, sr, cands)
            for label, kind in want.items():
                results.append({"clip": clip, "label": label, "kind": kind, "heard": label in heard,
                                "candidates": cands, "reply": reply, "other": other})
                print(f"  {kind:8s} {clip[:38]:38s} {label:32s} -> {'HEARD' if label in heard else 'not heard'}   | {reply[:70]!r}", flush=True)
    ph = [r for r in results if r["kind"] == "phantoms"]; re_ = [r for r in results if r["kind"] == "real"]
    vetoed = sum(1 for r in ph if not r["heard"]); kept = sum(1 for r in re_ if r["heard"])
    passed = vetoed >= 5 and kept >= 8
    print(f"[smoke] phantoms vetoed {vetoed}/{len(ph)}; real kept {kept}/{len(re_)}; {'PASSED' if passed else 'FAILED'}")
    OUT.write_text(json.dumps({"vetoed": [vetoed, len(ph)], "kept": [kept, len(re_)], "passed": passed,
                               "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

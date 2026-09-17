"""Granite Speech 4.1-2B as the Stage 3 transcriber (v4, docs/prereg_v4.md).

IBM, April 2026, Apache-2.0; first on the Open ASR leaderboard (mean WER 5.33 vs Whisper
large-v3's 6.43). A speech-language model: a Conformer encoder feeding a Granite LLM, so it
transcribes a chunk of audio in one generate call and gives no timestamps of its own. The
transcript is context only here (the reference builder and the speech-as-gate-context
question), so the clip is transcribed in fixed chunks and each chunk becomes one segment
with the chunk's bounds; a chunk with no speech yields nothing. English is assumed (the
model covers six languages; Whisper covered 99).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.types import SpeechSegment

MODEL = "ibm-granite/granite-speech-4.1-2b"
SR = 16000
CHUNK = 15.0          # seconds per generate call; the segment granularity
PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

_MODEL = None


def _load(device: str):
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL)
        mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL, device_map=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
        _MODEL = (mdl, proc)
    return _MODEL


def unload():
    global _MODEL
    _MODEL = None


def transcribe_granite(wav_path: Path, device: str = "cuda") -> List[SpeechSegment]:
    import torch, librosa
    mdl, proc = _load(device)
    tok = proc.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    n = int(CHUNK * SR)
    segments: List[SpeechSegment] = []
    for k, start in enumerate(range(0, max(1, len(audio)), n)):
        chunk = audio[start:start + n]
        if len(chunk) < SR // 2:
            break
        wav = torch.from_numpy(chunk).float().unsqueeze(0)
        inputs = proc(prompt, wav, device=device, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
        text = tok.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()
        if text and text.lower() not in {"", "[silence]", "(silence)", "..."}:
            segments.append(SpeechSegment(index=k, start=start / SR, end=(start + len(chunk)) / SR, text=text))
    return segments

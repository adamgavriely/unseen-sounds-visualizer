"""Stage 2 (v2-b) - full VLM visibility check with Qwen2.5-VL. GPU only.

Same contract as the CLIP backend (`analyze_video`): given a video, return a
SceneContext whose ``visible_entities`` names the sound-source concepts actually
visible on screen, so the Stage-5 gate can stay silent about them.

Why this upgrade: the benchmark error analysis showed the CLIP gate's dominant
failure is missing on-screen traffic in busy street scenes (11 Vehicle
false-augments) -- a *scene understanding* problem CLIP's single global embedding
handles poorly. A VLM reasons over the whole frame and answers a grounded
multiple-choice question instead, which is why this is the main GPU experiment.

The model is asked a CLOSED question (pick from our concept list, answer "none"
if unsure) rather than an open one, to keep hallucination low and the output
parseable -- the same grounded-decision principle used in Stage 5.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image

from src.types import SceneContext
from src.stage1_audio_extraction import media_duration
from src.stage2_video_understanding import VISIBLE_CONCEPTS

_MODEL = None
_PROCESSOR = None

_PROMPT = (
    "Look at this video frame. From the list below, name ONLY the items that are "
    "clearly visible in the frame right now.\n"
    "List: {options}\n"
    "Rules: answer with a comma-separated subset of the list, copying the names "
    "exactly. If none are visible, answer exactly: none. Do not explain."
)


def _load(model_name: str, device: str):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        _PROCESSOR = AutoProcessor.from_pretrained(model_name)
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ).eval()
    return _MODEL, _PROCESSOR


def _frames(video: Path, n: int) -> List[Image.Image]:
    dur = media_duration(video) or 10.0
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            t = dur * (i + 0.5) / n
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video),
                            "-frames:v", "1", "-vf", "scale=640:-2", str(fp)],
                           capture_output=True, timeout=120)
            if fp.exists():
                out.append(Image.open(fp).convert("RGB").copy())
    return out


def analyze_video_vlm(video_path: Path, num_frames: int = 6,
                      model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                      device: str = "cuda",
                      candidates: Optional[List[str]] = None) -> SceneContext:
    """VLM visibility check. ``candidates`` restricts the question to the sound
    labels actually detected (faster + more accurate); defaults to all concepts."""
    labels = [c for c in (candidates or list(VISIBLE_CONCEPTS)) if c in VISIBLE_CONCEPTS]
    if not labels:
        return SceneContext(summary="no candidate concepts", visible_entities=[],
                            frames_analyzed=0, raw={"backend": "qwen2.5-vl"})

    mdl, proc = _load(model, device)
    import torch

    # human-readable option names, mapped back to our canonical labels
    options = {lb: VISIBLE_CONCEPTS[lb].replace("a photo of ", "") for lb in labels}
    prompt = _PROMPT.format(options="; ".join(f"{lb} ({d})" for lb, d in options.items()))

    votes = {lb: 0 for lb in labels}
    frames = _frames(video_path, num_frames)
    for img in frames:
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=64, do_sample=False)
        answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].lower()
        if "none" in answer[:12]:
            continue
        for lb in labels:                      # substring match on the canonical name
            if re.search(rf"\b{re.escape(lb.lower())}\b", answer):
                votes[lb] += 1

    # a concept counts as visible if seen in >=1/3 of frames (rejects one-off blips)
    need = max(1, len(frames) // 3)
    visible = [lb for lb, v in votes.items() if v >= need]
    print(f"       [stage2/vlm] {len(frames)} frames, visible: {visible or 'none'}")
    return SceneContext(
        summary=f"VLM visibility over {len(frames)} frames",
        visible_entities=visible, frames_analyzed=len(frames),
        raw={"backend": "qwen2.5-vl", "model": model, "votes": votes, "min_votes": need})

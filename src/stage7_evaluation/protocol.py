"""Stage 7 - the automatic evaluation protocol of proposal section 6.1.

The question this answers is the project's actual research question: *do the
generated visual augmentations communicate the audio semantics that the original
video does not?* It is deliberately not the gating metric -- gating is one component
(Stage 5), whereas this measures the system's output.

The five steps of proposal sec 6.1, implemented:

  1. ANALYSE the original clip -- the detected non-speech events (Stage 4), the scene
     context (Stage 2) and the transcript (Stage 3) are already produced by the
     pipeline and are read from the run's artifacts.
  2. GENERATE the augmentations -- done by the pipeline (Stage 6); this module
     consumes the rendered panel images.
  3. DESCRIBE with a VLM -- a vision-language model is shown ONLY the augmentation
     image and asked what a viewer would learn from it. It is never told which sound
     produced the image, so the description is independent of the reference.
  4. REFERENCE from the original multimodal input -- an LLM turns the detected audio
     events plus the visible scene into one sentence stating what a hearing viewer
     perceives that a deaf viewer would miss. This is the semantic target.
  5. JUDGE -- a separate LLM call scores how much of the reference the description
     conveys, on a 0-4 scale with a written justification.

Two properties make the protocol defensible rather than self-confirming:

  * the describer sees the image only, so it cannot copy the reference wording;
  * the judge sees description and reference but not which system produced them,
    so the same judge scores the proposed method and the baselines identically.

BASELINES (proposal sec 7) share every step except how the image is produced:
  proposed        -- gate first, depict only sounds whose source is not visible
  blind_a2i       -- depict every detected sound, ignoring the video (this is the
                     direct audio-to-image baseline)
  audio_caption   -- text only, no image; the "description" is the caption itself

Backends are pluggable so the protocol can run with a local VLM/LLM on the cluster
or with a hosted API; see config.JUDGE_MODEL and config.VLM_MODEL.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# prompts -- kept as module constants so the exact wording is reviewable and
# citable in the report, and so a change of wording is a visible diff.
# ----------------------------------------------------------------------
DESCRIBE_PROMPT = (
    "This image is shown beside a video to help a deaf viewer understand a sound "
    "they cannot hear. Describe, in one sentence, what a viewer would learn from "
    "this image about the sound. Name the sound source if you can identify it. "
    "Do not describe artistic style."
)

REFERENCE_PROMPT = (
    "A video contains these non-speech sounds: {sounds}.\n"
    "The following things are already visible on screen: {visible}.\n"
    "{speech}\n"
    "In ONE sentence, state the information a hearing viewer gets from the sounds "
    "that a deaf viewer would miss from the picture alone. Name the specific sound "
    "sources. If the sounds add nothing beyond what is visible, say exactly: "
    "'nothing beyond the picture'."
)

JUDGE_PROMPT = (
    "REFERENCE (what a deaf viewer is missing): {reference}\n"
    "CANDIDATE (what an added visual actually conveys): {candidate}\n\n"
    "How much of the reference information does the candidate convey to a viewer?\n"
    "Score strictly on this scale:\n"
    "  4 = conveys the reference information, correct source\n"
    "  3 = conveys most of it, minor omission or vagueness\n"
    "  2 = partially related, the main sound source is wrong or missing\n"
    "  1 = barely related\n"
    "  0 = unrelated, or actively misleading\n"
    "Answer as JSON only: {{\"score\": <0-4>, \"why\": \"<one short sentence>\"}}"
)


@dataclass
class ClipEvaluation:
    clip: str
    system: str
    reference: str
    description: str
    score: int
    why: str
    n_augmentations: int

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------
# model backends
# ----------------------------------------------------------------------
class Backends:
    """Lazily-loaded VLM (describe) and LLM (reference + judge).

    Both default to Qwen2.5-VL, which can do vision and text, so one weight load
    serves all three calls on the cluster. A hosted API can be dropped in by
    replacing describe_image/complete without touching the protocol.
    """

    def __init__(self, vlm_model: str, device: str = "cuda"):
        self.vlm_model = vlm_model
        self.device = device
        self._vlm = None

    def _load(self):
        if self._vlm is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            proc = AutoProcessor.from_pretrained(self.vlm_model)
            mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None).eval()
            self._vlm = (mdl, proc)
        return self._vlm

    def describe_image(self, image_path: Path, prompt: str = DESCRIBE_PROMPT) -> str:
        import torch
        from PIL import Image
        mdl, proc = self._load()
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=80, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()

    def complete(self, prompt: str, max_new_tokens: int = 120) -> str:
        """Text-only completion, used for the reference and the judge."""
        import torch
        mdl, proc = self._load()
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------------
# the five steps
# ----------------------------------------------------------------------
def build_reference(events: List[str], visible: List[str], transcript: str,
                    backends: Backends) -> str:
    """Step 4: what does a hearing viewer get that a deaf viewer misses?"""
    speech = (f"Someone is speaking; the dialogue is already captioned, so ignore it."
              if transcript.strip() else "There is no speech.")
    prompt = REFERENCE_PROMPT.format(
        sounds=", ".join(events) if events else "no clear non-speech sound",
        visible=", ".join(visible) if visible else "nothing relevant",
        speech=speech)
    return backends.complete(prompt, max_new_tokens=80)


def describe_augmentation(image_paths: List[Path], backends: Backends) -> str:
    """Step 3: what does the augmentation itself convey? Image only, no context."""
    if not image_paths:
        return "no augmentation was shown"
    parts = [backends.describe_image(p) for p in image_paths[:3]]
    return " ".join(parts)


def judge(reference: str, candidate: str, backends: Backends) -> tuple[int, str]:
    """Step 5: independent scoring of semantic consistency."""
    raw = backends.complete(JUDGE_PROMPT.format(reference=reference,
                                                candidate=candidate),
                            max_new_tokens=100)
    m = re.search(r'\{.*\}', raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d.get("score", 0)), str(d.get("why", ""))[:200]
        except Exception:
            pass
    m = re.search(r'([0-4])', raw)          # fall back to the first digit
    return (int(m.group(1)) if m else 0), raw[:200]


def evaluate_clip(clip_name: str, system: str, work_dir: Path,
                  backends: Backends) -> Optional[ClipEvaluation]:
    """Run the whole protocol for one clip whose pipeline artifacts exist."""
    try:
        events_f = work_dir / "events.json"
        scene_f = work_dir / "scene.json"
        augs_f = work_dir / "augmentations.json"
        if not (events_f.exists() and augs_f.exists()):
            return None
        from src.labels import is_salient_nonspeech, consolidate_families
        from src.types import AudioEvent
        import config

        raw_events = json.loads(events_f.read_text(encoding="utf-8"))
        evs = [AudioEvent(e["label"], e["start"], e["end"], e["confidence"])
               for e in raw_events]
        salient = [e.label for e in consolidate_families(
            [x for x in evs if is_salient_nonspeech(x.label)])
            if e.confidence >= config.DISPLAY_THRESHOLD]
        scene = json.loads(scene_f.read_text(encoding="utf-8")) if scene_f.exists() else {}
        visible = scene.get("visible_entities", [])
        seg_f = work_dir / "segments.json"
        transcript = " ".join(s.get("text", "") for s in
                              json.loads(seg_f.read_text(encoding="utf-8"))) \
            if seg_f.exists() else ""

        specs = json.loads(augs_f.read_text(encoding="utf-8"))
        images = [Path(s["image_path"]) for s in specs
                  if s.get("augment") and s.get("image_path")
                  and Path(s["image_path"]).exists()]

        reference = build_reference(salient, visible, transcript, backends)
        description = describe_augmentation(images, backends)
        score, why = judge(reference, description, backends)
        return ClipEvaluation(clip=clip_name, system=system, reference=reference,
                              description=description, score=score, why=why,
                              n_augmentations=len(images))
    except Exception as e:
        print(f"    ! {clip_name}: {type(e).__name__}: {e}")
        return None

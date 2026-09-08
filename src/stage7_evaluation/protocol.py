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

Three properties make the protocol defensible rather than self-confirming:

  * the describer sees the image only, so it cannot copy the reference wording;
  * the judge sees description and reference but not which system produced them,
    so the same judge scores the proposed method and the baselines identically;
  * the JUDGE IS A DIFFERENT MODEL from the describer. A model scoring its own
    descriptions is a self-evaluation bias an examiner would rightly challenge, so
    the describing VLM and the judging LLM are separate weights (config.VLM_MODEL
    and config.JUDGE_MODEL) and the protocol refuses to run if they are the same.

BASELINES (proposal sec 7) share every step except how the image is produced:
  proposed        -- gate first, depict only sounds whose source is not visible
  blind_a2i       -- depict every detected sound, ignoring the video (this is the
                     direct audio-to-image baseline)
  audio_caption   -- text only, no image; the "description" is the caption itself

The protocol runs in TWO SEQUENTIAL PASSES so the describer and the judge never sit in
GPU memory at the same time (Qwen2.5-VL ~16 GB + Mistral ~15 GB will not co-fit on a
24 GB card):

  pass 1 "describe"  load the VLM, produce the reference and the description for every
                     clip, write them to benchmark/protocol_descriptions.json, then
                     free the weights;
  pass 2 "judge"     load the judge alone and score the cached pairs.

Besides fitting anywhere, this makes the Day-6 judge-reliability experiment cheap: a
second judge can re-score the same cached descriptions without re-running any vision.

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
    """The VLM that DESCRIBES and the separate LLM that JUDGES.

    Deliberately two different models. Letting one model describe an image and then
    score its own description measures self-consistency, not quality, and inflates
    the result: the judge recognises its own phrasing. Keeping them separate costs a
    second weight load and removes the objection entirely.

    describe_image() uses the VLM (vision). complete() takes a `judging` flag: the
    reference sentence is built by the VLM's text side (it is derived from pipeline
    metadata, not from any image, so no bias is possible), while the scoring call is
    routed to the independent judge model.
    """

    def __init__(self, vlm_model: str, judge_model: str, device: str = "cuda"):
        if judge_model == vlm_model:
            raise ValueError(
                f"judge and describer must differ (both are {vlm_model}); "
                "set config.JUDGE_MODEL to a different model")
        self.vlm_model = vlm_model
        self.judge_model = judge_model
        self.device = device
        self._vlm = None
        self._judge = None

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

    def _load_judge(self):
        """The independent judge: a text-only LLM from a different family."""
        if self._judge is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.judge_model)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.judge_model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None).eval()
            self._judge = (mdl, tok)
        return self._judge

    def unload_vlm(self):
        """Free the describer before the judge is loaded (see the two-pass note)."""
        if self._vlm is not None:
            del self._vlm
            self._vlm = None
        self._free()

    def unload_judge(self):
        if self._judge is not None:
            del self._judge
            self._judge = None
        self._free()

    @staticmethod
    def _free():
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

    def complete(self, prompt: str, max_new_tokens: int = 120,
                 judging: bool = False) -> str:
        """Text completion. ``judging`` routes to the independent judge model."""
        import torch
        if judging:
            mdl, tok = self._load_judge()
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False)
            return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
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
                            max_new_tokens=100, judging=True)
    m = re.search(r'\{.*\}', raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d.get("score", 0)), str(d.get("why", ""))[:200]
        except Exception:
            pass
    m = re.search(r'([0-4])', raw)          # fall back to the first digit
    return (int(m.group(1)) if m else 0), raw[:200]


def describe_clip(clip_name: str, system: str, work_dir: Path,
                  backends: Backends) -> Optional[dict]:
    """Pass 1: build the reference and the description. No judging, no judge model.

    Returns a plain dict so it can be cached to JSON and scored later by any judge.
    """
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
                              json.loads(seg_f.read_text(encoding="utf-8")))             if seg_f.exists() else ""

        specs = json.loads(augs_f.read_text(encoding="utf-8"))
        images = [Path(s["image_path"]) for s in specs
                  if s.get("augment") and s.get("image_path")
                  and Path(s["image_path"]).exists()]

        return {"clip": clip_name, "system": system,
                "reference": build_reference(salient, visible, transcript, backends),
                "description": describe_augmentation(images, backends),
                "n_augmentations": len(images),
                "sounds": salient, "visible": visible}
    except Exception as e:
        print(f"    ! describe {clip_name}: {type(e).__name__}: {e}")
        return None


def judge_record(rec: dict, backends: Backends) -> ClipEvaluation:
    """Pass 2: score one cached (reference, description) pair."""
    score, why = judge(rec["reference"], rec["description"], backends)
    return ClipEvaluation(clip=rec["clip"], system=rec["system"],
                          reference=rec["reference"], description=rec["description"],
                          score=score, why=why,
                          n_augmentations=rec.get("n_augmentations", 0))


def evaluate_clip(clip_name: str, system: str, work_dir: Path,
                  backends: Backends) -> Optional[ClipEvaluation]:
    """Single-pass convenience path (both models resident). Prefer the two passes."""
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

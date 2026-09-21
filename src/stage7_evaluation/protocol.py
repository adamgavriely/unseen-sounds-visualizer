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

# Amendment 3 (2026-09-21, bug fix, docs/prereg_v4.md): the prompt used to receive Stage 2's
# whole-clip list of visible things and to offer the sentinel "nothing beyond the picture"
# when the sounds added nothing to it. The gate uses that same Stage-2 list, so the gate was
# graded against its own input: on 9 of the 50 clips the annotator tagged as needing a
# picture, the gated system stayed silent and scored 4. Visibility is now settled only by
# the human clip tag (grounded_reference); the prompt just names what the sounds tell.
REFERENCE_PROMPT = (
    "A video contains these non-speech sounds: {sounds}.\n"
    "{speech}\n"
    "In ONE sentence, state the information a hearing viewer gets from these sounds. "
    "Name the specific sound sources. Do not mention the picture."
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
            from transformers import AutoProcessor
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            dev = "auto" if self.device == "cuda" else None
            proc = AutoProcessor.from_pretrained(self.vlm_model)
            if "Qwen2.5-VL" in self.vlm_model:
                from transformers import Qwen2_5_VLForConditionalGeneration
                mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.vlm_model, torch_dtype=dtype, device_map=dev).eval()
            else:
                # v4: Qwen3.8-27B and any other native VL model (transformers >= 5.8)
                from transformers import AutoModelForImageTextToText
                mdl = AutoModelForImageTextToText.from_pretrained(
                    self.vlm_model, dtype=dtype, device_map=dev).eval()
            self._vlm = (mdl, proc)
        return self._vlm

    def _load_judge(self):
        """The independent judge: a text-only LLM from a different family."""
        if self._judge is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.judge_model)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            dev = "auto" if self.device == "cuda" else None
            try:
                mdl = AutoModelForCausalLM.from_pretrained(self.judge_model, dtype=dtype, device_map=dev).eval()
            except (ValueError, KeyError):
                # v4 judge Gemma-4-31B-it is a multimodal checkpoint used in text mode
                from transformers import AutoModelForImageTextToText
                mdl = AutoModelForImageTextToText.from_pretrained(self.judge_model, dtype=dtype, device_map=dev).eval()
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

    @staticmethod
    def _template(proc, msgs) -> str:
        """Chat template with the reasoning preamble off where the model has one (Qwen3.x):
        the describer and the reference-builder are asked for one sentence."""
        try:
            return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def describe_image(self, image_path: Path, prompt: str = DESCRIBE_PROMPT) -> str:
        import torch
        from PIL import Image
        mdl, proc = self._load()
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        text = self._template(proc, msgs)
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
            try:
                # Qwen3 and friends default to emitting a <think> block; the judge is
                # asked for one JSON object, so turn it off where the template allows.
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
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
        text = self._template(proc, msgs)
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
    """Step 4: what does a hearing viewer get that a deaf viewer misses?

    If Stage 4 detected NO salient non-speech sound, the answer is fixed and the LLM is
    not asked. It was asked, in the pilot, and it invented one every time: on a clip
    with no non-speech audio it wrote "a hearing viewer would notice the crowd's
    movement and potential excitement" -- in the same sentence as "there are no
    non-speech sounds present". The sentinel the prompt asks for was produced 0 times
    in 36 records.

    That is not a cosmetic failure. The sentinel is the only route by which staying
    silent can score above zero, so while it is unreachable, a system that correctly
    shows nothing is punished exactly as hard as one that missed a siren -- and the
    gated system abstains far more than the baselines by construction. Deciding this
    case from the detector's own output removes the LLM's compliance from the loop.
    """
    if not events:
        return NOTHING_MISSING
    speech = (f"Someone is speaking; the dialogue is already captioned, so ignore it."
              if transcript.strip() else "There is no speech.")
    prompt = REFERENCE_PROMPT.format(
        sounds=", ".join(events) if events else "no clear non-speech sound",
        speech=speech)
    return backends.complete(prompt, max_new_tokens=80)


def describe_augmentation(image_paths: List[Path], backends: Backends) -> str:
    """Step 3: what does the augmentation itself convey? Image only, no context."""
    if not image_paths:
        return "no augmentation was shown"
    parts = [backends.describe_image(p) for p in image_paths[:3]]
    return " ".join(parts)


NO_AUG = "no augmentation was shown"
# The caption baseline has its own way of saying "I showed nothing", so both empty
# forms must hit the same rule -- otherwise the baselines are scored under different
# conventions and the comparison is not like-for-like.
EMPTY_CANDIDATES = (NO_AUG, "no notable non-speech sound")
NOTHING_MISSING = "nothing beyond the picture"


def is_empty_candidate(candidate: str) -> bool:
    c = candidate.strip().lower()
    return any(c.startswith(m) for m in EMPTY_CANDIDATES)


# Reasoning models spend their budget thinking before answering, so the judge needs
# room to reach the JSON at all: Qwen3-8B, given 100 tokens, never emitted a score.
JUDGE_TOKENS = 400
UNPARSED = "UNPARSED: "
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_judge_output(raw: str):
    """Pull (score, why) out of a judge's reply, or None if it did not answer.

    Returning None rather than a default matters more than the parsing does. The first
    judge-agreement run scored 263 of 300 records as 0 and reported quadratic kappa
    0.164 -- which reads as "the two judges disagree almost entirely", a publishable
    claim about how far LLM-judged numbers can be trusted. It was nothing of the kind:
    the second judge is a reasoning model, every reply opened with a <think> block that
    never finished inside the token budget, no JSON was ever produced, and the fallback
    "first digit found" rule turned each non-answer into a confident 0.

    So: strip the reasoning block, look for JSON, then for an explicit score, and if
    none of that is present say so instead of inventing a number.
    """
    if not raw:
        return None
    txt = _THINK.sub("", raw).strip()
    if "<think>" in txt:                    # opened a block and never closed it
        txt = txt.split("<think>")[0].strip()
    m = re.search(r'\{[^{}]*"score".*?\}', txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d.get("score", 0)), str(d.get("why", ""))[:200]
        except Exception:
            pass
    m = re.search(r'"?score"?\s*[:=]\s*([0-4])', txt, re.I)
    if m:
        return int(m.group(1)), txt[:200]
    m = re.match(r'\s*([0-4])', txt)      # a bare score on its own
    if m:
        return int(m.group(1)), txt[:200]
    return None


def judge(reference: str, candidate: str, backends: Backends) -> tuple[int, str]:
    """Step 5: independent scoring of semantic consistency.

    The EMPTY-CANDIDATE case is decided in code, not by the judge. In the pilot the
    judge handed "no augmentation was shown" a charitable 2/4, which both rewarded the
    gated system for showing nothing and denied it credit on clips where showing
    nothing is the right answer. Silence is not partially correct -- it is either
    exactly right or a total miss:

      reference says nothing is missing  -> staying silent is the correct behaviour, 4
      reference names missing information -> the system conveyed none of it, 0

    Scoring this deterministically also means the gate's core claim (stay silent when
    the source is already visible) is actually measured rather than blurred.
    """
    if is_empty_candidate(candidate):
        # exact match, as scripts/rubric_enforce.py tests it (amendment 3: a substring test
        # let a longer sentence containing the words score silence 4 but escape the cap)
        if reference.strip().lower().rstrip(".") == NOTHING_MISSING:
            return 4, "nothing was missing and the system correctly showed nothing"
        return 0, "information was missing but no augmentation was shown"
    raw = backends.complete(JUDGE_PROMPT.format(reference=reference,
                                                candidate=candidate),
                            max_new_tokens=JUDGE_TOKENS, judging=True)
    parsed = parse_judge_output(raw)
    if parsed is None:
        # Never silently score an unparsed answer as 0. The first judge-agreement run
        # did exactly that and produced a confident kappa of 0.164 that was entirely an
        # artefact of the parser (see parse_judge_output).
        return 0, UNPARSED + raw[:180]
    return parsed


# Human tags under which the annotator judged that the soundtrack adds nothing a
# viewer cannot already see: there is no ambient sound at all, or its source is on
# screen. Both are cases where showing nothing is the correct output.
NOTHING_MISSING_TAGS = ("no_ambient", "seen_ambient")
# Human tags under which a picture is needed. If the detector heard nothing there, the
# model-derived reference collapses to the sentinel and silence would score 4; the
# annotator says otherwise, so the reference becomes this fixed sentence instead
# (amendment 3): an empty panel then scores 0 by the coded rule, a shown picture is judged
# against it by the LLM.
NEEDED_TAGS = ("unseen_ambient", "mixed_ambient")
NEEDED_UNKNOWN = "An off-screen sound matters here; its source is not known."


def grounded_reference(llm_reference: str, human_tag: Optional[str]) -> str:
    """The reference, corrected by the annotator's label where that label settles it.

    The model-derived reference names whatever Stage 4 heard, so on a clip whose sound
    source is plainly on screen it still reports the sound as "missing". A system that
    correctly stays silent is then scored 0 -- and `seen_ambient` is 128 of the 274
    labelled clips, so this is not an edge case: it would systematically punish the
    behaviour the thesis is about and hand the win to the blind baseline for the wrong
    reason.

    Deriving that correction from Stage 2's visibility output instead would be
    circular: the gate would be graded against its own decision and would agree with
    the reference by construction. The human tag is independent of every model in the
    pipeline, so it is the one non-circular source available.

    Both references are stored, and the judge can be pointed at either, so the
    proposal-faithful number and the human-grounded number are both reportable from a
    single (expensive) describe pass.
    """
    if human_tag in NOTHING_MISSING_TAGS:
        return NOTHING_MISSING
    if human_tag in NEEDED_TAGS and (not llm_reference.strip()
                                     or llm_reference.strip().lower().rstrip(".") == NOTHING_MISSING):
        return NEEDED_UNKNOWN
    return llm_reference


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

        reference = build_reference(salient, visible, transcript, backends)
        return {"clip": clip_name, "system": system,
                "reference": reference,
                "description": describe_augmentation(images, backends),
                "n_augmentations": len(images),
                "sounds": salient, "visible": visible}
    except Exception as e:
        print(f"    ! describe {clip_name}: {type(e).__name__}: {e}")
        return None


def judge_record(rec: dict, backends: Backends, grounded: bool = False,
                 reference_override: Optional[str] = None) -> ClipEvaluation:
    """Pass 2: score one cached (reference, description) pair.

    ``grounded`` scores against the human-corrected reference instead of the purely
    model-derived one; see grounded_reference(). ``reference_override`` scores against a
    reference built by models that never saw the system's output (see
    independent_reference.py) -- the corrected primary since 2026-09-14, after a
    review found the default reference was written from the detector's own events by
    the proposed system's own VLM. The choice is made here, at judging time, so every
    number comes out of one describe pass.
    """
    if reference_override is not None:
        reference = reference_override
    else:
        reference = (grounded_reference(rec["reference"], rec.get("human_tag"))
                     if grounded else rec["reference"])
    score, why = judge(reference, rec["description"], backends)
    return ClipEvaluation(clip=rec["clip"], system=rec["system"],
                          reference=reference, description=rec["description"],
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

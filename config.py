"""Central defaults for the pipeline. Override per-run via main.py CLI flags.

Every stage is real; the rationale for each choice is in the comment beside it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INPUT_DIR = DATA / "input"
WORK_DIR = DATA / "work"
OUTPUT_DIR = DATA / "output"

# Compute
DEVICE = "cpu"               # "cpu" locally; "cuda" on the university GPU

# Stage 1 — audio extraction
SAMPLE_RATE = 16000

# Stage 2 — video understanding
# "siglip" (default) | "clip" (baseline) | "vlm" (Qwen2.5-VL, university GPU)
# SigLIP scores each concept with an independent sigmoid, so visibility is a real
# per-source yes/no; CLIP's softmax made concepts compete (see stage2/siglip.py).
# "owlv2" is the strongest but needs a GPU (~12 s/frame on CPU); "siglip" is the
# CPU default. Measured on 209 tagged clips: CLIP 49.3% acc, SigLIP 50.2% -- the
# embedding models both miss on-screen sources (67 of 103 seen_ambient clips were
# predicted unseen), which is what detection is meant to fix.
VIDEO_BACKEND = "siglip"
VIDEO_MODEL = "openai/clip-vit-base-patch32"          # used when backend == "clip"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_THRESHOLD = -7.5   # raw logit, not a probability (see stage2/siglip.py)
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_THRESHOLD = 0.20
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"      # used when VIDEO_BACKEND == "vlm"
VLM_THINKING = False          # Qwen3.x: reason before answering (docs/prereg_qwen38_visibility.md); slower
VLM_THINKING_TOKENS = 1024    # generation budget when thinking is on
NUM_FRAMES = 6              # frames sampled from the clip for scene analysis
VISIBILITY_THRESHOLD = 0.30  # CLIP prob for a source to count as "visible on screen"

# Stage 3 — speech recognition (Whisper)
WHISPER_MODEL = "base"       # tiny | base | small | medium | large-v3
WHISPER_COMPUTE = "int8"     # "int8" on cpu, "float16" on cuda
# Whether to run Whisper. The transcript feeds two things and only two: the Stage 7
# reference (so the judge neither credits nor penalises speech, which captions already
# cover) and SPEECH_CONTEXT below. It is deliberately kept OUT of the depiction prompt,
# where it leaked dialogue into pictures ("Hiccup on the phone, slow down, how many").
TRANSCRIBE = True
# Speech as gate context. For each sound with speech within a few seconds of it, the VLM
# is asked one text-only forced choice: are the people on the soundtrack reacting to
# this sound? A yes makes the sound (a) first in line when more sounds overlap than the
# panel can carry and (b) shown even if its confidence was marginal -- confidence
# measures loudness, and "did you hear that?" about a quiet sound is better evidence
# that it matters. The transcript never reaches a picture, and this never overrides
# visibility: a siren that is on screen is not shown however much people talk about it.
SPEECH_CONTEXT = True
# When the detector gives two labels to one acoustic event (same start and end, neither
# a kind of the other -- Sheep and Baby cry, Fire and Water), the frames from that
# moment choose between them. The video may pick among what the audio proposed; it may
# never add a label the audio did not hear, and if it cannot tell, both stay.
DISAMBIGUATE = True
# BEATs scores a 2 s window every 0.25 s, so a single-window blip is a 0.25 s span.
# Two consecutive windows drops those. Not more: a phone ring clears the display bar
# for only 0.75 s, and at 1.0 s it vanished.
AED_MIN_DUR = 0.5

# Stage 4 — audio event detection (PANNs CNN14 SED)
# "beats" = BEATs iter3+ fine-tuned on AudioSet-2M (Microsoft, 2022; 0.486 mAP), scored
# over a 2 s sliding window. Anything else = PANNs CNN14 (2019; 0.431 mAP), framewise.
# Same 527 labels either way. Switched 2026-09-12 after a demo review found eleven of
# fourteen visible errors were PANNs mislabelling the sound (siren -> "truck horn",
# engine rotor -> "printer", crying baby -> "sheep") and Stage 5 drawing it faithfully.
AED_MODEL = "beats"
# Onset by prefix-silencing occlusion on BEATs (beats_infer.occlusion_onset), inside the
# first window that fired: silence the opening t seconds for 26 values of t, and the
# onset is where the class starts losing evidence. The sliding-window stamp is within
# ~1 s and its sign depends on how the sound starts; this is class-conditional and within
# 80 ms for abrupt sounds. (A CAM on BEATs' tokens was tried first and cannot work: the
# model was trained only through the token mean, so its tokens are not local.)
ONSET_CAM = True
# 2026-09-24, docs/onset_timing.md. Adam watched the rendered clips and found the picture out of
# step with its sound; the render and the panel are exact, so the error is the onset. Every stage
# that can move a span moves it EARLIER and none ever moves one later, while the metric forgives
# late twice as much as early. The rule all three reviewers signed: a later stage may sharpen an
# onset inside its own evidence window, extend an end, or merge spans, but may never produce a
# start earlier than the anchor it was given. Off by default until the DEV numbers are in.
ONSET_MONOTONE = False
# "min" (as shipped), "bounded" or "beats": how the FlexSED twin rule may move a BEATs start.
UNION_START = "min"
# "earliest" (as shipped) or "strongest": where a burst of chained firings is deemed to start.
MERGE_START = "earliest"
# End of a span (2026-09-24). Hysteresis ends a span the moment the score dips below its bar, and a
# sustained sound dips: measured on the rendered panel the picture leaves 2.30 s before the sound
# stops. With this set, the span extends through any later stretch at or above this absolute score,
# tolerating silences shorter than AED_RELEASE_GAP. None keeps the shipped behaviour.
AED_RELEASE = None
AED_RELEASE_GAP = 1.0
# Hysteresis: a sound must reach DISPLAY_THRESHOLD to count, and then extends through any
# contiguous stretch above this fraction of it. Standard SED post-processing; it is what
# lets an approaching helicopter start when the ear hears it, not when it gets loud.
# Amendment 3 (2026-09-21, bug fix, docs/prereg_v4.md): the code halved AED_THRESHOLD (0.05) instead
# of DISPLAY_THRESHOLD, so spans grew from a 0.025 noise floor, started near 0 s and were then
# cut by the 8-s cap before the loud part. Now one pass at DISPLAY_THRESHOLD/2: a span is every
# stretch >= 0.175 (its confidence = its peak); peaks >= DISPLAY_THRESHOLD are shown, peaks in
# [0.175, 0.35) are the speech-rescue band; nothing below 0.175 is kept (it was never shown).
AED_HYSTERESIS = 1.0
AED_THRESHOLD = 0.175        # = 0.5 * DISPLAY_THRESHOLD (keep in sync)
# AED_MIN_DUR is set further down, next to the detector choice.
AED_PLOT_TOP_K = 15          # classes shown in the timeline plot

# Stage 5 — cross-modal semantic analysis (the gate)  [stub in skeleton]
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or Qwen3; or a hosted API
USE_LOCALIZATION = False     # feed an on/off-screen localization signal into the gate
GATE_ENABLED = True          # v2-(a): gate on the CLIP seen/not-seen check (Stage 2)

# 0.12 was the PANNs sweep optimum. BEATs is calibrated differently: over the eight demo
# clips every real sound scored 0.30 or above (Glass 0.94, Fire alarm 0.70, Telephone
# 0.58, Crowd 0.45) and every phantom 0.28 or below (Basketball bounce 0.28, Roaring
# cats 0.19, Keys jangling 0.17). Then 0.40 lost a real crow at 0.37 and a real insect at
# 0.39 on the hard set: real sounds and phantoms overlap in 0.30-0.42 and no bar is clean.
# 0.35, erring towards showing, since a missed real sound is the worse error for a deaf
# viewer. Sound-level gold labels are what would settle this properly. The sweep over the 256 tagged clips
# (job 28902699, benchmark/eval_results_owlv2_beats.json) puts the rule-only gate's F1 on
# a plateau from 0.30 to 0.35 (44.3-45.8%); recall falls off above 0.35 (77% -> 64% at
# 0.40). Note the sweep scores Stage 2's concept list, not the per-sound VLM check, so
# its precision (31%) is the concept list's blindness, which the VLM step exists to fix.
DISPLAY_THRESHOLD = 0.35
# Asymmetric bar for the OFF-screen (augment) claim. The PANNs sweep said raising it
# above DISPLAY_THRESHOLD hurts -- faint off-screen sounds are the true positives
# (distant == unseen). Kept as a knob, set equal.
AUGMENT_THRESHOLD = 0.35

# Stage 6 — visual augmentation generation
GEN_BACKEND = "diffusion"    # diffusion | retrieve (Openverse) | placeholder
# Generation, not retrieval, despite retrieval scoring better in the ablation: a
# retrieved image only exists if somebody photographed that sound source and licensed
# it, which is a hard ceiling for arbitrary future sounds. Generation trades score for
# coverage, and the pictogram prompt (stage 6, icon_prompt) is the attempt to win the
# score back by asking for something unambiguous rather than something photographic.
# Presentation preset (DHH evidence: users want choices; see notes "Rendering improvements"):
#   full    = stable per-sound slots with imagery, opacity weighted by confidence
#   minimal = same slots, compact label chips instead of imagery (icon set is a TODO)
#   off     = no augmentation panel (control condition; output = original video)
RENDER_MODE = "full"
# The panel is read at a glance beside a video the viewer is already watching, so every
# element has to earn its place against the divided-attention cost (see the literature
# review). The word under each picture restates what the picture shows, and the drawn
# sound symbol was a workaround from when the images were unreadable pictograms; with a
# generator that depicts the action, neither pays for itself.
# Stage 5 asks a VLM what to depict, from a text description of the scene, instead of
# consulting hand-written setting/phrasing tables. The transcript is deliberately NOT
# part of that prompt: it was, and dialogue leaked into the pictures -- a Hiccup became
# "Hiccup on the phone, slow down, how many". The tables
# broke where presets always break: a motorcycle POV shot scored "vehicle interior" at
# 0.98 -- true of the camera, false of the scene -- and every depiction came out
# "indoors" over an outdoor street.
DEPICTION_REASONING = True
SHOW_LABELS = False
# Pictures fade with detector confidence (alpha 0.45-1.0). use_shipped() switches it off: every shown picture is
# drawn at full opacity, as the glance test rated it (panel 3, topic 2).
CONFIDENCE_FADE = True
# Show a picture only if its detector confidence is >= this (None = off). 0.40 chosen on DEV and confirmed on the held-out
# 415 AudioSet-Strong clips (docs/prereg_v4.md, 2026-09-28); use_shipped() switches it on.
PICTURE_MIN_CONF = None
# A picture may stay on screen at most this long after its sound's real end (the 1.5-s minimum only lengthens short
# sounds, and never by more than this). None = off. use_shipped() sets 1.0 s and joins repeats if the gap <= 1.5 s
# (Adam, 28 Sept 2026).
MAX_AFTER_END = None
# Stage 5 family rule: a visible sound silences a related one only in the right direction (see reason.py). The scored
# runs had it off; use_shipped() switches it on (Adam, 28 Sept 2026).
KINSHIP_DIRECTED = False
# Print the generator's prompt under each picture. Debugging only -- it is how a bad
# picture gets traced to the words that produced it. Off for anything a viewer sees.
SHOW_PROMPT = False
# Under the panel, every raw detection active at that moment with its confidence and
# the gate's verdict. Debugging only: it separates "the detector heard the wrong thing"
# from "the gate did the wrong thing with the right one", which are different repairs.
SHOW_DEBUG_SOUNDS = False
SHOW_SOUND_GLYPH = False
# Stage 5 asks the VLM, on the frames spanning EACH sound, to name the thing making it,
# and stays silent when it can. This replaces the ~30-concept visibility table in
# Stage 2, which could only ever mark a sound visible if somebody had written that
# sound into it -- so laughter, footsteps, a telephone and an owl all got a picture
# beside a video that was already showing the source. Off-screen is the whole premise
# of the system, so this check is what makes the output honest.
VLM_VISIBILITY = True
# A sound longer than this is judged for visibility in stretches of this length, each
# on its own frames; the picture is shown only for the stretches where the source is
# off screen. A siren over a whole clip is off screen while the car approaches and on
# screen once it arrives.
VISIBILITY_STRETCH = 5.0
# How the three visibility votes per stretch combine: "majority" (v3) or "unanimous".
# Set once on the dev split by benchmark/gate_dev_sweep.py, never on test.
VISIBILITY_RULE = "majority"
# When the detector gives no sub-label, the frames from the sound's own moment are asked
# what KIND of that sound it is -- a crowd chanting, not a crowd cheering -- with the
# label fixed in the question and "unknown" as an answer. The answer must still name the
# sound or it is discarded, so the frames can qualify a label but never replace it.
KIND_FROM_FRAMES = True
# 2026-09-24 picture quality (docs/NIGHT_REPORT_2026-09-24.md). Both off = as shipped.
# KIND_ALWAYS: ask the frames which KIND of source it is even when the detector gave a sub-label.
# DEPICT_V2: the object that makes the sound first, never an abstract subject, homonyms qualified,
# and the kind word read from the frames is not stripped as if it were the place.
KIND_ALWAYS = False
DEPICT_V2 = False
# PICTURE_V3 (2026-09-24, picture panel of five, docs/picture_panel_round2.md). Draw the most specific
# sound the detector really heard (labels.choose_source), with a depiction prompt that asks for the
# whole recognisable source caught making THAT sound, no place in it, and a negative prompt for the
# generator. The gate is untouched. The two numbers are fixed a priori, never tuned on the 33 bench
# sounds: a child must reach half the family's peak in the burst, and two siblings within 80% of each
# other are not guessed between (their common parent is drawn).
PICTURE_V3 = False
SOURCE_REL_FLOOR = 0.5
SOURCE_TIE_MARGIN = 0.8
# V3.1 (docs/panel_2026-09-25_round2.md): the scene may add up to two qualifier words to the heard
# source ("car door", "farm vehicle"), never a new noun; a list guard refuses kinds the audio never
# established. Needs PICTURE_V3. Off until rated on a fresh set.
PICTURE_SCENE = False
# GP-4 3(b): refuse a scene qualifier that names a maker from another ontology branch, and a depiction that
# drops the heard head word (slice B, prereg amendment 2b). Screened on picture-DEV; off until frozen.
PICTURE_SCENE_GUARD2 = False
# Week plan C.1 (docs/WEEK_PLAN_2026-09-26.md): the frozen final picture setup inside stage 6 (templates, burst cards,
# rules tail, blank guard). Display only, for demo videos; never a scored row. See use_final_pictures().
PICTURE_FINAL = False
# 2026-09-28: check each final picture before it is shown (src/stage6_visual_augmentation/verify.py): a shuffled
# multiple-choice question to VLM_MODEL (the intended thing vs its known look-alikes) and an OCR text check; up to
# PICTURE_VERIFY_TRIES draws (new seed each, a clearer fixed rewrite from try 3), then a word card. Needs
# PICTURE_FINAL, and the VLM next to the generator (one H200, or two cards). Off by default; use_shipped() switches
# it on (Adam, 28 Sept 2026, after validation).
PICTURE_VERIFY = False
PICTURE_VERIFY_TRIES = 5      # Adam, 28 Sept: 5 tries, each refined by the last refusal, then a word card
# 2026-09-28 (Adam's rule, docs/freeze_picture_setup_2026-09-25.md, amendment "after the sitting was cancelled"): an
# action sound (laughter, applause, run, typing, honk) is drawn as the OBJECT that makes it; several possible makers ->
# the VLM picks from the sound's frames, else the maker the subject names, else a fixed default (reason.MAKERS).
PICTURE_MAKER = False
# 2026-09-28 (Adam: the hand-written rewrite wording is "too specific"): the clearer wording used by the redraw loop is
# written by the VLM (text only) from the maker object and the sound, with list guards; fallback the plain subject
# (verify.describe). Off = the hand-written AMBIGUOUS rewrites.
PICTURE_LOOK_VLM = False
# the checker's look-alike options for the AMBIGUOUS entries written by the VLM too (verify.lookalikes); adopted only if
# scripts/verify_validate.py still meets the four validation targets
PICTURE_LOOKALIKE_VLM = False
# 2026-09-28 (Adam: a GENERIC method instead of the per-word table): PICTURE_SENSE draws from a 4-slot form the VLM fills
# from the label, its ontology path and its official AudioSet description, adds negative words mined from 4 test
# pictures of the plain subject, and checks with the mined look-alikes (src/stage6_visual_augmentation/sense.py; the
# AMBIGUOUS table is not used). Off until docs/picture_sense_test_2026-09-28.md decides.
PICTURE_SENSE = False
# The place may veto a sound that plainly does not belong in it (a horse at a quarry
# blast, an ice-cream truck on a train platform), never a sound people are reacting to
# and never one the detector is more than 90% sure of. An assumption made at run time
# from the frames, not from a list; bounded because a surprising confident sound is
# exactly what a hearing viewer reacts to.
# OFF (2026-09-13). Measured on the demo sets it fired four times: horse at a quarry and
# ice-cream truck at a station (right), a dog in a parking lot and an aircraft over a
# railway (wrong) -- a rule that deletes a real sound half the time it fires is worse
# than the phantoms it catches. Adam's original "no assumptions" stands; the code stays
# for the ablation.
PLAUSIBILITY_CHECK = False
# A corroboration band (0.35-0.45, shown only with a second signal) was tried and cut
# the wrong way: the one real sound in the train clip, the horn at 0.42, had nothing
# behind it and was dropped, while the phantom Bird at 0.38 was "corroborated" by its own
# sub-label Crow -- from the same detector, and just as wrong. Phantoms come in families.
# A flat bar at 0.40 keeps the horn and clears every phantom seen on the demo sets
# (Footsteps 0.31, Gush 0.30, Bird 0.38, Crow 0.37, Owl 0.32); the one it does not
# clear is a Sheep at 0.41 in a jungle. Set to 0 to disable.
CORROBORATE_BELOW = 0.0
# Two sounds get one picture when they are one source. That is decided in two steps.
#
# First the AudioSet ontology -- the taxonomy PANNs' own label space comes from, vendored
# as src/audioset_parents.json. If one detected label is a more specific kind of another
# detected label (Giggle under Laughter, Hoot under Owl), they are the same source and no
# threshold is involved. It gets 14/14 on every pair these demos produced, including the
# ones that must NOT merge: Air horn against Siren, Dog against Sheep.
#
# Only then does depiction similarity apply, for paraphrases the ontology cannot see
# (two unrelated labels that happened to be drawn the same way). The bar is high because
# this is now the secondary signal, not the primary one, and because it is fragile in a
# way the ontology is not: it has to be recalibrated whenever the depiction prompt
# changes. At 0.70, tuned on scene-based depictions, the shorter place-based depictions
# that replaced them shared so much wording that "Dog barking on roadside" and "Air horn
# blaring on roadside" scored 0.72 and merged. 0.80 is above everything a wrong pair has
# scored across four runs. Pairs above DEDUP_REPORT are logged but not merged, so the
# evidence keeps arriving.
DEDUP_SIM = 0.80
DEDUP_REPORT = 0.45
# A picture appears when its sound is heard and leaves when it stops, but a detected
# span can be 0.2 s (AED_MIN_DUR) and a picture flashed for 0.2 s costs more attention
# than it returns. MIN_DWELL is the floor on how long a picture stays; MERGE_GAP joins
# two bursts of the same sound into one appearance instead of a flicker.
MIN_DWELL = 1.5
MERGE_GAP = 2.0   # amendment 3 (2026-09-21): measured from the sound's real end (was the stretched end,
                  # so barks 2.2 s apart chained into one picture); 2 s = the annotation rule
                  # "one row per continuous sound, split at pauses > 2 s" (docs/metric_per_sound.md)
# Rows in the panel = the most sounds heard AT ONCE, capped here. Three sounds that
# never overlap share one full-size cell in turn rather than splitting the panel into
# thin strips that are empty most of the time.
MAX_SLOTS = 3
# Chosen from a 3-model x 4-prompt grid (scripts/icon_grid.py; sheets in
# data/output/icon_grid). SDXL base was the WORST of the three for this job: asked
# for a pictogram of a dog it returned paw-print wallpaper and a 12-panel contact
# sheet; for a fire engine, a garage door. Turbo returned a clean readable subject
# on white for every subject tried, and is ~7x cheaper per image.
# PixArt-Sigma beat SDXL-Turbo and FLUX.1-schnell head to head on the six benchmark
# sounds (6/6 vs 6/6-with-3-losses vs 3/6): it was the only one that showed the ACTION
# -- an open beak for chirping -- and FLUX returned a blank white image for "rain
# falling". Ungated, and ~10x faster than FLUX, which needs CPU offload on a 23 GB card.
# FLUX.1-schnell ships (Adam, 2026-09-13: "in general flux is better than pixart"), after
# a side-by-side on video: a real red alarm bell where PixArt drew a red box, a police
# car with a lightbar where PixArt drew a dark sedan, a legible clock face. It is the
# proposal's named model. PixArt-Sigma stays selectable for the ablation.
GEN_MODEL = "black-forest-labs/FLUX.1-schnell"
GEN_MODEL_PIXART = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
# 768 for FLUX under sequential CPU offload (~25 s/image; 1024 would be ~100 s). The
# panel is 720 px, so nothing is lost.
RESOLUTION = (768, 768)
PANEL_SIZE = 720             # side-by-side augmentation panel size (px)
FPS = 25

# Stage 7 — automatic evaluation protocol (proposal sec 6.1)
# The judge MUST be a different model from the describing VLM: a model scoring its
# own descriptions measures self-consistency, not quality. Mistral-7B-Instruct is a
# different family from Qwen2.5-VL, text-only, and fits alongside it on one GPU.
# 2026-09-24: Gemma-4-31B is the only judge that passes both trust checks against Adam's labels
# (tracks his viewer cost AND separates clips with a known-wrong picture, 139 clips); Mistral-7B fails
# the second even on 139 (docs/NIGHT_REPORT_2026-09-24.md, Result 7). Adam: "if Gemma is stronger use it".
JUDGE_MODEL = "google/gemma-4-31B-it"

# ---------------------------------------------------------------------------------------
# v4 (docs/prereg_v4.md): the SOTA configuration, applied one stage at a time so that each
# swap is attributable. Nothing above changes -- v3 stays reproducible from the defaults;
# a job calls config.use_v4("4"), ("45"), ("456") or ("23456") before running.
SAM3_MODEL = "facebook/sam3"
SAM3_THRESHOLD = 0.5          # SAM 3's own default presence bar; not tuned on our clips
V4 = {
    "2": {"VIDEO_BACKEND": "sam3"},
    "3": {"WHISPER_MODEL": "ibm-granite/granite-speech-4.1-2b"},
    # no stage-4 swap has passed its bars (FLAM x2: docs/prereg_v4.md; PretrainedSED:
    # docs/prereg_psed.md), so "4" is not part of the cumulative rows; kept for a future pass
    "4": {"AED_MODEL": "psed", "ONSET_CAM": False, "PSED_BAR": 0.15},   # frame-level already; bar from the AudioSet-Strong calibration
    # thinking off: the DCASE check (docs/prereg_qwen38_visibility.md) shows the thinking arm
    # no better on visibility and ~3x slower; the pre-registered rule keeps the faster arm
    "5": {"VLM_MODEL": "Qwen/Qwen3.8-27B", "VLM_THINKING": False},
    "6": {"GEN_MODEL": "Qwen/Qwen-Image-2512", "RESOLUTION": (1024, 1024)},
    "7": {"JUDGE_MODEL": "google/gemma-4-31B-it"},
    # "8": the detector-neutral label filter (src/labels.py) and PSED's own operating point
    # (span-level F1 on the AudioSet-Strong calibration set, benchmark/psed_f1_bar.json), for
    # the fair re-test v4ab2 / v4b2 (docs/prereg_v4.md)
    "8": {"LABEL_FILTER": "branch"},
    # "9": the depictable allow-list by ontology branch + an 8-s picture cap (v4ab3 / v4b3, the
    # ten-Fable panel's decisive test, docs/prereg_v4.md)
    "9": {"LABEL_FILTER": "depictable", "MAX_SPAN": 8.0},
    # "0": the BEATs union with FlexSED (amendment 8), the adopted detector of row v4b6
    "0": {"FLEXSED_BAR": 0.8},
    # "1": PICTURE_V3 (2026-09-24, docs/picture_v3_prereg.md) -- the specific source, the v3 depiction
    # prompt and the scenery negative. Changes what is drawn, never whether (checked on DEV).
    "1": {"PICTURE_V3": True},
}
# Amendment 8 (2026-09-22): the second, open-vocabulary detector. 0 = off (every row up to v4b5);
# the adopted union row v4b6 sets 0.8, chosen on Adam's DEV half (onset-recall 0.45 -> 0.59 at
# 1.58 false labels per clip, where BEATs alone needs 3.05 to reach 0.57).
FLEXSED_BAR = 0.0
FLEXSED_FAMILY_BARS = None    # amendment 11: path to the per-family bars fitted on the AudioSet calibration set
PANNS_VETO = 0.0              # amendment 16: a third detector settles spans only FlexSED raised
BEATS_SELF_VETO = 0.0         # round 4: BEATs' own clip-max settles them instead (0.1218 in use_shipped; PANNs off)
FLEXSED_VETO = 0.0            # amendment 10: drop a label the second detector never hears in the clip
FLEXSED_CORROB = None         # amendment 22: (beats_min, panns_min, window_s) -- FlexSED-only spans need a nearby second detector
UNION_WEAK_TWIN = "absorb"      # amendment 22 cell F: "ignore" = a sub-display BEATs twin no longer swallows a FlexSED span
BEATS_LOWBAND_CORROB = None   # amendment 22 tier 3: (flexsed_min, panns_min, window_s) -- weak BEATs spans promoted when corroborated
MAX_SPAN = None               # seconds; None = no cap (v4ab3/v4b3 use 8.0)
# Round 13 detector push (docs/prereg_round13_detector_push.md; DEV-developed, all OFF = the scored/shipped behaviour).
TWIN_MAX = False              # R13-1: a BEATs span that absorbed a same-family FlexSED span keeps the stronger side's
                              # bar-normalised evidence (displayable if BEATs >= 0.35 OR FlexSED >= its 0.8 bar)
MIRROR_VETO = None            # R13-2: b; drop a BEATs-only span where FlexSED's top query is another family >= b ...
MIRROR_OWN_MAX = 0.4          #        ... and the span's own family's FlexSED score is < this
IMPULSE_MIN_SPAN = None       # R13-5: seconds (0.2); FlexSED min span for the impulsive queries below (else AED_MIN_DUR)
# the impulsive families, fixed by physics; FlexSED has queries for Gunshot, Gasp, Hammer, Explosion, Knock (no
# Whack/Clang/Slam/Bang query; "Slam" is not the Door family, so Door queries are not affected)
IMPULSE_FAMILIES = ("Gunshot", "Gunshot, gunfire", "Gasp", "Whack, thwack", "Clang", "Hammer", "Explosion", "Slam",
                    "Knock", "Bang")
RETRIGGER = None              # R13-6: (gap_s, flexsed_low, beats_low) = (1.5, 0.4, 0.175): a picture never bridges a
                              # stretch >= gap_s where its family has no evidence (a new onset gets a new appearance)
RETRIGGER_RAW = False         # R13-6 (Crowd mechanism): consolidate_families does not chain a later firing of a different
                              # sound (not same label / ancestor / descendant) into a burst; the boundary is a break
LISTENER_RESCUE = False       # R13-3: a Qwen3-Omni listener (cached per span, LISTENER_CACHE) rescues (a) FlexSED 0.4-runs
                              # with peak in [LISTENER_LO, FLEXSED_BAR) and (b) FlexSED spans the PANNs clip veto drops,
                              # when its yes-no logit > LISTENER_TH; a span with no cached score is not rescued
LISTENER_CACHE = None         # path of the listener cache (benchmark/gold/dev_listener.json format)
LISTENER_LO = 0.4
LISTENER_TH = 0.0
LISTENER_RULE = None          # amendment A: "V1" | "V2" | "V3" | "V4" | "V12" -- (a) and (b) use that rule's accept flag from
LISTENER_VCACHE = None        # this variants cache (benchmark/gold/listener_variants.py) instead of score > LISTENER_TH
LISTENER_BEATS_TH = None      # R13-3 (c): also a short BEATs run (peak 0.175-0.35, not covered) at the display bar if score > this
LABEL_FILTER = "lists"      # "lists" (v1-v4ab hand lists) | "branch" (speech, music, environment branch only)


def use_v4(stages: str = "23456") -> dict:
    """Apply the v4 settings for these stages (a string of digits); returns what changed."""
    import sys
    me = sys.modules[__name__]
    changed = {}
    for s in stages:
        for k, v in V4[s].items():
            changed[k] = (getattr(me, k, None), v)
            setattr(me, k, v)
    return changed


def use_scored() -> dict:
    """Exactly what the final TEST table (amendment 21, tag test_final_v33) ran with: slurm/job_protocol.sh
    sets VIDEO_BACKEND owlv2, then V4=590, FBAR=0.8, VETO=0.3, PVETO=0.05, MONO=1, MAXSPAN=none. Stage "9"
    sets MAX_SPAN 8.0, so the cap is switched off AFTER use_v4; ONSET_CAM stays on (the onset clamp needs the
    refinement step). Pictures FLUX.1-schnell, as rendered in every scored row (the per-sound metric does not look at
    the picture, so the scored numbers hold for either generator)."""
    import sys
    me = sys.modules[__name__]
    changed = use_v4("590")
    for k, v in (("VIDEO_BACKEND", "owlv2"), ("FLEXSED_BAR", 0.8), ("FLEXSED_VETO", 0.3), ("PANNS_VETO", 0.05),
                 ("ONSET_MONOTONE", True), ("ONSET_CAM", True), ("MAX_SPAN", None)):
        changed[k] = (getattr(me, k, None), v)
        setattr(me, k, v)
    return changed


def use_shipped() -> dict:
    """The shipped system (Adam, 28 Sept 2026: Qwen-Image is the picture model; FLUX is retired): the scored stack
    (use_scored) with the frozen final picture setup -- Qwen-Image-2512, V3.1 subject with guard 2, templates, cards,
    rules tail -- and pictures shown at full opacity (CONFIDENCE_FADE off: the glance test rated full-opacity pictures)."""
    import sys
    me = sys.modules[__name__]
    changed = use_scored()
    changed.update(use_final_pictures(2))
    changed["CONFIDENCE_FADE"] = (getattr(me, "CONFIDENCE_FADE", True), False)
    setattr(me, "CONFIDENCE_FADE", False)
    changed["PICTURE_MIN_CONF"] = (getattr(me, "PICTURE_MIN_CONF", None), 0.40)
    setattr(me, "PICTURE_MIN_CONF", 0.40)
    for k, v in (("MAX_AFTER_END", 1.0), ("MERGE_GAP", 1.5), ("KINSHIP_DIRECTED", True),
                 ("BEATS_SELF_VETO", 0.1218), ("PANNS_VETO", 0.0),
                 # picture check-and-redraw (Adam, 28 Sept: on after validation; round 1: 7/7 named bad, 32/33 wrong,
                 # 2/68 good rejected, 115/115 same on reshuffle; round 2, all generic options + trumpet-horn confusion
                 # + umbrella rain template: 7/7, 33/33, 3/68, 113/115; src/stage6_visual_augmentation/verify.py).
                 # Needs the VLM next to the generator (one H200).
                 ("PICTURE_VERIFY", True),
                 # Adam's maker rule (28 Sept, after the sitting was cancelled): an action sound is drawn as the object
                 # that makes it. Redraw of the 5 changed inspector pictures: all pass try 1, all fine by eye.
                 # PICTURE_LOOK_VLM stays off: its redraws were worse by eye (2 fire alarms + steam fell to word cards).
                 ("PICTURE_MAKER", True)):
        changed[k] = (getattr(me, k, None), v)
        setattr(me, k, v)
    return changed


def use_final_pictures(level: int) -> dict:
    """Demo videos only (week plan C.1, signed 6/6). Level 1 = Qwen-Image-2512 with the shipped subjects (arm N0 of
    the blind round 2); NOT triggered: the N0 rule-2 pass found one false message (phone drawn as a desk bell).
    Level 2 = the frozen final setup (V3.1 subject with guard 2 + templates + cards + rules tail), to be switched on
    only if the blind confirmation sitting passes (CI > 0, zero false messages). Call after use_shipped().
    In the pipeline the V3.1 guard runs with no raw firings (the strict side, stage-5 note)."""
    import sys
    me = sys.modules[__name__]
    sets = [("GEN_MODEL", "Qwen/Qwen-Image-2512"), ("RESOLUTION", (1024, 1024))]
    if level >= 2:
        sets += [("PICTURE_V3", True), ("PICTURE_SCENE", True), ("PICTURE_SCENE_GUARD2", True), ("PICTURE_FINAL", True)]
    changed = {}
    for k, v in sets:
        changed[k] = (getattr(me, k, None), v)
        setattr(me, k, v)
    return changed

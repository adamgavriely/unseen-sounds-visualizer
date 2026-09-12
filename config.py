"""Central defaults for the pipeline. Override per-run via main.py CLI flags.

Model names below are the *intended* backends; in the current skeleton most
stages run as stubs (see each stage module) and do not load these models yet.
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
AED_THRESHOLD = 0.05         # LOW on purpose: detect-everything-first (incl. faint background)
# AED_MIN_DUR is set further down, next to the detector choice.
AED_PLOT_TOP_K = 15          # classes shown in the timeline plot

# Stage 5 — cross-modal semantic analysis (the gate)  [stub in skeleton]
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or Qwen3; or a hosted API
USE_LOCALIZATION = False     # feed an on/off-screen localization signal into the gate
GATE_ENABLED = True          # v2-(a): gate on the CLIP seen/not-seen check (Stage 2)

# 0.12 was the PANNs sweep optimum. BEATs is calibrated differently: over the eight demo
# clips every real sound scored 0.30 or above (Glass 0.94, Fire alarm 0.70, Telephone
# 0.58, Crowd 0.45) and every phantom 0.28 or below (Basketball bounce 0.28, Roaring
# cats 0.19, Keys jangling 0.17). 0.30 is that gap; the full sweep over the 274 tagged
# clips (benchmark/evaluate.py) should confirm or move it.
DISPLAY_THRESHOLD = 0.30
# Asymmetric bar for the OFF-screen (augment) claim. The PANNs sweep said raising it
# above DISPLAY_THRESHOLD hurts -- faint off-screen sounds are the true positives
# (distant == unseen). Kept as a knob, set equal.
AUGMENT_THRESHOLD = 0.30

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
# "Visible" means the ACTION is visible, not just the object. A baby in its mother's
# arms and a fire-alarm pull station are both "the thing making the sound" and tell a
# deaf viewer nothing about it; a woman laughing with her mouth open does. Second
# frames question after the naming step; both orderings must agree before a sound is
# silenced on visibility.
EVENT_VISIBLE = True
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
MERGE_GAP = 0.8
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
GEN_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"   # for the v2 diffusion backend
# Side by side on the pipeline's own twelve prompts (2026-09-12, data/output/gen_compare):
# FLUX.1-schnell is cleaner and more literal -- a real gun with a muzzle flash, a legible
# clock face, an actual train wheel, clearer clapping hands -- while PixArt is more
# dramatic on impacts (both glass images, the crying baby) and garbles detail (clock
# numerals, wheels). FLUX needs sequential CPU offload on a 23 GB card, ~25 s per image
# against ~2 s. Selectable here; the demo job renders with FLUX for Adam to judge on
# video, the protocol ships whichever he picks.
GEN_MODEL_FLUX = "black-forest-labs/FLUX.1-schnell"
RESOLUTION = (1024, 1024)    # augmentation image size
PANEL_SIZE = 720             # side-by-side augmentation panel size (px)
FPS = 25

# Stage 7 — automatic evaluation protocol (proposal sec 6.1)
# The judge MUST be a different model from the describing VLM: a model scoring its
# own descriptions measures self-consistency, not quality. Mistral-7B-Instruct is a
# different family from Qwen2.5-VL, text-only, and fits alongside it on one GPU.
JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

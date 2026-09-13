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
# When the detector gives no sub-label, the frames from the sound's own moment are asked
# what KIND of that sound it is -- a crowd chanting, not a crowd cheering -- with the
# label fixed in the question and "unknown" as an answer. The answer must still name the
# sound or it is discarded, so the frames can qualify a label but never replace it.
KIND_FROM_FRAMES = True
# The place may veto a sound that plainly does not belong in it (a horse at a quarry
# blast, an ice-cream truck on a train platform), never a sound people are reacting to
# and never one the detector is more than 90% sure of. An assumption made at run time
# from the frames, not from a list; bounded because a surprising confident sound is
# exactly what a hearing viewer reacts to.
PLAUSIBILITY_CHECK = True
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
JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

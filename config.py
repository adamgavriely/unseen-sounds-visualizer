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

# Stage 4 — audio event detection (PANNs CNN14 SED)
AED_MODEL = "PANNs/Cnn14_DecisionLevelMax"    # -> BEATs for quality later
AED_THRESHOLD = 0.05         # LOW on purpose: detect-everything-first (incl. faint background)
AED_MIN_DUR = 0.2            # min span length (s) to count as an event
AED_PLOT_TOP_K = 15          # classes shown in the timeline plot

# Stage 5 — cross-modal semantic analysis (the gate)  [stub in skeleton]
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or Qwen3; or a hosted API
USE_LOCALIZATION = False     # feed an on/off-screen localization signal into the gate
GATE_ENABLED = True          # v2-(a): gate on the CLIP seen/not-seen check (Stage 2)

# Stage 5 display threshold — min confidence for a sound to get an image.
# 0.12 chosen from the benchmark sweep (best F1/precision balance; see eval_results.json)
DISPLAY_THRESHOLD = 0.12
# Asymmetric bar for the OFF-screen (augment) claim. Benchmark sweep verdict:
# raising it above DISPLAY_THRESHOLD *hurts* — faint off-screen sounds are the
# true positives (distant == unseen). Kept as a knob, set equal (F1 55.2%).
AUGMENT_THRESHOLD = 0.12

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
SHOW_SOUND_GLYPH = False
# Stage 5 asks the VLM, on the frames spanning EACH sound, to name the thing making it,
# and stays silent when it can. This replaces the ~30-concept visibility table in
# Stage 2, which could only ever mark a sound visible if somebody had written that
# sound into it -- so laughter, footsteps, a telephone and an owl all got a picture
# beside a video that was already showing the source. Off-screen is the whole premise
# of the system, so this check is what makes the output honest.
VLM_VISIBILITY = True
# Two sounds get one picture when they would be drawn the same way. Similarity is
# measured between the DEPICTIONS with SigLIP's text tower, never between the labels:
# single words embed too tightly to separate (Dog/Cat 0.90 against Laughter/Snicker
# 0.80), while descriptive phrases separate usefully.
#
# The bar comes from three demo runs, not from taste. Measured on real pipeline output:
#   true duplicates      0.57 - 0.87   (Chuckle/Laughter 0.87, Owl/Hoot 0.82)
#   true non-duplicates  0.46 - 0.63   (Siren/Hoot 0.63, Dog/Laughter 0.59)
# 0.70 merges three of five duplicates with no false merges. The two misses sit at 0.57,
# inside the non-duplicate range, so no threshold reaches them.
#
# There was meant to be a model deciding the overlapping band. There is not, because it
# did not work: asked ~40 times across two framings it answered "different" to about 90%
# of pairs, including Owl against Hoot, and its few "same" answers were arbitrary enough
# to merge a barking dog into a laughing baby. Pairs above DEDUP_REPORT are logged but
# not merged, so evidence for moving the bar keeps arriving.
DEDUP_SIM = 0.70
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
RESOLUTION = (1024, 1024)    # augmentation image size
PANEL_SIZE = 720             # side-by-side augmentation panel size (px)
FPS = 25

# Stage 7 — automatic evaluation protocol (proposal sec 6.1)
# The judge MUST be a different model from the describing VLM: a model scoring its
# own descriptions measures self-consistency, not quality. Mistral-7B-Instruct is a
# different family from Qwen2.5-VL, text-only, and fits alongside it on one GPU.
JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

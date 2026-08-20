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

# Stage 2 — video understanding (lightweight CLIP visibility check)
VIDEO_MODEL = "openai/clip-vit-base-patch32"   # full VLM (Qwen2.5-VL) is a later upgrade
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
GEN_BACKEND = "retrieve"     # v1: retrieve (Openverse, free) | placeholder | diffusion (v2 TODO)
# Presentation preset (DHH evidence: users want choices; see notes "Rendering improvements"):
#   full    = stable per-sound slots with imagery, opacity weighted by confidence
#   minimal = same slots, compact label chips instead of imagery (icon set is a TODO)
#   off     = no augmentation panel (control condition; output = original video)
RENDER_MODE = "full"
GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"   # for the v2 diffusion backend
RESOLUTION = (1024, 1024)    # augmentation image size
PANEL_SIZE = 720             # side-by-side augmentation panel size (px)
FPS = 25

# Stage 7 — evaluation  [stub in skeleton]
JUDGE_MODEL = "hosted-or-local-LLM"

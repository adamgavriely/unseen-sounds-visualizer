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

# Stage 2 — video understanding (VLM)  [stub in skeleton]
VIDEO_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"   # 7B on the uni GPU
NUM_FRAMES = 8               # frames sampled from the clip for scene analysis

# Stage 3 — speech recognition (Whisper)
WHISPER_MODEL = "base"       # tiny | base | small | medium | large-v3
WHISPER_COMPUTE = "int8"     # "int8" on cpu, "float16" on cuda

# Stage 4 — audio event detection  [stub in skeleton]
AED_MODEL = "PANNs/Cnn14"    # -> BEATs for quality
AED_THRESHOLD = 0.3          # min confidence to keep a detected event

# Stage 5 — cross-modal semantic analysis (the gate)  [stub in skeleton]
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or Qwen3; or a hosted API
USE_LOCALIZATION = False     # feed an on/off-screen localization signal into the gate

# Stage 6 — visual augmentation generation  [stub in skeleton]
GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"   # or FLUX.1
RESOLUTION = (1024, 1024)    # generated augmentation image size
FPS = 25

# Stage 7 — evaluation  [stub in skeleton]
JUDGE_MODEL = "hosted-or-local-LLM"

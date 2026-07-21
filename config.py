"""Central defaults. Override per-run via main.py CLI flags."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INPUT_DIR = DATA / "input"
WORK_DIR = DATA / "work"
OUTPUT_DIR = DATA / "output"

# Stage 2 — ASR
WHISPER_MODEL = "base"        # tiny | base | small | medium | large-v3
WHISPER_DEVICE = "cpu"        # "cpu" (robust) or "cuda" (needs CUDA-enabled ctranslate2)
WHISPER_COMPUTE = "int8"      # "int8" on cpu, "float16" on cuda

# Stage 3 — LLM planner (Gemini)
PLANNER_MODEL = "gemini-flash-latest"   # stable alias for the current flash model
PLANNER_CONTEXT = 3          # how many previous segments to feed as context
PLANNER_CHUNK = 25           # segments planned per API call (batching for rate limits)

# Stage 4 — visualizer
RESOLUTION = (1280, 720)     # all frames rendered at this size

# Stage 5 — compositor
FPS = 25

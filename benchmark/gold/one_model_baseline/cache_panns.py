"""PANNs CNN14 frame probabilities for the clips the old cache (benchmark/gold/panns_fw) lacks; same call as that cache
(src.stage4_audio_event_detection._infer), written only to benchmark/gold/one_model_baseline/panns_cache.
    python benchmark/gold/one_model_baseline/cache_panns.py          # msproj; stems from missing_panns.txt"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.stage4_audio_event_detection import _infer

WAV = Path.home() / "MscProj" / "data" / "work" / "gold_wav_flat"
OUT = HERE / "panns_cache"
OUT.mkdir(exist_ok=True)
device = "cuda" if len(sys.argv) < 2 else sys.argv[1]
for st in (HERE / "missing_panns.txt").read_text().split():
    dst = OUT / f"{st}.npz"
    if dst.exists():
        continue
    fw, times, labels = _infer(WAV / f"{st}.wav", device)
    np.savez_compressed(dst, fw=np.asarray(fw, dtype=np.float32), times=np.asarray(times, dtype=np.float64), labels=np.array(labels))
    print(st, np.asarray(fw).shape, flush=True)
print("DONE")

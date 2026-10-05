"""M2D-strong probabilities for the clips the old cache (data/work/psed_ens_cache/M2D) lacks; same code path
(psed_infer.cache_clips, backbone "M2D"), written only to benchmark/gold/one_model_baseline/m2d_cache.
    python benchmark/gold/one_model_baseline/cache_m2d.py tg_d007 tg_d016 ...     (GPU)"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.stage4_audio_event_detection import psed_infer as P
CLIPS = _ROOT / "data" / "input" / "tagger_set"
def find(stem):
    p = CLIPS / f"{stem}.mp4"
    if p.exists():
        return p
    hits = sorted((_ROOT / "data" / "input").rglob(f"{stem}.mp4"))
    return hits[0] if hits else p


src = [find(s) for s in sys.argv[1:]]
print(" ".join(str(p) for p in src if p.parent != CLIPS), flush=True)
assert all(p.exists() for p in src), [str(p) for p in src if not p.exists()]
print("cached", P.cache_clips(src, out_dir=Path(__file__).resolve().parent / "m2d_cache", backbone="M2D"))

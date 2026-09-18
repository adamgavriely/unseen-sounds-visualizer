"""Cache SSLAM window scores for the calibration set and slice B (env sota, GPU)."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_detector_eval as E
from src.stage4_audio_event_detection.sslam_infer import cache_clips

for name in ("calib", "sliceB"):
    E.use_set(name)
    items = [E.VIDEOS / f"{c['id']}.mp4" for c in E.clips()]
    n = cache_clips(items, out_dir=E.WIN / "sslam")
    print(f"[sslam] {name}: {len(items)} clips, {n} scored now -> {E.WIN / 'sslam'}", flush=True)

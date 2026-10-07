"""Step 5: frame probabilities of a PretrainedSED ear (original or fine-tuned) on the 71 DEV clips (and optionally the
415 held-out clips for calibration). Cluster GPU, env ~/venvs/psed2, run from ~/MscProj_tg.

    python benchmark/gold/coverage/ear_cache.py NAME BACKBONE [STATE_DICT.pt] [--set dev|heldout]
-> scratch_cov/ears/NAME/<stem>.npz  (fw float16 [frames, 447], times, labels = AudioSet-Strong names)
"""
import json
import sys
from pathlib import Path

_ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
import numpy as np

from src.stage4_audio_event_detection import psed_infer as P

HERE = Path(__file__).resolve().parent


def sources(which):
    if which == "dev":
        vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text(encoding="utf-8"))
        stems = [x.strip() for x in (HERE / "dev71_stems.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
        return [_ROOT / vids[s] for s in stems]
    d = json.loads((HERE.parent / "audioset_heldout.json").read_text(encoding="utf-8"))
    return [(HERE.parent / c["src"]).resolve() for c in d["clips"]]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = "heldout" if "--set=heldout" in sys.argv or ("--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "heldout") else "dev"
    args = [a for a in args if a not in ("dev", "heldout")]
    name, backbone = args[0], args[1]
    model, names, _ = P.load_model("cuda", backbone)
    if len(args) > 2:
        import torch
        sd = torch.load(args[2], map_location="cuda")
        print("load", model.load_state_dict(sd.get("model", sd), strict=True), flush=True)
    model.eval()
    out = _ROOT / "scratch_cov" / "ears" / name / which
    out.mkdir(parents=True, exist_ok=True)
    import tempfile
    n = 0
    with tempfile.TemporaryDirectory() as td:
        for src in sources(which):
            dst = out / f"{src.stem}.npz"
            if dst.exists():
                continue
            fw, t = P.score_file(model, P.wav16(src, Path(td) / "a.wav"), "cuda")
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=t.astype(np.float32), labels=np.array(names))
            n += 1
    print(name, which, "new", n, "->", out)


if __name__ == "__main__":
    main()

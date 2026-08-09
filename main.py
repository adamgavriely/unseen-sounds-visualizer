"""Visual Augmentation of Audio Semantics for Accessibility - pipeline entry point.

Walking skeleton. Run:
    python main.py --input data/input/clip.mp4

Stages 1 (ffmpeg audio) and 3 (whisper ASR) are real; stages 2/4/5/6 are stubs
(see src/stage*/), so this runs end-to-end with no GPU or downloaded models and
writes inspectable artifacts under data/work/<stem>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from src import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to input video clip")
    ap.add_argument("--work-dir", default=str(config.WORK_DIR),
                    help="where to write intermediate artifacts")
    ap.add_argument("--whisper-model", default=config.WHISPER_MODEL,
                    help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default=config.DEVICE, help="cpu|cuda")
    ap.add_argument("--generator", default=config.GEN_BACKEND,
                    choices=["retrieve", "placeholder", "diffusion"],
                    help="how Stage 6 produces images (v1=retrieve)")
    args = ap.parse_args()

    # allow CLI overrides of the shared config
    config.WHISPER_MODEL = args.whisper_model
    config.DEVICE = args.device
    config.GEN_BACKEND = args.generator

    try:
        from dotenv import load_dotenv
        load_dotenv()  # optional: makes any API keys in .env available to stages
    except ImportError:
        pass

    pipeline.run(Path(args.input), work_root=Path(args.work_dir))


if __name__ == "__main__":
    main()

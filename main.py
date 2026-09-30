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
    ap.add_argument("--raw-config", action="store_true",
                    help="use the bare config.py defaults instead of the shipped system (config.use_shipped())")
    ap.add_argument("--listener-split", default=None,
                    help="name given to slurm/run_best.sh for this clip's folder: where the audio-LLM answers and DASM "
                         "scores of the shipped detector are (without them stage 4 stops)")
    ap.add_argument("--whisper-model", default=None, help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default=None, help="cpu|cuda")
    ap.add_argument("--generator", default=None, choices=["retrieve", "placeholder", "diffusion"],
                    help="how Stage 6 produces images")
    ap.add_argument("--video-backend", default=None, choices=["siglip", "owlv2", "vlm"],
                    help="Stage 2 object finder")
    ap.add_argument("--debug-panel", action="store_true",
                    help="print the phrase and every raw detection with the gate's verdict under the panel")
    args = ap.parse_args()

    # the shipped system by default (Qwen-Image pictures, scored detector stack, PANNs veto); CLI flags override it
    if not args.raw_config:
        config.use_shipped()
        if args.listener_split:
            config.set_listener_split(args.listener_split)
    for k, v in (("WHISPER_MODEL", args.whisper_model), ("DEVICE", args.device), ("GEN_BACKEND", args.generator),
                 ("VIDEO_BACKEND", args.video_backend)):
        if v is not None:
            setattr(config, k, v)
    if args.debug_panel:
        config.SHOW_PROMPT = True
        config.SHOW_DEBUG_SOUNDS = True

    try:
        from dotenv import load_dotenv
        load_dotenv()  # optional: makes any API keys in .env available to stages
    except ImportError:
        pass

    pipeline.run(Path(args.input), work_root=Path(args.work_dir))


if __name__ == "__main__":
    main()

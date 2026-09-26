"""B.4 of docs/panel_2026-09-26_plan.md: the gated text-tag arm, DERIVED from ours (no re-render, no new gate pass).

Copies each clip's augmentations.json + media.json from protocol_proposed_<src> into protocol_audio_caption_<new>, and
links protocol_proposed_<new> -> protocol_proposed_<src>, so the judge sees ours and the text tags with the same gate
decisions. Then prints, per clip, whether the text arm's spans (require_image=False, no row limit) equal ours' placed
spans (P3/P4 guard); any difference is reported as a caveat, never fixed.

    python benchmark/gold/derive_gated_text.py --src dev_monocap_v31 --new dev_gtext_v34
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_per_sound as sps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    work = sps._ROOT / "data" / "work"
    src = work / f"protocol_proposed_{a.src}"
    dst = work / f"protocol_audio_caption_{a.new}"
    link = work / f"protocol_proposed_{a.new}"
    dst.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        os.symlink(src.resolve(), link)
    same = diff = 0
    for clip in sorted(p for p in src.iterdir() if (p / "augmentations.json").exists()):
        (dst / clip.name).mkdir(exist_ok=True)
        for f in ("augmentations.json", "media.json"):
            if (clip / f).exists():
                shutil.copy2(clip / f, dst / clip.name / f)
        ours = sorted(sps.load_pictures(src, clip.name, "proposed") or [])
        text = sorted(sps.load_pictures(dst, clip.name, "audio_caption") or [])
        if ours == text:
            same += 1
        else:
            diff += 1
            print(f"[caveat] {clip.name}: ours {len(ours)} spans, text {len(text)} spans -> "
                  f"only-text {sorted(set(text) - set(ours))} only-ours {sorted(set(ours) - set(text))}")
    print(f"[derive] {same + diff} clips; spans identical on {same}, different on {diff} -> {dst}")


if __name__ == "__main__":
    main()

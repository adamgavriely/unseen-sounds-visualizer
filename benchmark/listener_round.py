"""The audio "listener" (Qwen3-Omni-30B-A3B-Instruct) asked yes/no about a candidate span. This repository keeps only what the shipped listener harness imports: the model, the
question, the 215-family vocabulary and the AUROC helper (benchmark/gold/dev_listener.py, test_listener.py,
listener_variants.py, listener_afnext.py, clip_prep.py). The original study's steps (pool, score, screen, fit, heldout on
the AudioSet-Strong sets, with benchmark/detector_round2.py and audioset_stage4_report.py) are in release v1.2.0.

(design record: release v1.2.0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
VOCAB = json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text(encoding="utf-8"))["families"]
QUESTION = "Is the sound of {} present in this recording? Answer yes or no."


def auroc(s, y):
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

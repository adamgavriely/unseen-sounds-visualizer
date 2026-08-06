"""Stage 7 - Evaluation.

Measure whether generated augmentations communicate the intended audio semantics.
Runs over the benchmark rather than as a per-clip pipeline step.

Automatic protocol (TODO): generate augmentation -> VLM describes the augmented
scene -> compare vs a semantic reference derived from the original multimodal
input (LLM) -> independent LLM judge scores semantic consistency. Complement with
CLIP/CLAP relevance and a gating-accuracy metric (correctly silent when the sound
is visible, correctly augmenting when not). Baselines: direct audio->image
(Sound2Scene); audio captioning only; proposed method.
"""
from __future__ import annotations

from typing import List, Dict, Any

from src.types import AugmentationSpec


def gating_accuracy(specs: List[AugmentationSpec],
                    labels: Dict[int, bool]) -> Dict[str, Any]:
    """Precision/recall of the augment decision vs a hand-labelled ground truth.

    ``labels`` maps event index -> whether it *should* be augmented.
    """
    tp = fp = tn = fn = 0
    for s in specs:
        if s.index not in labels:
            continue
        gt, pred = labels[s.index], s.augment
        tp += gt and pred
        fp += (not gt) and pred
        tn += (not gt) and (not pred)
        fn += gt and (not pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(*args, **kwargs):  # noqa: D401
    # TODO(stage7): implement VLM-describe -> LLM-judge + CLIP/CLAP relevance.
    raise NotImplementedError(
        "Stage 7 automatic evaluation is a TODO (VLM-describe + LLM-judge). "
        "gating_accuracy() is available now for the core gate metric.")

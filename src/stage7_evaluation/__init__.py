"""Stage 7 - Evaluation.

Measure whether generated augmentations communicate the intended audio semantics.
Runs over the benchmark rather than as a per-clip pipeline step.

The automatic protocol of the project proposal (its Section 6.1; not this report's) lives in stage7_evaluation.protocol and
is driven by benchmark/run_protocol.py: generate augmentation -> a VLM describes the
augmented output -> an LLM builds a semantic reference from the original audio+video
-> an independent LLM judge scores how much of the reference the augmentation
conveys. The same judge scores all three systems of the proposal's Section 7 (proposed gate,
blind audio-to-image, audio captioning).

gating_accuracy() below measures Stage 5 alone -- a component diagnostic, not the
project's headline metric.
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


def evaluate(clip_name, system, work_dir, backends):
    """Run the sec-6.1 protocol on one clip. See stage7_evaluation.protocol."""
    from src.stage7_evaluation.protocol import evaluate_clip
    return evaluate_clip(clip_name, system, work_dir, backends)

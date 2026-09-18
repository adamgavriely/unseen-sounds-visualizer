"""Judge v4 (rubric-enforced), docs/prereg_v4.md "Judge v4" section: a deterministic code
rule on top of the LLM judge, applied identically to every system and every row.

For clips whose grounded reference is "nothing beyond the picture" (human tag seen / no
ambient): an empty panel or empty caption stays 4 (the existing coded rule); a NON-EMPTY
panel or a caption that names a sound is capped at 2 -- it shows what the video already
shows and costs attention. Every other clip is unchanged. The permissive (uncapped) judge
stays on disk and is reported beside it on every row.

    python scripts/rubric_enforce.py v3_grounded v3_q38_grounded v4b_n20_grounded
      -> benchmark/protocol_results_<tag>_rubric.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from src.stage7_evaluation.protocol import is_empty_candidate

NOTHING = "nothing beyond the picture"
CAP = 2


def enforce(tag: str):
    src = _ROOT / "benchmark" / f"protocol_results_{tag}.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    capped = 0
    for r in rows:
        if not isinstance(r.get("score"), (int, float)):
            continue
        r["score_permissive"] = r["score"]
        if (r.get("reference") or "").strip().lower() == NOTHING:
            shown = (r.get("n_augmentations", 0) or 0) > 0 if r["system"] != "audio_caption" \
                else not is_empty_candidate(r.get("description") or "")
            if shown and r["score"] > CAP:
                r["score"] = CAP; capped += 1
        r["judge_model"] = (r.get("judge_model") or "") + " + rubric cap"
    dst = src.with_name(f"protocol_results_{tag}_rubric.json")
    dst.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{tag}: {len(rows)} records, {capped} capped -> {dst.name}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        enforce(t)

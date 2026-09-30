"""DEV2/TEST2 membership for tagger-set clips (docs/prereg_round13_detector_push.md, "Confirmation set 1").

Batch 1 (16 clips) was split by sha256(stem) % 2 and is frozen. From batch 2 on (Adam, 2026-09-30: "use the tags to
split the sounds between the sets") each NEW batch is split tag-balanced before any pipeline output exists for it:
its clips are ranked by (needed sounds [not visible, not obvious, importance >= 2], all scored sounds, sha256(stem)),
taken in consecutive pairs, and in each pair the clip with the smaller sha256 goes to DEV2 if the pair index is even,
TEST2 if odd (so neither set always gets the richer clip); an unpaired last clip goes by sha256 % 2. Only the new
batch's tags are read; batch-1 TEST2 tags never are.

    python benchmark/gold/tagger_split.py --batch 2 tg_d009 tg_d014 ...   -> benchmark/gold/tagger_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

TAGS = _ROOT / "benchmark" / "gold" / "annotations" / "tagger_AG.json"
OUT = _ROOT / "benchmark" / "gold" / "tagger_split.json"
BATCH1 = {"dev2": ["tg_d007", "tg_d016", "tg_d020", "tg_d033", "tg_d075", "tg_d088"],
          "test2": ["tg_d001", "tg_d011", "tg_d013", "tg_d017", "tg_d040", "tg_d076", "tg_d077", "tg_d078",
                    "tg_d079", "tg_d080"]}


def h(stem: str) -> int:
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("stems", nargs="+")
    a = ap.parse_args()
    split = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"batches": {"1": dict(BATCH1, rule="sha256 % 2")}}
    done = {s for b in split["batches"].values() for k in ("dev2", "test2") for s in b[k]}
    assert not done & set(a.stems), f"already split: {sorted(done & set(a.stems))}"
    d = json.loads(TAGS.read_text(encoding="utf-8"))
    sub = {"clips": [c for c in d["clips"] if Path(c["clip"]).stem in set(a.stems)]}
    assert len(sub["clips"]) == len(a.stems), "stem not in the tag file"
    tmp = _ROOT / "data" / "work" / "_tagger_split_batch.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    gold = S.load_gold([tmp])
    tmp.unlink()
    key = {st: (sum(1 for s in gold.get(st, []) if s["needed"] and s["importance"] >= 2), len(gold.get(st, [])), h(st))
           for st in a.stems}
    order = sorted(a.stems, key=lambda st: key[st], reverse=True)
    dev, test = [], []
    for i in range(0, len(order) - 1, 2):
        lo, hi = sorted(order[i:i + 2], key=h)
        (dev, test)[(i // 2) % 2].append(lo)
        (test, dev)[(i // 2) % 2].append(hi)
    if len(order) % 2:
        (dev if h(order[-1]) % 2 == 0 else test).append(order[-1])
    split["batches"][str(a.batch)] = {"dev2": sorted(dev), "test2": sorted(test),
                                      "rule": "tag-balanced pairs (needed, all, sha256), alternating"}
    split["dev2"] = sorted(s for b in split["batches"].values() for s in b["dev2"])
    split["test2"] = sorted(s for b in split["batches"].values() for s in b["test2"])
    OUT.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"batch {a.batch}: DEV2 {len(dev)} (needed {sum(key[s][0] for s in dev)}), "
          f"TEST2 {len(test)} (needed {sum(key[s][0] for s in test)}); totals DEV2 {len(split['dev2'])}, TEST2 {len(split['test2'])}")
    print("DEV2:", " ".join(sorted(dev))); print("TEST2:", " ".join(sorted(test)))


if __name__ == "__main__":
    main()

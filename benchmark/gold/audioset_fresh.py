"""A FRESH AudioSet-Strong confirmation set (docs/prereg_fresh_confirm_set.md), built 2026-09-28.

Why: the held-out 415 (benchmark/gold/audioset_heldout.py) has been read by six detector rounds. New detector ideas
are picked on the 280 and checked on the 415; a candidate that passes the 415 gets ONE final check on this set.
Nobody looks at detector results on this set before that.

Same recipe as the 415: AudioSet-Strong EVAL split only, 300 COMPLEX + 200 RANDOM with the 415's rules
(audioset_heldout.is_complex), a new seed. Exclusion is wider than the 415's and works at the YouTube-id level:
  - every id the 415's builder excludes (the 280 + missing, slice B + missing, gold segment ids),
  - all 500 ids chosen for the 415 (the 415 fetched + the 85 gone),
  - every eval-split YouTube id that appears in any text file or file name under benchmark/ or data/
    (so DEV, TEST, pilots, caches, the WavCaps list, ...), label TSVs and sealed / key files not opened,
  - every eval YouTube id that starts with the 8-character stub of an 'as_<class>_<8 chars>' gold clip
    (scripts/source_audioset.py names clips that way).

    python benchmark/gold/audioset_fresh.py --dry     # selection only -> audioset_fresh.json (ids, strata)
    python benchmark/gold/audioset_fresh.py           # download exactly that list (needs internet), write events
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold.audioset_slice import load_labels, fetch as _fetch
from benchmark.gold.audioset_heldout import is_complex, records, excluded_ids as heldout_excluded
import benchmark.gold.audioset_slice as S

VIDEOS = _ROOT / "data" / "input" / "audioset_fresh"
OUT = _ROOT / "benchmark" / "gold" / "audioset_fresh.json"
HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
SEED, N_COMPLEX, N_RANDOM = 20260928, 300, 200
SCAN_DIRS = ("benchmark", "data")
TEXT_EXT = {".json", ".jsonl", ".tsv", ".csv", ".txt", ".md", ".py", ".log", ".yaml", ".yml", ".html", ".sh"}
NO_OPEN = ("confirm_hash", "sealed", "ARM_KEY", "rate_confirm")       # never opened (names only are listed)
SKIP_FILES = {"audioset_eval_strong.tsv", "mid_to_display_name.tsv", "audioset_fresh.json", "fresh_download.log",
              "eval_strong.tsv", "train_strong.tsv", "mid_to_name.tsv"}      # the label files (data/work/audioset copy too)
_WIN11 = re.compile(r"(?=([A-Za-z0-9_-]{11}))")
_AS_STUB = re.compile(r"\bas_[A-Za-z0-9_-]+")


def ytid(seg: str) -> str:
    return seg.rsplit("_", 1)[0]


def repo_scan(eval_ytids: set[str]):
    """eval YouTube ids named anywhere under benchmark/ and data/ (text contents + file names), and 'as_' stubs"""
    hits, stubs, n_files, skipped = set(), set(), 0, []
    for top in SCAN_DIRS:
        for dp, dns, fns in os.walk(_ROOT / top):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                p = Path(dp) / fn
                rel = p.relative_to(_ROOT).as_posix()
                texts = [rel]
                if any(k in rel for k in NO_OPEN):
                    skipped.append(rel)
                elif fn not in SKIP_FILES and p.suffix.lower() in TEXT_EXT and p.stat().st_size < 50_000_000:
                    texts.append(p.read_text(encoding="utf-8", errors="ignore")); n_files += 1
                for t in texts:
                    hits |= set(_WIN11.findall(t)) & eval_ytids
                    stubs |= {m[-8:] for m in _AS_STUB.findall(t)}
    return hits, stubs, n_files, skipped


def exclusion(clips):
    eval_ytids = {ytid(s) for s in clips}
    src = {"heldout_builder": {ytid(s) for s in heldout_excluded()}}
    h = json.loads(HELDOUT.read_text(encoding="utf-8"))
    src["heldout_500"] = {ytid(s) for s in h["strata"]} | {ytid(c["id"]) for c in h["clips"]} | {ytid(s) for s in h["missing"]}
    hits, stubs, n_files, skipped = repo_scan(eval_ytids)
    src["repo_scan"] = hits
    src["as_stub_prefix"] = {y for y in eval_ytids if y[:8] in stubs}
    ex = set().union(*src.values())
    info = {k: len(v & eval_ytids) for k, v in src.items()}
    info |= {"excluded_eval_ytids": len(ex & eval_ytids), "scanned_text_files": n_files, "not_opened": len(skipped)}
    return ex, info


def select(clips):
    ex, info = exclusion(clips)
    pool = sorted(s for s in clips if ytid(s) not in ex)
    rng = random.Random(SEED)
    cx = [s for s in pool if is_complex(clips[s])]
    rng.shuffle(cx)
    assert len(cx) >= N_COMPLEX, f"complex pool {len(cx)} < {N_COMPLEX}: stop and report, do not shrink"
    complex_ids = cx[:N_COMPLEX]
    rest = [s for s in pool if s not in set(complex_ids)]
    rng.shuffle(rest)
    random_ids = rest[:N_RANDOM]
    return complex_ids, random_ids, info | {"eval_clips": len(clips), "pool": len(pool), "complex_pool": len(cx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="write the id list only (fix it in the prereg before downloading)")
    a = ap.parse_args()
    clips = load_labels()
    if a.dry:
        cx, rd, info = select(clips)
        strata = {s: "complex" for s in cx} | {s: "random" for s in rd}
        ids_sha = hashlib.sha256("\n".join(sorted(strata)).encode()).hexdigest()
        OUT.write_text(json.dumps({"seed": SEED, "n_complex": N_COMPLEX, "n_random": N_RANDOM, "info": info,
                                   "ids_sha256": ids_sha, "strata": strata, "clips": [], "missing": []}, indent=1),
                       encoding="utf-8")
        n_cx_in_rd = sum(is_complex(clips[s]) for s in rd)
        print(f"[fresh] {info}")
        print(f"[fresh] chosen {len(cx)} complex + {len(rd)} random ({n_cx_in_rd} of the random are complex too); "
              f"ids sha256 {ids_sha} -> {OUT}")
        return
    d = json.loads(OUT.read_text(encoding="utf-8"))
    strata = d["strata"]
    assert strata, "run --dry and write the prereg first"
    VIDEOS.mkdir(parents=True, exist_ok=True)
    S.VIDEOS = VIDEOS                      # fetch() writes into this folder
    done = {c["id"] for c in d["clips"]} | set(d["missing"])
    for i, seg in enumerate(strata, 1):
        if seg in done:
            continue
        if _fetch(seg):
            d["clips"].append({"id": seg, "stratum": strata[seg], "src": f"../../data/input/audioset_fresh/{seg}.mp4",
                               "duration": 10.0, "events": records(clips[seg])})
        else:
            d["missing"].append(seg)
        if i % 20 == 0:
            print(f"[fresh] {i}/{len(strata)}: {len(d['clips'])} fetched, {len(d['missing'])} gone", flush=True)
            OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"[fresh] done: {len(d['clips'])} clips, {len(d['missing'])} gone -> {OUT}")


if __name__ == "__main__":
    main()

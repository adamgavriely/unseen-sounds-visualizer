"""Prepare a partial re-render of a v4 row after a label-filter change (docs/prereg_v4.md,
amendment 2026-09-21): only the clips whose drawn set changes are re-run; the other clips'
renders, descriptions and judge scores stay. Run on the cluster, from ~/MscProj, then submit
the same job as before (stamp-and-resume does the rest):

    python scripts/rerender_affected.py --tag v4b3  --affected logs/affected_by_human_sound_amendment_v4b3.json
    STAGES=59  JUDGE=mistralai/Mistral-7B-Instruct-v0.3 sbatch slurm/job_v4.sh
    python scripts/rerender_affected.py --tag v4ab3 --affected logs/affected_by_human_sound_amendment_v4ab3.json
    STAGES=459 JUDGE=mistralai/Mistral-7B-Instruct-v0.3 sbatch slurm/job_v4.sh

Then, on the results: python benchmark/rubric_enforce.py --tag <tag> ; python scripts/v4_table.py.
What it does: clears the render/describe/judge/grounded stamps of the tag, deletes the work
dirs of the affected clips for the three systems, and drops their (clip, system) records from
the description cache and the two result files. Nothing else is touched. --dry-run to look.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("proposed", "blind_a2i", "audio_caption")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--affected", required=True, help="json: {clip stem: [labels]} or [clip stems]")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    aff = json.loads(Path(a.affected).read_text(encoding="utf-8"))
    stems = sorted(aff if isinstance(aff, list) else aff.keys())
    print(f"[rerender] tag {a.tag}: {len(stems)} clips")

    stamps = ROOT / "benchmark" / ".chain" / a.tag
    for name in ("render", "describe", "judge", "grounded"):
        p = stamps / name
        if p.exists():
            print("  stamp", p.relative_to(ROOT))
            if not a.dry_run:
                p.unlink()
    for system in SYSTEMS:
        for stem in stems:
            wd = ROOT / "data" / "work" / f"protocol_{system}_{a.tag}" / stem
            if wd.exists():
                print("  work ", wd.relative_to(ROOT))
                if not a.dry_run:
                    shutil.rmtree(wd)
    names = {f"{s}.mp4" for s in stems} | set(stems)
    for f in (ROOT / "benchmark" / f"protocol_descriptions_{a.tag}.json",
              ROOT / "benchmark" / f"protocol_results_{a.tag}.json",
              ROOT / "benchmark" / f"protocol_results_{a.tag}_grounded.json"):
        if not f.exists():
            print("  (missing)", f.name); continue
        recs = json.loads(f.read_text(encoding="utf-8"))
        keep = [r for r in recs if r.get("clip") not in names and Path(r.get("clip", "")).stem not in stems]
        print(f"  {f.name}: {len(recs)} -> {len(keep)} records")
        if not a.dry_run:
            f.write_text(json.dumps(keep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[rerender] now: STAGES=<459|59> JUDGE=mistralai/Mistral-7B-Instruct-v0.3 sbatch slurm/job_v4.sh")


if __name__ == "__main__":
    main()

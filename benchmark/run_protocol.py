"""Run the Stage-7 evaluation protocol over the benchmark, for every system.

This is the experiment the proposal is actually about (sec 6.1 + sec 7): for each
clip, produce augmentations under each system, then score how well those
augmentations convey the audio information a deaf viewer is missing.

Systems compared (proposal sec 7):
  proposed       -- cross-modal gate: depict only sounds whose source is NOT visible
  blind_a2i      -- direct audio-to-image: depict every detected sound, ignoring the
                    video entirely. This is the Sound2Scene-style baseline.
  audio_caption  -- text only: no image is produced; the "augmentation" is a caption
                    of the detected sounds, standing in for enriched subtitles.

All three are scored by the SAME judge against the SAME reference, so the only
variable is what the system chose to show.

Usage:
    python -m benchmark.run_protocol                     # all systems, all clips
    python -m benchmark.run_protocol --systems proposed  # one system
    python -m benchmark.run_protocol --limit 20          # quick pass
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src import pipeline
from src.stage7_evaluation.protocol import Backends, evaluate_clip, ClipEvaluation

BENCH = _ROOT / "data" / "input" / "benchmark"
OUT = _ROOT / "benchmark" / "protocol_results.json"
TAGS = _ROOT / "benchmark" / "tags.json"
SYSTEMS = ("proposed", "blind_a2i", "audio_caption")
# proposal sec 5.1 scenarios, so results can be reported per scenario
SCENARIO_OF = {"unseen_ambient": "acoustic_event_or_ambient",
               "mixed": "mixed_audio", "seen_ambient": "ambient_environmental",
               "no_ambient": "ambient_environmental"}


def clips_to_run(limit: int | None):
    """Benchmark clips that carry a human label, so results can be broken down."""
    tags = json.loads(TAGS.read_text(encoding="utf-8")) if TAGS.exists() else {}
    out = []
    for key, val in tags.items():
        if val.get("tag") not in SCENARIO_OF:
            continue
        folder, _, base = key.partition("/")
        p = BENCH / folder / base
        if p.exists():
            out.append((p, val["tag"]))
    out.sort()
    return out[:limit] if limit else out


def configure(system: str):
    """Each system differs only in how Stage 5/6 decide what to show."""
    if system == "proposed":
        config.GATE_ENABLED = True          # depict only non-visible sources
        config.RENDER_MODE = "full"
    elif system == "blind_a2i":
        config.GATE_ENABLED = False         # depict everything heard, ignore the video
        config.RENDER_MODE = "full"
    elif system == "audio_caption":
        config.GATE_ENABLED = False         # no imagery: a caption stands in
        config.RENDER_MODE = "minimal"


def caption_from_artifacts(work: Path) -> str:
    """The audio-captioning baseline's 'augmentation': the sounds, as text."""
    f = work / "augmentations.json"
    if not f.exists():
        return "no augmentation was shown"
    specs = json.loads(f.read_text(encoding="utf-8"))
    labels = [s["event_label"] for s in specs if s.get("augment")]
    return ("The soundtrack contains: " + ", ".join(dict.fromkeys(labels))
            if labels else "no notable non-speech sound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse existing pipeline artifacts instead of re-running it")
    args = ap.parse_args()

    clips = clips_to_run(args.limit)
    print(f"[protocol] {len(clips)} labelled clips x {len(args.systems)} systems",
          flush=True)
    backends = Backends(config.VLM_MODEL, config.JUDGE_MODEL, config.DEVICE)
    print(f"[protocol] describer={config.VLM_MODEL}
           judge     ={config.JUDGE_MODEL}", flush=True)
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["clip"], r["system"]) for r in results}

    for system in args.systems:
        configure(system)
        work_root = _ROOT / "data" / "work" / f"protocol_{system}"
        for i, (clip, tag) in enumerate(clips, 1):
            if (clip.name, system) in done:
                continue
            work = work_root / clip.stem
            if not args.skip_render or not (work / "augmentations.json").exists():
                try:
                    pipeline.run(clip, work_root=work_root)
                except Exception as e:
                    print(f"  ! render {clip.name}: {type(e).__name__}: {e}", flush=True)
                    continue
            ev = evaluate_clip(clip.name, system, work, backends)
            if ev is None:
                continue
            if system == "audio_caption":       # score the caption text, not an image
                from src.stage7_evaluation.protocol import judge
                cand = caption_from_artifacts(work)
                sc, why = judge(ev.reference, cand, backends)
                ev = ClipEvaluation(ev.clip, system, ev.reference, cand, sc, why, 0)
            rec = ev.to_dict()
            rec["human_tag"] = tag
            rec["scenario"] = SCENARIO_OF[tag]
            results.append(rec)
            OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                           encoding="utf-8")
            print(f"  [{system} {i}/{len(clips)}] {clip.name[:34]:34} "
                  f"score={ev.score}  {ev.description[:46]}", flush=True)

    report(results)


def report(results):
    by_sys = defaultdict(list)
    by_sys_scn = defaultdict(list)
    for r in results:
        by_sys[r["system"]].append(r["score"])
        by_sys_scn[(r["system"], r.get("scenario", "?"))].append(r["score"])
    print("\n" + "=" * 66)
    print("STAGE-7 SEMANTIC CONSISTENCY (proposal sec 6.1), judge score 0-4\n")
    print(f"{'system':16}{'n':>5}{'mean':>8}{'>=3 (conveys it)':>20}")
    for s in SYSTEMS:
        v = by_sys.get(s, [])
        if not v:
            continue
        good = sum(1 for x in v if x >= 3)
        print(f"{s:16}{len(v):>5}{sum(v)/len(v):>8.2f}{100*good/len(v):>19.0f}%")
    print("\nper scenario (proposal sec 5.1):")
    scns = sorted({k[1] for k in by_sys_scn})
    print(f"  {'scenario':30}" + "".join(f"{s[:12]:>14}" for s in SYSTEMS))
    for scn in scns:
        row = "".join(
            f"{(sum(by_sys_scn[(s, scn)]) / len(by_sys_scn[(s, scn)])):>14.2f}"
            if by_sys_scn.get((s, scn)) else f"{'-':>14}" for s in SYSTEMS)
        print(f"  {scn:30}{row}")
    print(f"\nfull records -> {OUT}")


if __name__ == "__main__":
    main()

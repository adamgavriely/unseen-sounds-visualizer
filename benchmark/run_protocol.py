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

TWO PASSES, and why
-------------------
The describing VLM (Qwen2.5-VL, ~16 GB) and the judging LLM (Mistral-7B, ~15 GB) do
not co-fit on a 24 GB card, and the cluster's large-memory partitions are frequently
draining. So the run is split so that only one model is resident at a time:

  --phase describe   render, then build the reference and the description for every
                     clip, cache to benchmark/protocol_descriptions.json, free the VLM
  --phase judge      load the judge alone and score the cached pairs
  --phase all        (default) both, sequentially, unloading in between

The split also makes judge reliability cheap to measure: a second judge re-scores the
same cached descriptions with no vision work at all --
    python -m benchmark.run_protocol --phase judge --judge Qwen/Qwen3-8B --tag judge2
which is exactly the Day-6 experiment in docs/PLAN.md.

Usage:
    python -m benchmark.run_protocol --limit 12          # small end-to-end pass
    python -m benchmark.run_protocol --phase describe    # vision pass only
    python -m benchmark.run_protocol --phase judge       # scoring pass only
    python -m benchmark.run_protocol --systems proposed  # one system
"""
from __future__ import annotations

import argparse
import json
import random
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
from src.stage7_evaluation.protocol import (Backends, describe_clip, judge_record,
                                            UNPARSED)

BENCH = _ROOT / "data" / "input" / "benchmark"
DESCRIPTIONS = _ROOT / "benchmark" / "protocol_descriptions.json"
OUT = _ROOT / "benchmark" / "protocol_results.json"
TAGS = _ROOT / "benchmark" / "tags.json"
SYSTEMS = ("proposed", "blind_a2i", "audio_caption")
# proposal sec 5.1 scenarios, so results can be reported per scenario
SCENARIO_OF = {"unseen_ambient": "acoustic_event_or_ambient",
               "mixed": "mixed_audio", "seen_ambient": "ambient_environmental",
               "no_ambient": "ambient_environmental"}


def desc_file(tag: str) -> Path:
    """Cached descriptions, per run tag.

    The cache is keyed by (clip, system) only, so an ablation that changes HOW the
    image is produced -- SDXL instead of retrieval, say -- would otherwise be skipped
    as already-described and would silently score the other run's images. A tagged
    run gets its own cache; an untagged run reads the main one.
    """
    return DESCRIPTIONS if not tag else DESCRIPTIONS.with_name(
        f"protocol_descriptions_{tag}.json")


CLIP_DIR = None      # set from --clip-dir: run on every video of a folder instead (slice B, gold set 2)


def clips_to_run(limit):
    """Benchmark clips that carry a human label, sampled EVENLY across the tags.

    Taking the first N alphabetically silently returned only `mixed/` clips (that
    folder sorts first), so the pilot measured a single scenario and the comparison
    was not interpretable -- on mixed clips the gate suppresses sounds by design, so
    the blind baseline is flattered. Sample round-robin across the four tags instead,
    with a fixed seed so runs are reproducible and resumable.
    """
    if CLIP_DIR:
        vids = sorted(p for p in Path(CLIP_DIR).iterdir() if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"))
        # the human tag (read by the grounded judge: seen / no-ambient -> silence is right) comes
        # from the per-sound gold when the clip is in it (category from the ticks, amendment 5);
        # slice B and unlabelled folders keep the fixed "unseen" tag as before
        gold_tags = {}
        gold_file = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
        if gold_file.exists():
            try:
                from benchmark.gold.score_per_sound import load_gold, category
                names = {"no_ambient": "no_ambient", "seen": "seen_ambient", "mixed": "mixed", "unseen": "unseen_ambient"}
                gold_tags = {stem: names[category(snds)] for stem, snds in load_gold([gold_file]).items()}
            except Exception as e:
                print(f"[clips] gold tags unavailable ({type(e).__name__}: {e})", flush=True)
        return [(p, gold_tags.get(p.stem, "unseen_ambient")) for p in vids][: limit or None]
    tags = json.loads(TAGS.read_text(encoding="utf-8")) if TAGS.exists() else {}
    by_tag = {}
    for key, val in tags.items():
        tag = val.get("tag")
        if tag not in SCENARIO_OF:
            continue
        folder, _, base = key.partition("/")
        p = BENCH / folder / base
        if p.exists():
            by_tag.setdefault(tag, []).append((p, tag))
    rng = random.Random(7)
    for v in by_tag.values():
        v.sort()
        rng.shuffle(v)
    if not limit:
        return [c for v in by_tag.values() for c in v]
    out, i = [], 0
    order = sorted(by_tag)                       # deterministic tag order
    while len(out) < limit and any(len(by_tag[t]) > i for t in order):
        for t in order:
            if len(by_tag[t]) > i and len(out) < limit:
                out.append(by_tag[t][i])
        i += 1
    return out


def configure(system: str):
    """Each system differs only in how Stage 5/6 decide what to show.

    The video-reading parts of Stage 5 -- the per-sound visibility question, the place
    that makes a depiction specific, the speech question -- are switched off for both
    baselines and not just the gate flag. They run inside the reasoner regardless of
    GATE_ENABLED, so leaving them on would have made the "blind" baseline look at the
    video, and the comparison would have measured nothing.
    """
    if system == "proposed":
        config.GATE_ENABLED = True          # depict only non-visible sources
        config.DEPICTION_REASONING = True   # place-specific event depictions
        config.VLM_VISIBILITY = True        # per-sound visibility from the frames
        config.SPEECH_CONTEXT = True        # people reacting to a sound raises it
        config.RENDER_MODE = "full"
    elif system == "blind_a2i":
        config.GATE_ENABLED = False         # depict everything heard, ignore the video
        config.DEPICTION_REASONING = False  # the bare label is the depiction
        config.VLM_VISIBILITY = False
        config.SPEECH_CONTEXT = False
        config.RENDER_MODE = "full"
    elif system == "audio_caption":
        config.GATE_ENABLED = False         # no imagery: a caption stands in
        config.DEPICTION_REASONING = False
        config.VLM_VISIBILITY = False
        config.SPEECH_CONTEXT = False
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


def work_root_for(system: str, tag: str) -> Path:
    return _ROOT / "data" / "work" / (f"protocol_{system}" + (f"_{tag}" if tag else ""))


# ----------------------------------------------------------------------
# pass 0 -- render only. SDXL is loaded; the describing VLM is not.
# ----------------------------------------------------------------------
def phase_render(args):
    """Run the pipeline for every clip and system, and describe nothing.

    SDXL (~7 GB) is loaded by Stage 6 and stays resident, and Qwen2.5-VL (~16 GB) is
    loaded by the describer. On a 24 GB card they do not co-fit: the first attempt at
    this run rendered clip 1, loaded the VLM to describe it, and then every subsequent
    clip died at Stage 4 with "CUDA failed with error out of memory". The failures were
    caught per clip, so the job reported COMPLETED having produced 2 records out of 300.

    Splitting rendering from describing is the same separation already used for the
    describer and the judge, one level further down: each phase runs in its own process
    with one large model resident.
    """
    clips = clips_to_run(args.limit)
    print(f"[render] {len(clips)} clips x {len(args.systems)} systems", flush=True)
    ok = fail = 0
    for system in args.systems:
        configure(system)
        work_root = work_root_for(system, args.tag)
        # the composite mp4 (and its panel scratch dir) is keyed by clip stem only; two rows
        # rendering the same clip at once (v4b4 / v4ab4 shards, 2026-09-22) collided in
        # data/output -> one folder per (system, tag); nothing downstream reads the mp4
        config.OUTPUT_DIR = _ROOT / "data" / "output" / work_root.name
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for i, (clip, _tag) in enumerate(clips, 1):
            work = work_root / clip.stem
            if (work / "augmentations.json").exists():
                ok += 1
                continue
            try:
                pipeline.run(clip, work_root=work_root)
                ok += 1
            except Exception as e:
                # a CUDA OOM on a 22 GB card is fragmentation between the resident
                # models, not the clip (v3 job 28942456 lost london_protest_01 that way);
                # clear the allocator and try once more before counting a failure
                if "out of memory" in str(e).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        pipeline.run(clip, work_root=work_root)
                        ok += 1
                        print(f"  ~ render {clip.name}: recovered after OOM", flush=True)
                        continue
                    except Exception as e2:
                        e = e2
                fail += 1
                print(f"  ! render {clip.name}: {type(e).__name__}: {e}", flush=True)
            if i % 10 == 0:
                print(f"  [{system}] {i}/{len(clips)}  ok={ok} fail={fail}", flush=True)
    print(f"[render] done: ok={ok} fail={fail}", flush=True)
    guard(ok, fail, "render")


def guard(ok: int, fail: int, what: str):
    """Fail the job loudly when most clips failed.

    Each clip's exception is caught so that one bad file cannot end a ten-hour run.
    The cost of that is a run which fails almost completely and still exits 0 -- which
    is exactly what happened, and the chain went on to judge two records and declare
    itself finished. Past a quarter failed, something systematic is wrong and the chain
    should stop rather than produce a confident table built on nothing.
    """
    total = ok + fail
    if total and fail > total // 4:
        sys.exit(f"[{what}] ABORT: {fail}/{total} clips failed -- this is systematic, "
                 "not a handful of bad files. Fix the cause before continuing.")


# ----------------------------------------------------------------------
# pass 1 -- describe the rendered augmentations. Only the VLM is loaded.
# ----------------------------------------------------------------------
def phase_describe(args, backends):
    clips = clips_to_run(args.limit)
    cache = desc_file(args.tag)
    print(f"[describe] {len(clips)} clips x {len(args.systems)} systems", flush=True)
    recs = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    done = {(r["clip"], r["system"]) for r in recs}
    missing = 0

    for system in args.systems:
        configure(system)
        # --work-tag: describe another run's rendered panels under this tag (v4 re-describes
        # v3's panels with the v4 evaluation pair so the rows are comparable)
        work_root = work_root_for(system, getattr(args, "work_tag", None) or args.tag)
        for i, (clip, tag) in enumerate(clips, 1):
            if (clip.name, system) in done:
                continue
            work = work_root / clip.stem
            if not (work / "augmentations.json").exists() or not args.skip_render:
                try:
                    pipeline.run(clip, work_root=work_root)
                except Exception as e:
                    missing += 1
                    print(f"  ! render {clip.name}: {type(e).__name__}: {e}", flush=True)
                    continue
            rec = describe_clip(clip.name, system, work, backends)
            if rec is None:
                missing += 1
                continue
            if system == "audio_caption":     # the caption IS the augmentation
                rec["description"] = caption_from_artifacts(work)
                rec["n_augmentations"] = 0
            rec["human_tag"] = tag
            rec["scenario"] = SCENARIO_OF[tag]
            recs.append(rec)
            cache.write_text(json.dumps(recs, indent=1, ensure_ascii=False),
                             encoding="utf-8")
            print(f"  [{system} {i}/{len(clips)}] {clip.name[:32]:32} "
                  f"{rec['description'][:52]}", flush=True)
    print(f"[describe] cached -> {cache}", flush=True)
    guard(len(recs), missing, "describe")


# ----------------------------------------------------------------------
# pass 2 -- judge the cached pairs. Only the judge is loaded.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# pass 1b -- an independent reference. Audio LM, CLAP, a non-Qwen VLM and a third-party
# writer, each loaded in turn; none of them sees the system's output. Reads the same
# description cache to know which clips are in the run.
# ----------------------------------------------------------------------
def _find_clip(stem: str):
    """The benchmark clip with this stem, whichever tag folder and extension."""
    for p in BENCH.rglob(stem + ".*"):
        if p.suffix.lower() in (".mp4", ".webm", ".ogv", ".mkv"):
            return p
    return None


def ref_file(tag: str) -> Path:
    return DESCRIPTIONS.with_name(f"protocol_reference_indep{('_' + tag) if tag else ''}.json")


def phase_reference(tag: str = "", desc_tag=None, device: str = "cuda"):
    from src.stage7_evaluation import independent_reference as IR
    cache = desc_file(desc_tag if desc_tag is not None else tag)
    if not cache.exists():
        sys.exit("no descriptions cached -- run --phase describe first")
    recs = json.loads(cache.read_text(encoding="utf-8"))
    clips = sorted({r["clip"] for r in recs})
    wavs, videos, transcripts = {}, {}, {}
    for c in clips:
        stem = Path(c).stem                     # cache names carry the extension
        # any system's work dir has the same audio, video reference and transcript
        for system in SYSTEMS:
            wd = work_root_for(system, desc_tag if desc_tag is not None else tag) / stem
            if (wd / "audio.wav").exists():
                wavs[c] = wd / "audio.wav"
                seg = wd / "segments.json"
                transcripts[c] = " ".join(x.get("text", "") for x in
                                          json.loads(seg.read_text("utf-8"))) if seg.exists() else ""
                break
        vid = _find_clip(stem)
        if vid is not None:
            videos[c] = vid
    missing = [c for c in clips if c not in wavs or c not in videos]
    if missing:
        print(f"[reference] {len(missing)} clips without audio/video; skipped: {missing[:5]}")
    keep = [c for c in clips if c in wavs and c in videos]
    print(f"[reference] building an independent reference for {len(keep)} clips", flush=True)
    IR.build_all({c: wavs[c] for c in keep}, {c: videos[c] for c in keep},
                 transcripts, ref_file(tag), device=device)
    print("[reference] ->", ref_file(tag), flush=True)


def phase_judge(backends, tag: str = "", rescore: bool = False,
                desc_tag=None, grounded: bool = False, independent: bool = False, ref_tag=None):
    # A judge-agreement run re-scores the MAIN descriptions but writes its results
    # under a new tag, so cache and results file are tagged independently.
    cache = desc_file(desc_tag if desc_tag is not None else tag)
    if not cache.exists():
        cache = DESCRIPTIONS
    if not cache.exists():
        sys.exit("no descriptions cached -- run --phase describe first")
    recs = json.loads(cache.read_text(encoding="utf-8"))
    out_file = OUT if not tag else OUT.with_name(f"protocol_results_{tag}.json")
    # --rescore discards previous scores and judges the cached pairs again. Needed
    # whenever the scoring RULE changes (as when empty augmentations stopped getting
    # a charitable 2/4): the descriptions are unaffected, so re-rendering is waste.
    results = [] if rescore else (json.loads(out_file.read_text(encoding="utf-8"))
                                  if out_file.exists() else [])
    done = {(r["clip"], r["system"]) for r in results}
    print(f"[judge] {len(recs)} cached pairs, {len(done)} already scored", flush=True)
    refs = {}
    if independent:
        rf = ref_file(desc_tag if desc_tag is not None else tag)
        if not rf.exists():
            sys.exit(f"no independent reference cached at {rf} -- run --phase reference first")
        refs = json.loads(rf.read_text(encoding="utf-8")).get("references", {})
        print(f"[judge] independent references for {len(refs)} clips", flush=True)
    # --ref-tag: judge every system of THIS row against the grounded references of another
    # row (same answer sheet). Amendment 3 (2026-09-21): the flag was accepted since 2026-09-20
    # but never read here, so the two *_xref_* rows were judged against their own references
    # and are void. Missing clips are refused, not skipped, so n cannot change silently.
    if ref_tag:
        if independent:
            sys.exit("--ref-tag and --independent are exclusive")
        from src.stage7_evaluation.protocol import grounded_reference
        other = desc_file(ref_tag)
        if not other.exists():
            sys.exit(f"no descriptions cached for --ref-tag {ref_tag} at {other}")
        by_clip = {}
        for r in json.loads(other.read_text(encoding="utf-8")):
            if r["system"] == "proposed":
                by_clip[r["clip"]] = grounded_reference(r["reference"], r.get("human_tag"))
        missing = sorted({r["clip"] for r in recs} - set(by_clip))
        if missing:
            sys.exit(f"--ref-tag {ref_tag}: {len(missing)} clips have no reference there: {missing[:5]}")
        refs = {c: {"reference": v} for c, v in by_clip.items()}
        print(f"[judge] references of row {ref_tag} for {len(refs)} clips", flush=True)

    for i, rec in enumerate(recs, 1):
        if (rec["clip"], rec["system"]) in done:
            continue
        if independent and rec["clip"] not in refs:
            continue
        ev = judge_record(rec, backends, grounded=grounded,
                          reference_override=refs[rec["clip"]]["reference"] if (independent or ref_tag) else None)
        row = ev.to_dict()
        row["human_tag"] = rec.get("human_tag")
        row["scenario"] = rec.get("scenario")
        row["judge_model"] = backends.judge_model
        results.append(row)
        out_file.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  [{i}/{len(recs)}] {rec['system']:14} {rec['clip'][:30]:30} "
              f"score={ev.score}  {ev.why[:40]}", flush=True)
    # An unparsed judge reply is not a score of 0 and must never be averaged as one.
    # The first judge-agreement run averaged 263 of them and produced a confident
    # kappa of 0.164 that was purely an artefact of the parser.
    bad = sum(1 for r in results if str(r.get("why", "")).startswith(UNPARSED))
    if bad:
        print(f"[judge] {bad}/{len(results)} replies could not be parsed", flush=True)
    guard(len(results) - bad, bad, "judge")
    report(results)
    print(f"\nfull records -> {out_file}")


def report(results):
    from src.stage7_evaluation.protocol import is_empty_candidate
    by_sys = defaultdict(list)
    by_sys_scn = defaultdict(list)
    silent = defaultdict(list)          # per system: was the augmentation empty?
    for r in results:
        by_sys[r["system"]].append(r["score"])
        by_sys_scn[(r["system"], r.get("scenario", "?"))].append(r["score"])
        silent[r["system"]].append(is_empty_candidate(r.get("description", "")))
    print("\n" + "=" * 76)
    print("STAGE-7 SEMANTIC CONSISTENCY (proposal sec 6.1), judge score 0-4\n")
    # "silent" and "right to be" are reported because the gate's whole claim is about
    # WHEN to show nothing, and a mean score hides that: a system can reach a decent
    # mean by abstaining on the clips where abstaining happens to be correct.
    print(f"{'system':16}{'n':>5}{'mean':>8}{'>=3':>7}{'silent':>9}{'right to be':>13}")
    for s in SYSTEMS:
        v = by_sys.get(s, [])
        if not v:
            continue
        good = sum(1 for x in v if x >= 3)
        sil = silent[s]
        n_sil = sum(sil)
        # among the clips it stayed silent on, how often was silence the right call?
        ok_sil = sum(1 for q, z in zip(v, sil) if z and q == 4)
        rt = f"{100 * ok_sil / n_sil:>12.0f}%" if n_sil else f"{'-':>13}"
        print(f"{s:16}{len(v):>5}{sum(v) / len(v):>8.2f}{100 * good / len(v):>6.0f}%"
              f"{100 * n_sil / len(v):>8.0f}%{rt}")
    print("\nper scenario (proposal sec 5.1):")
    scns = sorted({k[1] for k in by_sys_scn})
    print(f"  {'scenario':30}" + "".join(f"{s[:12]:>14}" for s in SYSTEMS))
    for scn in scns:
        row = "".join(
            f"{(sum(by_sys_scn[(s, scn)]) / len(by_sys_scn[(s, scn)])):>14.2f}"
            if by_sys_scn.get((s, scn)) else f"{'-':>14}" for s in SYSTEMS)
        print(f"  {scn:30}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", choices=("all", "render", "describe", "judge", "reference"),
                    default="all")
    ap.add_argument("--judge", default=None,
                    help="override config.JUDGE_MODEL (for judge-agreement runs)")
    ap.add_argument("--tag", default="",
                    help="suffix for the results file, e.g. --tag judge2")
    ap.add_argument("--desc-tag", default=None,
                    help="read descriptions from this run's cache; use with --tag "
                         "to score the main descriptions under a second judge")
    ap.add_argument("--independent", action="store_true",
                    help="judge against the independent reference (--phase reference first)")
    ap.add_argument("--grounded", action="store_true",
                    help="score against the human-corrected reference: on clips the "
                         "annotator marked seen_ambient or no_ambient, nothing is "
                         "missing, so staying silent is the correct output")
    ap.add_argument("--rescore", action="store_true",
                    help="discard existing scores and re-judge the cached descriptions")
    ap.add_argument("--work-tag", default=None,
                    help="describe the rendered artifacts of this other tag (with --skip-render)")
    ap.add_argument("--ref-tag", default=None,
                    help="judge against another row's references (same answer sheet for two detectors)")
    ap.add_argument("--clip-dir", default=None,
                    help="run on every video in this folder instead of the tagged benchmark (slice B)")
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse existing pipeline artifacts instead of re-rendering")
    args = ap.parse_args()
    global CLIP_DIR
    CLIP_DIR = args.clip_dir

    judge_model = args.judge or config.JUDGE_MODEL
    backends = Backends(config.VLM_MODEL, judge_model, config.DEVICE)
    print(f"[protocol] describer = {config.VLM_MODEL}", flush=True)
    print(f"[protocol] judge     = {judge_model}   (independent model)", flush=True)

    if args.phase == "render":
        phase_render(args)
        return
    if args.phase in ("all", "describe"):
        phase_describe(args, backends)
        backends.unload_vlm()        # free ~16 GB before the judge is loaded
        print("[protocol] describer unloaded", flush=True)
    if args.phase == "reference":
        phase_reference(args.tag, args.desc_tag, device=config.DEVICE)
        return
    if args.phase in ("all", "judge"):
        phase_judge(backends, args.tag, args.rescore, args.desc_tag, args.grounded,
                    independent=args.independent, ref_tag=args.ref_tag)


if __name__ == "__main__":
    main()

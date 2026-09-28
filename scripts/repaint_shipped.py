"""Repaint a scored render with the shipped picture setup (config.use_shipped): same sounds, same times, new pictures.

Adam, 28 Sept 2026: Qwen-Image-2512 with the frozen final picture setup is the shipped picture model; FLUX is retired.
The inspector page needs every scored clip rendered with it, WITHOUT changing any scored decision. So this re-runs
ONLY what the shipped pipeline does after the gate:

  * spec.source   -- labels.choose_source on the clip's own raw detections (events.json), exactly as
                     consolidate_families does under PICTURE_V3: raw = the salient non-speech firings
                     (the marginal band only, for a family with no firing at the strong bar), burst = the
                     spec's start/end;
  * spec.subject  -- reason._depict_v31 on the frames decide_subjects keeps for the sound (first stretch of
                     its first burst, a second either side, six frames) with the clip's place, fired = []
                     (the strict side, as in the pipeline); fallback _depict_v3, then "<source> making its sound";
  * the picture   -- stage6._final_picture (card / template / Qwen-Image-2512 with the rules tail).

The dedup step that follows depiction in reason.py is NOT re-run: it could flip `augment`. start, end, augment,
spans, confidence and reason are copied from the scored file and checked identical after writing.

The place is read from the scored proposed run's own log ("[stage5] place: ...") when it is there, else asked again
with PLACE_PROMPT (then SCENE_PROMPT), as decide_subjects does.

    python scripts/repaint_shipped.py --tag dev_monocap_v31 [--shard 0 --of 2] [--limit 2] [--clips a b]
Output: data/work/shipped_<tag>/<clip>/ (json copies, augmentations/aug_XXX.png, repaint.json)

--verify (2026-09-28): the same repaint with config.PICTURE_VERIFY on (check each picture, redraw up to 4 times, else
a word card; src/stage6_visual_augmentation/verify.py), into data/work/shipped_v_<tag>/. The subjects are copied from
shipped_<tag> (phase A is not re-run), so try 1 is the shipped picture's own prompt and seed. Per picture,
repaint.json gets the tries, what the VLM picked and saw, and the final kind (picture / rewritten / word card).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

JSONS = ("augmentations.json", "events.json", "gate_votes.json", "media.json", "scene.json", "segments.json",
         "onset_trace.json", "credits.json")
KEEP = ("index", "event_label", "start", "end", "augment", "spans", "confidence", "reason", "talked_about", "detail")


def places_from_logs(tag: str) -> dict:
    """clip -> the place the scored proposed run printed for it (last run wins)."""
    out = {}
    head = re.compile(r"^\[gold\].*systems=proposed tag=" + re.escape(tag) + r"\b")
    for f in sorted((_ROOT / "logs").glob("gold_*.out")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if not any(head.match(x) for x in lines[:40]):
            continue
        place = None
        for x in lines:
            if x.startswith("[1/7]"):
                place = None
            m = re.search(r"\[stage5\] place: (.*)$", x)
            if m:
                place = m.group(1).strip()
            m = re.search(r"Done -> .*/([^/]+)_augmented\.mp4", x)
            if m and place:
                out[m.group(1)] = place
    return out


def spec_frames_times(spec) -> list:
    """decide_subjects: first stretch of the first burst, a second either side, six frames."""
    stretch = float(getattr(config, "VISIBILITY_STRETCH", 5.0))
    bursts = list(spec.spans or [(spec.start, spec.end)])
    a, b = bursts[0]
    k = max(1, int(round((b - a) / stretch)))
    b1 = a + (b - a) / k
    n = 6
    lo, hi = a - 1.0, b1 + 1.0
    return [lo + (hi - lo) * t / (n - 1) for t in range(n)]


def raw_for(events, fam: str) -> list:
    """The `raw` list consolidate_families saw for this family (plan_augmentations)."""
    from src.labels import is_salient_nonspeech, min_confidence, canonical
    drawable = [e for e in events if is_salient_nonspeech(e.label)]
    strong_bar = min_confidence("", config.DISPLAY_THRESHOLD)
    strong = {canonical(e.label) for e in drawable if e.confidence >= strong_bar}
    if fam in strong:
        return drawable
    return [e for e in drawable if 0.5 * strong_bar <= e.confidence < strong_bar and canonical(e.label) not in strong]


def load_specs(path: Path):
    from src.types import AugmentationSpec
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, [AugmentationSpec(**{k: v for k, v in s.items() if k in AugmentationSpec.__dataclass_fields__})
                 for s in raw]


def kind_of(spec) -> str:
    from benchmark.gold.gen_screen import TEMPLATES
    if spec.backend == "card":
        return "word card" if (spec.image_prompt or "").startswith("VCARD:") else "card"
    if spec.backend == "placeholder":
        return "PLACEHOLDER"
    if (spec.source or spec.event_label) in TEMPLATES or spec.event_label in TEMPLATES:
        return "template"
    return "diffusion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--phase", choices=["all", "subjects", "draw"], default="all")
    ap.add_argument("--verify", action="store_true", help="PICTURE_VERIFY on, into data/work/shipped_v_<tag>")
    a = ap.parse_args()

    print("[cfg]", config.use_shipped(), flush=True)
    config.DEVICE = "cuda"
    if a.verify:
        config.PICTURE_VERIFY = True
        print("[cfg] PICTURE_VERIFY on, tries", config.PICTURE_VERIFY_TRIES, "VLM", config.VLM_MODEL, flush=True)
    src_root = _ROOT / "data" / "work" / f"protocol_proposed_{a.tag}"
    dst_root = _ROOT / "data" / "work" / (f"shipped_v_{a.tag}" if a.verify else f"shipped_{a.tag}")
    stems = sorted(p.name for p in src_root.iterdir() if (p / "augmentations.json").exists())
    if a.clips:
        stems = [s for s in stems if s in set(a.clips)]
    stems = [s for i, s in enumerate(stems) if i % a.of == a.shard]
    if a.limit:
        stems = stems[: a.limit]
    print(f"[repaint] tag {a.tag}: {len(stems)} clip(s), shard {a.shard}/{a.of}", flush=True)

    # --- copy the scored clip dirs (json only)
    for stem in stems:
        d = dst_root / stem
        (d / "augmentations").mkdir(parents=True, exist_ok=True)
        for f in JSONS:
            if (src_root / stem / f).exists() and not (d / f).exists():
                shutil.copy2(src_root / stem / f, d / f)
        if not (d / "scored_augmentations.json").exists():
            shutil.copy2(src_root / stem / "augmentations.json", d / "scored_augmentations.json")
        old = _ROOT / "data" / "work" / f"shipped_{a.tag}" / stem / "subjects.json"
        if a.verify and old.exists() and not (d / "subjects.json").exists():
            shutil.copy2(old, d / "subjects.json")          # the shipped subjects: try 1 == the shipped picture

    # --- phase A: subjects (VLM), one file per clip so a later crash never redoes them
    if a.phase in ("all", "subjects"):
        from src.types import AudioEvent
        from src.labels import choose_source
        from src.stage2_video_understanding import _sample_frames, _sample_frames_at
        from src.stage5_cross_modal_analysis import reason as R
        places = places_from_logs(a.tag)
        print(f"[repaint] places from the scored log: {len(places)}", flush=True)
        mdl = proc = None
        for stem in stems:
            d = dst_root / stem
            if (d / "subjects.json").exists():
                continue
            media = json.loads((d / "media.json").read_text(encoding="utf-8"))
            vp = Path(media["video_path"])
            _, specs = load_specs(d / "scored_augmentations.json")
            events = [AudioEvent(**{k: v for k, v in e.items() if k in AudioEvent.__dataclass_fields__})
                      for e in json.loads((d / "events.json").read_text(encoding="utf-8"))]
            drawn = [s for s in specs if s.augment]
            out = {"place": None, "place_from": None, "specs": {}}
            if drawn:
                if mdl is None:
                    t0 = time.time()
                    mdl, proc = R._load(config.VLM_MODEL, "cuda")
                    print(f"[repaint] VLM {config.VLM_MODEL} loaded in {time.time() - t0:.0f}s", flush=True)
                place, how = places.get(stem), "log"
                if not place:
                    frames = _sample_frames(vp, 4)
                    scene = " ".join(R._ask(mdl, proc, R.SCENE_PROMPT, images=frames, max_new=40).split())[:160] \
                        if frames else ""
                    scene = scene or "an unknown place"
                    place = (R._clean_phrase(R._ask(mdl, proc, R.PLACE_PROMPT, images=frames, max_new=16),
                                             max_words=4) if frames else "") or scene
                    how = "asked"
                out["place"], out["place_from"] = place, how
                print(f"== {stem}  place: {place} ({how})", flush=True)
                for s in drawn:
                    s.source = choose_source(raw_for(events, s.event_label), s.event_label, (s.start, s.end))
                    frames = _sample_frames_at(vp, spec_frames_times(s))
                    phrase = R._depict_v31(s, place, frames, [], mdl, proc) or ""
                    via = "v3.1"
                    if not phrase:
                        phrase, via = R._depict_v3(s, place, mdl, proc) or "", "v3"
                    if not phrase:
                        phrase, via = (s.source or s.event_label).split(",")[0].split("(")[0].strip() \
                            + " making its sound", "fallback"
                    out["specs"][str(s.index)] = {"label": s.event_label, "source": s.source, "subject": phrase,
                                                  "via": via, "old_subject": s.subject, "frames": len(frames)}
                    print(f"   [{s.index}] {s.event_label} [{s.source}] {s.subject!r} -> {phrase!r} ({via})",
                          flush=True)
            (d / "subjects.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        if mdl is not None:
            del mdl, proc
            R.unload()
            print("[repaint] VLM unloaded", flush=True)

    # --- phase B: pictures (Qwen-Image-2512 via the frozen final setup)
    if a.phase in ("all", "draw"):
        from src.labels import search_query
        from src.stage6_visual_augmentation import _final_picture, VERIFY_LOG
        n = {"card": 0, "template": 0, "diffusion": 0, "PLACEHOLDER": 0}
        nv = {}
        for stem in stems:
            d = dst_root / stem
            if (d / "repaint.json").exists():
                for r in json.loads((d / "repaint.json").read_text(encoding="utf-8")):
                    n[r["kind"]] = n.get(r["kind"], 0) + 1
                    if "verify" in r:
                        nv[r["verify"]["final"]] = nv.get(r["verify"]["final"], 0) + 1
                continue
            subj = json.loads((d / "subjects.json").read_text(encoding="utf-8"))
            raw, specs = load_specs(d / "scored_augmentations.json")
            log = []
            for s in specs:
                if not s.augment:
                    continue
                info = subj["specs"][str(s.index)]
                s.source, s.subject = info["source"], info["subject"]
                path = d / "augmentations" / f"aug_{s.index:03d}.png"
                t0 = time.time()
                _final_picture(s, path, d, search_query(s.subject or s.event_label), config.RESOLUTION,
                               config.GEN_MODEL, "cuda")
                k = kind_of(s)
                n[k] = n.get(k, 0) + 1
                log.append({"index": s.index, "label": s.event_label, "source": s.source, "subject": s.subject,
                            "old_subject": info["old_subject"], "kind": k, "prompt": s.image_prompt,
                            "image": path.name, "seconds": round(time.time() - t0, 1)})
                if a.verify:
                    v = next((r for r in reversed(VERIFY_LOG) if r["clip"] == stem and r["index"] == s.index), None)
                    if v is None:                     # a CARDS sound: never drawn, never checked
                        v = {"tries": [], "final": "card"}
                    log[-1]["verify"] = {"n_tries": len(v["tries"]), "final": v["final"], "tries": v["tries"]}
                    nv[v["final"]] = nv.get(v["final"], 0) + 1
                print(f"   {stem} [{s.index}] {s.event_label} -> {s.subject} -> {k} ({time.time() - t0:.0f}s)",
                      flush=True)
            # write the copy: only subject/source/image_prompt/image_path/backend may differ from the scored file
            by_i = {s.index: s for s in specs}
            new = []
            for r in raw:
                r = dict(r)
                s = by_i[r["index"]]
                if s.augment:
                    r.update(subject=s.subject, source=s.source, image_prompt=s.image_prompt,
                             image_path=s.image_path, backend=s.backend)
                new.append(r)
            for r0, r1 in zip(raw, new):
                for k in KEEP:
                    assert r0.get(k) == r1.get(k), f"{stem}: {k} changed for spec {r0.get('index')}"
            assert len(raw) == len(new)
            (d / "augmentations.json").write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
            (d / "repaint.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[repaint] done: {n}" + (f" verify: {nv}" if a.verify else ""), flush=True)


if __name__ == "__main__":
    main()

"""Detector SUGGESTIONS for the human tagging tool (tagger/, clips d030-d100).

Runs the scored stage-4 stack (config.use_scored(): BEATs + FlexSED 215 family queries at 0.8 + PANNs veto 0.05 +
onset refinement) through the same call as src/pipeline.py (detect_events), then keeps what the pipeline would put
on screen: salient non-speech, non-music labels (labels.is_salient_nonspeech, LABEL_FILTER "depictable"), firings
at the display bar (min_confidence("", DISPLAY_THRESHOLD)), and a family only if its best firing clears
min_confidence(family, DISPLAY_THRESHOLD) -- as in stage5.plan_augmentations. Firings are grouped per family into
bursts (gap 1.5 s, a superset of the pipeline's 1.0-s merge); each burst is one suggestion, named by its strongest
raw label when that name is in tagger/src/vocab.js, else by the family name.

The videos are copied to data/work/tagger_suggest/videos as tsug_<id>.mp4 (a unique stem: the FlexSED cache and the
flat wav folder are keyed by the stem). Steps:
    python benchmark/gold/flexsed_run.py --clip-dir data/work/tagger_suggest/videos --out data/work/flexsed_tagger
    python benchmark/gold/tagger_suggest.py            -> benchmark/gold/tagger_suggest.json
Nothing here reads gold labels; the output is for display in the tagger only (hidden until the human asks).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

VID = _ROOT / "data" / "work" / "tagger_suggest" / "videos"
WORK = _ROOT / "data" / "work" / "tagger_suggest" / "work"
FXDIR = _ROOT / "data" / "work" / "flexsed_tagger"
OUT = _ROOT / "benchmark" / "gold" / "tagger_suggest.json"
VOCAB_JS = _ROOT / "tagger" / "src" / "vocab.js"
GAP = 1.5
NOT_TAGGED = re.compile(r"^\s*(speech|speaking|talk|talking|voice|voices|conversation|narration|narrator|music|song|singing|soundtrack)\b", re.I)


def vocab():
    t = VOCAB_JS.read_text(encoding="utf-8")
    t = t[t.index("["): t.rindex("]") + 1]
    v = json.loads(t)
    return {x.lower(): x for x in v}


def duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 1e9


def main():
    changed = config.use_scored()
    config.DEVICE = "cuda"
    print("use_scored:", {k: v[1] for k, v in changed.items()}, flush=True)
    from src.stage1_audio_extraction import extract_audio
    import src.stage4_audio_event_detection as S
    from src.stage4_audio_event_detection import flexsed_infer as FX
    from src.labels import is_salient_nonspeech, min_confidence, canonical, is_music, is_descendant
    FX.CACHE_DIR = FXDIR                                  # cache_path reads the module global at call time
    VOC = vocab()
    print(f"vocab {len(VOC)} names; FlexSED cache {FXDIR}", flush=True)

    # detect_events does not return which spans FlexSED raised alone; the wrapper records them (behaviour unchanged)
    flex = {"ids": set()}
    _fuse = S.fuse_flexsed

    def fuse(*a, **k):
        ev, ids, ffw = _fuse(*a, **k)
        flex["ids"] = set(ids)
        return ev, ids, ffw
    S.fuse_flexsed = fuse

    bar = min_confidence("", config.DISPLAY_THRESHOLD)
    out = {}
    vids = sorted(VID.glob("tsug_*.mp4"))
    print(f"{len(vids)} clips, display bar {bar}", flush=True)
    for i, p in enumerate(vids, 1):
        cid = p.name[len("tsug_"):]
        stem = p.stem
        if not FX.cache_path(stem).exists():
            raise SystemExit(f"no FlexSED cache for {stem}: run flexsed_run.py first (the run would not be the scored stack)")
        media = extract_audio(p, WORK / stem / "audio.wav", config.SAMPLE_RATE)
        flex["ids"] = set()
        events = S.detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR,
                                 model=config.AED_MODEL, device="cuda")
        dur = duration(p)
        firings = []
        for e in events:
            if not is_salient_nonspeech(e.label) or e.confidence < bar:
                continue
            fam = canonical(e.label)
            if NOT_TAGGED.match(fam) or is_music(fam) or is_descendant(fam, "Speech") or is_descendant(fam, "Music"):
                continue
            firings.append((fam, float(e.start), float(e.end), float(e.confidence), e.label,
                            "FlexSED" if id(e) in flex["ids"] else "BEATs"))
        best = {}
        for f in firings:
            best[f[0]] = max(best.get(f[0], 0.0), f[3])
        sugg = []
        for fam in sorted(best):
            if best[fam] < min_confidence(fam, config.DISPLAY_THRESHOLD):
                continue
            bursts = []
            for f in sorted((x for x in firings if x[0] == fam), key=lambda x: x[1]):
                if bursts and f[1] - bursts[-1]["end"] <= GAP:
                    b = bursts[-1]
                    b["end"] = max(b["end"], f[2]); b["src"].add(f[5])
                    if f[3] > b["conf"]:
                        b["conf"], b["raw"] = f[3], f[4]
                else:
                    bursts.append({"start": f[1], "end": f[2], "conf": f[3], "raw": f[4], "src": {f[5]}})
            for b in bursts:
                raw = b["raw"]
                label = VOC.get(raw.lower()) or VOC.get(fam.lower()) or fam
                sugg.append({"label": label, "family": fam, "raw": raw,
                             "start": round(max(0.0, b["start"]), 1), "end": round(min(b["end"], dur), 1),
                             "conf": round(b["conf"], 2), "source": "+".join(sorted(b["src"]))})
        sugg.sort(key=lambda s: (s["start"], s["end"], s["label"]))
        out[cid] = sugg
        print(f"[{i}/{len(vids)}] {cid}: {len(sugg)} suggestion(s) "
              + "; ".join(f"{s['label']} {s['start']}-{s['end']} {s['conf']} {s['source']}" for s in sugg), flush=True)
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

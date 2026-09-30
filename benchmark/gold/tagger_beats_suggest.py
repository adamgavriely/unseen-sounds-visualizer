"""BEATs-only detector SUGGESTIONS for Adam's tagging tool (tagger/), and the "2+ sounds" check for new clips.

Adam, 30 Sept 2026: "use only BEATs just to give suggestions"; every clip in the tagger must have at least 2 detected
sounds. Same post-processing as tagger_suggest.py (salient non-speech, non-music labels at the display bar, a family
only if its best firing clears its own bar, bursts per family with a 1.5-s gap), but the detector is BEATs alone:
config.use_scored() with FLEXSED_BAR, FLEXSED_VETO and PANNS_VETO set to 0 (no second detector, no vetoes).

A clip's "sounds" = its suggested families grouped by labels.same_source (Duck and Quack are one sound, Duck and
Goose are two). n_sounds >= 2 is the keep rule for the tagger.

    python benchmark/gold/tagger_beats_suggest.py --videos tagger/videos --out benchmark/gold/tagger_beats_suggest.json
Nothing here reads gold labels.
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

WORK = _ROOT / "data" / "work" / "tagger_beats"
OUT = _ROOT / "benchmark" / "gold" / "tagger_beats_suggest.json"
VOCAB_JS = _ROOT / "tagger" / "src" / "vocab.js"
GAP = 1.5
NOT_TAGGED = re.compile(r"^\s*(speech|speaking|talk|talking|voice|voices|conversation|narration|narrator|music|song|singing|soundtrack)\b", re.I)


def vocab():
    t = VOCAB_JS.read_text(encoding="utf-8")
    return {x.lower(): x for x in json.loads(t[t.index("["): t.rindex("]") + 1])}


def duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 1e9


def sound_groups(families, same_source):
    groups = []
    for f in sorted(set(families)):
        hit = [g for g in groups if any(f == x or same_source(f, x) for x in g)]
        for g in hit:
            groups.remove(g)
        groups.append(sorted(set().union({f}, *hit)))
    return groups


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of .mp4 clips")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    vid_dir, out_path = Path(a.videos), Path(a.out)
    config.use_scored()
    for k in ("FLEXSED_BAR", "FLEXSED_VETO", "PANNS_VETO"):
        setattr(config, k, 0.0)
    config.DEVICE = a.device
    from src.stage1_audio_extraction import extract_audio
    import src.stage4_audio_event_detection as S
    from src.labels import is_salient_nonspeech, min_confidence, canonical, is_music, is_descendant, same_source
    VOC = vocab()
    bar = min_confidence("", config.DISPLAY_THRESHOLD)
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    vids = sorted(vid_dir.glob("*.mp4"))
    print(f"{len(vids)} clips, BEATs only, display bar {bar}", flush=True)
    for i, p in enumerate(vids, 1):
        if p.name in out:
            continue
        media = extract_audio(p, WORK / ("tb_" + p.stem) / "audio.wav", config.SAMPLE_RATE)
        events = S.detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR,
                                 model="beats", device=a.device)
        dur = duration(p)
        firings = []
        for e in events:
            if not is_salient_nonspeech(e.label) or e.confidence < bar:
                continue
            fam = canonical(e.label)
            if NOT_TAGGED.match(fam) or is_music(fam) or is_descendant(fam, "Speech") or is_descendant(fam, "Music"):
                continue
            firings.append((fam, float(e.start), float(e.end), float(e.confidence), e.label))
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
                    b["end"] = max(b["end"], f[2])
                    if f[3] > b["conf"]:
                        b["conf"], b["raw"] = f[3], f[4]
                else:
                    bursts.append({"start": f[1], "end": f[2], "conf": f[3], "raw": f[4]})
            for b in bursts:
                label = VOC.get(b["raw"].lower()) or VOC.get(fam.lower()) or fam
                sugg.append({"label": label, "family": fam, "raw": b["raw"],
                             "start": round(max(0.0, b["start"]), 1), "end": round(min(b["end"], dur), 1),
                             "conf": round(b["conf"], 2), "source": "BEATs"})
        sugg.sort(key=lambda s: (s["start"], s["end"], s["label"]))
        groups = sound_groups([s["family"] for s in sugg], same_source)
        out[p.name] = {"suggestions": sugg, "n_sounds": len(groups), "sounds": groups, "duration": round(dur, 2)}
        print(f"[{i}/{len(vids)}] {p.name}: {len(groups)} sound(s) {groups}", flush=True)
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()

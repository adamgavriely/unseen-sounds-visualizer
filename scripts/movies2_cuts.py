"""Wave 5b: three cuts per film scene + an audio-only check (2026-09-22, Adam: the single 40 % cuts
"did not contain unseen or mixed").

For every scene video already chosen by scripts/source_movies2.py (benchmark/sources_movies2.json) and
for scenes still without a video, take 15-s cuts at 25 %, 50 % and 75 % of the upload (the earlier
40 % cut is kept as it is). A cut is kept only if BEATs hears at least one drawable, non-speech,
non-music sound above 0.25 whose keyword importance is >= 2 (scripts/source_mixed2.audio_families).
Detector only: no gate, no visibility model. Cuts land in data/input/benchmark/unsorted/ as
m5_<genre>_<film>_<n><b|c|d>.mp4; the audio families go to benchmark/gold/movies2_audio.json (for
the VLM pre-screen and for ranking), sources to benchmark/sources_movies2.json.

Usage: python scripts/movies2_cuts.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import _accept, STAGE
import scripts.source_batch3 as b3
from scripts.source_movies2 import QUERIES, _pick, SEC, FMT, LOG
from scripts.source_mixed2 import audio_families
import os
NOAUDIO = bool(os.environ.get("NOAUDIO"))

AUDIO = ROOT / "benchmark" / "gold" / "movies2_audio.json"
BENCH = ROOT / "data" / "input" / "benchmark"
FRACS = {"b": 0.25, "c": 0.50, "d": 0.75}


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _known() -> dict[str, dict]:
    """name -> {url, title} for the scene videos already fetched (from the source log)"""
    out = {}
    if LOG.exists():
        for e in json.loads(LOG.read_text(encoding="utf-8")):
            n = e.get("name") or e.get("file", "").replace(".mp4", "")
            if n and e.get("url"):
                out[n] = e
    return out


def _duration(vid: str) -> float:
    r = _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--simulate", "--no-warnings", "--print", "%(duration)s"], timeout=120)
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return 0.0


def _grab(name: str, vid: str, start: float) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--download-sections", f"*{start:.0f}-{start + SEC:.0f}",
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def _exists(name: str) -> bool:
    return any((BENCH / d / f"{name}.mp4").exists() for d in ("unsorted", "_dropped", "_bad", "mixed", "seen_ambient", "unseen_ambient", "no_ambient"))


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    known = _known()
    audio = json.loads(AUDIO.read_text(encoding="utf-8")) if AUDIO.exists() else {}
    kept = 0
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 1        # start at this query index (e.g. 107 = the Fable scenes)
    for i, (genre, film, query) in enumerate(QUERIES, 1):
        if query is None or i < first:
            continue
        base = f"m5_{genre}_{film}_{i}"
        if base in known:
            vid = known[base]["url"].split("v=")[-1]; title = known[base].get("title", "")
        else:
            hit = _pick(query)
            if not hit:
                print(f"  {base}: no hit", flush=True); continue
            vid, _, title = hit
        dur = _duration(vid)
        if dur < 45:
            print(f"  {base}: duration unknown/short", flush=True); continue
        for suf, frac in FRACS.items():
            name = base + suf
            if _exists(name):
                continue
            start = max(15.0, frac * dur - SEC / 2)
            raw = _grab(name, vid, start)
            if raw is None:
                print(f"  {name}: skip (no download)", flush=True); continue
            if NOAUDIO:                      # Adam 2026-09-22: "don't even check audio, just give me scenes"
                fams = [{"family": "?", "detail": "?", "conf": 0, "start": 0, "end": 0}]
            try:
                fams = fams if NOAUDIO else audio_families(raw)
            except Exception as e:
                print(f"  {name}: audio failed ({type(e).__name__})", flush=True); raw.unlink(missing_ok=True); continue
            if not fams:
                print(f"  {name}: drop (no drawable sound event)", flush=True); raw.unlink(missing_ok=True); continue
            if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                   "targeted": "movies2 cuts (audio check only)", "genre": genre, "film": film, "cut": f"{int(frac * 100)}%",
                                   "license": "research use; not redistributed"}):
                kept += 1
                if not NOAUDIO:
                    audio[name] = fams
                AUDIO.write_text(json.dumps(audio, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"  {name}: KEEP ({kept}) {[f['family'] for f in fams]}", flush=True)
    print(f"done: {kept} kept")


if __name__ == "__main__":
    main()

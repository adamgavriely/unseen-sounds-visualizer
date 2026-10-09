"""Step 13: AudioSet-Strong TRAIN audio from YouTube (Adam's standing OK, 8 Oct), audio only, the labelled 10-s segment,
16 kHz mono FLAC; video never kept. Every benchmark YouTube id (the name scan, ~/open_data/leak_names.txt) is skipped.
Resumable: done / failed ids are logged. Cluster CPU, env msproj (yt-dlp, ffmpeg).

    python benchmark/gold/coverage/fetch_as_strong.py [LIMIT] [WORKERS]
-> ~/open_data/as_strong/audio/<segment_id>.flac, fetch_log.tsv (segment, status)
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path.home() / "open_data" / "as_strong"
OUT = D / "audio"
LOG = D / "fetch_log.tsv"


def leak():
    rx = re.compile(r"[A-Za-z0-9_-]{11}")
    ids = set()
    for line in (Path.home() / "open_data" / "leak_names.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        ids.update(rx.findall(line))
    return ids


def one(seg):
    yt, ms = seg[:11], int(seg[12:])
    s = ms / 1000.0
    dst = OUT / f"{seg}.flac"
    if dst.exists():
        return seg, "ok"
    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "--download-sections", f"*{s}-{s + 10}",
                                "--force-keyframes-at-cuts", "-o", f"{td}/a.%(ext)s", f"https://www.youtube.com/watch?v={yt}"],
                               capture_output=True, text=True, timeout=180)
            src = [p for p in Path(td).iterdir()]
            if r.returncode != 0 or not src:
                return seg, "fail:" + (r.stderr.strip().splitlines() or ["?"])[-1][:120].replace("\t", " ")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src[0]), "-t", "10", "-ac", "1", "-ar", "16000",
                            str(dst)], check=True, timeout=120)
            return seg, "ok"
        except Exception as e:
            return seg, "fail:" + str(e)[:120].replace("\t", " ")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUT.mkdir(parents=True, exist_ok=True)
    segs = sorted({l.split("\t")[0] for l in (D / "audioset_train_strong.tsv").read_text().splitlines()[1:] if l})
    bad = leak()
    skip = [s for s in segs if s[:11] in bad]
    segs = [s for s in segs if s[:11] not in bad]
    seen = {}
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            a, b = l.split("\t", 1)
            seen[a] = b
    todo = [s for s in segs if s not in seen]
    if limit:
        todo = todo[:limit]
    print(f"segments {len(segs) + len(skip)}, benchmark ids skipped {len(skip)}, already tried {len(seen)}, to do {len(todo)}", flush=True)
    with open(LOG, "a") as lg, ThreadPoolExecutor(workers) as ex:
        for n, (seg, st) in enumerate(ex.map(one, todo), 1):
            lg.write(f"{seg}\t{st}\n"); lg.flush()
            if n % 200 == 0:
                ok = sum(1 for _ in OUT.iterdir())
                print(f"{n}/{len(todo)} tried, {ok} files", flush=True)
    print("FETCH_AS_DONE")


if __name__ == "__main__":
    main()

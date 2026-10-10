"""AS100 step 1: pick the 100 clips, collect gold + our stage-4 events (frozen D' arm SHIP8+MD3+WW5+SL, 'proposed').
Run on the login node (light):  python as100_prep.py   -> ~/as100_eval/{clips.txt, gold.json, ours.json, wav_paths.json}
"""
import glob, json, os
from pathlib import Path

H = Path.home()
WORK = H / "MscProj_tg" / "data" / "work"
OUT = H / "as100_eval"
ARM = "SHIP8+MD3+WW5+SL|proposed"

ours_all, wav = {}, {}
for f in sorted(glob.glob(str(WORK / "r13freshf0*" / "stage4.json"))):
    a = json.load(open(f))["arms"][ARM]
    for st, ev in a.items():
        ours_all[st] = ev
        wav[st] = str(Path(f).parent / "wav16" / f"{st}.wav")
assert len(ours_all) == 340, len(ours_all)
clips = sorted(ours_all)[:100]
for st in clips:
    assert os.path.exists(wav[st]), wav[st]

mid2name = {}
for line in open(H / "open_data" / "as_strong" / "mid_to_display_name.tsv", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        mid2name[p[0]] = p[1]
gold = {st: [] for st in clips}
for i, line in enumerate(open(H / "open_data" / "as_strong" / "audioset_eval_strong.tsv", encoding="utf-8")):
    if i == 0:
        continue
    seg, s, e, mid = line.rstrip("\n").split("\t")
    if seg in gold:
        gold[seg].append([float(s), float(e), mid2name[mid]])
OUT.mkdir(exist_ok=True)
(OUT / "clips.txt").write_text("\n".join(clips) + "\n")
json.dump(gold, open(OUT / "gold.json", "w"), indent=0)
json.dump({st: [[e["start"], e["end"], e["label"], e["conf"]] for e in ours_all[st]] for st in clips},
          open(OUT / "ours.json", "w"), indent=0)
json.dump({st: wav[st] for st in clips}, open(OUT / "wav_paths.json", "w"), indent=0)
print("clips", len(clips), "gold events", sum(len(v) for v in gold.values()),
      "ours events", sum(len(ours_all[s]) for s in clips), "clips without gold", sum(1 for v in gold.values() if not v))

# our stage-4 vocabulary (BEATs 527 tagger labels + FlexSED query labels), read from their cached outputs.
# Needs numpy (run in env sota for this part).
try:
    import numpy as np
    b = sorted(glob.glob(str(WORK / "j2_freshf00_beats" / "*.npz")))[0]
    f = sorted(glob.glob(str(WORK / "flexsed_fresh" / "*.npz")))[0]
    json.dump({"beats": [str(x) for x in np.load(b)["labels"]], "flexsed": [str(x) for x in np.load(f)["labels"]]},
              open(OUT / "our_vocab.json", "w"))
    print("our_vocab.json written")
except ImportError:
    print("numpy missing: our_vocab.json not written (run this file in env sota)")

"""Energy-rise salience test (Fable idea 7; Adam 10 Oct: keep working). On top of v1.4, all 158 clips.
A heard event usually rises above the background; a phantom tag often does not. For each picture: rise = 10 log10 of
the mean energy over [start - 0.25, start + 0.75] s divided by the median 100-ms-frame energy over [start - 3, start - 0.5]
s (whole band, and the 1-8 kHz band), clip audio at 16 kHz. Rule: drop when rise < r dB; band in {full, 1-8k},
r in {-3, 0, 2, 4} or off, chosen by clip-grouped 5-fold CV on onset cost. Cluster CPU from ~/wt_slice.

    python benchmark/gold/coverage/salience.py -> salience.md
"""
import itertools, json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
WAV = Path.home() / "MscProj" / "data" / "work" / "gold_wav_flat"
GRID = [None] + list(itertools.product(("full", "hi"), (-3.0, 0.0, 2.0, 4.0)))


def rises(st, pics):
    import soundfile as sf, librosa
    from scipy.signal import butter, sosfilt
    x, sr = sf.read(str(WAV / f"{st}.wav"), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000); sr = 16000
    hi = sosfilt(butter(4, [1000, 7900], btype="band", fs=sr, output="sos"), x)
    out = []
    for lab, a, b in pics:
        r = {}
        for nm, y in (("full", x), ("hi", hi)):
            seg = y[max(0, int((a - 0.25) * sr)):int((a + 0.75) * sr)]
            bg = y[max(0, int((a - 3.0) * sr)):max(0, int((a - 0.5) * sr))]
            if len(seg) < sr // 10 or len(bg) < sr // 5:
                r[nm] = 99.0; continue
            fr = bg[: len(bg) // 1600 * 1600].reshape(-1, 1600)
            base = float(np.median((fr ** 2).mean(1))) + 1e-10
            r[nm] = float(10 * np.log10((seg ** 2).mean() / base + 1e-10))
        out.append(r)
    return out


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    pics = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    R = {st: rises(st, pics[st]) for st in allc}
    apply = lambda st, g: pics[st] if g is None else [p for p, r in zip(pics[st], R[st]) if r[g[0]] >= g[1]]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: S.viewer_cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    L = ["# Energy-rise salience test, all 158 clips", "", f"v1.4: {summ([row(st, None) for st in allc])}",
         f"CV choices {ch}; out of fold: {summ([oof[st] for st in allc])}", "", "| band, r | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "salience.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()

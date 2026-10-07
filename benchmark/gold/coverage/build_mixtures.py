"""Step 5 (PREREG_step5_detector_finetune.md): 10-s training mixtures, quiet drawable events under speech / music /
crowd / noise beds, frame labels on the PretrainedSED grid (40 ms). Cluster CPU, env msproj, run from ~/MscProj_tg.

    python benchmark/gold/coverage/build_mixtures.py [N_TRAIN=20000] [N_VAL=1000]
-> ~/open_data/mix/{train,val}_NNN.npz  (audio int16 [n, 160000]; events: rows (mix, class index, f0, f1)); classes.json
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_salient_nonspeech, is_music, ancestors

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("OPEN_DATA", Path.home() / "open_data"))
OUT = D / "mix"
SR, SEG, HOP = 16000, 160000, 640          # 40-ms label frames: hop 160 x pooling 4
NF = SEG // HOP                            # 250
SHARD = 500
CROWD = {"Crowd", "Chatter", "Hubbub, speech noise, speech babble"}


def strong_classes():
    sys.path.insert(0, str(Path.home() / "PretrainedSED"))
    from data_util.audioset_classes import as_strong_train_classes
    return list(as_strong_train_classes)


def to_onto(name):
    from src.stage4_audio_event_detection.psed_infer import STRONG_TO_ONTOLOGY
    return STRONG_TO_ONTOLOGY.get(name, name)


def drawable(onto):
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return bool(is_salient_nonspeech(onto)) and not is_music(onto) and canonical(onto) not in ("Speech", "Music")
    finally:
        config.LABEL_FILTER = old


def leak_ids():
    """every 11-char YouTube-like id and every file stem under benchmark/ and data/ (as audioset_fresh.py scans)"""
    ids = set()
    rx = re.compile(r"[A-Za-z0-9_-]{11}")
    for root in (_ROOT / "benchmark", _ROOT / "data"):
        for p in root.rglob("*"):
            ids.add(p.stem)
            ids.update(rx.findall(p.name))
            if p.is_file() and p.suffix in (".txt", ".json", ".csv", ".tsv") and p.stat().st_size < 50e6:
                try:
                    ids.update(rx.findall(p.read_text(encoding="utf-8", errors="ignore")))
                except Exception:
                    pass
    return ids


def load(path, n=None):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def activity(x):
    """40-ms frames with RMS above -30 dB of the max; gaps < 0.2 s bridged, runs < 0.2 s dropped"""
    n = len(x) // HOP
    if n == 0:
        return np.zeros(0, bool)
    r = np.sqrt((x[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1) + 1e-12)
    a = 20 * np.log10(r / r.max()) > -30
    k = 5                                       # 0.2 s = 5 frames
    i = 0
    while i < n:                                # bridge short gaps
        if not a[i]:
            j = i
            while j < n and not a[j]:
                j += 1
            if 0 < i and j < n and j - i < k:
                a[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:                                # drop short runs
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i < k:
                a[i:j] = False
            i = j
        else:
            i += 1
    return a


def rms(x):
    return float(np.sqrt((x ** 2).mean() + 1e-12))


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng = random.Random(0)
    classes = strong_classes()
    cidx = {c: i for i, c in enumerate(classes)}
    onto2strong = {}
    for c in classes:
        onto2strong.setdefault(to_onto(c), []).append(c)
    leaks = leak_ids()
    # foreground pool: (path, [strong classes incl. ancestors present]) ; bed pool: (path, kind)
    fg, beds, removed = [], [], 0
    gt = D / "fsd50k" / "FSD50K.ground_truth"
    for split, adir in (("dev", "FSD50K.dev_audio"), ("eval", "FSD50K.eval_audio")):
        for line in (gt / f"{split}.csv").read_text(encoding="utf-8").splitlines()[1:]:
            m = re.match(r'^(\d+),"?([^"]*)"?,', line)
            if not m:
                continue
            fname, labs = m.group(1), [l.replace("_", " ") for l in m.group(2).split(",")]
            path = D / "fsd50k" / adir / f"{fname}.wav"
            if fname in leaks:
                removed += 1; continue
            if any(l in ("Speech", "Music") or is_music(l) for l in labs):
                continue
            if set(labs) & CROWD:
                beds.append((str(path), "crowd")); continue
            ok = [l for l in labs if l in onto2strong and drawable(l)]
            if not ok:
                continue
            leaf = [l for l in ok if not any(o != l and l in ancestors(o) for o in ok)]
            tgt = set()
            for l in leaf:
                for a in [l] + list(ancestors(l)):
                    tgt.update(onto2strong.get(a, []))
            fg.append((str(path), sorted(tgt)))
    esc = json.loads((HERE / "esc50_map.json").read_text(encoding="utf-8"))
    edir = D / "esc50" / "ESC-50-master"
    for line in (edir / "meta" / "esc50.csv").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split(",")
        lab = esc.get(f[3])
        onto = to_onto(lab) if lab else None
        if not lab or Path(f[0]).stem.split("-")[1] in leaks:
            continue
        tgt = set(onto2strong.get(onto, [])) | ({lab} if lab in cidx else set())
        for a in ancestors(onto):
            tgt.update(onto2strong.get(a, []))
        if tgt and drawable(onto):
            fg.append((str(edir / "audio" / f[0]), sorted(tgt)))
    for kind, sub in (("speech", "speech"), ("music", "music"), ("noise", "noise")):
        beds += [(str(p), kind) for p in (D / "musan" / "musan" / sub).rglob("*.wav")]
    beds += [(str(p), "speech") for p in (D / "librispeech" / "LibriSpeech" / "train-clean-100").rglob("*.flac")]
    rng.shuffle(fg); rng.shuffle(beds)
    cut_f, cut_b = int(len(fg) * 0.9), int(len(beds) * 0.9)
    pools = {"train": (fg[:cut_f], beds[:cut_b], n_train), "val": (fg[cut_f:], beds[cut_b:], n_val)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.json").write_text(json.dumps({"classes": classes, "fg": len(fg), "beds": len(beds), "leak_removed": removed,
                                                  "bed_kinds": {k: sum(1 for _, b in beds if b == k) for k in ("speech", "music", "noise", "crowd")}},
                                                 indent=1), encoding="utf-8")
    print(f"foreground files {len(fg)}, beds {len(beds)}, removed for leakage {removed}", flush=True)
    kinds = ["speech", "music", "noise", "crowd"]
    for split, (F, Bd, n) in pools.items():
        bykind = {k: [p for p, b in Bd if b == k] for k in kinds}
        for s0 in range(0, n, SHARD):
            dst = OUT / f"{split}_{s0 // SHARD:03d}.npz"
            if dst.exists():
                continue
            audio, events = np.zeros((min(SHARD, n - s0), SEG), np.int16), []
            for i in range(audio.shape[0]):
                kind = rng.choice([k for k in kinds if bykind[k]])
                bed = load(rng.choice(bykind[kind]))
                while len(bed) < SEG:
                    bed = np.concatenate([bed, load(rng.choice(bykind[kind]))])
                o = rng.randrange(0, len(bed) - SEG + 1)
                mix = bed[o:o + SEG].copy()
                mix *= 10 ** (-20 / 20) / max(rms(mix), 1e-6)
                for _ in range(rng.randint(1, 3)):
                    path, tgt = rng.choice(F)
                    x = load(path)[: SEG]
                    act = activity(x)
                    if not act.any():
                        continue
                    xa = x[: len(act) * HOP].reshape(len(act), HOP)[act].ravel()
                    snr = rng.uniform(-20.0, -5.0)
                    x = x * (rms(mix) * 10 ** (snr / 20) / max(rms(xa), 1e-6))
                    st = rng.randrange(0, max(1, SEG - len(x)))
                    mix[st:st + len(x)] += x
                    f0 = st // HOP
                    on = np.flatnonzero(act)
                    for c in tgt:
                        # runs of activity, shifted to the event's place in the mixture
                        run_s = on[0]
                        for a, b in zip(on, list(on[1:]) + [None]):
                            if b is None or b != a + 1:
                                events.append((i, cidx[c], min(NF, f0 + run_s), min(NF, f0 + a + 1)))
                                if b is not None:
                                    run_s = b
                pk = np.abs(mix).max()
                if pk > 0.99:
                    mix *= 0.99 / pk
                audio[i] = (mix * 32767).astype(np.int16)
            np.savez(dst, audio=audio, events=np.array(events, np.int32).reshape(-1, 4))
            print(dst.name, audio.shape[0], len(events), flush=True)
    print("MIX_DONE")


if __name__ == "__main__":
    main()

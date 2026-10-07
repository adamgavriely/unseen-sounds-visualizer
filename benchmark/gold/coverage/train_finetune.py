"""Step 5 (PREREG_step5_detector_finetune.md): fine-tune a PretrainedSED strong checkpoint on the open-data mixtures,
with the frozen original as teacher. Cluster GPU, env ~/venvs/psed2, run from ~/MscProj_tg. Resumes from its last epoch.

    python benchmark/gold/coverage/train_finetune.py ATST-F [OUT_NAME] [SEED]
-> ~/open_data/ft/OUT_NAME/epoch_E.pt (state dict), log.json (train / held-out mixture loss per epoch)
"""
import glob
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from src.stage4_audio_event_detection import psed_infer as P

D = Path(os.environ.get("OPEN_DATA", Path.home() / "open_data"))
EPOCHS = int(os.environ.get("FT_EPOCHS", 10))
_E, BATCH, LR_BACKBONE, LR_HEAD = 10, 32, 1e-5, 1e-4


def build(backbone):
    sys.path.insert(0, str(P.PSED_ROOT))
    cwd = os.getcwd(); os.chdir(P.PSED_ROOT)
    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "config"}
    try:
        from models.prediction_wrapper import PredictionsWrapper
        mod, cls, ckpt = P.BACKBONES[backbone]
        wrapper = getattr(importlib.import_module(mod), cls)()
        extra = {"embed_dim": wrapper.m2d.cfg.feature_d} if backbone == "M2D" else {}
        return PredictionsWrapper(wrapper, checkpoint=ckpt, **extra)
    finally:
        os.chdir(cwd)
        sys.modules.pop("config", None)
        sys.modules.update(ours)


def shards(split):
    return sorted(glob.glob(str(D / "mix" / f"{split}_*.npz")))


def batches(split, rng, shuffle=True):
    files = shards(split)
    if shuffle:
        rng.shuffle(files)
    for f in files:
        z = np.load(f)
        a, ev = z["audio"], z["events"]
        idx = list(range(len(a)))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), BATCH):
            ii = idx[s:s + BATCH]
            pos = {j: k for k, j in enumerate(ii)}
            x = torch.from_numpy(a[ii].astype(np.float32) / 32767.0)
            e = [(pos[m], c, f0, f1) for m, c, f0, f1 in ev if m in pos]
            yield x, e


def target(teacher_p, events):
    t = teacher_p.clone()
    for b, c, f0, f1 in events:
        t[b, c, f0:f1] = 1.0
    return t


def run_epoch(student, teacher, split, opt, sched, rng, dev):
    train = opt is not None
    student.train(train)
    tot, n = 0.0, 0
    lossf = torch.nn.BCEWithLogitsLoss()
    for x, ev in batches(split, rng, shuffle=train):
        x = x.to(dev)
        with torch.no_grad():
            tp = torch.sigmoid(teacher(teacher.mel_forward(x))[0].float())
        y = target(tp, ev)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
            logit = student(student.mel_forward(x))[0]
        loss = lossf(logit.float(), y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()
        tot += float(loss) * len(x); n += len(x)
    return tot / max(n, 1)


def main():
    backbone = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else f"{backbone}_ft_s0"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed); rng = random.Random(seed)
    dev = "cuda"
    out = D / "ft" / name
    out.mkdir(parents=True, exist_ok=True)
    teacher = build(backbone).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build(backbone).to(dev)
    head = [p for n_, p in student.named_parameters() if not n_.startswith("model.")]
    body = [p for n_, p in student.named_parameters() if n_.startswith("model.")]
    opt = torch.optim.AdamW([{"params": body, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    n_train = sum(len(np.load(f)["audio"]) for f in shards("train"))
    steps = EPOCHS * ((n_train + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    logp = out / "log.json"
    log = json.loads(logp.read_text()) if logp.exists() else {"backbone": backbone, "seed": seed, "epochs": []}
    start = len(log["epochs"])
    if start:
        st = torch.load(out / f"state_{start}.pt", map_location=dev, weights_only=False)   # our own resume file
        student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.setstate(tuple(st["rng"]) if isinstance(st["rng"], list) else st["rng"])
    else:
        log["epochs"].append({"epoch": 0, "val_loss": run_epoch(student, teacher, "val", None, None, rng, dev)})
        logp.write_text(json.dumps(log, indent=1)); start = 1
        print("epoch 0 val", log["epochs"][-1], flush=True)
    for ep in range(start, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(student, teacher, "train", opt, sched, rng, dev)
        va = run_epoch(student, teacher, "val", None, None, rng, dev)
        torch.save({"model": student.state_dict()}, out / f"epoch_{ep}.pt")
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.getstate()},
                   out / f"state_{ep + 1}.pt")
        old = out / f"state_{ep}.pt"
        if old.exists():
            old.unlink()
        log["epochs"].append({"epoch": ep, "train_loss": tr, "val_loss": va, "seconds": time.time() - t0})
        logp.write_text(json.dumps(log, indent=1))
        print(f"epoch {ep} train {tr:.5f} val {va:.5f} ({time.time() - t0:.0f} s)", flush=True)
    best = min(log["epochs"][1:], key=lambda e: e["val_loss"])
    log["best_epoch"] = best["epoch"]
    logp.write_text(json.dumps(log, indent=1))
    print("TRAIN_DONE best epoch", best, flush=True)


if __name__ == "__main__":
    main()

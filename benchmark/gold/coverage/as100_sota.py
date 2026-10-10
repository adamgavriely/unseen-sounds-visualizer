"""AS100 step 2: PretrainedSED (Schmid et al.) AudioSet-Strong models on the 100 clips -> frame probabilities.
GPU, env sota:  python as100_sota.py ATST-F BEATs fpasst M2D ASIT
Writes ~/as100_eval/sota_<model>.npz: per clip an array [frames, 447] of sigmoid probabilities (40 ms frames,
raw = no median filter) + 'classes'. Same chunking as PretrainedSED/inference.py (10 s chunks, zero-pad).
"""
import json, sys, os
from pathlib import Path
import numpy as np
import torch, librosa

PS = Path.home() / "PretrainedSED"
sys.path.insert(0, str(PS)); os.chdir(PS)
from data_util import audioset_classes
from models.prediction_wrapper import PredictionsWrapper

OUT = Path.home() / "as100_eval"
SR, SEG = 16000, 160000


def build(name):
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1")
    if name == "fpasst":
        from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
        return PredictionsWrapper(FPaSSTWrapper(), checkpoint="fpasst_strong_1")
    if name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        m = M2DWrapper()
        return PredictionsWrapper(m, checkpoint="M2D_strong_1", embed_dim=m.m2d.cfg.feature_d)
    if name == "ASIT":
        from models.asit.ASIT_wrapper import ASiTWrapper
        return PredictionsWrapper(ASiTWrapper(), checkpoint="ASIT_strong_1")
    raise ValueError(name)


def main():
    wav = json.load(open(OUT / "wav_paths.json"))
    clips = [l.strip() for l in open(OUT / "clips.txt") if l.strip()]
    dev = torch.device("cuda")
    for name in sys.argv[1:]:
        model = build(name).eval().to(dev)
        res = {}
        for st in clips:
            w, _ = librosa.load(wav[st], sr=SR, mono=True)
            n = len(w)
            x = torch.from_numpy(w[None, :]).to(dev)
            preds = []
            for i in range(n // SEG + (n % SEG != 0)):
                ch = x[:, i * SEG:(i + 1) * SEG]
                if ch.shape[1] < SEG:
                    ch = torch.nn.functional.pad(ch, (0, SEG - ch.shape[1]))
                with torch.no_grad():
                    y, _ = model(model.mel_forward(ch))
                preds.append(y)
            y = torch.sigmoid(torch.cat(preds, dim=2))[0].T.float().cpu().numpy()   # [frames, 447]
            nf = int(np.ceil(n / SR / 0.04))
            res[st] = y[:nf]
        np.savez_compressed(OUT / f"sota_{name}.npz", classes=np.array(audioset_classes.as_strong_train_classes), **res)
        print(name, "done", len(res), res[clips[0]].shape, flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

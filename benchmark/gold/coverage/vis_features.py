"""Step 12 version 1 (PREREG_step12_visible_source_head.md, amendment 1): per-second visibility features.
Cluster GPU, env ~/venvs/denseav, run from ~/MscProj_tg.

    python benchmark/gold/coverage/vis_features.py items     # CPU: scratch_cov/s12/items.json (AVATAR train + DEV seconds)
    python benchmark/gold/coverage/vis_features.py run [i/n] # GPU: scratch_cov/s12/feats_<i>of<n>.json (resumes)
"""
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
HERE = Path(__file__).resolve().parent
S12 = _ROOT / "scratch_cov" / "s12"
AV = Path.home() / "open_data" / "avatar"
CKPT = "https://marhamilresearch4.blob.core.windows.net/denseav-public/hub/denseav_2head.ckpt"
FPS = 6


def items():
    rx = re.compile(r"[A-Za-z0-9_-]{11}")
    leak = set()
    for line in (Path.home() / "open_data" / "leak_names.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        leak.update(rx.findall(line))
    rng = random.Random(0)
    neg, pos, dropped = [], {}, 0
    for f in sorted(glob.glob(str(AV / "*" / "*.json"))):
        vid = Path(f).parent.name
        if vid[:11] in leak:
            dropped += 1; continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        t = float(d["frame_number"]) / float(d.get("fps") or 25.0)
        for a in d.get("annotations", []):
            it = {"src": "avatar", "video": str(AV / "video" / f"{vid}.mp4"), "t": round(t, 3),
                  "text": a.get("audio_visual_category", ""), "dur": float(d.get("duration") or 10.0)}
            if a.get("task") == "Off-Screen":
                neg.append({**it, "y": 0})
            elif a.get("task") == "Single-Sound" and a.get("bbox"):
                pos.setdefault(vid, []).append({**it, "y": 1})
    vids = sorted(pos)
    rng.shuffle(vids)
    P = [rng.choice(pos[v]) for v in vids[:2000]]
    dev = []
    a = json.loads((HERE / "omni_items_dev.json").read_text(encoding="utf-8"))
    for x in a["a"]:
        dev.append({"src": "dev", "id": x["id"], "clip": x["clip"], "gate": x["gate"], "video": str(_ROOT / a["videos"][x["clip"]]),
                    "t": x["t"], "text": x["label"], "dur": None})
    S12.mkdir(parents=True, exist_ok=True)
    out = neg + P + dev
    for i, x in enumerate(out):
        x["k"] = i
    (S12 / "items.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"AVATAR negatives {len(neg)}, positives {len(P)} (videos removed by the benchmark scan: {dropped} frames); DEV seconds {len(dev)}")


def run(shard):
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    from torchaudio.functional import resample
    from denseav.train import LitAVAligner
    from denseav.shared import norm, crop_to_divisor
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    si, sn = (int(v) for v in shard.split("/"))
    src = os.environ.get("S12_ITEMS", "items.json")
    its = [x for x in json.loads((S12 / src).read_text(encoding="utf-8")) if x["k"] % sn == si]
    outp = S12 / f"feats_{Path(src).stem}_{si}of{sn}.json"
    done = json.loads(outp.read_text()) if outp.exists() else {}
    m = LitAVAligner.load_from_checkpoint(CKPT, loss_leak=0.0, use_cached_embs=False, strict=True)
    m.set_full_train(True)
    m = m.cuda().eval()
    op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval()
    tf = T.Compose([T.Resize(224, Image.BILINEAR), lambda x: crop_to_divisor(x, 8), lambda x: x.to(torch.float32) / 255, norm])
    for n, x in enumerate(its):
        if str(x["k"]) in done:
            continue
        t = float(x["t"])
        a0 = max(0.0, t - 1.0)
        try:
            fr, au, info = torchvision.io.read_video(x["video"], start_pts=a0, end_pts=t + 2.0, pts_unit="sec")
        except Exception as e:
            done[str(x["k"])] = {"error": str(e)[:200]}; continue
        if fr.shape[0] < 2 or au.numel() == 0:
            done[str(x["k"])] = {"error": "no frames or audio"}; continue
        vf = float(info.get("video_fps") or 25.0)
        step = max(1, int(round(vf / FPS)))
        fsel = list(range(0, fr.shape[0], step))
        ftimes = np.array([a0 + i / vf for i in fsel])
        au = resample(au.float(), int(info["audio_fps"]), 16000).mean(0, keepdim=True)
        frames = torch.cat([tf(fr[i].permute(2, 0, 1)).unsqueeze(0) for i in fsel], 0)
        with torch.no_grad():
            af = m.forward_audio({"audio": au.cuda()})
            imf = m.forward_image({"frames": frames.unsqueeze(0).cuda()}, max_batch_size=4)
            s = m.sim_agg.get_pairwise_sims({**imf, **af}, raw=False, agg_sim=False, agg_heads=False)[:, :, :, :, 0, :].float().cpu().numpy()
        # s: [frames, heads, H, W, audio steps]; audio steps span the cut uniformly
        na = s.shape[-1]
        atimes = a0 + (np.arange(na) + 0.5) * (au.shape[1] / 16000.0) / na
        fi = np.where((ftimes >= t) & (ftimes < t + 1.0))[0]
        ai = np.where((atimes >= t) & (atimes < t + 1.0))[0]
        if not len(fi) or not len(ai):
            done[str(x["k"])] = {"error": "empty second"}; continue
        sub = s[fi][..., ai]                                     # [f, heads, H, W, a]
        feat = {}
        for h in range(sub.shape[1]):
            mp = sub[:, h]                                       # [f, H, W, a]
            mx = mp.max(axis=(1, 2)).mean(); mn = mp.mean(axis=(1, 2)).mean()
            feat[f"h{h}_max"], feat[f"h{h}_mean"], feat[f"h{h}_peak"] = float(mx), float(mn), float(mx - mn)
        mid = fsel[fi[len(fi) // 2]]
        img = Image.fromarray(fr[mid].numpy())
        q = [" ".join(f"a photo of a {x['text'].split(',')[0].lower()}".split()[:8])]
        inp = op(text=q, images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = om(**inp)
        tsz = torch.tensor([[max(img.size), max(img.size)]], device="cuda")
        r = op.post_process_object_detection(o, threshold=0.0, target_sizes=tsz)[0]
        feat["box_score"] = float(r["scores"].max()) if len(r["scores"]) else 0.0
        feat["box_in_out"] = None
        feat["motion_in_out"] = None
        if len(r["scores"]) and feat["box_score"] >= 0.1:
            bx = r["boxes"][int(r["scores"].argmax())].cpu().numpy()
            W0, H0 = img.size
            hmap = sub[:, 1].mean(axis=(0, 3))                   # sound head (head 1), [H, W]
            Hs, Ws = hmap.shape
            x0, y0 = int(np.clip(bx[0] / W0 * Ws, 0, Ws - 1)), int(np.clip(bx[1] / H0 * Hs, 0, Hs - 1))
            x1, y1 = int(np.clip(np.ceil(bx[2] / W0 * Ws), x0 + 1, Ws)), int(np.clip(np.ceil(bx[3] / H0 * Hs), y0 + 1, Hs))
            inside = hmap[y0:y1, x0:x1].mean(); mask = np.ones_like(hmap, bool); mask[y0:y1, x0:x1] = False
            outside = hmap[mask].mean() if mask.any() else inside
            feat["box_in_out"] = float(inside - outside)
            g = fr.float().mean(-1).numpy()                      # grey frames [N, H0, W0]
            d = np.abs(np.diff(g, axis=0))
            ft = np.array([a0 + i / vf for i in range(1, len(g))])
            xa, ya, xb, yb = [int(v) for v in np.clip(bx, 0, [W0, H0, W0, H0])]
            def ratio(sel):
                if not sel.any():
                    return None
                dd = d[sel]
                inn = dd[:, ya:yb, xa:xb].mean() if yb > ya and xb > xa else 0.0
                return float(inn / (dd.mean() + 1e-6))
            now = ratio((ft >= t) & (ft < t + 1)); before = ratio((ft >= t - 1) & (ft < t))
            feat["motion_in_out"] = None if now is None or before is None else now - before
        y = fr.float().mean(dim=(1, 2, 3)).numpy()
        fl = 0.0
        for i in range(len(y)):
            if a0 + i / vf < t or a0 + i / vf >= t + 1:
                continue
            base = y[max(0, i - int(vf)):i]
            if len(base) >= 3:
                med = np.median(base); sig = max(2.0, 1.4826 * np.median(np.abs(base - med)))
                fl = max(fl, float((y[i] - med) / sig))
        feat["flash"] = fl
        done[str(x["k"])] = feat
        if n % 50 == 0:
            outp.write_text(json.dumps(done))
            print(n, len(its), flush=True)
    outp.write_text(json.dumps(done))
    print("FEATS_DONE", len(done))


if __name__ == "__main__":
    items() if sys.argv[1] == "items" else run(sys.argv[2] if len(sys.argv) > 2 else "0/1")

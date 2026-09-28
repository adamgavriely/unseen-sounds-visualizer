"""Audio-to-image examples for the inspector's "Audio-to-image" tab (display only, no scoring).

Model: AudioToken (Yariv et al., InterSpeech 2023), official code https://github.com/guyyariv/AudioToken,
official weights from the authors' HF space GuyYariv/AudioToken (BEATs encoder + embedder, no LoRA,
as in the authors' demo and inference.py defaults), on CompVis/stable-diffusion-v1-4.

Each clip is cut into fixed 2-s windows (hop 2 s) over the whole clip, as the model would run with no
detector. A last partial window is kept if it is at least 1 s long. One 512x512 picture per window,
same seed (0) for every window, so only the audio differs between pictures.

Run on the cluster:  sbatch slurm/job_a2i.sh
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
AT = _ROOT / "ckpts" / "audiotoken"          # repo/ (git clone) + models/ (weights from the HF space)
sys.path.insert(0, str(AT / "repo"))
from modules.BEATs.BEATs import BEATs, BEATsConfig  # noqa: E402
from modules.AudioToken.embedder import FGAEmbedder  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402

sys.path.insert(0, str(_ROOT / "benchmark" / "gold"))
from flexsed_run import clip_path  # noqa: E402

CLIPS = ["m4_film_1917_33a", "w8_dashcam_ambulance_behind_1a", "m4_clay_shoot_11a", "m4_live_fire_26a",
         "m5_war_fury_25b", "mc_bridge_scene", "w8_kids_fire_alarm_school_1b", "m4_fire_bodycam_23a"]
WIN, HOP, MIN_LAST = 2.0, 2.0, 1.0
SR = 16000
SEED = 0
PROMPT = "a photo of <*>, 4k, high resolution"   # inference.py default
STEPS, CFG, RES = 50, 7.5, 512
SD = "CompVis/stable-diffusion-v1-4"
OUT = _ROOT / "docs" / "inspector" / "media" / "A2I"
JSON_OUT = _ROOT / "docs" / "inspector" / "a2i.json"


def load_wav(p: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def windows(n_samples: int):
    dur = n_samples / SR
    k, s = 0, 0.0
    while s < dur:
        e = min(s + WIN, dur)
        if e - s >= MIN_LAST:
            yield k, s, e
        k += 1
        s += HOP


def main():
    dev = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(SD, safety_checker=None, requires_safety_checker=False,
                                                   torch_dtype=torch.float32).to(dev)
    pipe.set_progress_bar_config(disable=True)
    tok, te = pipe.tokenizer, pipe.text_encoder
    assert tok.add_tokens("<*>") == 1
    te.resize_token_embeddings(len(tok))
    tid = tok.convert_tokens_to_ids("<*>")

    ck = torch.load(AT / "models" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", map_location="cpu")
    beats = BEATs(BEATsConfig(ck["cfg"]))
    beats.load_state_dict(ck["model"])
    beats.predictor = None
    beats = beats.to(dev).eval()
    emb = FGAEmbedder(input_size=768 * 3, output_size=768)
    emb.load_state_dict(torch.load(AT / "models" / "embedder_learned_embeds.bin", map_location="cpu"))
    emb = emb.to(dev).eval()

    clips_out = []
    for c in CLIPS:
        p = clip_path(c + ".mp4") or clip_path(c)   # clip_path wants the file name
        assert p is not None, c
        wav = load_wav(p)
        od = OUT / c
        od.mkdir(parents=True, exist_ok=True)
        wins = []
        for k, s, e in windows(len(wav)):
            seg = wav[int(round(s * SR)):int(round(e * SR))]
            x = torch.from_numpy(seg).unsqueeze(0).to(dev)
            torch.manual_seed(SEED)   # FGA uses F.dropout even in eval: fix it too
            with torch.no_grad():
                feats = beats.extract_features(x)[1]
                token = emb(feats)
                te.get_input_embeddings().weight.data[tid] = token.reshape(-1).to(te.dtype)
                g = torch.Generator(device=dev).manual_seed(SEED)
                img = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=CFG, height=RES, width=RES,
                           generator=g).images[0]
            img.save(od / f"{k}.png", optimize=True)
            wins.append({"k": k, "start": round(s, 3), "end": round(e, 3), "image": f"media/A2I/{c}/{k}.png"})
            print(f"[a2i] {c} k={k} {s:.1f}-{e:.2f}s", flush=True)
        clips_out.append({"clip": c, "split": "TEST", "video": f"media/TEST/{c}.mp4", "windows": wins})

    doc = {
        "model": "AudioToken (Yariv et al., InterSpeech 2023), official embedder + BEATs weights from "
                 "huggingface.co/spaces/GuyYariv/AudioToken on Stable Diffusion v1.4 (no LoRA); "
                 "https://github.com/guyyariv/AudioToken",
        "window_s": int(WIN),
        "note": ("AudioToken turns a sound into one extra word for Stable Diffusion (BEATs audio encoder -> "
                 "learned token, trained on VGGSound); we ran it with no detector on fixed 2-s windows "
                 "(hop 2 s, raw 2-s audio at 16 kHz mono, last window kept if >= 1 s), prompt "
                 f"'{PROMPT}', {STEPS} steps, guidance {CFG}, {RES}x{RES}, the same seed ({SEED}) for every window."),
        "clips": clips_out,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(doc, indent=1))
    print("[a2i] wrote", JSON_OUT, flush=True)


if __name__ == "__main__":
    main()

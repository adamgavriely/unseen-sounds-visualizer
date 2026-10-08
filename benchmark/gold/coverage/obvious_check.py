"""Step 9 (PREREG_step9_obvious_check.md): Qwen3-Omni, video only, on every a/b-candidate DEV picture. Cluster GPU.
    python benchmark/gold/coverage/obvious_check.py run     -> obvious_dev.json   (needs step2_pics_dev.json, videos)
    python benchmark/gold/coverage/obvious_check.py report  (local)"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "obvious_dev.json"
AB = "SHIP8+MD3+WW5+SL|AB-m"
Q = ("These frames are from a video shown without sound. Would a viewer already expect to hear a {label} sound here, "
     "just from what is on screen? Answer yes or no.")


def run():
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold.coverage.omni_probe import MODEL
    pics = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"][AB]["clips"]
    vids = json.loads((HERE / "verify_items_dev.json").read_text(encoding="utf-8"))["videos"]
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    tmp = Path(tempfile.mkdtemp())
    for st, v in pics.items():
        for lab, a, b in v["pics_none"]:
            k = f"{st}|{lab}|{a:.3f}"
            if k in done:
                continue
            cut = tmp / "c.mp4"
            subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-ss", str(max(0.0, a - 1.0)),
                            "-to", str(a + 2.0), "-i", str(_ROOT / vids[st]), "-an", "-c:v", "libx264", "-preset", "veryfast", str(cut)],
                           check=True)
            conv = [{"role": "user", "content": [{"type": "video", "video": str(cut)}, {"type": "text", "text": Q.format(label=lab.lower())}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
            inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inp = inp.to(model.thinker.device).to(torch.bfloat16)
            with torch.inference_mode():
                lg = model.thinker(**inp).logits[0, -1].float()
            done[k] = float(torch.sigmoid(lg[yes].max() - lg[no].max()))
            OUT.write_text(json.dumps(done))
    print("DONE", len(done))


def report():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.coverage import score_coverage as V
    from benchmark.gold.coverage.omni_probe import auroc
    from benchmark.gold.inspector_data import classify
    gold = S.load_gold([V.GOLD]); dev = V.stems("dev")
    pics = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"][AB]["clips"]
    ans = json.loads(OUT.read_text())
    sc, y = [], []
    for st in dev:
        _, pp = classify(gold[st], [tuple(x) for x in pics[st]["pics_none"]])
        for p in pp:
            k = f"{st}|{p['label']}|{p['start']:.3f}"
            if p["class"].startswith("hit") or p["class"] == "wrong: source visible or obvious":
                sc.append(ans[k]); y.append(p["class"] == "wrong: source visible or obvious")
    L = ["# Step 9: 'expected from the picture' check on a/b-candidate pictures (DEV)", "",
         f"AUROC (wrong visible/obvious vs hit pictures): {auroc(sc, y):.3f} ({sum(y)} vs {len(y) - sum(y)})", "",
         "| remove when P(yes) >= q | hits | wrong | onset cost |", "|---|---|---|---|", "| a/b candidate | 32 | 16 | 1.915 |"]
    for q in (0.5, 0.7, 0.9):
        rows = [V.score_clip_v2(gold[st], [tuple(x) for x in pics[st]["pics_none"] if ans[f"{st}|{x[0]}|{x[1]:.3f}"] < q]) for st in dev]
        A = V.aggregate(rows)
        L.append(f"| {q} | {A['hits']} | {A['wrong']} | {A['onset_cost']:.3f} |")
    (HERE / "obvious_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()

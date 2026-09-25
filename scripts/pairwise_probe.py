"""Check that GLM-4.6V-Flash really receives both pictures in the pairwise prompt (3 pairs)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from benchmark.gold.pairwise import pairs_round1, MODEL
proc = AutoProcessor.from_pretrained(MODEL)
mdl = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
for p in pairs_round1()[:3]:
    ims = [Image.open(x).convert("RGB").resize((384, 384)) for x in (p["good"], p["bad"])]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
             {"type": "text", "text": "Describe the first picture in five words, then the second picture in five words."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
    print({k: tuple(v.shape) for k, v in inp.items() if hasattr(v, "shape")})
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=200, do_sample=False)
    print(repr(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0][-300:]))

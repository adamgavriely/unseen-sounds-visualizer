import sys, json; sys.path.insert(0, ".")
import config
config.DEVICE = "cuda"
from pathlib import Path
import benchmark.gold.picture_bench as PB
from src.stage6_visual_augmentation import _diffusion_image, plain_prompt, _is_blank
B = Path("data/work/picture_bench_confirm"); it = [x for x in json.load(open(B/"specs.json")) if x["i"] == 12][0]
p = B/"A0"/"12.png"
print("blank before:", _is_blank(p))
ok = _diffusion_image(p, plain_prompt(it["subject"]), (768, 768), model="black-forest-labs/FLUX.1-schnell", device="cuda", seed=PB.seed_of(it) + 1)
print("redrawn", ok, "blank after:", _is_blank(p), it["subject"])
m = json.load(open(B/"A0"/"manifest.json"))
for x in m:
    if x["i"] == 12: x["shipped_blank_redraw"] = True; x["seed_redraw"] = PB.seed_of(it) + 1
json.dump(m, open(B/"A0"/"manifest.json","w"), indent=1)

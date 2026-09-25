"""Labelled sheets of one arm (source + prompt under each picture) for the by-eye rule-2 check (GP-4)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
bench = ROOT / sys.argv[1]
arm = sys.argv[2]
subjects = sys.argv[3]
out = ROOT / "data" / "work" / f"check_{arm}"
out.mkdir(parents=True, exist_ok=True)
specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
subj = json.loads((bench / f"subjects_{subjects}.json").read_text(encoding="utf-8"))
man = {m["i"]: m for m in json.loads((bench / arm / "manifest.json").read_text(encoding="utf-8"))}
specs = [s for s in specs if s["i"] in man]
W = 240
for k in range(0, len(specs), 12):
    sheet = Image.new("RGB", (4 * (W + 8), 3 * (W + 44)), "white")
    d = ImageDraw.Draw(sheet)
    for j, it in enumerate(specs[k:k + 12]):
        i = it["i"]
        x, y = (j % 4) * (W + 8), (j // 4) * (W + 44)
        sheet.paste(Image.open(bench / arm / f"{i:02d}.png").convert("RGB").resize((W, W)), (x, y + 40))
        d.text((x + 2, y + 2), f"{i} src: {subj.get(str(i), {}).get('source', it['label'])[:26]}", fill=(0, 0, 0))
        d.text((x + 2, y + 18), man[i]["prompt"][:38], fill=(120, 0, 0))
    sheet.save(out / f"sheet_{k // 12 + 1}.jpg", quality=85)
print(len(specs), "pictures ->", out)

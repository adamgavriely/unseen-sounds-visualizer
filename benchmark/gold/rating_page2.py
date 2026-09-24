"""Rating page, round 2: the picture alone, and Adam types what he thinks is making the sound.
(2026-09-24, picture panel of five, docs/picture_panel_round2.md point 8)

Round 1's page showed the sound's FAMILY name under each picture, which both helped and hurt the
pictures and hid the very problem Adam then raised. This page shows no name at all: a deaf viewer gets
no name either. Adam types what he thinks is making a sound; the answers are scored afterwards against
the specific sound, with the key sealed. Ten pictures appear twice, far apart, to measure how
consistent a single rater is. Today's picture of every sound is included, so the comparison is paired.

    python benchmark/gold/rating_page2.py --bench <bench dir> --arms today,N --out data/work/rate_pictures2
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image

N_DUPLICATES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--today", required=True, help="folder of <clip>/augmentations/*.png of today's run")
    ap.add_argument("--arms", default="today,N")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bench, out = Path(a.bench), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))

    def src(arm, it):
        if arm == "today":
            return Path(a.today) / it["clip"] / "augmentations" / Path(it["shipped_image"]).name
        return bench / arm / f"{it['i']:02d}.png"

    rng = random.Random(24092027)
    pool = [(arm, it) for arm in a.arms.split(",") for it in specs if src(arm, it).exists()]
    rng.shuffle(pool)
    # ten repeats, each placed at least a third of the page away from its first appearance
    dup_from = rng.sample(range(len(pool) // 2), min(N_DUPLICATES, len(pool) // 2))
    order = list(pool)
    for k in sorted(dup_from, reverse=True):
        at = min(len(order), k + len(pool) // 3 + rng.randint(0, len(pool) // 3))
        order.insert(at, pool[k])

    cards, key = [], {}
    for n, (arm, it) in enumerate(order, 1):
        code = f"P{n:03d}"
        Image.open(src(arm, it)).convert("RGB").resize((384, 384)).save(out / "img" / f"{code}.jpg",
                                                                        quality=86)
        cards.append(code)
        key[code] = {"arm": arm, "i": it["i"], "clip": it["clip"], "family": it["label"],
                     "detail": it.get("detail", ""), "start": it["start"]}
    (out.parent / f"{out.name}_KEY_do_not_open_before_rating.json").write_text(
        json.dumps(key, indent=1), encoding="utf-8")
    (out / "index.html").write_text(page(cards), encoding="utf-8")
    print(f"{len(cards)} cards ({len(pool)} pictures + {len(order) - len(pool)} repeats) -> {out / 'index.html'}")


def page(cards) -> str:
    grid = "\n".join(f"""
<figure class="card" data-code="{c}">
  <img src="img/{c}.jpg" alt="picture {c}" loading="lazy">
  <figcaption class="code">{c}</figcaption>
  <label for="t{c}">What is making a sound here?</label>
  <input id="t{c}" type="text" autocomplete="off" placeholder="a few words">
  <button type="button" class="cant">Can't tell</button>
</figure>""" for c in cards)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Picture Rating Two</title>
<style>
 :root {{ --ink:#1c2230; --dim:#5d6678; --line:#dde1e8; --bg:#f4f5f8; --card:#fff; --done:#1f7a4d; --cant:#8a6d00; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
 header {{ background:var(--card); border-bottom:1px solid var(--line); padding:16px; position:sticky; top:0; z-index:2; }}
 header h1 {{ margin:0 0 4px; font-size:20px; }}
 .bar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; color:var(--dim); }}
 .bar button {{ font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:6px; background:#eef1f5; cursor:pointer; }}
 main {{ max-width:1200px; margin:0 auto; padding:16px; }}
 .how {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; max-width:70ch; margin-bottom:16px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; }}
 .card {{ margin:0; background:var(--card); border:2px solid var(--line); border-radius:8px; padding:8px; display:flex; flex-direction:column; gap:5px; }}
 .card img {{ width:100%; aspect-ratio:1; object-fit:contain; background:#fff; display:block; }}
 .code {{ font:12px ui-monospace,Menlo,monospace; color:var(--dim); }}
 label {{ font-size:13px; color:var(--dim); }}
 input {{ font:inherit; padding:5px 7px; border:1px solid var(--line); border-radius:6px; }}
 input:focus-visible, button:focus-visible {{ outline:2px solid #3b6fd8; }}
 .cant {{ font:inherit; padding:4px 0; border:1px solid var(--line); border-radius:6px; background:#f7f8fa; cursor:pointer; }}
 .card.done {{ border-color:var(--done); }}
 .card.cantell {{ border-color:var(--cant); }} .card.cantell .cant {{ background:var(--cant); color:#fff; }}
</style></head><body>
<header><h1>What is making a sound in this picture?</h1>
 <div class="bar"><span id="count">0 answered</span>
  <button id="copy">Copy my answers</button><button id="save">Save my answers as a file</button></div></header>
<main>
 <div class="how">
  <b>How to answer.</b> Each picture would appear beside a video, for a viewer who cannot hear. There
  is no name under it on purpose &mdash; a deaf viewer does not get one either. Glance at it for a
  second and type what you think is making a sound (for example the thing and what it is doing).
  If you cannot tell, press <i>Can't tell</i>. Some pictures appear twice; answer each time as if it
  were new. Your answers are kept in this browser as you go.
 </div>
 <div class="grid">{grid}</div>
</main>
<script>
 const KEY = "picture-rating-2-2026-09-24";
 let ans = {{}};
 try {{ ans = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ ans = {{}}; }}
 const cards = document.querySelectorAll(".card");
 function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(ans)); }} catch (e) {{}} paint(); }}
 function paint() {{
   let n = 0;
   cards.forEach(c => {{
     const v = ans[c.dataset.code];
     c.classList.toggle("done", !!v && v !== "__cant__");
     c.classList.toggle("cantell", v === "__cant__");
     if (v) n++;
   }});
   document.getElementById("count").textContent = n + " of " + cards.length + " answered";
 }}
 cards.forEach(c => {{
   const inp = c.querySelector("input"), code = c.dataset.code;
   if (ans[code] && ans[code] !== "__cant__") inp.value = ans[code];
   inp.addEventListener("input", () => {{ const v = inp.value.trim(); if (v) ans[code] = v; else delete ans[code]; save(); }});
   c.querySelector(".cant").addEventListener("click", () => {{ ans[code] = "__cant__"; inp.value = ""; save(); }});
 }});
 const payload = () => JSON.stringify({{when: new Date().toISOString(), answers: ans}}, null, 1);
 document.getElementById("copy").addEventListener("click", async () => {{
   try {{ await navigator.clipboard.writeText(payload()); alert("Copied. Paste it into the chat."); }}
   catch (e) {{ prompt("Copy this:", payload()); }}
 }});
 document.getElementById("save").addEventListener("click", () => {{
   const a = document.createElement("a");
   a.href = URL.createObjectURL(new Blob([payload()], {{type: "application/json"}}));
   a.download = "picture_answers_2.json"; a.click();
 }});
 paint();
</script>
</body></html>"""


if __name__ == "__main__":
    main()

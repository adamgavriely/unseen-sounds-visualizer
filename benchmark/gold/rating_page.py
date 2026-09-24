"""A blind rating page for the pictures, for someone who is not me. (2026-09-24)

Every picture verdict so far is mine, and I wrote the rules the new pictures come from. The
reviewers were unanimous that the result becomes defensible only when a second person rates the
pictures without knowing which version drew them. This builds that page: today's pictures and the
new ones, shuffled under codes, the sound's name shown, one question per picture. The key that says
which version drew each code is written to a separate file the page never loads.

    python benchmark/gold/rating_page.py --bench <dir with specs.json, arm folders> --out data/work/rate_pictures

Part 1 is the comparison that matters (today's pictures vs the new ones, one of each per sound).
Part 2 is optional: the same sounds drawn with two more random seeds each, which measures how much
of any difference is luck.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image

PART1 = [("shipped", 0), ("A3c", 0)]
PART2 = [("A0", 1), ("A0", 2), ("A3b", 1), ("A3b", 2)]
# A3c is the configuration recommended from the night: the new subject rules with the place-phrase
# strip, drawn by Qwen-Image, with ONLY the blank guard. A3b is the same plus an upper "full-frame"
# guard, which turned out to throw away the best thunder pictures (storm clouds with lightning) and
# drop both thunders, so it is not recommended. For A3c the picture is A3b's, except where A3b's upper
# guard fired: there the subject is unchanged from A2 (checked below) and the seed-0 Qwen-Image draw of
# that subject is A3's picture, i.e. exactly what A3c would have drawn.


def _a3c(bench: Path, it) -> Path:
    import json as _j
    man = {m["i"]: m for m in _j.loads((bench / "A3b" / "manifest.json").read_text(encoding="utf-8"))}
    sa = _j.loads((bench / "subjects_A2.json").read_text(encoding="utf-8"))
    sb = _j.loads((bench / "subjects_A2b.json").read_text(encoding="utf-8"))
    m = man[it["i"]]
    if m["guard_fired"] and sa[str(it["i"])]["subject"] == sb[str(it["i"])]["subject"]:
        return bench / "A3" / f"{it['i']:02d}.png"
    return bench / "A3b" / f"{it['i']:02d}.png"


def picture(bench: Path, shipped_dir: Path, arm: str, seed: int, it) -> Path:
    if arm == "A3c":
        return _a3c(bench, it)
    if arm == "shipped":
        return shipped_dir / it["clip"] / "augmentations" / Path(it["shipped_image"]).name
    folder = arm if seed == 0 else f"{arm}_s{seed}"
    return bench / folder / f"{it['i']:02d}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--shipped", required=True, help="folder of <clip>/augmentations/*.png")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bench, out = Path(a.bench), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))

    cards, key = {"1": [], "2": []}, {}
    rng = random.Random(24092026)
    for part, arms in (("1", PART1), ("2", PART2)):
        pool = [(arm, seed, it) for arm, seed in arms for it in specs]
        rng.shuffle(pool)
        for arm, seed, it in pool:
            src = picture(bench, Path(a.shipped), arm, seed, it)
            if not src.exists():
                continue
            if arm != "A3c" and arm != "shipped":
                folder = arm if seed == 0 else f"{arm}_s{seed}"
                mf = bench / folder / "manifest.json"
                if mf.exists() and any(x["i"] == it["i"] and x.get("dropped")
                                       for x in json.loads(mf.read_text(encoding="utf-8"))):
                    continue             # the system would show nothing here, so there is nothing to rate
            code = f"{part}-{len(cards[part]) + 1:03d}"
            Image.open(src).convert("RGB").resize((384, 384)).save(out / "img" / f"{code}.jpg", quality=86)
            cards[part].append({"code": code, "sound": it["label"]})
            key[code] = {"arm": arm, "seed": seed, "i": it["i"], "clip": it["clip"], "label": it["label"]}
    # the key lives OUTSIDE the page's folder, so opening the page cannot reveal it
    (out.parent / f"{out.name}_KEY_do_not_open_before_rating.json").write_text(
        json.dumps(key, indent=1), encoding="utf-8")
    (out / "index.html").write_text(page(cards), encoding="utf-8")
    print(f"part 1: {len(cards['1'])} pictures, part 2: {len(cards['2'])} -> {out / 'index.html'}")


def page(cards) -> str:
    def grid(part):
        return "\n".join(f"""
<figure class="card" data-code="{c['code']}">
  <img src="img/{c['code']}.jpg" alt="picture {c['code']}" loading="lazy">
  <figcaption><span class="code">{c['code']}</span> <span class="snd">{html.escape(c['sound'])}</span></figcaption>
  <div class="btns">
    <button data-v="yes">Yes</button><button data-v="no">No</button><button data-v="unsure">Not sure</button>
  </div>
</figure>""" for c in cards[part])
    n1, n2 = len(cards["1"]), len(cards["2"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Picture Rating</title>
<style>
 :root {{ --ink:#1c2230; --dim:#5d6678; --line:#dde1e8; --bg:#f4f5f8; --card:#fff;
          --yes:#1f7a4d; --no:#b3261e; --unsure:#8a6d00; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
 header {{ background:var(--card); border-bottom:1px solid var(--line); padding:18px 16px; position:sticky; top:0; z-index:2; }}
 header h1 {{ margin:0 0 4px; font-size:20px; }}
 .bar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; color:var(--dim); }}
 .bar button {{ font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:6px; background:#eef1f5; cursor:pointer; }}
 main {{ max-width:1200px; margin:0 auto; padding:16px; }}
 .how {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; max-width:70ch; }}
 h2 {{ font-size:17px; margin:28px 0 10px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:12px; }}
 .card {{ margin:0; background:var(--card); border:2px solid var(--line); border-radius:8px; padding:8px; }}
 .card img {{ width:100%; aspect-ratio:1; object-fit:contain; background:#fff; display:block; }}
 figcaption {{ margin:6px 0; }}
 .code {{ font:12px ui-monospace,Menlo,monospace; color:var(--dim); }}
 .snd {{ font-weight:600; }}
 .btns {{ display:flex; gap:6px; }}
 .btns button {{ flex:1; font:inherit; padding:5px 0; border:1px solid var(--line); border-radius:6px; background:#f7f8fa; cursor:pointer; }}
 .btns button:focus-visible {{ outline:2px solid #3b6fd8; }}
 .card[data-v="yes"] {{ border-color:var(--yes); }} .card[data-v="yes"] [data-v="yes"] {{ background:var(--yes); color:#fff; }}
 .card[data-v="no"] {{ border-color:var(--no); }}   .card[data-v="no"] [data-v="no"] {{ background:var(--no); color:#fff; }}
 .card[data-v="unsure"] {{ border-color:var(--unsure); }} .card[data-v="unsure"] [data-v="unsure"] {{ background:var(--unsure); color:#fff; }}
</style></head><body>
<header><h1>Would a viewer get the sound from this picture?</h1>
 <div class="bar"><span id="count">0 rated</span>
  <button id="copy">Copy my answers</button><button id="save">Save my answers as a file</button></div></header>
<main>
 <div class="how">
  <b>How to rate.</b> Each picture would appear beside a video to tell a deaf viewer about a sound
  they cannot hear. Under it is the sound's name. Imagine glancing at it for one second:
  <b>would you understand that this is the sound named?</b> Yes, no, or not sure.
  <br><br>
  The pictures come from different versions of the system, mixed together; the page does not say
  which is which, on purpose. Please do not look at the key file until you are done. Your answers
  are kept in this browser as you go.
 </div>
 <h2>Part 1 &mdash; {n1} pictures (the comparison that matters)</h2>
 <div class="grid">{grid("1")}</div>
 <h2>Part 2 &mdash; {n2} pictures (optional: the same sounds drawn again, to measure luck)</h2>
 <div class="grid">{grid("2")}</div>
</main>
<script>
 const KEY = "picture-rating-2026-09-24";
 let ans = {{}};
 try {{ ans = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ ans = {{}}; }}
 const cards = document.querySelectorAll(".card");
 function paint() {{
   cards.forEach(c => {{ const v = ans[c.dataset.code]; if (v) c.dataset.v = v; else delete c.dataset.v; }});
   document.getElementById("count").textContent = Object.keys(ans).length + " of " + cards.length + " rated";
 }}
 cards.forEach(c => c.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {{
   ans[c.dataset.code] = b.dataset.v;
   try {{ localStorage.setItem(KEY, JSON.stringify(ans)); }} catch (e) {{}}
   paint();
 }})));
 const payload = () => JSON.stringify({{rater: "", when: new Date().toISOString(), answers: ans}}, null, 1);
 document.getElementById("copy").addEventListener("click", async () => {{
   try {{ await navigator.clipboard.writeText(payload()); alert("Copied. Paste it into the chat."); }}
   catch (e) {{ prompt("Copy this:", payload()); }}
 }});
 document.getElementById("save").addEventListener("click", () => {{
   const a = document.createElement("a");
   a.href = URL.createObjectURL(new Blob([payload()], {{type: "application/json"}}));
   a.download = "picture_ratings.json"; a.click();
 }});
 paint();
</script>
</body></html>"""


if __name__ == "__main__":
    main()

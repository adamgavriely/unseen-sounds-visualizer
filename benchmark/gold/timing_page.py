"""Before / after, by eye: every sound whose picture moved under the onset rule. (2026-09-24)

Adam found the timing problem by watching the videos, so the fix should be checkable the same way:
for each needed sound whose picture start changed between two runs, the two rendered videos side by
side, a button that jumps both to a second before the sound, and one plain line saying when the sound
starts and when each picture appeared.

    python benchmark/gold/timing_page.py --before dev_symgen_v30 --after dev_mono_v31 --out data/work/timing_page
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD


def first_pic(pics, label, onset):
    cand = [(x, y) for l, x, y in pics if S.same_family(l, label)]
    return min(cand, key=lambda p: abs(p[0] - onset)) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = S.load_gold([GOLD])
    dev = S.subsets_of(gold)["dev"]
    out = Path(a.out)
    (out / "v").mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in sorted(dev):
        wb = _ROOT / "data" / "work" / f"protocol_proposed_{a.before}"
        wa = _ROOT / "data" / "work" / f"protocol_proposed_{a.after}"
        pb, pa = S.load_pictures(wb, stem, "proposed"), S.load_pictures(wa, stem, "proposed")
        if pb is None or pa is None:
            continue
        for g in gold[stem]:
            if not (g["needed"] and g["importance"] >= 2):
                continue
            b, f = first_pic(pb, g["label"], g["start"]), first_pic(pa, g["label"], g["start"])
            if b == f or (b and f and abs(b[0] - f[0]) < 0.1):
                continue
            hit = lambda p: bool(p) and S.in_window(p[0], g["start"], S.EARLY, S.LATE)
            rows.append({"clip": stem, "label": g["label"], "onset": g["start"], "end": g["end"],
                         "before": b, "after": f, "hit_before": hit(b), "hit_after": hit(f)})
    for r in rows:
        for tag, side in ((a.before, "before"), (a.after, "after")):
            src = _ROOT / "data" / "output" / f"protocol_proposed_{tag}" / f"{r['clip']}_augmented.mp4"
            dst = out / "v" / f"{r['clip']}__{side}.mp4"
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
    (out / "index.html").write_text(page(rows), encoding="utf-8")
    print(f"{len(rows)} sounds whose picture moved -> {out / 'index.html'}")


def when(p):
    return "no picture" if not p else f"{p[0]:.1f} s"


def verdict(r):
    if r["hit_after"] and not r["hit_before"]:
        return '<span class="good">now on time</span>'
    if r["hit_before"] and not r["hit_after"]:
        return '<span class="bad">was on time, now not</span>'
    return '<span class="same">moved, same verdict</span>'


def page(rows) -> str:
    cards = []
    for k, r in enumerate(rows):
        t0 = max(0.0, r["onset"] - 1.0)
        early = lambda p: "" if not p else (f" ({r['onset'] - p[0]:.1f} s before the sound)" if p[0] < r["onset"] - 0.05
                                             else f" ({p[0] - r['onset']:.1f} s after the sound starts)")
        cards.append(f"""
<section class="card" id="c{k}">
  <h3>{html.escape(r['label'])} <span class="clip">{html.escape(r['clip'])}</span> {verdict(r)}</h3>
  <p>The sound starts at <b>{r['onset']:.1f} s</b>. Before the fix the picture appeared at
     <b>{when(r['before'])}</b>{early(r['before'])}; after the fix at <b>{when(r['after'])}</b>{early(r['after'])}.</p>
  <button onclick="jump({k}, {t0:.1f})">Play both from {t0:.1f} s</button>
  <div class="pair">
    <figure><figcaption>before</figcaption><video id="b{k}" controls preload="metadata" src="v/{html.escape(r['clip'])}__before.mp4"></video></figure>
    <figure><figcaption>after the fix</figcaption><video id="a{k}" controls preload="metadata" src="v/{html.escape(r['clip'])}__after.mp4"></video></figure>
  </div>
</section>""")
    n_good = sum(1 for r in rows if r["hit_after"] and not r["hit_before"])
    n_bad = sum(1 for r in rows if r["hit_before"] and not r["hit_after"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Picture Timing Before and After</title>
<style>
 :root {{ --ink:#1d2330; --dim:#5f6879; --line:#dfe3ea; --bg:#f5f6f9; --card:#fff; --good:#1d7a4c; --bad:#b3261e; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }}
 header {{ background:var(--card); border-bottom:1px solid var(--line); padding:18px 16px; }}
 h1 {{ margin:0 0 4px; font-size:21px; }} .sub {{ color:var(--dim); max-width:75ch; }}
 main {{ max-width:1200px; margin:0 auto; padding:16px; display:grid; gap:16px; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; }}
 h3 {{ margin:0 0 4px; font-size:17px; }} .clip {{ font:12px ui-monospace,Menlo,monospace; color:var(--dim); font-weight:400; }}
 .good {{ color:var(--good); font-size:14px; }} .bad {{ color:var(--bad); font-size:14px; }} .same {{ color:var(--dim); font-size:14px; }}
 button {{ font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:6px; background:#eef1f5; cursor:pointer; margin:4px 0 10px; }}
 .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
 @media (max-width:800px) {{ .pair {{ grid-template-columns:1fr; }} }}
 figure {{ margin:0; }} figcaption {{ color:var(--dim); font-size:13px; margin-bottom:3px; }}
 video {{ width:100%; background:#000; border-radius:5px; display:block; }}
</style></head><body>
<header><h1>Did the pictures stop coming too early?</h1>
<div class="sub">Every sound whose picture moved after the timing fix: {len(rows)} in all,
<b>{n_good}</b> now on time that were not, <b>{n_bad}</b> that were on time and now are not.
Each pair is the same clip rendered before and after. Press the button to play both from a second
before the sound starts, and watch when the picture appears on the right.</div></header>
<main>{''.join(cards)}</main>
<script>
 function jump(k, t) {{
   for (const id of ["b" + k, "a" + k]) {{ const v = document.getElementById(id); v.currentTime = t; v.play(); }}
 }}
</script></body></html>"""


if __name__ == "__main__":
    main()

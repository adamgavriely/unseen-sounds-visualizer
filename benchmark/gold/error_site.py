"""Build a page for looking at what the system got wrong, in plain words.

Every number in this project says how often the system is wrong. None of them let anyone SEE it.
This collects each mistake on the DEV clips, pairs it with the clip and the moment it happened, and
writes a self-contained page: press play, jump to the second where it went wrong, read one sentence
saying what should have happened and what we did instead.

Two kinds of mistake:

  MISSED    a sound the annotator said a deaf viewer needs, and we showed no picture for it
  WRONG     a picture we showed that should not have been there

    python benchmark/gold/error_site.py --out data/work/error_site
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import bucket, GOLD, EARLY, LATE

TAG = "dev_symgen_v30"


def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


# plain-language reasons, one per cause, written for a reader who is not in the code
WHY = {
    "never detected": ("Our sound detector never heard it at all.",
                       "Nothing to fix in the rest of the pipeline - if the ear misses it, no picture can be drawn."),
    "heard elsewhere": ("Our detector heard this kind of sound in the clip, but at a different moment, not this one.",
                        "Usually the sound happens several times and we only catch one of them."),
    "gate silenced": ("We heard it, but decided the viewer can already SEE what is making the sound, so we stayed quiet.",
                      "The annotator disagreed - they marked this source as not visible."),
    "mistimed": ("We heard it and we drew it, but the picture came up at the wrong moment.",
                 "It counts as a miss because a picture that arrives late does not help."),
    "filtered": ("We heard it, but a rule removed it before a picture was made.", ""),
    "invented": ("We drew a picture of a sound that is not in this clip at all.",
                 "The detector named something that was never there."),
    "wrong-family": ("There IS a real sound at this moment, but we named it as the wrong kind of thing.",
                     ""),
    "gate-leak": ("The sound is real, but the annotator said the viewer can already see what is making it.",
                  "We should have stayed quiet and did not."),
    "late": ("The right sound, the right kind - but the picture appears outside the moment it was heard.", ""),
    "level-1": ("A real sound, but the annotator rated it as just background noise.", ""),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data" / "work" / "error_site"))
    ap.add_argument("--half", default="dev")
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.half]) & set(subs["bench"]))
    root = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"
    # what the VIEWER saw: the composite the pipeline rendered for this same run, original video
    # on the left and our panel on the right. Adam asked for this instead of the clean clip -- a
    # mistake is only judgeable next to the picture the system actually put on screen.
    rend = _ROOT / "data" / "output" / f"protocol_proposed_{TAG}"
    out = Path(a.out)
    vids = out / "clips"
    vids.mkdir(parents=True, exist_ok=True)

    items, used = [], set()
    for stem in stems:
        pics = S.load_pictures(root, stem, "proposed")
        if pics is None:
            continue
        vp = clip_path(stem)
        if vp is None:
            continue
        snds = gold[stem]
        ev = json.loads((root / stem / "events.json").read_text(encoding="utf-8")) if (root / stem / "events.json").exists() else []
        au = json.loads((root / stem / "augmentations.json").read_text(encoding="utf-8")) if (root / stem / "augmentations.json").exists() else []

        def covers(u, t):
            sp = u.get("spans") or [[u.get("start", -1), u.get("end", -1)]]
            return any(x - 1.0 <= t <= y + 1.0 for x, y in sp)

        # --- sounds we should have drawn and did not
        for s in snds:
            if not (s["needed"] and s["importance"] >= 2):
                continue
            if any(S.same_family(l, s["label"]) and S.in_window(x, s["start"], EARLY, LATE) for l, x, y in pics):
                continue
            cov = [u for u in au if S.same_family(u.get("event_label", ""), s["label"]) and covers(u, s["start"])]
            at = [e for e in ev if S.same_family(e["label"], s["label"]) and e["start"] - 1.0 <= s["start"] <= e["end"] + 1.0]
            anywhere = [e for e in ev if S.same_family(e["label"], s["label"])]
            if cov and not any(u.get("augment") for u in cov):
                cause = "gate silenced"
            elif cov:
                cause = "mistimed"
            elif at:
                cause = "filtered"
            elif anywhere:
                cause = "heard elsewhere"
            else:
                cause = "never detected"
            items.append({"kind": "MISSED", "clip": stem, "t": round(float(s["start"]), 1),
                          "label": s["label"], "importance": int(s["importance"]), "cause": cause})
            used.add(stem)

        # --- pictures we showed that should not have been there
        for lab, x, y in pics:
            if any(S.same_family(lab, s["label"]) and S.in_window(x, s["start"], EARLY, LATE)
                   and s["needed"] and s["importance"] >= 2 for s in snds):
                continue
            other = [s for s in snds if not S.same_family(lab, s["label"]) and s["start"] - 1.0 <= y and x <= s["end"] + 1.0]
            bk = bucket(lab, x, y, snds)
            if bk == "invented" and other:
                bk = "wrong-family"
            if bk == "counted-hit":
                continue
            items.append({"kind": "WRONG", "clip": stem, "t": round(float(x), 1),
                          "label": lab, "importance": 0, "cause": bk,
                          "real_now": sorted({s["label"] for s in other})[:3]})
            used.add(stem)

    missing = []
    for stem in sorted(used):
        src = rend / f"{stem}_augmented.mp4"
        if not src.exists():                       # fall back to the clean clip, and say so
            src = clip_path(stem)
            missing.append(stem)
        dst = vids / f"{stem}{src.suffix}"
        if not dst.exists():
            shutil.copy(src, dst)
    if missing:
        print(f"no rendered video for {len(missing)} clip(s): {', '.join(missing[:5])}")

    items.sort(key=lambda i: (i["kind"], i["cause"], i["clip"], i["t"]))
    (out / "errors.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    write_page(out, items, vids)
    print(f"{len(items)} mistakes on {len(used)} clips -> {out / 'index.html'}")


def write_page(out: Path, items, vids: Path):
    by_cause = {}
    for it in items:
        by_cause.setdefault((it["kind"], it["cause"]), []).append(it)
    nmiss = sum(1 for i in items if i["kind"] == "MISSED")
    nwrong = len(items) - nmiss

    cards = []
    for (kind, cause), group in sorted(by_cause.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        head, extra = WHY.get(cause, (cause, ""))
        cards.append(f'<h2 class="{ "miss" if kind == "MISSED" else "wrong" }">'
                     f'{"Sounds we missed" if kind == "MISSED" else "Pictures that were wrong"}'
                     f' &mdash; {html.escape(cause)} <span class="n">{len(group)}</span></h2>')
        cards.append(f'<p class="why">{html.escape(head)}'
                     + (f' <span class="dim">{html.escape(extra)}</span>' if extra else "") + "</p>")
        for it in group:
            f = next((p.name for p in vids.iterdir() if p.stem == it["clip"]), None)
            if not f:
                continue
            t = it["t"]
            line = (f'The annotator marked <b>{html.escape(it["label"])}</b> here and we showed nothing.'
                    if kind == "MISSED" else
                    f'We showed a picture of <b>{html.escape(it["label"])}</b> here.')
            if kind == "MISSED" and it.get("importance") == 3:
                line += ' <span class="imp">the annotator rated this one important</span>'
            if kind == "WRONG" and it.get("real_now"):
                line += f' The real sound at that moment was: <i>{html.escape(", ".join(it["real_now"]))}</i>.'
            cards.append(f"""
<div class="card">
  <video controls preload="metadata" src="clips/{html.escape(f)}#t={max(0, t - 2):.1f}"></video>
  <div class="meta">
    <div class="t">jump to <button onclick="this.closest('.card').querySelector('video').currentTime={t}; this.closest('.card').querySelector('video').play()">{t:.1f}s</button></div>
    <div class="clip">{html.escape(it["clip"])}</div>
    <p>{line}</p>
  </div>
</div>""")

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>What the system got wrong</title>
<style>
 body {{ font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#f6f7f9; color:#1d2125; }}
 header {{ background:#fff; border-bottom:1px solid #dfe3e8; padding:22px 28px; }}
 h1 {{ margin:0 0 6px; font-size:22px; }}
 .sub {{ color:#5a6672; }}
 main {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }}
 h2 {{ font-size:17px; margin:34px 0 4px; padding-left:10px; border-left:4px solid #999; }}
 h2.miss {{ border-color:#d9534f; }}
 h2.wrong {{ border-color:#e8a33d; }}
 h2 .n {{ color:#5a6672; font-weight:400; }}
 p.why {{ margin:4px 0 14px 14px; color:#3d4852; }}
 .dim {{ color:#7b8794; }}
 .card {{ background:#fff; border:1px solid #e2e6ea; border-radius:8px;
          padding:12px; margin-bottom:16px; }}
 /* the video is the original beside our panel, so it is very wide -- it gets the whole card */
 .card video {{ width:100%; display:block; border-radius:5px; background:#000; }}
 .meta {{ margin-top:10px; }}
 .clip {{ font-family: ui-monospace, Menlo, monospace; font-size:12px; color:#7b8794; margin:2px 0 8px; }}
 .t button {{ font:inherit; border:1px solid #c8cfd6; background:#eef1f4; border-radius:5px;
              padding:2px 9px; cursor:pointer; }}
 .t button:hover {{ background:#e2e8ee; }}
 .imp {{ background:#fff3cd; padding:1px 6px; border-radius:4px; font-size:13px; }}
 .legend {{ background:#fff; border:1px solid #e2e6ea; border-radius:8px; padding:14px 18px; margin-top:16px; }}
</style></head><body>
<header>
 <h1>What the system got wrong</h1>
 <div class="sub">Every mistake on the {len(set(i['clip'] for i in items))} development clips &mdash;
   <b>{nmiss}</b> sounds we missed, <b>{nwrong}</b> pictures that should not have been shown.</div>
</header>
<main>
 <div class="legend">
  <b>How to read this.</b> Each card is one mistake. Press the button to jump to the second where it
  happened. The heading above each group says, in plain words, why that group went wrong.
  <br><br>
  <b>Missed</b> means the annotator said a deaf viewer needs to know about that sound and we showed
  nothing. <b>Wrong</b> means we put a picture on screen that should not have been there.
  <br><br>
  <b>These are our own videos.</b> Each one is the original clip on the left and the panel the
  system produced on the right &mdash; so you see exactly what a viewer would have seen at that
  second, not the clean clip.
 </div>
 {''.join(cards)}
</main></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()

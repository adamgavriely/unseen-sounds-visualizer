"""Round 46 GOLD-SCOPE (docs/prereg_round13_detector_push.md): blind human recheck of EXPECT-A4's added wrong pictures (E),
a sample of SHIP8's own wrong pictures (S) and random no-picture moments (C).

    python benchmark/gold/gold_scope_items.py build    -> gold_scope_items.json (hidden key), docs/review/gold_scope_recheck.html, media
    python benchmark/gold/gold_scope_items.py score    -> reads gold_scope_answers.json, prints the pre-registered table
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
G = _ROOT / "benchmark" / "gold"
KEY, ANS = G / "gold_scope_items.json", G / "gold_scope_answers.json"
HTML, MEDIA = _ROOT / "docs" / "review" / "gold_scope_recheck.html", _ROOT / "docs" / "review" / "gold_scope_media"
PRE, POST = 0.5, 2.5
CLIPS = {"dev": 71, "test": 88}


def video(clip):
    for ext in ("mp4", "webm", "mkv", "mov"):
        hits = [p for p in (_ROOT / "data" / "input").rglob(f"{clip}.{ext}") if "_bad" not in p.parts]
        if hits:
            return hits[0]
    raise FileNotFoundError(clip)


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def gold_by_clip():
    d = json.loads((G / "annotations" / "gold_AG.json").read_text(encoding="utf-8"))
    out = {}
    for c in d["clips"]:
        if not isinstance(c, dict):
            continue
        st = Path(str(c.get("clip", ""))).stem
        out[st] = [v for e in c.get("sounds", []) for v in (e.get("label", ""), e.get("family", "")) if v]
    return out


def build():
    rng = random.Random(0)
    E = []
    for p in json.loads((G / "expect_a4_screen.json").read_text(encoding="utf-8"))["pictures"]:
        if p["outcome"] == "kept" and p["class_40d"] != "hit":
            E.append({"set": "E", "split": "dev", "clip": p["clip"], "family": p["family"], "at": p["start"], "cls": p["class_40d"]})
    for p in json.loads((G / "expect_test.json").read_text(encoding="utf-8"))["added"]:
        if p["class"] != "hit":
            E.append({"set": "E", "split": "test", "clip": p["clip"], "family": p["family"], "at": p["start"], "cls": p["class"]})
    pool = [dict(x, split="dev") for x in json.loads((G / "ledger_ship8.json").read_text(encoding="utf-8"))["wrong"]] + \
           [dict(x, split="test") for x in json.loads((G / "ledger_ship8_test.json").read_text(encoding="utf-8"))["wrong"]]
    pool = [x for x in pool if x["type"] in ("cross", "phantom")]
    S_ = [{"set": "S", "split": x["split"], "clip": x["clip"], "family": x["picture"], "at": x["at"], "cls": x["type"]}
          for x in rng.sample(pool, 20)]
    fams = sorted({x["family"] for x in E + S_})
    gold = gold_by_clip()
    pics = {}
    for x in pool + [dict(clip=e["clip"], at=e["at"]) for e in E]:
        pics.setdefault(x["clip"], []).append(x["at"])
    clips = sorted({x["clip"] for x in pool + E})
    C = []
    while len(C) < 10:
        clip = rng.choice(clips)
        dur = duration(video(clip))
        t = round(rng.uniform(0, max(0.0, dur - POST)), 2)
        if any(abs(t - a) < 2.0 for a in pics.get(clip, [])):
            continue
        fam = rng.choice([f for f in fams if not any(f.lower() in g.lower() or g.lower() in f.lower() for g in gold.get(clip, []))])
        C.append({"set": "C", "split": "?", "clip": clip, "family": fam, "at": t, "cls": "control"})
        pics.setdefault(clip, []).append(t)
    items = E + S_ + C
    rng.shuffle(items)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(items, 1):
        x["id"] = f"g{i:02d}"
        s = max(0.0, x["at"] - PRE)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-i", str(video(x["clip"])), "-t", f"{PRE + POST:.2f}",
                        "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a", "128k",
                        str(MEDIA / f"{x['id']}.mp4")], check=True)
    KEY.write_text(json.dumps({"items": items}, indent=1), encoding="utf-8")
    pub = [{"id": x["id"], "file": f"{x['id']}.mp4", "family": x["family"]} for x in items]
    HTML.write_text(TEMPLATE.replace("__ITEMS__", json.dumps(pub)), encoding="utf-8")
    print(f"{len(items)} items: E {len(E)}, S {len(S_)}, C {len(C)} -> {HTML}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def score():
    items = {x["id"]: x for x in json.loads(KEY.read_text(encoding="utf-8"))["items"]}
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    ans = ans.get("answers", ans)
    real = lambda a: a.get("q1") == "yes" and a.get("q2") == "no" and a.get("q3") == "yes"
    C = [i for i, x in items.items() if x["set"] == "C"]
    c_yes = sum(ans.get(i, {}).get("q1") == "yes" for i in C)
    print(f"control Q1-yes {c_yes}/{len(C)} (bar <= 2)")
    if c_yes > 2:
        print("STOP: control fails; nothing else computed (pre-registered)"); return
    from benchmark.gold import score_per_sound as S
    gold = gold_by_clip()
    listed = lambda x: any(S.same_family(x["family"], g) for g in gold.get(x["clip"], []))   # amendment 1: gold times it elsewhere
    for s in ("E", "S"):
        ids = [i for i, x in items.items() if x["set"] == s]
        lst = [i for i in ids if real(ans.get(i, {})) and listed(items[i])]
        print(f"{s}: real, but the gold lists this family elsewhere in the clip: {len(lst)} {lst}")
        k = sum(real(ans.get(i, {})) for i in ids if not listed(items[i]))
        print(f"{s}: real-unlisted (no same-family gold in the clip) {k}/{len(ids)}")
        print(f"{s}: real-unlisted {k}/{len(ids)} = {k / len(ids):.2f} {wilson(k, len(ids))}")
        if s == "E":
            for sp, d in (("dev", -0.028), ("test", +0.045)):
                kk = sum(real(ans.get(i, {})) for i in ids if items[i]["split"] == sp and not listed(items[i]))
                print(f"  EXPECT-A4 {sp}: d {d:+.3f} -> d_corr {d - 2 * kk / CLIPS[sp]:+.3f} (k = {kk})")
        for i in ids:
            x = items[i]
            print(f"   {i} {x['split']} {x['clip']} {x['family']} {x['at']} {x['cls']}: {ans.get(i, {})}")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sound Re-check</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#2f6fde;--ok:#2e8b57}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--card:#242422;--ink:#ecece8;--muted:#a3a39c;--line:#3a3a36;--accent:#6f9df0;--ok:#5cc28a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}h1{font-size:22px;margin:8px 0 4px}
.guide,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}
.card.done{border-color:var(--ok)}.sound{font-size:19px;font-weight:600}.meta{color:var(--muted);font-size:14px}
video{width:100%;max-height:380px;background:#000;border-radius:8px;margin:8px 0}.q{margin:10px 0 4px;font-weight:600}
.opts{display:flex;gap:8px;flex-wrap:wrap}.opts label{border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button{font:inherit;padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
textarea{width:100%;font:inherit;min-height:34px;border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--bg);color:var(--ink)}
.bar{position:sticky;bottom:0;background:var(--bg);padding:10px 0;display:flex;gap:10px;align-items:center;border-top:1px solid var(--line)}
.hide{display:none}
</style></head><body><main>
<h1>Sound re-check (41 short moments)</h1>
<div class="guide"><b>Each clip is 3 seconds.</b> A sound name is shown. Play the clip with sound on.
<ol><li><b>Q1.</b> Do you hear this sound start or play here?</li>
<li><b>Q2.</b> If yes: is the thing making it on screen?</li>
<li><b>Q3.</b> If yes: would a deaf viewer want a picture for it?</li></ol>
Pick "unsure" if you can't tell. Answers save in this browser. At the end press <b>Download answers</b> and put the file in
<code>P:\\MscProj\\benchmark\\gold\\</code> as <code>gold_scope_answers.json</code>.</div>
<div id="list"></div>
<div class="bar"><span id="count" class="meta"></span><span style="flex:1"></span><button class="primary" id="dl">Download answers</button></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "gold_scope_r46";
let A = {}; try { A = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(A)); } catch (e) {} count(); }
function count(){ const n = ITEMS.filter(x => done(A[x.id])).length; document.getElementById("count").textContent = n + " / " + ITEMS.length + " done"; }
function done(a){ return a && a.q1 && (a.q1 !== "yes" || (a.q2 && a.q3)); }
const QS = [["q1","Q1. Do you hear a <b>%s</b> sound start or play here?"],["q2","Q2. Is the thing making it on screen?"],["q3","Q3. Would a deaf viewer want a picture for it?"]];
const list = document.getElementById("list");
ITEMS.forEach((x, i) => {
  const a = A[x.id] = A[x.id] || {};
  const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<div class="sound">${i + 1}. ${x.family}</div><video controls preload="none" src="gold_scope_media/${x.file}"></video>` +
    QS.map(([k, t]) => `<div class="blk" data-k="${k}"><div class="q">${t.replace("%s", x.family)}</div><div class="opts">` +
      ["yes","no","unsure"].map(v => `<label><input type="radio" name="${x.id}_${k}" value="${v}" ${a[k] === v ? "checked" : ""}>${v}</label>`).join("") +
      `</div></div>`).join("") + `<div class="q">Note (optional)</div><textarea>${a.note || ""}</textarea>`;
  const sync = () => {
    c.querySelectorAll(".blk").forEach(b => { if (b.dataset.k !== "q1") b.classList.toggle("hide", a.q1 !== "yes"); });
    c.classList.toggle("done", !!done(a));
  };
  c.querySelectorAll("input").forEach(r => r.addEventListener("change", () => { a[r.name.split("_").pop()] = r.value; sync(); save(); }));
  c.querySelector("textarea").addEventListener("input", e => { a.note = e.target.value; save(); });
  sync(); list.appendChild(c);
});
count();
document.getElementById("dl").onclick = () => {
  const b = new Blob([JSON.stringify({answers: A}, null, 1)], {type: "application/json"});
  const u = URL.createObjectURL(b); const l = document.createElement("a"); l.href = u; l.download = "gold_scope_answers.json"; l.click();
};
</script></body></html>
"""

if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()

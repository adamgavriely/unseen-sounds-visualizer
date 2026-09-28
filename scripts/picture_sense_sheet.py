"""Picture sense test, local step (CPU): blind folder for the by-eye verdicts, the results table, and the contact sheet.

    python scripts/picture_sense_sheet.py --src <pulled data/work/picture_sense_test> --mine <pulled sense_cache/mine/..>
        --phase blind      -> <blind>/<id>.jpg + list.json (id, label, subject) and key.json (not to be opened first)
        --phase table      -> needs <blind>/verdicts.json; prints the per-arm table, writes <src>/summary.json
        --phase sheet      -> docs/picture_sense_test/index.html + img/
Plan: docs/picture_sense_test_2026-09-28.md.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DOC = _ROOT / "docs" / "picture_sense_test"


def load(src: Path):
    items = json.loads((src / "items.json").read_text("utf-8"))["items"]
    res = {}
    for arm in "ABC":
        p = src / f"results_{arm}.json"
        res[arm] = {r["key"]: r for r in json.loads(p.read_text("utf-8"))["rows"]} if p.exists() else {}
    for k, r in res["A"].items():                     # B = A on the unseen by construction
        if r["group"] == "unseen" and k not in res["B"]:
            res["B"][k] = {**r, "arm": "B", "same_as_A": True}
    return items, res


def phase_blind(src: Path, blind: Path):
    from PIL import Image
    items, res = load(src)
    blind.mkdir(parents=True, exist_ok=True)
    rng = random.Random(4242)
    rows = []
    for it in items:
        for arm in "ABC":
            r = res[arm].get(it["key"])
            if not r or r.get("same_as_A"):
                continue
            rows.append((it, arm, r))
    rng.shuffle(rows)
    lst, key = [], {}
    for n, (it, arm, r) in enumerate(rows):
        i = f"p{n:03d}"
        Image.open(src / r["image"]).convert("RGB").save(blind / f"{i}.jpg", quality=88)
        lst.append({"id": i, "label": it["source"], "case": it["case"], "group": it["group"]})
        key[i] = {"arm": arm, "key": it["key"]}
    lst.sort(key=lambda x: (x["group"], x["case"], x["id"]))
    (blind / "list.json").write_text(json.dumps(lst, indent=1), encoding="utf-8")
    (blind / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(len(lst), "blind pictures")


def phase_table(src: Path, blind: Path):
    items, res = load(src)
    verd = json.loads((blind / "verdicts.json").read_text("utf-8"))
    key = json.loads((blind / "key.json").read_text("utf-8"))
    eye = {}
    for i, v in verd.items():
        eye[(key[i]["arm"], key[i]["key"])] = v
    summ = {}
    for arm in "ABC":
        for grp in ("known", "unseen"):
            rs = [r for r in res[arm].values() if r["group"] == grp]
            src_arm = "A" if arm == "B" and grp == "unseen" else arm
            e = [eye.get((src_arm, r["key"]), {}) for r in rs]
            s = {"n": len(rs), "pass1": sum(r["pass1"] for r in rs), "pass5": sum(r["passed"] for r in rs),
                 "cards": sum(r["card"] for r in rs),
                 "eye_right": sum(1 for x in e if x.get("verdict") == "yes"),
                 "eye_wrong": sum(1 for x in e if x.get("verdict") == "no"),
                 "tries_mean": round(sum(r["n_tries"] for r in rs) / max(1, len(rs)), 2)}
            if arm == "C":
                own = [r["own_check"]["ok"] for r in rs if "own_check" in r]
                s["own_check_pass"] = f"{sum(own)}/{len(own)}"
            summ[f"{arm}/{grp}"] = s
    for k, s in summ.items():
        print(k, s)
    kn = lambda a, f: summ[f"{a}/known"][f]
    un = lambda a, f: summ[f"{a}/unseen"][f]
    c1 = kn("C", "eye_right") >= kn("A", "eye_right")
    c2 = un("C", "eye_right") >= un("A", "eye_right") and un("C", "cards") <= un("A", "cards")
    dec = {"known_C_ge_A": c1, "unseen_C_ge_A_and_cards_le": c2, "C_replaces_A": bool(c1 and c2)}
    print("decision", dec)
    per = {}
    for it in items:
        per[it["key"]] = {arm: eye.get(("A" if arm == "B" and it["group"] == "unseen" else arm, it["key"]), {})
                          for arm in "ABC"}
    (src / "summary.json").write_text(json.dumps({"summary": summ, "decision": dec, "by_eye": per}, indent=1),
                                      encoding="utf-8")


def _jpg(src_png: Path, dst: Path, px: int):
    from PIL import Image
    if src_png.exists():
        im = Image.open(src_png).convert("RGB")
        im.thumbnail((px, px))
        im.save(dst, quality=82)
        return True
    return False


def phase_sheet(src: Path, mine: Path):
    items, res = load(src)
    summ = json.loads((src / "summary.json").read_text("utf-8")) if (src / "summary.json").exists() else {}
    per = summ.get("by_eye", {})
    img = DOC / "img"
    img.mkdir(parents=True, exist_ok=True)
    e = html.escape
    rows = []
    for it in items:
        tag = it["tag"]
        cells = []
        for arm in "ABC":
            r = res[arm].get(it["key"])
            if not r:
                cells.append("<td class=arm>-</td>")
                continue
            srcarm = "A" if r.get("same_as_A") else arm
            fin = img / f"{srcarm}_{tag}.jpg"
            _jpg(src / r["image"], fin, 360)
            strip = []
            p = src / r["image"]
            for t in range(1, r["n_tries"]):
                tp = p.with_name(f"{p.stem}_try{t}.png")
                tj = img / f"{srcarm}_{tag}_t{t}.jpg"
                if _jpg(tp, tj, 120):
                    strip.append(f"<img class=t src='img/{tj.name}' title='try {t}'>")
            v = per.get(it["key"], {}).get(arm, {})
            badge = ("card" if r["card"] else ("pass@1" if r["pass1"] else f"pass@{r['n_tries']}" if r["passed"]
                                                else "no pass"))
            eye = {"yes": "<b class=ok>right</b>", "no": "<b class=bad>wrong</b>"}.get(v.get("verdict"), "")
            note = e(v.get("note", ""))
            same = " <i>(= A)</i>" if r.get("same_as_A") else ""
            cells.append(f"<td class=arm><img class=f src='img/{fin.name}' title='{e(r['final_prompt'])}'>"
                         f"<div class=meta>{badge} &middot; {r['n_tries']} tr {eye}{same}<br>{note}</div>"
                         f"<div class=strip>{''.join(strip)}</div></td>")
        cp = it.get("c_plan", {})
        slots = it.get("slots", {})
        mimg = []
        slug = "".join(c if c.isalnum() else "_" for c in it["source"].lower()).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        for k in range(4):
            mj = img / f"mine_{slug}_{k}.jpg"
            if _jpg(mine / f"{slug}_{k}.png", mj, 90):
                mimg.append(f"<img class=m src='img/{mj.name}'>")
        info = (f"<div class=lab>{e(it['source'])}</div><div class=g>{e(it['group'])}"
                f"{' &middot; ' + e(it['case']) if it['group'] == 'known' else ''}</div>"
                f"<div><span class=k>plain</span> {e(it['drawn'])}</div>"
                f"<div><span class=k>C</span> {e(slots.get('sentence') or '(slots refused: ' + '; '.join(slots.get('why') or []) + ')')}</div>"
                f"<div><span class=k>mined neg</span> {e(cp.get('neg', '') or '-')}</div>"
                f"<div><span class=k>checker</span> {e(it['union']['intended'])} vs {len(it['union']['confusions'])}"
                f" look-alikes</div><div class=mine>{''.join(mimg)}</div>")
        rows.append(f"<tr><td class=info>{info}</td>{''.join(cells)}</tr>")
    table = ""
    if summ:
        hdr = "<tr><th>arm / sounds</th><th>pass@1</th><th>pass &le;5</th><th>word cards</th><th>by eye right</th><th>by eye wrong</th></tr>"
        body = "".join(f"<tr><td>{k}</td><td>{s['pass1']}/{s['n']}</td><td>{s['pass5']}/{s['n']}</td>"
                       f"<td>{s['cards']}/{s['n']}</td><td>{s['eye_right']}/{s['n']}</td><td>{s['eye_wrong']}/{s['n']}</td></tr>"
                       for k, s in summ["summary"].items())
        d = summ["decision"]
        table = (f"<table class=sum>{hdr}{body}</table><p>Decision rule (fixed before drawing): C replaces A = "
                 f"<b>{'YES' if d['C_replaces_A'] else 'NO'}</b> (known: C &ge; A {d['known_C_ge_A']}; unseen: C &ge; A "
                 f"and cards &le; {d['unseen_C_ge_A_and_cards_le']}).</p>")
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Picture sense test</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--line:#e2e0da;--ok:#1f7a3a;--bad:#b3261e;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161614;--fg:#ecebe6;--mute:#9b9a94;--line:#34332f;--ok:#6fcf8a;--bad:#ff8a80;--card:#1f1f1c}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} p{{max-width:900px;color:var(--mute)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse}} td,th{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
th{{position:sticky;top:0;background:var(--bg)}} .info{{width:280px;font-size:12px}} .lab{{font-weight:600;font-size:14px}}
.g{{color:var(--mute);margin-bottom:4px}} .k{{color:var(--mute);font-size:11px;text-transform:uppercase}}
.arm{{width:250px}} img.f{{width:240px;height:240px;border-radius:6px;background:var(--card);display:block}}
.meta{{font-size:12px;color:var(--mute);margin-top:4px}} img.t{{width:46px;height:46px;margin:2px 2px 0 0;border-radius:3px}}
img.m{{width:60px;height:60px;margin:2px 2px 0 0;border-radius:3px}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
table.sum td,table.sum th{{padding:4px 10px}}
</style></head><body>
<h1>Picture sense test &mdash; 28 Sept 2026</h1>
<p>A = hand table (shipped), B = VLM free text, C = new generic method (slot form + mistake mining). Same seeds, 5 tries,
the same union checker for all arms. Small pictures under "plain" are C's 4 mining pictures; the strip under each final
picture shows the refused tries. Hover a picture for its prompt. By-eye verdicts are Claude's, made blind; Adam's look
overrides them. Plan: docs/picture_sense_test_2026-09-28.md.</p>
{table}
<div class=wrap><table><tr><th>sound</th><th>A &middot; hand table</th><th>B &middot; VLM free text</th><th>C &middot; generic</th></tr>
{''.join(rows)}</table></div></body></html>"""
    (DOC / "index.html").write_text(page, encoding="utf-8")
    print("wrote", DOC / "index.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["blind", "table", "sheet"], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--mine", default="")
    ap.add_argument("--blind", default="")
    a = ap.parse_args()
    src = Path(a.src)
    if a.phase == "blind":
        phase_blind(src, Path(a.blind))
    elif a.phase == "table":
        phase_table(src, Path(a.blind))
    else:
        phase_sheet(src, Path(a.mine))


if __name__ == "__main__":
    main()

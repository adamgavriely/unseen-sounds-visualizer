"""Collect attribution for every retrieved augmentation image actually used.

Stage 6 can retrieve its images from Openverse instead of generating them, and the
ablation showed retrieval scores better -- so the images are part of the system's output
and their licences are a real constraint, not a footnote. Each pipeline run writes a
credits.json; this walks them, de-duplicates by source URL, and writes a single
attribution table plus a licence summary.

Usage:
    python scripts/build_attributions.py [--roots data/work] [--out benchmark/IMAGE_ATTRIBUTIONS.md]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREE = {"by", "by-sa", "pdm", "cc0"}          # usable without a NonCommercial restriction


def main():
    roots = ROOT / "data" / "work"
    if "--roots" in sys.argv:
        roots = Path(sys.argv[sys.argv.index("--roots") + 1])
    out = ROOT / "benchmark" / "IMAGE_ATTRIBUTIONS.md"
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])

    uses = 0
    by_src = {}
    labels = defaultdict(set)
    for f in roots.rglob("credits.json"):
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in rows:
            src = r.get("source") or ""
            if not src:
                continue
            uses += 1
            by_src.setdefault(src, r)
            if r.get("label"):
                labels[src].add(r["label"])

    lic = Counter((r.get("license") or "?").lower() for r in by_src.values())
    restricted = sum(v for k, v in lic.items() if k not in FREE)

    lines = [
        "# Attribution for retrieved augmentation images",
        "",
        "Stage 6 retrieves its images from [Openverse](https://openverse.org) rather than",
        "generating them, a choice the generator ablation supports on quality as well as cost.",
        "Retrieved images carry licences, so every image the system used is listed here with its",
        "creator, licence and source, as CC-BY and CC-BY-SA require.",
        "",
        f"**{len(by_src)} distinct images across {uses} uses.**",
        "",
        "## Licence summary",
        "",
        "| licence | images |",
        "|---|---|",
    ]
    for k, v in lic.most_common():
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        f"**{restricted} of {len(by_src)} images ({100 * restricted // max(len(by_src), 1)}%) carry a",
        "NonCommercial or NoDerivatives term.** That is fine for an academic prototype and for the",
        "evaluation reported here, but it constrains deployment: a released accessibility tool could",
        "not ship these images, and compositing one beside a video is plausibly a derivative work.",
        "A deployed system would need to restrict retrieval to the permissive licences, license a",
        "stock library, or fall back to generation -- which is the one real argument left for",
        "diffusion, and it is a legal argument rather than a quality one.",
        "",
        "## Images",
        "",
        "| image | creator | licence | used for | source |",
        "|---|---|---|---|---|",
    ]
    for src, r in sorted(by_src.items(), key=lambda kv: (kv[1].get("label") or "")):
        used = ", ".join(sorted(labels[src])[:4]) or "-"
        lic_url = r.get("license_url") or ""
        lic_txt = f"[{r.get('license', '?')}]({lic_url})" if lic_url else str(r.get("license"))
        lines.append(f"| {r.get('title') or '(untitled)'} | {r.get('creator') or 'unknown'} | "
                     f"{lic_txt} | {used} | [link]({src}) |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(by_src)} distinct images, {uses} uses -> {out}")
    print("licences:", dict(lic.most_common()))


if __name__ == "__main__":
    main()

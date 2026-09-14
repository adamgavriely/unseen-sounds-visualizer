"""Render a Markdown document to a readable A4 PDF (large type) with PyMuPDF.

    python scripts/md_to_pdf.py docs/EXECUTIVE_SUMMARY.md docs/EXECUTIVE_SUMMARY.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
import pymupdf

CSS = """
body { font-family: sans-serif; font-size: 13pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; margin: 0 0 10pt 0; }
h2 { font-size: 16pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #999; }
p { margin: 0 0 8pt 0; }
li { margin: 0 0 5pt 0; }
ul, ol { margin: 4pt 0 10pt 0; padding-left: 22pt; }
code { font-family: monospace; font-size: 11.5pt; background: #f0f0f0; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 11.5pt; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; vertical-align: top; }
th { background: #eee; }
em { color: #333; }
hr { border: 0; border-top: 1px solid #999; margin: 10pt 0; }
"""


def main(src: str, dst: str):
    text = Path(src).read_text(encoding="utf-8")
    # Python-Markdown needs a blank line before a list; the summary often starts a list
    # right after a paragraph, which otherwise renders as one run-on paragraph
    lines, out = text.splitlines(), []
    for i, ln in enumerate(lines):
        is_item = re.match(r"^\s*(-|\d+\.)\s", ln)
        prev = out[-1] if out else ""
        if is_item and prev.strip() and not re.match(r"^\s*(-|\d+\.)\s", prev) and not prev.startswith("  "):
            out.append("")
        out.append(ln)
    text = "\n".join(out)
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(dst)
    page = pymupdf.paper_rect("a4")
    where = page + (48, 48, -48, -54)
    more = True
    n = 0
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        n += 1
    writer.close()
    print(f"{dst}: {n} pages")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Wrap page.html (the published artifact body) into a standalone index.html that opens
offline by double-click. page.html is the source of truth; run this after editing it.

    python docs/supervisor_meeting/meeting_page/build_local.py
"""
from pathlib import Path
here = Path(__file__).resolve().parent
body = (here / "page.html").read_text(encoding="utf-8")
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<style>body{margin:0;font-size:14px;font-family:system-ui,sans-serif;background:#f6f7f5}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(here / "index.html").write_text(html, encoding="utf-8")
print("->", here / "index.html")

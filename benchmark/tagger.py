"""Simple local web app to tag benchmark clips.

Walk back and forth through every clip in data/input/benchmark/<folders>, play it
with normal video controls (play/pause/seek/volume), and tag it with one click:
  1 = sound source NOT visible, 2 = source IS visible, 3 = no ambient sound,
  plus a red "BAD" button (with a required reason) to discard a clip.

Tags save immediately to benchmark/tags.json.

RUN IT IN YOUR OWN TERMINAL (not through the Claude app):
    python benchmark/tagger.py
It opens http://localhost:8000 in your default browser. Press Ctrl+C in the
terminal to stop it when you are done.
"""
from __future__ import annotations

import http.server
import json
import re
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "input" / "benchmark"
FOLDERS = ["unseen_ambient", "seen_ambient", "no_ambient"]
TAGS_FILE = ROOT / "benchmark" / "tags.json"
EXT = {".webm", ".ogv", ".mp4"}
PORT = 8000
_LOCK = threading.Lock()

TAG_OPTIONS = [
    ("unseen_ambient", "1  Sound source NOT visible", "#2e7d32"),
    ("seen_ambient",   "2  Sound source IS visible",  "#1565c0"),
    ("no_ambient",     "3  No ambient sound (speech/music)", "#6a1b9a"),
]


def list_clips():
    out = []
    for f in FOLDERS:
        d = BENCH / f
        if d.exists():
            for v in sorted(d.glob("*")):
                if v.suffix.lower() in EXT:
                    out.append(f"{f}/{v.name}")
    return out


def load_tags():
    if TAGS_FILE.exists():
        try:
            return json.loads(TAGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_tags(tags):
    TAGS_FILE.write_text(json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8")


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>Benchmark tagger</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,Arial,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:16px}
 .top{display:flex;justify-content:space-between;align-items:center;gap:12px}
 h1{font-size:16px;margin:0;font-weight:600;color:#bbb}
 .prog{font-size:14px;color:#9ad}
 video{width:100%;max-height:60vh;background:#000;border-radius:8px;margin:10px 0}
 .name{font-size:15px;color:#ddd;word-break:break-all}
 .cur{font-size:13px;color:#888;margin-top:2px;min-height:18px}
 .tagged{color:#7c7}.badt{color:#f77}
 .btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:12px 0 8px}
 button.tag{padding:16px 10px;font-size:15px;border:none;border-radius:8px;color:#fff;cursor:pointer;opacity:.92}
 button.tag:hover{opacity:1}button.tag.sel{outline:3px solid #fff}
 .badrow{display:flex;gap:8px;margin:4px 0}
 .badrow input{flex:1;padding:10px;border-radius:6px;border:1px solid #444;background:#1b1b1b;color:#eee;font-size:14px}
 button.bad{padding:10px 16px;font-size:14px;border:none;border-radius:6px;background:#b71c1c;color:#fff;cursor:pointer}
 .nav{display:flex;justify-content:space-between;gap:10px;margin-top:10px}
 .nav button{padding:10px 18px;font-size:14px;border:none;border-radius:6px;background:#333;color:#eee;cursor:pointer}
 .hint{font-size:12px;color:#777;margin-top:14px;line-height:1.6}
</style></head><body><div class=wrap>
 <div class=top><h1>Benchmark tagger</h1><div class=prog id=prog></div></div>
 <video id=vid controls></video>
 <div class=name id=name></div>
 <div class=cur id=cur></div>
 <div class=btns id=btns></div>
 <div class=badrow>
   <input id=reason placeholder="reason this clip is bad (required for BAD)">
   <button class=bad onclick="markBad()">&#10007; BAD &mdash; discard (B)</button>
 </div>
 <div class=nav>
   <button onclick="go(-1)">&larr; Prev (P)</button>
   <button onclick="nextUntagged()">Next untagged</button>
   <button onclick="go(1)">Next (N) &rarr;</button>
 </div>
 <div class=hint>Keys: <b>Space</b> play/pause &middot; <b>&larr;/&rarr;</b> or P/N prev/next &middot;
   <b>1 / 2 / 3</b> tag &middot; <b>B</b> mark bad &middot; drag video bar to seek, right side = volume.
   Tags save to <code>benchmark/tags.json</code>; tagging auto-advances to the next untagged clip.</div>
</div>
<script>
let clips=[],tags={},i=0;const TAGS=__TAGS__;
async function boot(){
  const d=await (await fetch('/api/clips')).json();
  clips=d.clips;tags=d.tags||{};
  const b=document.getElementById('btns');b.innerHTML='';
  TAGS.forEach(t=>{const el=document.createElement('button');el.className='tag';el.style.background=t[2];
    el.textContent=t[1];el.dataset.k=t[0];el.onclick=()=>tag(t[0]);b.appendChild(el);});
  i=clips.findIndex(c=>!tags[c]);if(i<0)i=0;show();
}
function show(){
  const c=clips[i],v=document.getElementById('vid');
  v.src='/media/'+c.split('/').map(encodeURIComponent).join('/');v.load();
  document.getElementById('name').textContent=(i+1)+' / '+clips.length+'   '+c.split('/')[1];
  const cur=tags[c];let s='folder guess: <b>'+c.split('/')[0]+'</b> &middot; ';
  if(!cur)s+='not tagged yet';
  else if(cur.tag==='bad')s+='<span class=badt>BAD: '+(cur.reason||'')+'</span>';
  else s+='<span class=tagged>tagged: '+cur.tag+'</span>';
  document.getElementById('cur').innerHTML=s;
  document.querySelectorAll('button.tag').forEach(el=>el.classList.toggle('sel',cur&&cur.tag===el.dataset.k));
  document.getElementById('reason').value=(cur&&cur.tag==='bad')?(cur.reason||''):'';
  document.getElementById('prog').textContent=Object.keys(tags).length+' / '+clips.length+' tagged';
}
async function post(body){await fetch('/api/tag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
async function tag(k){const c=clips[i];await post({clip:c,tag:k});tags[c]={tag:k};show();setTimeout(nextUntagged,250);}
async function markBad(){const r=document.getElementById('reason').value.trim();
  if(!r){alert('Please type why this clip is bad first.');document.getElementById('reason').focus();return;}
  const c=clips[i];await post({clip:c,tag:'bad',reason:r});tags[c]={tag:'bad',reason:r};show();setTimeout(nextUntagged,250);}
function go(d){i=Math.max(0,Math.min(clips.length-1,i+d));show();}
function nextUntagged(){let j=clips.findIndex((c,idx)=>idx>i&&!tags[c]);
  if(j<0)j=clips.findIndex(c=>!tags[c]);if(j>=0){i=j;show();}else{i=Math.min(i+1,clips.length-1);show();}}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();const v=document.getElementById('vid');v.paused?v.play():v.pause();}
  else if(e.key==='ArrowRight'||e.key==='n'||e.key==='N')go(1);
  else if(e.key==='ArrowLeft'||e.key==='p'||e.key==='P')go(-1);
  else if(e.key==='1')tag(TAGS[0][0]);else if(e.key==='2')tag(TAGS[1][0]);else if(e.key==='3')tag(TAGS[2][0]);
  else if(e.key==='b'||e.key==='B')markBad();
});
boot();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _write(self, data):
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # browser aborted the (range) request -- normal for <video>

    def _send(self, code, ctype, body):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return
        self._write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8",
                       PAGE.replace("__TAGS__", json.dumps(TAG_OPTIONS)).encode("utf-8"))
        elif path == "/api/clips":
            self._send(200, "application/json",
                       json.dumps({"clips": list_clips(), "tags": load_tags()}).encode())
        elif path.startswith("/media/"):
            self._serve_media(BENCH / urllib.parse.unquote(path[len("/media/"):]))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path == "/api/tag":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            entry = {"tag": data["tag"]}
            if data.get("reason"):
                entry["reason"] = data["reason"]
            with _LOCK:
                tags = load_tags()
                tags[data["clip"]] = entry
                save_tags(tags)
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"not found")

    def _serve_media(self, fp: Path):
        if not fp.exists() or fp.suffix.lower() not in EXT:
            self._send(404, "text/plain", b"no file")
            return
        size = fp.stat().st_size
        ctype = {".webm": "video/webm", ".ogv": "video/ogg", ".mp4": "video/mp4"}[fp.suffix.lower()]
        rng = self.headers.get("Range")
        try:
            if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self._stream(fp, start, end - start + 1)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                self._stream(fp, 0, size)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def _stream(self, fp: Path, start: int, length: int):
        with open(fp, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(262144, remaining))
                if not chunk:
                    break
                self._write(chunk)
                remaining -= len(chunk)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    n = len(list_clips())
    url = f"http://localhost:{port}"
    try:
        httpd = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"Could not start on port {port} ({e}). Try another: python benchmark/tagger.py 8010")
        return
    print(f"Tagger running: {url}   ({n} clips)")
    print("Opening your browser... press Ctrl+C here to stop.")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()

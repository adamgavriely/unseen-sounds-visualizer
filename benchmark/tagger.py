"""Simple local web app to tag benchmark clips.

By default it shows ONLY clips you have not tagged yet, so nothing you have already
done can reappear. Play each clip (play/pause/seek/volume) and tag it:
  1 = source NOT visible, 2 = source IS visible, 3 = no ambient sound,
  ? = "I don't know" (revisit later, via the Unsure filter),
  BAD (red, key B) = discard this clip (a reason is required).

Tags save immediately to benchmark/tags.json (existing tags are never deleted).

RUN IT IN YOUR OWN TERMINAL (or double-click tag_videos.bat):
    python benchmark/tagger.py
It opens http://localhost:8000 in your default browser. Ctrl+C to stop.
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
FOLDERS = ["unseen_ambient", "seen_ambient", "mixed", "no_ambient", "unsorted"]
# where a tagged clip's file should live (unknown -> stay put; bad -> _bad, out of view)
TAG_FOLDER = {"unseen_ambient": "unseen_ambient", "seen_ambient": "seen_ambient",
              "mixed": "mixed", "no_ambient": "no_ambient", "bad": "_bad"}
TAGS_FILE = ROOT / "benchmark" / "tags.json"
EXT = {".webm", ".ogv", ".mp4"}
PORT = 8000
_LOCK = threading.Lock()

TAG_OPTIONS = [
    ("unseen_ambient", "Heard, not seen", "#2e7d32",
     "A meaningful sound whose source is OFF-screen (e.g. traffic behind a crowd)."),
    ("seen_ambient", "Heard and seen", "#1565c0",
     "Every real sound's source is visible ON screen (e.g. a waterfall you can see)."),
    ("mixed", "Both / mixed", "#00838f",
     "Some sources on-screen, some off (e.g. visible train + off-screen birds)."),
    ("no_ambient", "No ambient sound", "#6a1b9a",
     "Only speech or music — no environmental/background sound."),
]


def list_clips():
    seen, out = set(), []
    for f in FOLDERS:
        d = BENCH / f
        if d.exists():
            for v in sorted(d.glob("*")):
                if v.suffix.lower() in EXT:
                    key = f"{f}/{v.name}"
                    if key not in seen:      # defensive de-dup
                        seen.add(key)
                        out.append(key)
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


SUGG_FILE = ROOT / "benchmark" / "suggestions.json"


def load_suggestions():
    if SUGG_FILE.exists():
        try:
            return json.loads(SUGG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def move_for_tag(clip_key, tag):
    """Move a tagged clip's FILE into its category folder; return the new key.
    unknown -> stays put (revisit later); bad -> _bad/ (out of view)."""
    tf = TAG_FOLDER.get(tag)
    if not tf:
        return clip_key
    src = BENCH / clip_key
    if not src.exists():
        return clip_key
    (BENCH / tf).mkdir(exist_ok=True)
    dest = BENCH / tf / src.name
    try:
        if src.resolve() == dest.resolve():
            return clip_key
        if dest.exists():
            dest.unlink()
        src.rename(dest)
    except OSError:
        return clip_key
    return f"{tf}/{src.name}"


def reconcile():
    """One-time at startup: move already-tagged clips into their folders."""
    with _LOCK:
        tags = load_tags()
        new, changed = {}, False
        for k, v in tags.items():
            nk = move_for_tag(k, v.get("tag"))
            new[nk] = v
            changed = changed or (nk != k)
        if changed:
            save_tags(new)
            print(f"reconciled: moved tagged clips into their folders")


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>Benchmark tagger</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,Arial,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:16px}
 .top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:600;color:#bbb}
 .prog{font-size:14px;color:#9ad}
 select{background:#1b1b1b;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
 video{width:100%;max-height:58vh;background:#000;border-radius:8px;margin:10px 0}
 .name{font-size:15px;color:#ddd;word-break:break-all}
 .cur{font-size:13px;color:#888;margin-top:2px;min-height:18px}
 .tagged{color:#7c7}.badt{color:#f77}.unk{color:#fb3}.sug{color:#e0b25a}
 .btns{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0 6px}
 .cell{display:flex;flex-direction:column;gap:5px}
 button.tag{padding:15px 8px;font-size:15px;border:none;border-radius:8px;color:#fff;cursor:pointer;opacity:.92}
 button.tag b{opacity:.7;margin-right:4px}
 button.tag:hover{opacity:1}button.tag.sel{outline:3px solid #fff}
 .cap{font-size:11.5px;color:#9a9a9a;line-height:1.35;padding:0 2px;min-height:44px}
 .cap2{font-size:11.5px;color:#8a8a8a;margin:2px 0 2px}
 .row2{display:flex;gap:8px;margin:4px 0;align-items:center}
 button.unknown{padding:12px 14px;font-size:14px;border:none;border-radius:6px;background:#ef6c00;color:#fff;cursor:pointer;white-space:nowrap}
 .row2 input{flex:1;padding:10px;border-radius:6px;border:1px solid #444;background:#1b1b1b;color:#eee;font-size:14px}
 button.bad{padding:10px 16px;font-size:14px;border:none;border-radius:6px;background:#b71c1c;color:#fff;cursor:pointer;white-space:nowrap}
 .nav{display:flex;justify-content:space-between;gap:10px;margin-top:10px}
 .nav button{padding:10px 18px;font-size:14px;border:none;border-radius:6px;background:#333;color:#eee;cursor:pointer}
 .done{text-align:center;padding:40px;color:#7c7;font-size:18px}
 .hint{font-size:12px;color:#777;margin-top:14px;line-height:1.6}
</style></head><body><div class=wrap>
 <div class=top><h1>Benchmark tagger</h1>
   <div><label style="font-size:13px;color:#999">show:
     <select id=filter onchange="setFilter(this.value)">
       <option value=untagged>Untagged only</option>
       <option value=unsure>Unsure (I don't know)</option>
       <option value=all>All</option>
     </select></label></div>
   <div class=prog id=prog></div>
 </div>
 <div id=main>
   <video id=vid controls></video>
   <div class=name id=name></div>
   <div class=cur id=cur></div>
   <div class=btns id=btns></div>
   <div class=row2>
     <button class=unknown onclick="act({tag:'unknown'})">?  I don't know (U)</button>
     <input id=reason placeholder="reason this clip is bad (required for BAD)">
     <button class=bad onclick="markBad()">&#10007; BAD (B)</button>
   </div>
   <div class=cap2><b>U</b> = not sure, revisit later &nbsp;&middot;&nbsp; <b>B</b> = broken/unusable clip (type why; it gets discarded).</div>
   <div class=nav>
     <button onclick="go(-1)">&larr; Prev (P)</button>
     <button onclick="go(1)">Next (N) &rarr;</button>
   </div>
 </div>
 <div class=done id=done style=display:none></div>
 <div class=hint>Default shows <b>only untagged</b> clips, so nothing you've tagged comes back.
   Keys: <b>Space</b> play/pause &middot; <b>&larr;/&rarr;</b> or P/N move &middot; <b>1&ndash;4</b> tag &middot;
   <b>U</b> I don't know &middot; <b>B</b> bad. Saved to <code>benchmark/tags.json</code>.</div>
</div>
<script>
let clips=[],tags={},suggestions={},filter='untagged',view=[],vi=0;const TAGS=__TAGS__;
const SUGNAME={unseen_ambient:'Heard, not seen',seen_ambient:'Heard and seen',mixed:'Both/mixed',no_ambient:'No ambient','?':'?'};
async function boot(){
  const d=await (await fetch('/api/clips')).json();clips=d.clips;tags=d.tags||{};suggestions=d.suggestions||{};
  const b=document.getElementById('btns');b.innerHTML='';
  TAGS.forEach((t,idx)=>{
    const cell=document.createElement('div');cell.className='cell';
    const el=document.createElement('button');el.className='tag';el.style.background=t[2];
    el.innerHTML='<b>'+(idx+1)+'</b>'+t[1];el.dataset.k=t[0];el.onclick=()=>act({tag:t[0]});
    const cap=document.createElement('div');cap.className='cap';cap.textContent=t[3];
    cell.appendChild(el);cell.appendChild(cap);b.appendChild(cell);});
  buildView();vi=0;show();
}
function buildView(){
  if(filter==='untagged')view=clips.filter(c=>!tags[c]);
  else if(filter==='unsure')view=clips.filter(c=>tags[c]&&tags[c].tag==='unknown');
  else view=clips.slice();
}
function setFilter(f){filter=f;buildView();vi=0;show();}
function show(){
  const total=clips.length,done=Object.keys(tags).length;
  document.getElementById('prog').textContent=done+' / '+total+' tagged  ('+view.length+' in view)';
  if(view.length===0){document.getElementById('main').style.display='none';
    const dv=document.getElementById('done');dv.style.display='block';
    dv.textContent=(filter==='untagged')?'✓ All clips tagged. Nothing left untagged!':'Nothing in this view.';return;}
  document.getElementById('main').style.display='';document.getElementById('done').style.display='none';
  if(vi>=view.length)vi=view.length-1;if(vi<0)vi=0;
  const c=view[vi],v=document.getElementById('vid');
  v.src='/media/'+c.split('/').map(encodeURIComponent).join('/');v.load();
  document.getElementById('name').textContent=(vi+1)+' / '+view.length+'   '+c.split('/')[1];
  const cur=tags[c];let s='folder: <b>'+c.split('/')[0]+'</b> &middot; ';
  if(!cur){s+='not tagged';
    const sg=suggestions[c];
    if(sg&&sg.suggest)s+=' &middot; <span class=sug>suggest: <b>'+(SUGNAME[sg.suggest]||sg.suggest)+'</b> &mdash; '+sg.reason+'</span>';}
  else if(cur.tag==='bad')s+='<span class=badt>BAD: '+(cur.reason||'')+'</span>';
  else if(cur.tag==='unknown')s+='<span class=unk>marked: unsure</span>';
  else s+='<span class=tagged>tagged: '+cur.tag+'</span>';
  document.getElementById('cur').innerHTML=s;
  document.querySelectorAll('button.tag').forEach(el=>el.classList.toggle('sel',cur&&cur.tag===el.dataset.k));
  document.getElementById('reason').value=(cur&&cur.tag==='bad')?(cur.reason||''):'';
}
async function post(body){await fetch('/api/tag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
async function act(entry){
  const c=view[vi];if(!c)return;await post({clip:c,...entry});tags[c]=entry;
  buildView();show();                     // in 'untagged' view the clip drops out -> auto next
}
function markBad(){const r=document.getElementById('reason').value.trim();
  if(!r){alert('Type why this clip is bad first.');document.getElementById('reason').focus();return;}
  act({tag:'bad',reason:r});}
function go(d){vi=Math.max(0,Math.min(view.length-1,vi+d));show();}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();const v=document.getElementById('vid');v.paused?v.play():v.pause();}
  else if(e.key==='ArrowRight'||e.key==='n'||e.key==='N')go(1);
  else if(e.key==='ArrowLeft'||e.key==='p'||e.key==='P')go(-1);
  else if(['1','2','3','4'].includes(e.key)){const t=TAGS[+e.key-1];if(t)act({tag:t[0]});}
  else if(e.key==='u'||e.key==='U')act({tag:'unknown'});
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
            pass

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
                       json.dumps({"clips": list_clips(), "tags": load_tags(),
                                   "suggestions": load_suggestions()}).encode())
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
                newkey = move_for_tag(data["clip"], data["tag"])   # move file to its folder
                tags = load_tags()
                if newkey != data["clip"]:
                    tags.pop(data["clip"], None)
                tags[newkey] = entry
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
    reconcile()   # move any already-tagged clips into their folders on startup
    n = len(list_clips())
    url = f"http://localhost:{port}"
    try:
        httpd = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"Could not start on port {port} ({e}). Try: python benchmark/tagger.py 8010")
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

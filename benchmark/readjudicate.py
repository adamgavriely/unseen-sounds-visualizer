"""Blind re-adjudication: is the labelling criterion stable, and what is it really?

Why this exists. Across ~450 tagged clips the useful rate FELL (22.2% -> 19.7% ->
15.4%) while the DROP rate rose from 0% to 46%, and the same sound types appear in
both DROP and unseen_ambient. Meanwhile 36% of clips that contain ambient sound are
unseen/mixed, so the phenomenon is not rare -- the funnel is discarding it. That
pattern points at the labelling criterion drifting, not at the sourcing.

This tool measures that directly. It replays 60 clips ALREADY TAGGED (20 drop,
20 seen_ambient, 20 unseen_ambient), shuffled, with the original label hidden, and:

  * removes the DROP button, so every clip must receive a real category -- DROP was
    absorbing a quarter of the corpus with no recorded reason (107 of 113 blank);
  * makes a one-line reason MANDATORY, which is the only way to recover the criterion
    actually in use;
  * asks the question at the level the gate answers it: which sound was judged.

Read-outs (scripts/readjudicate_report.py):
  1. self-agreement with the original labels -- if low, the labels are not stable
     enough for any gating metric computed against them to be interpretable;
  2. what fraction of DROPs become unseen/mixed -- if high, the benchmark can be
     filled from material already on disk, with no further sourcing;
  3. the free-text reasons, which are the actual definition of the task.

Results go to benchmark/readjudication.json; the original tags.json is NEVER
modified by this tool.

Run:  python benchmark/readjudicate.py     (or double-click readjudicate.bat)
"""
from __future__ import annotations

import json
import os
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "input" / "benchmark"
SET_FILE = ROOT / "benchmark" / "readjudicate_set.json"
OUT = ROOT / "benchmark" / "readjudication.json"
PORT = 8010
_LOCK = threading.Lock()

OPTIONS = [
    ("unseen_ambient", "1 &nbsp;Heard, NOT seen", "#2e7d32",
     "A non-speech sound whose source is not visible anywhere in the clip."),
    ("seen_ambient", "2 &nbsp;Heard AND seen", "#1565c0",
     "The thing making the sound is visible on screen."),
    ("mixed", "3 &nbsp;Both", "#6a1b9a",
     "At least one source visible AND at least one not."),
    ("no_ambient", "4 &nbsp;No real sound", "#455a64",
     "Only speech, music or silence -- nothing ambient to judge."),
]

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Re-adjudication</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,Segoe UI,Arial;margin:0;padding:14px}
 .wrap{max-width:1080px;margin:0 auto}
 h1{font-size:17px;margin:0 0 4px}
 .sub{font-size:12.5px;color:#999;margin-bottom:10px;line-height:1.5}
 video{width:100%;max-height:56vh;background:#000;border-radius:8px}
 .name{font-size:12px;color:#888;margin:6px 0}
 .btns{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
 button.tag{padding:14px 8px;font-size:14px;border:none;border-radius:8px;color:#fff;cursor:pointer;opacity:.92}
 button.tag:hover{opacity:1}
 .cap{font-size:11px;color:#9a9a9a;line-height:1.35;min-height:40px;padding:0 2px}
 .row{display:flex;gap:8px;align-items:center;margin:8px 0}
 .row input{flex:1;padding:11px;border-radius:6px;border:1px solid #444;background:#1b1b1b;color:#eee;font-size:14px}
 .prog{font-size:13px;color:#9ad;margin-bottom:8px}
 .done{text-align:center;padding:50px;color:#7c7;font-size:18px;line-height:1.6}
 .warn{color:#e8a33d;font-size:12px;min-height:16px}
</style></head><body><div class=wrap>
 <h1>Blind re-adjudication</h1>
 <div class=sub>These are clips you already tagged, shuffled with the old label hidden.
   There is <b>no Drop button</b> on purpose &mdash; give every clip a real category, and
   say in a few words <b>why</b>. That sentence is what tells us what the task actually is.</div>
 <div class=prog id=prog></div>
 <div id=main>
   <video id=vid controls></video>
   <div class=name id=name></div>
   <div class=btns id=btns></div>
   <div class=row>
     <input id=reason placeholder="why? e.g. 'siren clearly off screen' / 'boring, no image would help' / 'cant tell what the sound is'">
   </div>
   <div class=warn id=warn></div>
 </div>
 <div class=done id=done style=display:none></div>
</div>
<script>
let items=[],i=0,done={};const OPTS=__OPTS__;
async function boot(){
  const d=await (await fetch('/api/set')).json();
  items=d.items;done=d.done||{};
  const b=document.getElementById('btns');b.innerHTML='';
  OPTS.forEach((o,ix)=>{
    const cell=document.createElement('div');
    const el=document.createElement('button');el.className='tag';el.style.background=o[2];
    el.innerHTML=o[1];el.onclick=()=>save(o[0]);
    const c=document.createElement('div');c.className='cap';c.textContent=o[3];
    cell.appendChild(el);cell.appendChild(c);b.appendChild(cell);});
  i=items.findIndex(x=>!done[x.clip]); if(i<0)i=items.length;
  show();
}
function show(){
  const p=document.getElementById('prog');
  p.textContent=Object.keys(done).length+' / '+items.length+' re-judged';
  if(i>=items.length){document.getElementById('main').style.display='none';
    const dv=document.getElementById('done');dv.style.display='block';
    dv.innerHTML='✓ All 60 re-judged.<br>Tell Claude it is finished and he will run the report.';return;}
  const it=items[i];
  const v=document.getElementById('vid');
  v.src='/media/'+it.clip.split('/').map(encodeURIComponent).join('/');v.load();
  document.getElementById('name').textContent=(i+1)+' / '+items.length+'   '+it.clip.split('/')[1];
  document.getElementById('reason').value='';
  document.getElementById('warn').textContent='';
}
async function save(tag){
  const r=document.getElementById('reason').value.trim();
  if(r.length<3){document.getElementById('warn').textContent='Please type a short reason first.';
    document.getElementById('reason').focus();return;}
  const it=items[i];done[it.clip]={tag:tag,reason:r};
  i++;show();
  try{await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clip:it.clip,tag:tag,reason:r})});}catch(e){console.error(e);}
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&e.key!=='Enter')return;
  if(e.key==='Enter'){document.getElementById('reason').focus();return;}
  if(['1','2','3','4'].includes(e.key)){const o=OPTS[+e.key-1];if(o)save(o[0]);}
});
boot();
</script></body></html>"""


def load_set():
    return json.loads(SET_FILE.read_text(encoding="utf-8"))


def load_done():
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_one(clip, tag, reason):
    with _LOCK:
        d = load_done()
        d[clip] = {"tag": tag, "reason": reason}
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, OUT)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            page = PAGE.replace("__OPTS__", json.dumps(OPTIONS))
            return self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
        if path == "/api/set":
            return self._send(200, "application/json",
                              json.dumps({"items": load_set(),
                                          "done": load_done()}).encode())
        if path.startswith("/media/"):
            rel = urllib.parse.unquote(path[len("/media/"):])
            f = BENCH / rel
            if not f.exists():
                return self._send(404, "text/plain", b"not found")
            data = f.read_bytes()
            ctype = "video/mp4" if f.suffix.lower() == ".mp4" else "video/webm"
            return self._send(200, ctype, data)
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/save":
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        save_one(d.get("clip"), d.get("tag"), d.get("reason", ""))
        self._send(200, "application/json", b'{"ok":true}')


def main():
    items = load_set()
    print("=" * 62)
    print("  BLIND RE-ADJUDICATION")
    print(f"  {len(items)} clips you already tagged, shuffled, labels hidden.")
    print("  No Drop button. A short reason is required for every clip.")
    print(f"  Results -> {OUT}   (tags.json is NOT touched)")
    print(f"  Open http://localhost:{PORT}   -- keep this window open")
    print("=" * 62)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

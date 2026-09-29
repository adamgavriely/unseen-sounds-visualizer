/* Read-only viewer for tagger exports (tagger_*.json). Works from file:// : the files are read with FileReader.
   Several exports can be loaded at once; each annotator gets its own lane on the timeline. */
(function () {
  "use strict";
  var CLIPS = window.DELEGATION_CLIPS || [];
  var DUR = {};
  CLIPS.forEach(function (c) { DUR[c.id] = c.duration; });
  var runs = [];                 // [{name, data:{clipId: clipRecord}}]
  var shown = [], idx = 0;
  var COLORS = ["#3E5C76", "#2F6B4F", "#9A6B00", "#A33A2E", "#6B4E8A"];

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function f1(v) { return v == null || v === "" ? "–" : Number(v).toFixed(1); }
  var video = $("video");

  function addExport(name, j) {
    if (!j || !Array.isArray(j.clips)) throw new Error(name + " is not a tagger export");
    var data = {};
    j.clips.forEach(function (c) { data[c.clip] = c; });
    var who = (j.annotator || name).trim();
    runs = runs.filter(function (r) { return r.name !== who; });
    runs.push({ name: who, file: name, exported: j.exported, data: data });
    rebuild();
  }

  function readFiles(files) {
    Array.prototype.forEach.call(files, function (f) {
      var fr = new FileReader();
      fr.onload = function () {
        try { addExport(f.name, JSON.parse(fr.result)); }
        catch (e) { alert("Could not read " + f.name + ": " + e.message); }
      };
      fr.readAsText(f);
    });
  }

  function tagged(id) { return runs.some(function (r) { return r.data[id]; }); }

  function rebuild() {
    var all = $("showAll").checked;
    shown = CLIPS.filter(function (c) { return all || tagged(c.id); });
    if (idx >= shown.length) idx = 0;
    renderStats(); renderList(); show();
    $("drop").hidden = runs.length > 0;
  }

  function renderStats() {
    var parts = runs.map(function (r) {
      var cl = Object.keys(r.data).map(function (k) { return r.data[k]; });
      var done = cl.filter(function (c) { return c.done; }).length;
      var rows = cl.reduce(function (a, c) { return a + (c.sounds || []).length; }, 0);
      var secs = cl.filter(function (c) { return c.done && c.seconds_on_clip; }).map(function (c) { return c.seconds_on_clip; }).sort(function (a, b) { return a - b; });
      var med = secs.length ? secs[Math.floor(secs.length / 2)] : null;
      return r.name + ": " + done + " done, " + rows + " sounds" + (med != null ? ", median " + (med / 60).toFixed(1) + " min per clip" : "");
    });
    $("stats").textContent = parts.join("  ·  ");
  }

  function renderList() {
    var ul = $("list"); ul.innerHTML = "";
    $("nclips").textContent = shown.length + " / " + CLIPS.length;
    shown.forEach(function (c, i) {
      var li = el("li"); if (i === idx) li.className = "cur";
      var recs = runs.map(function (r) { return r.data[c.id]; }).filter(Boolean);
      var n = recs.reduce(function (a, r) { return a + (r.sounds || []).length; }, 0);
      var mk = el("span", "mk", recs.some(function (r) { return r.bad; }) ? "✗" : recs.some(function (r) { return r.done; }) ? "✓" : recs.length ? "●" : "○");
      li.appendChild(mk); li.appendChild(el("span", null, c.id.replace(".mp4", "")));
      if (recs.length) li.appendChild(el("span", "hint", n + " snd"));
      li.onclick = function () { idx = i; renderList(); show(); };
      ul.appendChild(li);
    });
  }

  function kind(s) { return s.obvious ? "obv" : s.visible ? "vis" : "need"; }
  function seek(t) { video.currentTime = Math.max(0, (Number(t) || 0) - 0.5); video.play().catch(function () {}); }

  function show() {
    var c = shown[idx];
    $("rows").innerHTML = ""; $("lanes").innerHTML = ""; $("axis").innerHTML = ""; $("notes").innerHTML = ""; $("flags").innerHTML = "";
    if (!c) { $("title").textContent = runs.length ? "No tagged clips" : "Load an export"; video.removeAttribute("src"); return; }
    $("title").textContent = c.id.replace(".mp4", "");
    if (video.getAttribute("src") !== c.src) video.src = c.src;
    var dur = DUR[c.id] || 20;
    for (var t = 0; t <= dur + 0.001; t += (dur > 12 ? 2 : 1)) {
      var s = el("span", null, t + "s"); s.style.left = (100 * t / dur) + "%"; $("axis").appendChild(s);
    }
    var thead = el("tr");
    ["who", "#", "sound", "start", "end", "visible", "obvious", "importance", "covered"].forEach(function (h) { thead.appendChild(el("th", null, h)); });
    $("rows").appendChild(thead);
    runs.forEach(function (r) {
      var rec = r.data[c.id];
      var lane = el("div", "lane"); lane.appendChild(el("div", "who", r.name));
      var track = el("div", "track"); lane.appendChild(track);
      var rows = rec ? (rec.sounds || []).slice().sort(function (a, b) { return (a.start || 0) - (b.start || 0); }) : [];
      // stack overlapping bars on sub-rows
      var ends = [];
      rows.forEach(function (s, k) {
        var a = Math.max(0, Number(s.start) || 0), b = Math.min(dur, Math.max(a + 0.3, Number(s.end) || a + 0.3));
        var level = 0; while (ends[level] != null && ends[level] > a) level++;
        ends[level] = b;
        var bar = el("div", "bar-s " + kind(s), (s.label || "?") + (s.importance ? " · " + s.importance : ""));
        bar.style.left = (100 * a / dur) + "%"; bar.style.width = Math.max(1.5, 100 * (b - a) / dur) + "%";
        bar.style.top = (4 + level * 26) + "px"; bar.tabIndex = 0;
        bar.title = (s.label || "") + "  " + f1(s.start) + "–" + f1(s.end) + " s  visible " + (s.visible ? "yes" : "no") + ", obvious " + (s.obvious ? "yes" : "no") + ", importance " + (s.importance || "–") + (s.masked ? ", covered by speech/music" : "");
        bar.onclick = function () { seek(s.start); };
        bar.onkeydown = function (e) { if (e.key === "Enter") seek(s.start); };
        track.appendChild(bar);
        var tr = el("tr", "click");
        [r.name, String(k + 1), s.label || "?", f1(s.start), f1(s.end)].forEach(function (v) { tr.appendChild(el("td", null, v)); });
        tr.appendChild(el("td", s.visible ? "yes" : "no", s.visible ? "yes" : "no"));
        tr.appendChild(el("td", s.obvious ? "yes" : "no", s.obvious ? "yes" : "no"));
        tr.appendChild(el("td", null, s.importance == null ? "–" : String(s.importance)));
        tr.appendChild(el("td", s.masked ? "yes" : "no", s.masked ? "yes" : "no"));
        tr.onclick = function () { seek(s.start); };
        $("rows").appendChild(tr);
      });
      track.style.height = (8 + Math.max(1, ends.length) * 26) + "px";
      if (!rows.length) track.appendChild(el("div", "hint", rec ? (rec.no_sounds ? "  no sounds besides speech/music" : "  no rows") : "  not tagged"));
      $("lanes").appendChild(lane);
      if (rec) {
        var p = el("p");
        p.appendChild(el("b", null, r.name + ": "));
        if (rec.done) p.appendChild(el("span", "flag done", "done"));
        if (rec.bad) p.appendChild(el("span", "flag bad", "marked broken"));
        if (rec.no_sounds) p.appendChild(el("span", "flag none", "no sounds besides speech/music"));
        if (rec.seconds_on_clip) p.appendChild(el("span", "hint", (rec.seconds_on_clip / 60).toFixed(1) + " min on this clip  "));
        if ((rec.note || "").trim()) p.appendChild(el("span", null, "Comment: " + rec.note));
        $("notes").appendChild(p);
      }
    });
    if (!$("notes").childNodes.length) $("notes").appendChild(el("span", "hint", "No comments or flags."));
    tick();
  }

  function tick() {
    var c = shown[idx]; if (!c) return;
    var dur = DUR[c.id] || video.duration || 20;
    $("time").textContent = (video.currentTime || 0).toFixed(1) + " s / " + dur.toFixed(1) + " s";
    var w = $("tl").clientWidth - 20 - 110;
    $("head").style.left = (10 + w * Math.min(1, (video.currentTime || 0) / dur)) + "px";
  }
  video.addEventListener("timeupdate", tick);
  video.addEventListener("seeked", tick);
  window.addEventListener("resize", tick);

  $("files").onchange = function (e) { readFiles(e.target.files); e.target.value = ""; };
  $("showAll").onchange = rebuild;
  $("prev").onclick = function () { if (idx > 0) { idx--; renderList(); show(); } };
  $("next").onclick = function () { if (idx < shown.length - 1) { idx++; renderList(); show(); } };
  var drop = $("drop");
  ["dragenter", "dragover"].forEach(function (ev) { document.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (ev) { document.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  document.addEventListener("drop", function (e) { if (e.dataTransfer && e.dataTransfer.files.length) readFiles(e.dataTransfer.files); });
  window.__inspectLoad = addExport;   // for testing from the console
  rebuild();
})();

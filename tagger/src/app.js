/* Sound tagging tool (manual listening, no pre-fill). Works from file:// : no fetch, no server.
   Data: window.DELEGATION_CLIPS (src/clips.js), window.DELEGATION_VOCAB (src/vocab.js),
   window.DELEGATION_SUGGEST (src/suggest.js: detector suggestions for some clips, hidden until the human asks;
   pressing "Show" records suggestions_shown_at; a row copied from one carries from_suggestion: true).
   Export schema = the project's gold export ({annotator, exported, clips:[{clip, sounds:[{label,
   family?, start, end, visible, obvious, importance, ...}], done, bad, picture_due, ...}]}),
   readable by benchmark/gold/score_per_sound.load_gold. */
(function () {
  "use strict";
  var CLIPS = window.DELEGATION_CLIPS || [];
  var VOCAB = window.DELEGATION_VOCAB || [];
  var SUGGEST = window.DELEGATION_SUGGEST || {};
  var ANNOT = window.DELEGATION_ANNOT || {};        // src/annot.js: other annotators' tags (AudioSet-Strong), hidden until asked
  var STORE_KEY = "delegation-sound-tagging-v1";   // distinct from the gold tool's "gold-..." keys
  var TOOL = "delegation-manual-v1";
  var VOCAB_LC = {};
  VOCAB.forEach(function (v) { VOCAB_LC[v.toLowerCase()] = v; });
  var NOT_TAGGED = /^\s*(speech|speaking|talk|talking|voice|voices|conversation|narration|narrator|music|song|singing|soundtrack)\b/i;

  function $(id) { return document.getElementById(id); }
  var video = $("video");
  var store = { v: 1, annotator: "", started: null, last: 0, clips: {} };
  var idx = 0, sel = -1, storageOk = true;

  // ------------------------------------------------------------------ storage
  function load() {
    try {
      var raw = window.localStorage.getItem(STORE_KEY);
      if (raw) {
        var s = JSON.parse(raw);
        if (s && typeof s === "object" && s.clips) store = s;
      }
    } catch (e) { storageOk = false; }
    if (!store.started) store.started = new Date().toISOString();
  }
  function save() {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(store));
      storageOk = true;
      $("saved").textContent = "saved " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch (e) {
      storageOk = false;
      $("saved").textContent = "cannot save in this browser — Export often!";
    }
  }
  function rec(c) {
    c = c || CLIPS[idx];
    if (!store.clips[c.id]) {
      store.clips[c.id] = { clip: c.id, sounds: [], no_sounds: false, bad: false, note: "", done: false,
                            first_opened: new Date().toISOString(), updated: null, done_at: null, seconds: 0 };
    }
    return store.clips[c.id];
  }
  function touch() { var r = rec(); r.updated = new Date().toISOString(); save(); renderList(); renderState(); }

  // ------------------------------------------------------------------ helpers
  function num(v) { if (v === "" || v == null) return null; var x = Number(v); return isFinite(x) ? Math.round(x * 10) / 10 : null; }
  function fmt(v) { return v == null ? "" : Number(v).toFixed(1); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function started(r) { return r && (r.sounds.length || r.no_sounds || r.bad || (r.note || "").trim()); }
  function msg(text, kind) { var m = $("msg"); m.textContent = text || ""; m.className = "msg" + (kind ? " " + kind : ""); }

  // ------------------------------------------------------------------ clip list + header
  function renderList() {
    var ol = $("clipList"); ol.innerHTML = "";
    var nd = 0;
    CLIPS.forEach(function (c, i) {
      var r = store.clips[c.id];
      var li = el("li");
      var state = r && r.bad && r.done ? "bad" : r && r.done ? "done" : started(r) ? "started" : "";
      if (r && r.done) nd++;
      li.className = (i === idx ? "cur " : "") + state;
      li.appendChild(el("span", "mk", state === "done" ? "✓" : state === "bad" ? "✗" : state === "started" ? "●" : "○"));
      li.appendChild(el("span", "", c.id.replace(/\.mp4$/, "")));
      li.addEventListener("click", function () { go(i); });
      ol.appendChild(li);
    });
    $("progress").textContent = nd + " / " + CLIPS.length + " done";
    $("sideCount").textContent = nd + " / " + CLIPS.length;
    var cur = ol.children[idx]; if (cur && cur.scrollIntoView) { try { cur.scrollIntoView({ block: "nearest" }); } catch (e) {} }
  }
  function renderState() {
    var r = rec();
    $("clipState").textContent = r.done ? (r.bad ? "✗ marked broken" : "✓ done") : "not done";
    $("clipState").style.color = r.done ? (r.bad ? "var(--bad)" : "var(--ok)") : "var(--muted)";
  }

  // ------------------------------------------------------------------ sound rows
  function seg(options, value, onPick, label) {
    var box = el("span", "seg");
    if (label) box.appendChild(el("span", "seglbl", label));
    options.forEach(function (o) {
      var b = el("button", o[0] === value ? "on" : "", o[1]); b.type = "button"; b.title = o[2] || "";
      b.addEventListener("click", function (ev) { ev.stopPropagation(); onPick(o[0]); });
      box.appendChild(b);
    });
    return box;
  }
  function renderSounds() {
    var r = rec(), box = $("sounds"); box.innerHTML = "";
    if (!r.sounds.length) {
      box.appendChild(el("p", "hint empty", "No sounds yet. Press “+ Add a sound” (or A), or tick “No sounds besides speech or music” below."));
    }
    r.sounds.forEach(function (s, i) {
      var row = el("div", "srow" + (i === sel ? " sel" : ""));
      row.addEventListener("mousedown", function () { if (sel !== i) { sel = i; markSel(); } });
      row.appendChild(el("span", "n", String(i + 1)));

      var name = el("input"); name.type = "text"; name.setAttribute("list", "vocab"); name.value = s.label || "";
      name.placeholder = "e.g. Siren, Bark, Door, Traffic noise";
      name.addEventListener("focus", function () { sel = i; markSel(); });
      name.addEventListener("input", function () { s.label = name.value; touch(); });
      name.addEventListener("change", function () { if (NOT_TAGGED.test(name.value)) msg("Speech and music are not tagged — only other sounds.", "err"); });
      row.appendChild(name);

      ["start", "end"].forEach(function (k) {
        var tm = el("span", "tm");
        var inp = el("input"); inp.type = "number"; inp.step = "0.1"; inp.min = "0"; inp.value = fmt(s[k]);
        inp.placeholder = k; inp.title = k + " time in seconds";
        inp.addEventListener("focus", function () { sel = i; markSel(); });
        inp.addEventListener("change", function () { s[k] = num(inp.value); inp.value = fmt(s[k]); touch(); });
        var b = el("button", "", "now"); b.type = "button"; b.title = "use the current video time";
        b.addEventListener("click", function (ev) { ev.stopPropagation(); sel = i; setTime(k); });
        var p = el("button", "", "▶"); p.type = "button"; p.title = "play from here";
        p.addEventListener("click", function (ev) { ev.stopPropagation(); if (s[k] != null) { video.currentTime = Math.max(0, s[k] - (k === "end" ? 1.5 : 0)); video.play(); } });
        tm.appendChild(inp); tm.appendChild(b); tm.appendChild(p);
        row.appendChild(tm);
      });

      row.appendChild(seg([[true, "Yes", "you can see what makes the sound"],
                           [false, "No", "you cannot see it, or it is too small or dark to recognise"]], s.visible,
        function (v) { sel = i; s.visible = v; touch(); renderSounds(); }, "Visible?"));
      row.appendChild(seg([[true, "Yes", "with the sound off, the picture still makes it clear this sound is happening"],
                           [false, "No", "with the sound off, you would not know"]], s.obvious,
        function (v) { sel = i; s.obvious = v; touch(); renderSounds(); }, "Obvious?"));
      row.appendChild(seg([[1, "1", "steady background noise, no clear start"], [2, "2", "something that happens (clear start)"], [3, "3", "danger or key story moment"]], s.importance,
        function (v) { sel = i; s.importance = v; touch(); renderSounds(); }, "Importance"));
      var mk = el("label", "masked"); mk.title = "tick if speech or music makes this sound hard to hear";
      var cb = el("input"); cb.type = "checkbox"; cb.checked = s.masked === true;
      cb.addEventListener("click", function (ev) { ev.stopPropagation(); });
      cb.addEventListener("change", function () { sel = i; s.masked = cb.checked; touch(); });
      mk.appendChild(cb); mk.appendChild(el("span", "", " covered by speech/music"));
      row.appendChild(mk);

      var ops = el("span", "ops");
      var play = el("button", "", "▶ row"); play.type = "button"; play.title = "play this sound from start to end";
      play.addEventListener("click", function (ev) { ev.stopPropagation(); sel = i; markSel(); playSpan(s.start, s.end); });
      var del = el("button", "", "✕"); del.type = "button"; del.title = "delete this row";
      del.addEventListener("click", function (ev) {
        ev.stopPropagation();
        if ((s.label || s.start != null) && !confirm("Delete row " + (i + 1) + (s.label ? " (" + s.label + ")" : "") + "?")) return;
        r.sounds.splice(i, 1); sel = Math.min(sel, r.sounds.length - 1); touch(); renderSounds();
      });
      ops.appendChild(play); ops.appendChild(del);
      row.appendChild(ops);
      if (s.from_suggestion) row.appendChild(el("span", "fromsug", "from a detector suggestion — please check it by ear"));
      if (s.from_annotators) row.appendChild(el("span", "fromsug", "from other annotators' tags — please check it by ear"));
      box.appendChild(row);
    });
    renderSuggest();
  }

  // ------------------------------------------------------------------ detector suggestions (hidden by default)
  // Shown only after the human presses the button (to avoid anchoring). They are never rows: they do not count for
  // done/started; "Add as a row" copies one into a normal row whose visible/obvious/importance stay empty.
  var LISTS = [
    { box: "suggestBox", data: SUGGEST, shown: "suggestions_shown_at", from: "from_suggestion",
      btn: "Show detector suggestions", head: "Detector suggestions — not checked; they can be wrong or missing sounds",
      tip: "Tag by ear first. The suggestions come from our detector and can be wrong or miss sounds." },
    { box: "annotBox", data: ANNOT, shown: "annotators_shown_at", from: "from_annotators",
      btn: "Show other annotators' tags", head: "Other annotators' tags (AudioSet-Strong) — their names and times; check each by ear",
      tip: "Tag by ear first. These are tags other people made for this clip; times can differ and sounds can be missing." }];
  function renderSuggest() { LISTS.forEach(renderSugList); }
  function renderSugList(L) {
    var box = $(L.box); if (!box) return;
    box.innerHTML = "";
    var c = CLIPS[idx], list = L.data[c.id];
    if (!list || !list.length) { box.hidden = true; return; }
    box.hidden = false;
    var r = rec(c);
    if (!r[L.shown]) {
      var b = el("button", "sugbtn", L.btn + " (" + list.length + ")"); b.type = "button";
      b.title = L.tip;
      b.addEventListener("click", function () {
        var q = rec(c); q[L.shown] = new Date().toISOString(); save(); renderSugList(L);
      });
      box.appendChild(b);
      box.appendChild(el("span", "hint", " Tag by ear first; open these only as a last check."));
      return;
    }
    var wrap = el("div", "suglist");
    wrap.appendChild(el("div", "sughead", L.head));
    list.forEach(function (g) {
      var row = el("div", "sugrow");
      row.appendChild(el("span", "suglabel", g.label));
      row.appendChild(el("span", "sugtime", fmt(g.start) + "–" + fmt(g.end) + " s"));
      var p = el("button", "", "▶ play"); p.type = "button"; p.title = "play from half a second before the start";
      p.addEventListener("click", function () {
        clearSpan(); video.currentTime = Math.max(0, Number(g.start) - 0.5);
        var pr = video.play(); if (pr && pr.catch) pr.catch(function () {});
      });
      var have = r.sounds.some(function (s) { return s[L.from] && s.label === g.label && s.start === num(g.start) && s.end === num(g.end); });
      var a = el("button", "", have ? "Added" : "Add as a row"); a.type = "button"; a.disabled = have;
      a.title = "copy this into the sound rows; you still answer Visible, Obvious and Importance";
      a.addEventListener("click", function () { addFromSuggestion(g, L.from); });
      row.appendChild(p); row.appendChild(a);
      wrap.appendChild(row);
    });
    box.appendChild(wrap);
  }
  function addFromSuggestion(g, from) {
    var r = rec();
    var row = { label: g.label, start: num(g.start), end: num(g.end), visible: null, obvious: null, importance: null, masked: false,
                added: new Date().toISOString(), from_suggestion: false, from_annotators: false };
    row[from || "from_suggestion"] = true;
    r.sounds.push(row);
    if (r.no_sounds) { r.no_sounds = false; $("noSounds").checked = false; }
    sel = r.sounds.length - 1; touch(); renderSounds();
  }
  function markSel() {
    var rows = document.querySelectorAll("#sounds .srow");
    for (var k = 0; k < rows.length; k++) rows[k].classList.toggle("sel", k === sel);
  }
  function addRow(startNow) {
    var r = rec();
    r.sounds.push({ label: "", start: startNow ? num(video.currentTime) : null, end: null, visible: null, obvious: null, importance: null, masked: false,
                    added: new Date().toISOString() });
    if (r.no_sounds) { r.no_sounds = false; $("noSounds").checked = false; }
    sel = r.sounds.length - 1; touch(); renderSounds();
    var inputs = document.querySelectorAll("#sounds .srow input[type=text]");
    if (inputs[sel]) inputs[sel].focus();
  }
  function setTime(k) {
    var r = rec();
    if (sel < 0 || !r.sounds[sel]) { if (k === "start") { addRow(true); return; } msg("Select a row first (click it).", "err"); return; }
    r.sounds[sel][k] = num(video.currentTime); touch(); renderSounds();
  }
  var spanStop = null, spanTimer = null;
  function clearSpan() {
    if (spanStop) video.removeEventListener("timeupdate", spanStop);
    if (spanTimer) clearInterval(spanTimer);
    spanStop = null; spanTimer = null;
  }
  function playSpan(a, b) {
    if (a == null) { msg("This row has no start time yet.", "err"); return; }
    clearSpan();
    var end = b != null && b > a ? b : a + 2;
    video.currentTime = Math.max(0, a);
    var p = video.play(); if (p && p.catch) p.catch(function () {});
    spanStop = function () { if (video.currentTime >= end) { video.pause(); clearSpan(); } };
    video.addEventListener("timeupdate", spanStop);
    spanTimer = setInterval(spanStop, 40);        // timeupdate alone fires only every ~250 ms
  }
  function replay2() {
    clearSpan();
    var t = video.currentTime; playSpan(Math.max(0, t - 2), t > 0.2 ? t : 2);
  }

  // ------------------------------------------------------------------ clip view
  function renderClip() {
    var c = CLIPS[idx], r = rec(c);
    sel = r.sounds.length ? 0 : -1;
    $("clipName").textContent = "Clip " + c.id.replace(/\.mp4$/, "") + "  (" + (idx + 1) + " of " + CLIPS.length + ")";
    clearSpan();
    video.src = c.src; video.load();
    video.playbackRate = Number($("speed").value) || 1;
    $("dur").textContent = c.duration ? Number(c.duration).toFixed(1) : "–";
    $("noSounds").checked = !!r.no_sounds; $("bad").checked = !!r.bad; $("note").value = r.note || "";
    msg("");
    renderSounds(); renderList(); renderState();
    store.last = idx; save();
  }
  function go(i) { if (i < 0 || i >= CLIPS.length) return; idx = i; renderClip(); window.scrollTo(0, 0); }

  // ------------------------------------------------------------------ done / validation
  function problems(r, c) {
    var out = [], bad = [];
    if (r.bad) { if (!(r.note || "").trim()) out.push("Please say in the comment what is wrong with the video."); return { out: out, rows: bad }; }
    if (r.no_sounds && r.sounds.length) out.push("You ticked “No sounds besides speech or music” but there are sound rows. Untick it or delete the rows.");
    if (!r.no_sounds && !r.sounds.length) out.push("Add at least one sound, or tick “No sounds besides speech or music”.");
    var dur = Number(c.duration) || 1e9;
    r.sounds.forEach(function (s, i) {
      var p = [];
      if (!(s.label || "").trim()) p.push("name");
      else if (NOT_TAGGED.test(s.label)) p.push("speech/music are not tagged");
      if (s.start == null) p.push("start");
      if (s.end == null) p.push("end");
      if (s.start != null && s.end != null && !(s.end > s.start)) p.push("end must be after start");
      if (s.end != null && s.end > dur + 0.5) p.push("end is after the clip ends (" + dur.toFixed(1) + " s)");
      if (typeof s.visible !== "boolean") p.push("visible?");
      if (typeof s.obvious !== "boolean") p.push("obvious?");
      if (!s.importance) p.push("importance");
      if (p.length) { out.push("Row " + (i + 1) + ": " + p.join(", ")); bad.push(i); }
    });
    return { out: out, rows: bad };
  }
  function markDone() {
    var c = CLIPS[idx], r = rec(c);
    var pr = problems(r, c);
    var rows = document.querySelectorAll("#sounds .srow");
    for (var k = 0; k < rows.length; k++) rows[k].classList.toggle("err", pr.rows.indexOf(k) >= 0);
    if (pr.out.length) { msg("Not done yet — " + pr.out.join(" · "), "err"); return; }
    r.done = true; r.done_at = new Date().toISOString();
    r.sounds.sort(function (a, b) { return (a.start - b.start) || (a.end - b.end); });
    touch();
    var next = -1;
    for (var j = 1; j <= CLIPS.length; j++) { var k2 = (idx + j) % CLIPS.length; var q = store.clips[CLIPS[k2].id]; if (!q || !q.done) { next = k2; break; } }
    if (next < 0) { renderClip(); msg("All clips are done. Please click Export and send the file.", "ok"); return; }
    go(next); msg("Saved. Clip " + c.id.replace(/\.mp4$/, "") + " is done.", "ok");
  }

  // ------------------------------------------------------------------ export / import
  function exportObject() {
    var who = ($("who").value || "").trim() || "anonymous";
    var clips = [];
    CLIPS.forEach(function (c) {
      var r = store.clips[c.id]; if (!r) return;
      var sounds = r.sounds.map(function (s) {
        var o = { label: (s.label || "").trim() };
        var fam = VOCAB_LC[o.label.toLowerCase()];
        if (fam) o.family = fam;                         // only an exact vocabulary name; free text is resolved later
        o.start = s.start; o.end = s.end;
        o.visible = s.visible === true;                  // gold schema: needed = not (visible or obvious)
        o.obvious = s.obvious === true;
        o.masked = s.masked === true;                    // speech or music makes it hard to hear (gold schema "masked")
        o.importance = s.importance || null;
        o.added = s.added || null;
        o.from_suggestion = s.from_suggestion === true;
        o.from_annotators = s.from_annotators === true;
        return o;
      });
      clips.push({ clip: c.id, sounds: sounds, no_sounds: !!r.no_sounds, bad: !!r.bad, note: r.note || "",
                   done: !!r.done, picture_due: sounds.some(function (s) { return s.label && !s.visible && !s.obvious; }),
                   annotator: who, first_opened: r.first_opened || null, updated: r.updated || null, done_at: r.done_at || null,
                   seconds_on_clip: Math.round(r.seconds || 0), suggestions_shown_at: r.suggestions_shown_at || null,
                   annotators_shown_at: r.annotators_shown_at || null });
    });
    return { annotator: who, exported: new Date().toISOString(), started: store.started, tool: TOOL, tool_version: 1,
             n_clips_total: CLIPS.length, n_done: clips.filter(function (c) { return c.done; }).length, clips: clips };
  }
  window.__delegationExport = exportObject;   // for testing from the console
  function doExport() {
    var out = exportObject();
    if (out.annotator === "anonymous") { msg("Please type your name at the top first.", "err"); $("who").focus(); return; }
    var d = new Date(), p = function (n) { return (n < 10 ? "0" : "") + n; };
    var fname = "tagger_" + out.annotator.replace(/[^A-Za-z0-9_-]+/g, "_") + "_" + d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + "_" + p(d.getHours()) + p(d.getMinutes()) + ".json";
    var blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = fname;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    msg("Exported " + fname + " (" + out.n_done + " clips done). Keep this file; send it when you finish.", "ok");
  }
  function doImport(file) {
    var fr = new FileReader();
    fr.onload = function () {
      try {
        var j = JSON.parse(fr.result);
        var ids = {}; CLIPS.forEach(function (c) { ids[c.id] = true; });
        var list = (j.clips || []).filter(function (c) { return c && ids[c.clip]; });
        if (!list.length) { msg("That file has no clips of this tool.", "err"); return; }
        var have = Object.keys(store.clips).filter(function (k) { return started(store.clips[k]); }).length;
        if (have && !confirm("Import " + list.length + " clips from this file? It replaces what is saved in this browser for those clips.")) return;
        list.forEach(function (c) {
          store.clips[c.clip] = {
            clip: c.clip, no_sounds: !!c.no_sounds, bad: !!c.bad, note: c.note || "", done: !!c.done,
            first_opened: c.first_opened || null, updated: c.updated || null, done_at: c.done_at || null, seconds: c.seconds_on_clip || 0,
            suggestions_shown_at: c.suggestions_shown_at || null, annotators_shown_at: c.annotators_shown_at || null,
            sounds: (c.sounds || []).map(function (s) {
              return { label: s.label || s.family || "", start: s.start == null ? null : Number(s.start), end: s.end == null ? null : Number(s.end),
                       visible: typeof s.visible === "boolean" ? s.visible : null,
                       obvious: typeof s.obvious === "boolean" ? s.obvious : null,
                       importance: [1, 2, 3].indexOf(Number(s.importance)) >= 0 ? Number(s.importance) : null, added: s.added || null,
                       masked: s.masked === true, from_suggestion: s.from_suggestion === true,
                       from_annotators: s.from_annotators === true };
            })
          };
        });
        if (j.annotator && j.annotator !== "anonymous") { $("who").value = j.annotator; store.annotator = j.annotator; }
        if (j.started && (!store.started || j.started < store.started)) store.started = j.started;
        save(); renderClip(); msg("Imported " + list.length + " clips.", "ok");
      } catch (e) { msg("Could not read that file (" + e.message + ").", "err"); }
    };
    fr.readAsText(file);
  }

  // ------------------------------------------------------------------ guide
  function openGuide() { $("guideOverlay").hidden = false; }
  function closeGuide() { $("guideOverlay").hidden = true; store.guide_seen = true; save(); }

  // ------------------------------------------------------------------ wiring
  function wire() {
    var dl = $("vocab");
    VOCAB.forEach(function (v) { var o = document.createElement("option"); o.value = v; dl.appendChild(o); });
    $("who").value = store.annotator || "";
    $("who").addEventListener("input", function () { store.annotator = $("who").value.trim(); save(); });
    $("prevBtn").addEventListener("click", function () { go(idx - 1); });
    $("nextBtn").addEventListener("click", function () { go(idx + 1); });
    $("replayBtn").addEventListener("click", replay2);
    $("setStartBtn").addEventListener("click", function () { setTime("start"); });
    $("setEndBtn").addEventListener("click", function () { setTime("end"); });
    $("speed").addEventListener("change", function () { video.playbackRate = Number($("speed").value) || 1; });
    $("addBtn").addEventListener("click", function () { addRow(false); });
    $("noSounds").addEventListener("change", function () { rec().no_sounds = $("noSounds").checked; touch(); });
    $("bad").addEventListener("change", function () { rec().bad = $("bad").checked; touch(); });
    $("note").addEventListener("input", function () { rec().note = $("note").value; touch(); });
    $("doneBtn").addEventListener("click", markDone);
    $("undoneBtn").addEventListener("click", function () { var r = rec(); r.done = false; r.done_at = null; touch(); msg("Marked as not done.", ""); });
    $("exportBtn").addEventListener("click", doExport);
    $("importBtn").addEventListener("click", function () { $("importFile").click(); });
    $("importFile").addEventListener("change", function (e) { var f = e.target.files[0]; if (f) doImport(f); e.target.value = ""; });
    $("guideBtn").addEventListener("click", openGuide);
    $("guideClose").addEventListener("click", closeGuide);
    $("guideOk").addEventListener("click", closeGuide);
    $("guideOverlay").addEventListener("click", function (e) { if (e.target === $("guideOverlay")) closeGuide(); });
    video.addEventListener("timeupdate", function () { $("now").textContent = video.currentTime.toFixed(1); });
    video.addEventListener("seeked", function () { $("now").textContent = video.currentTime.toFixed(1); });
    video.addEventListener("loadedmetadata", function () { if (isFinite(video.duration)) $("dur").textContent = video.duration.toFixed(1); });
    video.addEventListener("error", function () { msg("This video could not be played. Tick “broken” and write a comment, or try Chrome/Edge/Firefox.", "err"); });

    document.addEventListener("keydown", function (e) {
      if (!$("guideOverlay").hidden) { if (e.key === "Escape") closeGuide(); return; }
      var t = (e.target.tagName || "").toLowerCase();
      if (t === "input" || t === "textarea" || t === "select" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === " ") { e.preventDefault(); if (video.paused) video.play(); else video.pause(); }
      else if (e.key === "r" || e.key === "R") replay2();
      else if (e.key === "[") setTime("start");
      else if (e.key === "]") setTime("end");
      else if (e.key === "a" || e.key === "A") { e.preventDefault(); addRow(false); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); video.currentTime = Math.max(0, video.currentTime - 1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); video.currentTime = Math.min(video.duration || 1e9, video.currentTime + 1); }
      else if (e.key === "n" || e.key === "N") go(idx + 1);
      else if (e.key === "p" || e.key === "P") go(idx - 1);
    });

    // time spent on each clip (only while the page is visible), for the annotator's own record
    setInterval(function () {
      if (document.visibilityState !== "visible" || !CLIPS.length) return;
      var r = rec(); r.seconds = (r.seconds || 0) + 5;
    }, 5000);
    setInterval(save, 30000);
    window.addEventListener("beforeunload", save);
  }

  load();
  if (!CLIPS.length) { document.querySelector("main").innerHTML = "<p class='msg err'>No clips found: src/clips.js is missing.</p>"; return; }
  wire();
  idx = Number.isInteger(store.last) && store.last >= 0 && store.last < CLIPS.length ? store.last : 0;
  renderClip();
  if (!storageOk) $("saved").textContent = "cannot save in this browser — Export often!";
  if (!store.guide_seen) openGuide();
})();

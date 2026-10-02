/* Decision Inspector. Data: window.INSPECTOR2 from data.js (made by export.py). No network; opens by double-click. */
(function () {
  'use strict';
  var D = window.INSPECTOR2, app = document.getElementById('app');
  var C = window.INSPECTOR2_CONTENT || {};
  if (D) ['build', 'build_steps', 'tried', 'lesson'].forEach(function (k) { if (D[k] == null && C[k] != null) D[k] = C[k]; });
  if (!D) { app.innerHTML = '<div class="banner"><b>data.js is missing.</b> Run <code>python docs/inspector2/export.py</code>, then reload.</div>'; return; }

  // ---------------------------------------------------------------- helpers
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function t1(x) { return x == null ? '—' : (Math.round(x * 100) / 100).toFixed(2); }
  function shortWhy(w) { w = String(w || ''); var a = w.split(' | ')[0].split(' Also near the onset')[0]; return a.length < w.length ? a + ' … (open the row for the full trail)' : a; }
  function chip(k, txt) { return '<span class="chip c-' + esc(k) + '">' + esc(txt || k) + '</span>'; }
  var STEP = {}; (D.steps || []).forEach(function (s, i) { s.order = i; STEP[s.id] = s; });
  function stepName(id) { return STEP[id] ? STEP[id].name : (id || '—'); }
  function stepTag(id) { return id ? '<a class="steptag" href="#steps/' + encodeURIComponent(id) + '" title="' + esc(STEP[id] ? STEP[id].plain : '') + '">' + esc(stepName(id)) + '</a>' : '—'; }
  function clipHref(c) { return '#clip/' + encodeURIComponent(c.split) + '/' + encodeURIComponent(c.clip); }
  var setSel = document.getElementById('setsel');
  function store(k, v) { try { if (v === undefined) return localStorage.getItem('insp2-' + k); localStorage.setItem('insp2-' + k, v); } catch (e) { return null; } }
  setSel.value = store('set') || 'DEV';
  setSel.addEventListener('change', function () { store('set', setSel.value); route(); });
  function inSet(c) { var s = setSel.value; return s === 'ALL' || c.split === s; }
  function clips() { return (D.clips || []).filter(inSet); }
  function candOf(c, id) { for (var i = 0; i < (c.cands || []).length; i++) if (c.cands[i].id === id) return c.cands[i]; return null; }
  function goldOf(c, id) { for (var i = 0; i < (c.gold || []).length; i++) if (c.gold[i].id === id) return c.gold[i]; return null; }
  var testWarn = function () { return setSel.value !== 'DEV' ? '<div class="banner"><b>TEST is a report.</b> Look only; no setting may be chosen from what you see here.</div>' : ''; };

  // ---------------------------------------------------------------- trail rendering
  function renderTrail(cand) {
    if (!cand) return '<p class="empty">No candidate span reached the pipeline for this sound.</p>';
    var h = '<div class="trail">';
    (cand.trail || []).forEach(function (st) {
      var vb = '';
      if (st.value != null || st.bar != null) vb = '<span class="vb">' + (st.value != null ? 'value ' + esc(st.value) : '') + (st.bar != null ? '  ·  bar ' + esc(st.bar) : '') + '</span>';
      h += '<div class="tstep ' + esc(st.res) + '"><div class="thead"><b>' + stepTag(st.step) + '</b>' + chip(st.res) + vb + '</div>';
      (st.asks || []).forEach(function (a) {
        h += '<div class="ask">' + (a.who ? '<div class="note">' + esc(a.who) + '</div>' : '') + '<div class="q">' + esc(a.q) + '</div><div class="a">Answer: <code>' + esc(String(a.a || '').replace(/\s*\[raw (answers?|description) not stored[^\]]*\]/, '') || '—') + '</code>' + (/raw (answers?|description) not stored/.test(a.a || '') ? ' ' + chip('skip', 'vote only') : '') + (a.vote ? '  → ' + chip(a.vote === 'seen' || a.vote === 'no' ? 'drop' : 'pass', a.vote) : '') + '</div></div>';
      });
      if (st.note) h += '<div class="note">' + esc(st.note) + '</div>';
      h += '</div>';
    });
    return h + '</div>';
  }
  function candCard(c, cand) {
    var fate = cand.fate === 'dropped' ? 'dropped at ' + stepName(cand.at) : cand.fate;
    return '<div class="cand"><h4>' + esc(cand.label) + ' <span class="vb">' + t1(cand.start) + '–' + t1(cand.end) + ' s · from ' + esc(cand.origin || '?') + '</span>' + chip(cand.fate, fate) + '</h4>' + renderTrail(cand) + '</div>';
  }

  // ---------------------------------------------------------------- collected lists
  function misses() {
    var out = [];
    clips().forEach(function (c) { (c.gold || []).forEach(function (g) { if (g.needed && g.outcome === 'miss') out.push({ c: c, g: g }); }); });
    return out;
  }
  function wrongs() {
    var out = [];
    clips().forEach(function (c) { (c.pictures || []).forEach(function (p) { if (p.verdict !== 'hit') out.push({ c: c, p: p }); }); });
    return out;
  }


  function gapsBox() {
    var cs = clips(), nh = 0, sc = 0, noTrail = 0;
    cs.forEach(function (c) {
      (c.gold || []).forEach(function (g) { if (g.outcome === 'miss') { if (g.lost_at === 'never_heard') nh++; else if (g.lost_at === 'scorer') sc++; } });
      (c.pictures || []).forEach(function (p) { if (p.verdict !== 'hit' && !p.cand) noTrail++; });
    });
    return '<h2>What this data does not show</h2><div class="tw"><table><tbody>' +
      '<tr><td><b>On-screen check, raw answers</b></td><td>Across DEV + TEST, 86 of 156 on-screen decisions show the three votes and the object named, but not the model\'s full sentence. Those verdicts were reused from the stored run, which kept only the votes. The decision itself is exact. Marked <span class="chip c-skip">vote only</span> in the trail.</td></tr>' +
      '<tr><td><b>"Visibly happening" look-alike replies</b></td><td>For the display step that drops a picture whose event is visibly happening, only the final yes/no was stored, not the look-alike replies. The event questions are shown.</td></tr>' +
      '<tr><td><b>Never heard</b></td><td>' + nh + ' missed sounds: no detector produced a span of that type near the start, so no step ever decided anything. Their sub-bar scores are not logged.</td></tr>' +
      '<tr><td><b>Timing / matching</b></td><td>' + sc + ' missed sounds: a span of the right type existed, but no picture started inside −0.5 … +1.0 s. The note names the nearest span and what happened to it.</td></tr>' +
      (noTrail ? '<tr><td><b>Wrong picture without a trail</b></td><td>' + noTrail + ' wrong picture(s): no kept span of its type overlaps it, so it is shown without a trail.</td></tr>' : '') +
      '</tbody></table></div>';
  }

  // ---------------------------------------------------------------- views
  function vOverview() {
    var m = D.meta || {}, h = '';
    if (m.sample) h += '<div class="banner"><b>Sample data.</b> This page shows the layout with a few real cases from the 1 Oct error dissection; some values are illustrative. The full data for the final version comes from the pipeline log after the detector is frozen.</div>';
    h += testWarn();
    h += '<h1>Where each sound is won or lost</h1><p class="lede">Every needed sound and every picture, traced through the pipeline step by step: the value, the bar, and for model steps the exact question and answer. New here? Start with <a href="#build">the build in five parts</a> and <a href="#tried">what did not ship</a>.</p>';
    if (m.parity) h += '<p class="note">Parity check: re-running the frozen version with logging on gives exactly the reported numbers (DEV ' + esc(m.parity.DEV) + ', TEST ' + esc(m.parity.TEST) + '; 0 clips differ).</p>';
    var keys = setSel.value === 'ALL' ? ['DEV', 'TEST'] : [setSel.value];
    keys.forEach(function (k) {
      var s = (D.sets || {})[k]; if (!s) return;
      h += '<h2>' + k + ' · ' + s.clips + ' clips</h2><div class="tiles">' +
        tile('needed sounds found', s.hits + ' / ' + s.needed) + tile('wrong pictures', s.wrong + ' <small class="vb">' + s.visible + ' vis · ' + s.cross + ' cross · ' + s.phantom + ' phantom</small>') + tile('viewer cost', s.cost) + '</div>';
    });
    // funnel of misses by step
    var ms = misses(), by = {};
    ms.forEach(function (x) { var k = x.g.lost_at || 'never_heard'; by[k] = (by[k] || 0) + 1; });
    var rows = Object.keys(by).sort(function (a, b) { return ((STEP[a] || {}).order || 0) - ((STEP[b] || {}).order || 0); });
    var mx = Math.max.apply(null, [1].concat(rows.map(function (k) { return by[k]; })));
    h += '<h2>Where needed sounds are lost (' + ms.length + ' misses)</h2><div class="funnel">';
    rows.forEach(function (k) {
      var w = (by[k] / mx * 92).toFixed(1);
      h += '<div class="frow"><span class="lab">' + stepTag(k) + '</span><a class="ftrack" href="#misses/' + encodeURIComponent(k) + '" title="' + by[k] + ' misses lost at ' + esc(stepName(k)) + '"><div class="fbar" style="width:' + w + '%"></div><span class="fval" style="left:' + w + '%">' + by[k] + '</span></a></div>';
    });
    h += '</div><p class="note">Each miss is counted at the step where its last candidate span was removed. Click a bar for the list.</p>';
    var ws = wrongs(), wb = {};
    ws.forEach(function (x) { wb[x.p.verdict] = (wb[x.p.verdict] || 0) + 1; });
    h += '<h2>Wrong pictures (' + ws.length + ')</h2><div class="funnel">';
    var wm = Math.max.apply(null, [1].concat(Object.keys(wb).map(function (k) { return wb[k]; })));
    ['visible', 'cross', 'phantom'].forEach(function (k) {
      if (!wb[k]) return; var w = (wb[k] / wm * 92).toFixed(1);
      h += '<div class="frow"><span class="lab">' + chip(k) + '</span><a class="ftrack" href="#wrongs/' + k + '"><div class="fbar w" style="width:' + w + '%"></div><span class="fval" style="left:' + w + '%">' + wb[k] + '</span></a></div>';
    });
    h += '</div><p class="note">A wrong picture passed every step. Its trail shows how close each value was to the bar, which tells you which step could have caught it.</p>' + (m.sample ? '' : gapsBox());
    app.innerHTML = h;
  }
  function tile(k, v) { return '<div class="tile"><div class="k">' + esc(k) + '</div><div class="v">' + v + '</div></div>'; }

  function expandableTable(head, items, rowHtml, detailHtml, cols) {
    var h = '<div class="tw"><table><thead><tr>' + head.map(function (x) { return '<th>' + x + '</th>'; }).join('') + '</tr></thead><tbody>';
    items.forEach(function (it, i) { h += '<tr class="row" data-i="' + i + '" tabindex="0" aria-expanded="false">' + rowHtml(it) + '</tr>'; });
    if (!items.length) h += '<tr><td colspan="' + cols + '" class="empty">Nothing here for this set.</td></tr>';
    h += '</tbody></table></div>';
    setTimeout(function () {
      app.querySelectorAll('tr.row').forEach(function (tr) {
        function toggle() {
          var nx = tr.nextElementSibling;
          if (nx && nx.classList.contains('detail')) { nx.remove(); tr.classList.remove('open'); tr.setAttribute('aria-expanded', 'false'); return; }
          var d = document.createElement('tr'); d.className = 'detail';
          d.innerHTML = '<td colspan="' + cols + '">' + detailHtml(items[+tr.dataset.i]) + '</td>';
          tr.after(d); tr.classList.add('open'); tr.setAttribute('aria-expanded', 'true');
        }
        tr.addEventListener('click', function (e) { if (e.target.closest('a')) return; toggle(); });
        tr.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
      });
    }, 0);
    return h;
  }

  function vMisses(step) {
    var ms = misses();
    if (step) ms = ms.filter(function (x) { return (x.g.lost_at || 'never_heard') === step; });
    var h = testWarn() + '<h1>Misses' + (step ? ' lost at ' + esc(stepName(step)) : '') + '</h1><p class="lede">Needed sounds with no picture in time. Click a row to see every candidate span near the sound and the step that removed it.' + (step ? ' <a href="#misses">Show all</a>' : '') + '</p>';
    h += expandableTable(['Clip', 'Sound', 'Time (s)', 'Lost at', 'Exact reason'], ms, function (x) {
      return '<td><a href="' + clipHref(x.c) + '">' + esc(x.c.clip) + '</a></td><td>' + esc(x.g.label) + '</td><td class="n">' + t1(x.g.start) + '</td><td>' + stepTag(x.g.lost_at || 'never_heard') + '</td><td title="' + esc(x.g.why || '') + '">' + esc(shortWhy(x.g.why)) + '</td>';
    }, function (x) {
      var cs = (x.g.cands || []).map(function (id) { return candOf(x.c, id); }).filter(Boolean);
      return cs.length ? cs.map(function (cd) { return candCard(x.c, cd); }).join('') : '<p class="empty">No detector produced a candidate of this sound family near its start.</p>' + (x.g.heard ? '<div class="note">' + esc(x.g.heard) + '</div>' : '');
    }, 5);
    app.innerHTML = h;
  }

  function vWrongs(kind) {
    var ws = wrongs();
    if (kind) ws = ws.filter(function (x) { return x.p.verdict === kind; });
    var h = testWarn() + '<h1>Wrong pictures' + (kind ? ': ' + esc(kind) : '') + '</h1><p class="lede">Pictures that did not match a needed sound. Click a row to see every step the picture passed, with its value and bar.' + (kind ? ' <a href="#wrongs">Show all</a>' : '') + '</p>';
    h += expandableTable(['Type', 'Clip', 'Picture (start s)', 'What the gold has there'], ws, function (x) {
      return '<td>' + chip(x.p.verdict) + '</td><td><a href="' + clipHref(x.c) + '">' + esc(x.c.clip) + '</a></td><td>' + esc(x.p.label) + ' (' + t1(x.p.start) + ')</td><td>' + esc(x.p.gold_text || (x.p.gold ? (goldOf(x.c, x.p.gold) || {}).label : 'nothing')) + '</td>';
    }, function (x) { var cd = candOf(x.c, x.p.cand); return cd ? candCard(x.c, cd) : '<p class="empty">No trail recorded for this picture.</p>'; }, 4);
    app.innerHTML = h;
  }

  function vClips() {
    var cs = clips();
    var h = testWarn() + '<h1>Clips</h1><p class="lede">Open a clip for its video, a timeline of every sound and candidate, and each decision trail.</p>';
    h += '<div class="tw"><table><thead><tr><th>Clip</th><th>Set</th><th>Needed</th><th>Hits</th><th>Misses</th><th>Wrong</th></tr></thead><tbody>';
    cs.forEach(function (c) {
      var need = (c.gold || []).filter(function (g) { return g.needed; }), hit = need.filter(function (g) { return g.outcome === 'hit'; }).length;
      var wr = (c.pictures || []).filter(function (p) { return p.verdict !== 'hit'; }).length;
      h += '<tr><td><a href="' + clipHref(c) + '">' + esc(c.clip) + '</a></td><td>' + esc(c.split) + '</td><td class="n">' + need.length + '</td><td class="n">' + hit + '</td><td class="n">' + (need.length - hit) + '</td><td class="n">' + wr + '</td></tr>';
    });
    app.innerHTML = h + '</tbody></table></div>';
  }

  function timeline(c) {
    var dur = c.dur || Math.max.apply(null, [10].concat((c.gold || []).map(function (g) { return g.end; }), (c.cands || []).map(function (x) { return x.end; })));
    var W = 640, L = 64, R = 8, X = function (t) { return L + (W - L - R) * t / dur; };
    var lanes = [['Gold', c.gold || [], function (g) { return g.needed ? 'k-gold-need' : 'k-gold-vis'; }, 'g'],
                 ['Pictures', c.pictures || [], function (p) { return p.verdict === 'hit' ? 'k-hit' : p.verdict === 'phantom' ? 'k-phantom' : 'k-wrong'; }, 'p'],
                 ['Candidates', c.cands || [], function (x) { return 'k-' + (x.fate || 'dropped'); }, 'c']];
    var y = 6, h = "", rowH = 16;
    lanes.forEach(function (ln) {
      var items = ln[1].slice().sort(function (a, b) { return a.start - b.start; }), tracks = [];
      items.forEach(function (it) { var k = 0; while (tracks[k] != null && tracks[k] > it.start) k++; tracks[k] = it.end; it._tr = k; });
      var nT = Math.max(1, tracks.length), hh = nT * (rowH + 3);
      h += '<rect class="lane" x="' + L + '" y="' + y + '" width="' + (W - L - R) + '" height="' + hh + '"/><text x="0" y="' + (y + 11) + '">' + ln[0] + '</text>';
      items.forEach(function (it) {
        var x0 = X(it.start), w = Math.max(3, X(it.end) - x0), yy = y + it._tr * (rowH + 3);
        h += '<rect class="s ' + ln[2](it) + '" data-k="' + ln[3] + '" data-id="' + esc(it.id) + '" x="' + x0.toFixed(1) + '" y="' + yy + '" width="' + w.toFixed(1) + '" height="' + rowH + '" rx="2"><title>' + esc(it.label + ' ' + t1(it.start) + '–' + t1(it.end) + ' s' + (it.verdict ? ' · ' + it.verdict : '') + (it.fate ? ' · ' + it.fate + (it.at ? ' at ' + stepName(it.at) : '') : '') + (it.needed ? ' · needed' : '')) + '</title></rect>';
      });
      y += hh + 8;
    });
    for (var t = 0; t <= dur; t += (dur > 20 ? 5 : 2)) h += '<text x="' + X(t).toFixed(1) + '" y="' + (y + 8) + '" text-anchor="middle">' + t + 's</text>';
    return '<svg class="tl" viewBox="0 0 ' + W + ' ' + (y + 14) + '" role="img" aria-label="Timeline of gold sounds, pictures and candidates">' + h + '</svg>' +
      '<div class="legend"><span><i style="background:var(--accent)"></i>needed sound</span><span><i style="background:var(--muted)"></i>visible sound</span><span><i style="background:var(--good)"></i>hit / drawn</span><span><i style="background:var(--warn)"></i>wrong / merged</span><span><i style="background:var(--bad)"></i>phantom / dropped</span></div>';
  }

  function vClip(split, name) {
    var c = (D.clips || []).filter(function (x) { return x.split === split && x.clip === name; })[0];
    if (!c) { app.innerHTML = '<p class="empty">Clip not found.</p>'; return; }
    var h = (split === 'TEST' ? '<div class="banner"><b>TEST is a report.</b> Look only.</div>' : '') + '<h1>' + esc(c.clip) + ' <span class="vb">' + esc(c.split) + '</span></h1>';
    h += '<div class="clipgrid"><div><div class="player"><video id="vid" controls preload="metadata" playsinline></video><div class="vnote" id="vnote" hidden></div></div>' + timeline(c) + '</div><div id="side"><p class="note">Click a bar on the timeline to see its decision trail. The video jumps to that moment.</p></div></div>';
    h += '<h2>Needed sounds</h2>' + (c.gold || []).filter(function (g) { return g.needed; }).map(function (g) {
      return '<div class="cand"><h4>' + esc(g.label) + ' <span class="vb">' + t1(g.start) + '–' + t1(g.end) + ' s</span>' + chip(g.outcome) + (g.outcome === 'miss' ? ' lost at ' + stepTag(g.lost_at || 'never_heard') : '') + '</h4>' + (g.why ? '<div class="note">' + esc(g.why) + '</div>' : '') + '</div>';
    }).join('');
    app.innerHTML = h;
    var v = document.getElementById('vid'), vn = document.getElementById('vnote');
    v.onerror = function () { vn.hidden = false; vn.innerHTML = 'Video not available here.<br><code>' + esc(c.video || '') + '</code>'; };
    if (c.video) v.src = c.video; else v.onerror();
    var side = document.getElementById('side');
    app.querySelectorAll('rect.s').forEach(function (r) {
      r.addEventListener('click', function () {
        var k = r.dataset.k, id = r.dataset.id, html = '', t0 = null;
        if (k === 'c') { var cd = candOf(c, id); html = candCard(c, cd); t0 = cd.start; }
        else if (k === 'p') { var p = (c.pictures || []).filter(function (x) { return x.id === id; })[0]; t0 = p.start; html = '<p>' + chip(p.verdict) + ' picture of <b>' + esc(p.label) + '</b> at ' + t1(p.start) + ' s</p>' + (p.cand ? candCard(c, candOf(c, p.cand)) : ''); }
        else { var g = goldOf(c, id); t0 = g.start; html = '<p><b>' + esc(g.label) + '</b> ' + t1(g.start) + '–' + t1(g.end) + ' s · ' + (g.needed ? 'needed' : 'source visible') + (g.outcome ? ' · ' + chip(g.outcome) : '') + '</p>' + (g.cands || []).map(function (cid) { return candCard(c, candOf(c, cid)); }).join(''); }
        side.innerHTML = html;
        if (t0 != null && v.readyState >= 1) { try { v.currentTime = Math.max(0, t0 - 0.5); } catch (e) {} }
      });
    });
  }

  function vSteps(sel) {
    var h = '<h1>Pipeline steps</h1><p class="lede">Every decision point in the shipped version, in order. Counts are over the selected set: candidates of a needed sound removed here, and other candidates removed here.</p>';
    var cnt = {};
    clips().forEach(function (c) {
      var needCands = {};
      (c.gold || []).forEach(function (g) { if (g.needed) (g.cands || []).forEach(function (id) { needCands[id] = 1; }); });
      (c.cands || []).forEach(function (cd) { if (cd.fate === 'dropped' && cd.at) { cnt[cd.at] = cnt[cd.at] || [0, 0]; cnt[cd.at][needCands[cd.id] ? 0 : 1]++; } });
    });
    h += '<div class="tw"><table><thead><tr><th>#</th><th>Step</th><th>What it does</th><th>Bar / rule</th><th>Model and question</th><th>Needed lost</th><th>Others removed</th></tr></thead><tbody>';
    (D.steps || []).forEach(function (s, i) {
      var cc = cnt[s.id] || [0, 0];
      h += '<tr id="st-' + esc(s.id) + '"' + (sel === s.id ? ' class="open"' : '') + '><td class="n">' + (i + 1) + '</td><td><b>' + esc(s.name) + '</b><div class="vb">stage ' + esc(s.stage) + '</div></td><td>' + esc(s.plain) + '</td><td>' + esc(s.bar || '—') + '</td><td>' + (s.model ? '<b>' + esc(s.model) + '</b>' : '—') + (s.question ? '<div class="ask"><div class="q">' + esc(s.question) + '</div></div>' : '') + '</td><td class="n">' + (cc[0] ? '<a href="#misses/' + encodeURIComponent(s.id) + '">' + cc[0] + '</a>' : '0') + '</td><td class="n">' + cc[1] + '</td></tr>';
    });
    app.innerHTML = h + '</tbody></table></div>';
    if (sel) { var el = document.getElementById('st-' + sel); if (el) el.scrollIntoView({ block: 'center' }); }
  }

  function vBuild() {
    var m = D.meta || {}, h = '<h1>The current build: ' + esc(m.version || '?') + '</h1><p class="lede">What the system does with one video, in five parts. Each part runs per video, with fixed rules, so it can run live.</p>';
    h += '<div class="tw"><table><thead><tr><th>Part</th><th>Goal</th><th>How</th><th>Models</th></tr></thead><tbody>';
    (D.build || []).forEach(function (b) {
      h += '<tr><td><b>' + esc(b.part) + '</b><div class="vb">stage ' + esc((b.stages || []).join(', ')) + '</div></td><td>' + esc(b.what) + '</td><td>' + esc(b.how) + '</td><td>' + esc(b.models) + '</td></tr>';
    });
    h += '</tbody></table></div><p class="note">Every single decision point, with its bar and exact question, is in <a href="#steps">Steps</a>.</p>';
    h += '<h2>How we got here</h2><p class="lede">Each row adds one change. Numbers are hits / wrong pictures / cost (lower cost is better). Changes were chosen on DEV only; TEST was read afterwards as a report.</p>';
    h += '<div class="tw"><table><thead><tr><th>Change</th><th>DEV (71 clips, 58 needed)</th><th>TEST (88 clips, 65 needed)</th><th>Note</th></tr></thead><tbody>';
    var bs = D.build_steps || [];
    bs.forEach(function (r, i) { var last = i === bs.length - 1; h += '<tr' + (last ? ' class="open"' : '') + '><td>' + (last ? '<b>' + esc(r.change) + '</b>' : esc(r.change)) + '</td><td class="n">' + esc(r.dev) + '</td><td class="n">' + esc(r.test) + '</td><td>' + esc(r.note) + '</td></tr>'; });
    app.innerHTML = h + '</tbody></table></div><p class="note">Showing nothing at all would cost 3.268 on DEV and 2.955 on TEST.</p>';
  }

  function vTried() {
    var h = '<h1>What we tried that did not ship</h1><p class="lede">Every idea was set in advance with a pass rule and judged on DEV or the 415 held-out clips, never on TEST.</p>';
    if (D.lesson) h += '<div class="banner"><b>Main lesson.</b> ' + esc(D.lesson) + '</div>';
    var parts = [];
    (D.tried || []).forEach(function (t) { if (parts.indexOf(t.part) < 0) parts.push(t.part); });
    parts.forEach(function (pt) {
      var rows = (D.tried || []).filter(function (t) { return t.part === pt; });
      h += '<h2>' + esc(pt) + ' <span class="vb">' + rows.length + ' ideas</span></h2><div class="tw"><table><thead><tr><th>Idea</th><th>Why it failed</th><th>Shipped?</th></tr></thead><tbody>';
      rows.forEach(function (t) { h += '<tr><td>' + esc(t.idea) + '</td><td>' + esc(t.why) + '</td><td>' + chip(t.shipped === 'no' ? 'no' : 'visible', t.shipped) + '</td></tr>'; });
      h += '</tbody></table></div>';
    });
    app.innerHTML = h;
  }

  // ---------------------------------------------------------------- router, theme
  function route() {
    var hs = decodeURIComponent((location.hash || '#overview').slice(1)).split('/'), v = hs[0] || 'overview';
    document.querySelectorAll('#nav a').forEach(function (a) { a.classList.toggle('on', a.dataset.v === v || (v === 'clip' && a.dataset.v === 'clips')); });
    if (v === 'misses') vMisses(hs[1]); else if (v === 'wrongs') vWrongs(hs[1]); else if (v === 'clips') vClips();
    else if (v === 'clip') vClip(hs[1], hs.slice(2).join('/')); else if (v === 'steps') vSteps(hs[1]); else if (v === 'build') vBuild(); else if (v === 'tried') vTried(); else vOverview();
    window.scrollTo(0, 0);
  }
  var m = D.meta || {};
  document.getElementById('ver').textContent = 'version ' + (m.version || '?') + ' · ' + (m.arm || '') + (m.sample ? ' · SAMPLE DATA' : '') + (m.built ? ' · built ' + m.built : '');
  var th = store('theme'); if (th) document.documentElement.setAttribute('data-theme', th);
  document.getElementById('theme').addEventListener('click', function () {
    var cur = document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    var nx = cur === 'dark' ? 'light' : 'dark'; document.documentElement.setAttribute('data-theme', nx); store('theme', nx);
  });
  window.addEventListener('hashchange', route);
  route();
})();

/* Inspector page logic. Data: window.INSPECTOR from data.js (made by build.py from data.json). No network. */
(function () {
  'use strict';

  var D = window.INSPECTOR;
  var app = document.getElementById('app');
  if (!D) {
    app.innerHTML = '<div class="warn"><b>data.js is missing.</b> Run <code>python docs/inspector/build.py</code> ' +
      'from the project folder, then reload this page.</div>';
    return;
  }

  // ------------------------------------------------------------------ constants and small helpers
  var CAT = { unseen: 'off-screen', mixed: 'mixed', seen: 'on-screen', no_ambient: 'nothing to draw' };
  var CAT_ORDER = ['unseen', 'mixed', 'seen', 'no_ambient'];
  var SPLITS = ['DEV', 'TEST', 'sliceB'];
  var WIN = D.window || [-0.5, 1.0];
  var PCLASS = {
    'hit': { s: 'hit', k: 'hit' },
    'hit (+ a visible sound of the same family)': { s: 'hit (+ a visible sound of the same family)', k: 'hit' },
    'wrong: source visible or obvious': { s: 'source visible', k: 'vis' },
    'wrong: a different sound': { s: 'different sound', k: 'diff' },
    'wrong: no such sound': { s: 'no such sound', k: 'none' },
    'duplicate': { s: 'duplicate', k: 'dup' },
    "don't care": { s: "don't care (importance 1)", k: 'dup' }
  };
  var WRONG_KEYS = [['vis', 'source visible'], ['diff', 'different sound'], ['none', 'no such sound']];
  var REASONS = ['removed by the gate', 'timing', 'detected, not drawn', 'never detected', 'taken by another sound'];
  // display names: the data keeps the key "blind"; the page calls it "pipeline without gate"
  var SYSNAME = { ours: 'ours', blind: 'pipeline without gate', silence: 'silence' };
  function sysName(s) { return SYSNAME[s] || s; }
  var NOGATE_DEF = 'same detector stack, label filter, subject text and pictures as ours; only the visibility gate is removed — it draws every detected sound.';
  var REASON_TEXT = {
    'removed by the gate': 'The detector heard it and the pipeline without gate drew it in time, but the VLM said the source is visible, so ours stayed quiet.',
    'timing': 'A picture of this kind of sound was shown, but it did not start inside the window (0.5 s before to 1.0 s after the sound starts).',
    'detected, not drawn': 'The detector heard this kind of sound near the start, but no picture was made (a rule dropped it, e.g. below the display bar).',
    'never detected': 'The sound detector did not hear this kind of sound near the moment it starts. No picture is possible.',
    'taken by another sound': 'A picture was in time, but the scorer gave it to another sound of the same kind (one picture per sound).'
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function ft(x) { return x == null ? '—' : (Math.round(x * 10) / 10).toFixed(1); }
  function f2(x) { return x == null || isNaN(x) ? '—' : x.toFixed(2); }
  function pct(x) { return x == null || isNaN(x) ? '—' : x.toFixed(2); }
  function sgn(x) { return (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(1); }
  function lc(s) { return String(s || '').toLowerCase(); }
  function pclass(c) { return PCLASS[c] || { s: c, k: 'dup' }; }
  function isWrong(c) { return String(c).indexOf('wrong') === 0; }
  function badge() { return '<span class="testbadge">TEST — look only, do not tune</span>'; }
  function splitTag(s) { return '<span class="split-tag ' + esc(s) + '">' + esc(s) + '</span>'; }
  function soundKind(s) { return s.needed ? (s.importance >= 2 ? 'needed' : 'imp1') : (s.visible ? 'visible' : 'obvious'); }
  var KIND_TEXT = { needed: 'needed', imp1: 'needed, importance 1', visible: 'visible', obvious: 'obvious' };
  function kindTag(s) { var k = soundKind(s); return '<span class="tag ' + k + '">' + KIND_TEXT[k] + '</span>'; }
  function classTag(c) { var p = pclass(c); return '<span class="tag ' + p.k + '">' + esc(p.s) + '</span>'; }
  function clipHref(c) { return '#/clip/' + encodeURIComponent(c.split) + '/' + encodeURIComponent(c.clip); }
  function clipLink(c) { return '<a href="' + clipHref(c) + '">' + esc(c.clip) + '</a>'; }
  function media(split, clip, kind) {
    var base = 'media/' + encodeURIComponent(split) + '/' + encodeURIComponent(clip);
    return base + (kind === 'debug' ? '_debug.mp4' : kind === 'blind' ? '_blind.mp4' : '.mp4');
  }
  function picSrc(c, idx) { return 'media/' + encodeURIComponent(c.split) + '/' + encodeURIComponent(c.clip) + '_pics/' + idx + '.png'; }
  function playBtn(c, t, kind) {
    return '<button type="button" class="play" data-split="' + esc(c.split) + '" data-clip="' + esc(c.clip) + '" data-t="' + t +
      '" data-kind="' + (kind || 'ours') + '" title="Play from this moment">&#9654; ' + ft(t) + ' s</button>';
  }
  function votes(g) {
    function v(x) { return x === true ? 'visible' : x === false ? 'not' : '—'; }
    return 'name: ' + v(g.name) + ' · a/b: ' + v(g.ab) + ' · desc: ' + v(g.desc);
  }
  function verdict(g) {
    return g.seen ? '<span class="tag t-seen">VLM: visible</span>' : '<span class="tag t-unseen">VLM: not visible</span>';
  }
  function gateLine(g) {
    return '<span class="gl">' + verdict(g) + ' ' + esc(g.label) + ' ' + ft(g.stretch[0]) + '–' + ft(g.stretch[1]) +
      ' s · saw “' + esc(g.named) + '” <span class="muted small">(' + votes(g) + ')</span></span>';
  }
  function inWin(t, onset) { return onset + WIN[0] <= t && t <= onset + WIN[1]; }
  function soundsAt(c, t) {
    return c.sounds.map(function (s, i) { return { s: s, i: i }; })
      .filter(function (o) { return inWin(t, o.s.start) || (o.s.start <= t && t <= o.s.end); });
  }
  function ours(c) { return c.systems.ours || { sounds: [], pictures: [] }; }
  function blind(c) { return c.systems.blind || { sounds: [], pictures: [] }; }
  function gateOf(c) { return ours(c).gate || []; }
  function duration(c) {
    var m = ours(c).media;
    if (m && m.duration) return m.duration;
    var e = 0;
    c.sounds.forEach(function (s) { e = Math.max(e, s.end); });
    [ours(c), blind(c)].forEach(function (x) { (x.pictures || []).forEach(function (p) { e = Math.max(e, p.end); }); });
    return Math.max(e + 1, 5);
  }
  // ---- what the detectors heard around a sound's start ([onset - 0.5, onset + 1.0] s; from add_heard.py)
  var DETS = ['BEATs', 'FlexSED', 'PANNs'];
  var HEARD_NOTE = 'BEATs and PANNs report every AudioSet class they hear; FlexSED is asked about 215 drawable sounds by name. ' +
    'When Music or Speech is on top and the sound\'s own score is near 0, the sound was masked.';
  function sc2(x) { return x == null ? '—' : x.toFixed(2).replace(/^0\./, '.'); }
  function shortLab(l) { return String(l).split(',')[0]; }
  function heardCell(c, i) {
    var s = c.sounds[i], H = s.heard;
    if (!H) return '<span class="muted small">not in data.json (run add_heard.py)</span>';
    var fam = (c.derived.family || [])[i] || s.label;
    return '<div class="heard">' + DETS.map(function (k) {
      var x = H[k];
      if (!x) return '<div class="hl"><b>' + k + ':</b> <span class="muted">—</span></div>';
      var top = (x.top || []).map(function (t) {
        return '<span title="' + esc(t[0]) + '">' + esc(shortLab(t[0])) + ' ' + sc2(t[1]) + '</span>';
      }).join(', ');
      var own;
      if (x.own == null) own = '<span class="own low">own ' + esc(fam) + ': not in its list</span>';
      else {
        var cls = x.bar == null ? 'nobar' : x.own >= x.bar ? 'ok' : x.own >= x.bar / 2 ? 'half' : 'low';
        own = '<span class="own ' + cls + '">own ' + esc(fam) + ' ' + sc2(x.own) + ' / ' + (x.bar == null ? 'no bar (veto only)' : 'bar ' + sc2(x.bar)) + '</span>';
      }
      return '<div class="hl"><b>' + k + ':</b> ' + top + ' · ' + own + '</div>';
    }).join('') + '</div>';
  }
  function heardNote() {
    return '<p class="small muted heard-note">What the detectors heard = the 5 highest scores of each detector from 0.5 s before to 1.0 s after the sound starts. ' +
      '<b>own</b> = the highest score of the sound\'s own family; colour: <span class="own ok">at or above the bar</span> ' +
      '<span class="own half">at least half the bar</span> <span class="own low">below half</span>. ' + esc(HEARD_NOTE) + '</p>';
  }

  function outcomeText(c, sys, i) {
    var x = (c.systems[sys] && c.systems[sys].sounds || [])[i];
    if (!x) return '—';
    if (x.outcome === 'hit') return '<span class="tag hit">hit</span> <span class="muted small">' + sgn(x.late) + ' s</span>';
    if (x.outcome === 'miss') {
      var r = (c.derived.miss[sys] || {})[String(i)];
      return '<span class="tag miss">miss</span>' + (r ? ' <span class="small">' + esc(r.reason) + '</span>' : '');
    }
    return '<span class="muted small">' + esc(x.outcome) + '</span>';
  }

  var CLIPS = D.clips;
  var BYKEY = {};
  CLIPS.forEach(function (c) { BYKEY[c.split + '/' + c.clip] = c; });

  // per-clip counts for the list
  CLIPS.forEach(function (c) {
    function cnt(x) {
      var h = 0, m = 0, w = 0;
      (x.sounds || []).forEach(function (o) { if (o && o.outcome === 'hit') h++; if (o && o.outcome === 'miss') m++; });
      (x.pictures || []).forEach(function (p) { if (isWrong(p.class)) w++; });
      return { h: h, m: m, w: w, n: (x.pictures || []).length };
    }
    c._o = cnt(ours(c)); c._b = cnt(blind(c));
    c._needed = c.sounds.filter(function (s) { return s.needed && s.importance >= 2; }).length;
  });

  // ------------------------------------------------------------------ numbers
  function funnel(splits) {
    var cs = CLIPS.filter(function (c) { return splits.indexOf(c.split) >= 0; });
    var o = { clips: cs.length, sounds: 0, needed: 0, imp1: 0, visible: 0, obvious: 0, cats: {} };
    CAT_ORDER.forEach(function (k) { o.cats[k] = 0; });
    cs.forEach(function (c) {
      o.cats[c.category] = (o.cats[c.category] || 0) + 1;
      c.sounds.forEach(function (s) {
        o.sounds++;
        var k = soundKind(s);
        if (k === 'needed') o.needed++; else if (k === 'imp1') o.imp1++; else if (k === 'visible') o.visible++; else o.obvious++;
      });
    });
    return o;
  }
  // cross-check the page's counts against the sets block of data.json
  var CHECKS = [];
  SPLITS.forEach(function (sp) {
    var st = D.sets[sp]; if (!st) return;
    var f = funnel([sp]);
    var pairs = [['clips', f.clips, st.clips], ['sounds', f.sounds, st.sounds], ['needed 2-3', f.needed, st.needed_2_3],
      ['visible', f.visible, st.visible], ['obvious, not visible', f.obvious, st.obvious_not_visible]];
    CAT_ORDER.forEach(function (k) { pairs.push([CAT[k], f.cats[k], st.categories[k]]); });
    pairs.forEach(function (p) { if (p[1] !== p[2]) CHECKS.push(sp + ' ' + p[0] + ': page ' + p[1] + ', data.json ' + p[2]); });
  });

  function pooled(splits) {
    var out = {};
    ['ours', 'blind', 'silence'].forEach(function (sys) {
      var a = { hits: 0, misses: 0, visible: 0, cross: 0, phantom: 0, dup: 0, clips: 0, costsum: 0 };
      splits.forEach(function (sp) {
        var s = D.sets[sp] && D.sets[sp].stats[sys]; if (!s) return;
        ['hits', 'misses', 'visible', 'cross', 'phantom', 'dup'].forEach(function (k) { a[k] += s[k] || 0; });
        a.clips += s.clips; a.costsum += s.viewer_cost * s.clips;
      });
      a.wrong = a.visible + a.cross + a.phantom;
      a.shown = a.hits + a.wrong;
      a.P = a.shown ? a.hits / a.shown : null;
      a.R = (a.hits + a.misses) ? a.hits / (a.hits + a.misses) : null;
      a.F1 = (a.P && a.R) ? 2 * a.P * a.R / (a.P + a.R) : 0;
      a.cost = a.clips ? a.costsum / a.clips : null;
      out[sys] = a;
    });
    return out;
  }

  // ------------------------------------------------------------------ tables (sortable)
  var SORT = {};
  function mkTable(host, key, cols, rows, opts) {
    opts = opts || {};
    var st = SORT[key] || (SORT[key] = { k: null, dir: 1 });
    var rs = rows.slice();
    if (st.k != null) {
      var col = cols.filter(function (c) { return c.k === st.k; })[0];
      if (col && col.v) {
        rs.sort(function (a, b) {
          var x = col.v(a), y = col.v(b);
          if (x == null) x = typeof y === 'number' ? -Infinity : '';
          if (y == null) y = typeof x === 'number' ? -Infinity : '';
          return (x < y ? -1 : x > y ? 1 : 0) * st.dir;
        });
      }
    }
    var h = '<div class="tw"><table class="' + (opts.cls || '') + '"><thead><tr>' + cols.map(function (c) {
      var arrow = st.k === c.k ? (st.dir > 0 ? ' ▲' : ' ▼') : '';
      return '<th data-k="' + c.k + '" class="' + (c.num ? 'num ' : '') + (c.v ? 'sortable' : '') + '"' +
        (c.v ? ' title="Sort"' : '') + '>' + c.t + arrow + '</th>';
    }).join('') + '</tr></thead><tbody>';
    if (!rs.length) h += '<tr><td colspan="' + cols.length + '" class="empty">Nothing matches these filters.</td></tr>';
    rs.forEach(function (r) {
      h += '<tr' + (opts.href ? ' data-href="' + esc(opts.href(r)) + '"' : '') + '>' + cols.map(function (c) {
        return '<td class="' + (c.num ? 'num' : '') + (c.cls ? ' ' + c.cls : '') + '">' + c.h(r) + '</td>';
      }).join('') + '</tr>';
    });
    h += '</tbody></table></div>';
    if (!opts.nocount) h += '<p class="count">' + rs.length + ' row' + (rs.length === 1 ? '' : 's') + '</p>';
    host.innerHTML = h;
    Array.prototype.forEach.call(host.querySelectorAll('th.sortable'), function (th) {
      th.addEventListener('click', function () {
        var k = th.getAttribute('data-k');
        if (st.k === k) st.dir = -st.dir; else { st.k = k; st.dir = 1; }
        mkTable(host, key, cols, rows, opts);
      });
    });
  }

  // ------------------------------------------------------------------ tooltip for SVG charts
  var tip = document.getElementById('tip');
  function bindTips(root) {
    Array.prototype.forEach.call(root.querySelectorAll('[data-tip]'), function (n) {
      n.addEventListener('mousemove', function (e) {
        tip.innerHTML = n.getAttribute('data-tip');
        tip.style.display = 'block';
        var x = e.clientX + 14, y = e.clientY + 14;
        var w = tip.offsetWidth, hh = tip.offsetHeight;
        if (x + w > window.innerWidth - 8) x = e.clientX - w - 14;
        if (y + hh > window.innerHeight - 8) y = e.clientY - hh - 14;
        tip.style.left = x + 'px'; tip.style.top = y + 'px';
      });
      n.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    });
  }

  // redraw hooks for width-dependent SVGs
  var REDRAW = [];
  var rt = null;
  window.addEventListener('resize', function () {
    clearTimeout(rt);
    rt = setTimeout(function () { REDRAW.forEach(function (f) { f(); }); }, 120);
  });

  // ------------------------------------------------------------------ OVERVIEW
  function funnelCard(splits, title) {
    var f = funnel(splits);
    var hasTest = splits.indexOf('TEST') >= 0;
    var parts = [['needed', f.needed, 'var(--c-needed)', 'needed: off-screen, must be shown (importance 2–3; these are scored)'],
      ['visible', f.visible, 'var(--c-visible)', 'visible: the source is on screen'],
      ['obvious', f.obvious, 'var(--c-obvious)', 'obvious: not on screen, but you know it happens'],
      ['imp1', f.imp1, 'var(--c-imp1)', 'needed, but importance 1 (steady background; not scored)']];
    var h = '<div class="card"><h3>' + esc(title) + ' ' + (hasTest ? badge() : '') + '</h3>';
    h += '<div class="funnel-top"><span class="big">' + f.clips + '</span> clips <span class="arrow">&rarr;</span> <span class="big">' +
      f.sounds + '</span> gold sounds</div>';
    h += '<div class="stack" role="img" aria-label="sounds by type">' + parts.map(function (p) {
      return p[1] ? '<span style="flex:' + p[1] + ';background:' + p[2] + (p[0] === 'imp1' ? ';opacity:.55' : '') + '" title="' + esc(p[3]) + ': ' + p[1] + '"></span>' : '';
    }).join('') + '</div>';
    h += '<ul class="legend-list">' + parts.map(function (p) {
      return '<li><span class="sw' + (p[0] === 'imp1' ? ' dash' : '') + '" style="background:' + p[2] + '"></span><b>' + p[1] + '</b><span>' + esc(p[3]) + '</span></li>';
    }).join('') + '</ul>';
    h += '<div class="cats">' + CAT_ORDER.map(function (k) {
      return '<span class="chip">' + CAT[k] + ' <b>' + f.cats[k] + '</b></span>';
    }).join('') + '</div>';
    return h + '</div>';
  }

  function resultsTable(host, splits, key) {
    var P = pooled(splits);
    var rows = ['ours', 'blind', 'silence'].map(function (s) { var r = P[s]; r.sys = s; return r; });
    var cols = [
      { k: 'sys', t: 'system', h: function (r) { return sysName(r.sys); } },
      { k: 'h', t: 'hits', num: 1, h: function (r) { return r.hits; } },
      { k: 'm', t: 'misses', num: 1, h: function (r) { return r.misses; } },
      { k: 'v', t: 'wrong: source visible', num: 1, h: function (r) { return r.visible; } },
      { k: 'c', t: 'wrong: different sound', num: 1, h: function (r) { return r.cross; } },
      { k: 'p', t: 'wrong: no such sound', num: 1, h: function (r) { return r.phantom; } },
      { k: 'w', t: 'wrong total', num: 1, h: function (r) { return '<b>' + r.wrong + '</b>'; } },
      { k: 'P', t: 'precision', num: 1, h: function (r) { return r.shown ? pct(r.P) : '—'; } },
      { k: 'R', t: 'recall', num: 1, h: function (r) { return pct(r.R); } },
      { k: 'F', t: 'F1', num: 1, h: function (r) { return pct(r.F1); } },
      { k: 'cost', t: 'viewer cost / clip', num: 1, h: function (r) { return '<b>' + f2(r.cost) + '</b>'; } }
    ];
    mkTable(host, key, cols, rows, { cls: 'results', nocount: 1 });
    Array.prototype.forEach.call(host.querySelectorAll('tbody tr'), function (tr, i) { if (i === 0) tr.className = 'ours'; });
    var o = P.ours, b = P.blind, s = P.silence;
    var cheapest = [['ours', o.cost], ['blind', b.cost], ['silence', s.cost]].sort(function (x, y) { return x[1] - y[1]; })[0][0];
    var p = document.createElement('p');
    p.className = 'sentence';
    p.innerHTML = 'Ours showed a correct picture for <b>' + o.hits + ' of ' + (o.hits + o.misses) + '</b> needed sounds (pipeline without gate: ' + b.hits +
      ') and showed <b>' + o.wrong + '</b> wrong pictures (pipeline without gate: ' + b.wrong + '). With a miss costing 4 and a wrong picture 2, ours costs <b>' +
      f2(o.cost) + '</b> per clip, the pipeline without gate ' + f2(b.cost) + ', silence ' + f2(s.cost) + ' — lower is better, so the cheapest here is <b>' + sysName(cheapest) + '</b>.';
    host.appendChild(p);
  }

  function wrongChart(host) {
    function draw() {
      var groups = SPLITS.filter(function (sp) { return D.sets[sp]; });
      var W = Math.max(300, host.clientWidth || 600);
      var narrow = W < 520;
      var L = narrow ? 92 : 150, R = 44, rowH = 22, gap = 6, headH = 22, groupGap = 14;
      var max = 0;
      groups.forEach(function (sp) { ['ours', 'blind'].forEach(function (s) { var x = D.sets[sp].stats[s]; max = Math.max(max, x.visible + x.cross + x.phantom); }); });
      var H = groups.length * (headH + 2 * rowH + gap + groupGap) + 4;
      var sc = (W - L - R) / max;
      var y = 0, h = '';
      groups.forEach(function (sp) {
        var so = D.sets[sp].stats.ours, sb = D.sets[sp].stats.blind;
        var to = so.visible + so.cross + so.phantom, tb = sb.visible + sb.cross + sb.phantom;
        h += '<text x="0" y="' + (y + 15) + '" class="bold">' + esc(sp) + (sp === 'sliceB' && !narrow ? ' (AudioSet slice)' : '') + '</text>';
        h += '<text x="' + (W - 2) + '" y="' + (y + 15) + '" text-anchor="end" class="ink2">the gate removed ' + (tb - to) + ' of ' + tb + ' wrong pictures</text>';
        y += headH;
        [['blind', sb], ['ours', so]].forEach(function (row) {
          var s = row[1], x = L;
          if (narrow && row[0] === 'blind') {
            h += '<text x="' + (L - 8) + '" y="' + (y + 7) + '" text-anchor="end" class="ink2" style="font-size:10px">pipeline<tspan x="' + (L - 8) + '" dy="11">without gate</tspan></text>';
          } else {
            h += '<text x="' + (L - 8) + '" y="' + (y + rowH / 2 + 4) + '" text-anchor="end" class="ink2">' + sysName(row[0]) + '</text>';
          }
          [['vis', s.visible, 'source visible'], ['diff', s.cross, 'different sound'], ['none', s.phantom, 'no such sound']].forEach(function (seg, i, arr) {
            var w = seg[1] * sc;
            if (seg[1] > 0) {
              var ww = Math.max(1, w - 2);
              var tipTxt = '<b>' + esc(sp) + ' · ' + sysName(row[0]) + '</b><br>' + seg[2] + ': <b>' + seg[1] + '</b> of ' + (s.visible + s.cross + s.phantom) + ' wrong pictures';
              h += '<rect class="f-' + seg[0] + '" x="' + x + '" y="' + y + '" width="' + ww + '" height="' + (rowH - 4) + '" rx="3" data-tip="' + esc(tipTxt) + '"></rect>';
              if (ww > 22) h += '<text x="' + (x + ww / 2) + '" y="' + (y + rowH / 2 + 2) + '" text-anchor="middle" style="fill:#fff;font-weight:600;pointer-events:none">' + seg[1] + '</text>';
            }
            x += w;
          });
          h += '<text x="' + (x + 6) + '" y="' + (y + rowH / 2 + 2) + '" class="bold">' + (s.visible + s.cross + s.phantom) + '</text>';
          y += rowH + (row[0] === 'blind' ? gap : 0);
        });
        y += groupGap;
      });
      var txt = groups.map(function (sp) {
        var so = D.sets[sp].stats.ours, sb = D.sets[sp].stats.blind;
        var to = so.visible + so.cross + so.phantom;
        return '<li><b>' + esc(sp) + ':</b> compared with the pipeline without gate, ours has ' + (sb.visible - so.visible) + ' fewer “source visible”, ' + (sb.cross - so.cross) +
          ' fewer “different sound” and ' + (sb.phantom - so.phantom) + ' fewer “no such sound” pictures. Ours still shows ' + to + ': ' +
          so.visible + ' source visible, ' + so.cross + ' different sound, ' + so.phantom + ' no such sound.</li>';
      }).join('');
      host.innerHTML = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Wrong pictures by type, ours versus the pipeline without gate">' + h + '</svg>' +
        '<ul class="small" style="margin:6px 0 0;padding-left:18px">' + txt + '</ul>';
      bindTips(host);
    }
    draw();
    REDRAW.push(function () { if (document.body.contains(host)) draw(); });
  }

  function renderOverview() {
    var h = '<h2>What is in the benchmark</h2>';
    h += '<p class="lede">Each clip was watched by one annotator (Adam). For every non-speech, non-music sound he marked when it starts and ends, ' +
      'whether its source is <b>visible</b> on screen, whether it is <b>obvious</b> without sound, and how <b>important</b> it is (1–3). ' +
      'A sound is <b>needed</b> when it is neither visible nor obvious. Only needed sounds of importance 2–3 are scored.</p>';
    if (D.build_note) h += '<div class="warn">' + esc(D.build_note) + '</div>';
    if (CHECKS.length) h += '<div class="warn"><b>Data check:</b> ' + CHECKS.map(esc).join('; ') + '</div>';
    else h += '<p class="small muted">Data check: the counts on this page match the sets block of data.json.</p>';
    h += '<div class="grid funnels">' + funnelCard(['DEV'], 'DEV') + funnelCard(['TEST'], 'TEST') +
      funnelCard(['DEV', 'TEST'], 'All 109 (DEV + TEST)') + funnelCard(['sliceB'], 'Slice B (AudioSet, reported apart)') + '</div>' +
      '<p class="small muted">Clip types: <b>off-screen</b> = every important sound is needed; <b>mixed</b> = some needed, some seen; ' +
      '<b>on-screen</b> = sounds, but none needed; <b>nothing to draw</b> = no non-speech, non-music sound.</p>';
    h += '<h2>Results</h2><p class="lede">A <b>hit</b> is a picture of the right kind of sound that starts between 0.5 s before and 1.0 s after the sound starts. ' +
      'A <b>wrong picture</b> is any other picture: its source was visible, it showed a different sound, or there was no sound at all. ' +
      'Precision = hits / all pictures; recall = hits / needed sounds. Viewer cost = (4 × misses + 2 × wrong pictures) / clips; lower is better.</p>';
    h += '<div class="callout"><b>Three kinds of miss to look at first:</b>' +
      '<a class="bigbtn" href="#/mistakes/late">Late or early pictures &rarr;<small>' + countMiss('timing') + ' needed sounds (ours + pipeline without gate, DEV + TEST) had a picture of the right kind at the wrong moment</small></a>' +
      '<a class="bigbtn" href="#/mistakes/gated">Removed by the gate &rarr;<small>' + countMiss('removed by the gate') + ' needed sounds (DEV + TEST): the pipeline without gate drew them in time, ours stayed quiet</small></a>' +
      '<a class="bigbtn" href="#/mistakes/never">Never detected &rarr;<small>' + countMiss('never detected', ['ours']) + ' needed sounds (ours, DEV + TEST) that the detector did not hear near their start. See what each detector heard instead</small></a></div>';
    h += '<p class="small"><b>Pipeline without gate</b> = ' + esc(NOGATE_DEF) + '</p>';
    h += '<div id="res-DEV"></div><div id="res-TEST"></div><div id="res-base"></div><div id="res-POOL"></div><div id="res-sliceB"></div>';
    h += '<h2>Sensitivity: what if a late picture counted?</h2><p class="lede">The scoring rule counts a picture as correct only if it starts within 0.5 s before to 1 s after the sound\'s start. ' +
      'Here the rule is relaxed: a wrong picture of the same kind that starts <b>any time while the sound plays</b> (from 0.5 s before its start to its end) turns a miss into a hit ' +
      '(one picture per sound). This is only a check, not the score.</p><div id="sens"></div>';
    h += '<h2>What wrong pictures are left?</h2><p class="lede">The pipeline without gate draws every sound the detector hears. Ours draws the same pictures but asks the VLM first and ' +
      'skips a sound whose source is on screen. The bars show the wrong pictures of each system by type. ' +
      'A “different sound” picture means the detector named the wrong sound; the gate can only remove it by chance (when the VLM sees something that could make the wrong sound).</p>';
    h += '<div class="card"><div class="legend-row"><span><span class="sw" style="background:var(--c-vis)"></span>source visible</span>' +
      '<span><span class="sw" style="background:var(--c-diff)"></span>different sound</span>' +
      '<span><span class="sw" style="background:var(--c-none)"></span>no such sound</span></div><div class="chartbox" id="wchart"></div>' +
      '<p class="small muted">The exact numbers are in the tables above. The same data per picture: Mistakes &rarr; Wrong pictures.</p></div>';
    app.innerHTML = h;
    [['DEV', ['DEV'], 'DEV (49 clips)'], ['TEST', ['TEST'], 'TEST (60 clips)'], ['POOL', ['DEV', 'TEST'], 'DEV + TEST pooled (109 clips)'], ['sliceB', ['sliceB'], 'Slice B (30 AudioSet clips, reported apart)']].forEach(function (g) {
      var box = document.getElementById('res-' + g[0]);
      box.innerHTML = '<h3 style="margin-top:16px">' + esc(g[2]) + ' ' + (g[1].indexOf('TEST') >= 0 ? badge() : '') + '</h3><div></div>';
      resultsTable(box.lastChild, g[1], 'res-' + g[0]);
      if (g[0] === 'POOL') {
        var n = document.createElement('p'); n.className = 'small muted';
        n.textContent = 'Pooled: counts are added; precision, recall and F1 are computed from the added counts; the cost is the clip-weighted mean of the two splits.';
        box.appendChild(n);
      }
    });
    baselineBlock(document.getElementById('res-base'));
    wrongChart(document.getElementById('wchart'));
    sensTable(document.getElementById('sens'));
  }

  // ---- existing approaches: one detector alone + a picture for every detection
  // (build.py copies benchmark/gold/baseline_psed.json and baseline_panns.json; both use keys named panns_*)
  var BASE = [
    ['psed', 'PretrainedSED alone (CP-JKU, ICASSP 2025, state of the art frame-level detector on AudioSet-Strong) + a picture for every detection', 'PretrainedSED'],
    ['panns', 'PANNs alone (older, widely used) + a picture for every detection', 'PANNs']
  ];
  function baselineBlock(host) {
    var BS = D.baselines || {};
    var have = BASE.filter(function (b) { return BS[b[0]]; });
    var h = '<h3 style="margin-top:16px">Existing approach, TEST (60 clips) ' + badge() + '</h3>';
    if (!have.length) { host.innerHTML = h + '<p class="muted">Not in data.js: re-run build.py (it reads benchmark/gold/baseline_psed.json and baseline_panns.json).</p>'; return; }
    h += '<p class="lede">An existing single-detector approach: one sound detector alone, and a picture for every sound it detects above one threshold. ' +
      'The threshold was chosen on DEV (' + have.map(function (b) { return b[2] + ' ' + esc(BS[b[0]].threshold); }).join(', ') + '). There is no gate and no second detector.</p>';
    var first = BS[have[0][0]];
    var rows = have.map(function (b) { return [b[1] + ' (threshold chosen on DEV)', BS[b[0]].base, '']; })
      .concat([['ours', first.ours, 'ours'], ['silence', first.silence, '']]);
    h += '<div class="tw"><table class="results"><thead><tr><th>system</th><th class="num">hits</th><th class="num">misses</th>' +
      '<th class="num">wrong pictures</th><th class="num">precision</th><th class="num">recall</th><th class="num">F1</th><th class="num">viewer cost / clip</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      var x = r[1]; if (!x) return;
      var w = (x.visible || 0) + (x.cross || 0) + (x.phantom || 0);
      h += '<tr' + (r[2] ? ' class="ours"' : '') + '><td class="wrapcell">' + esc(r[0]) + '</td><td class="num nowrap">' + x.hits + ' <span class="muted small">of ' + (x.hits + x.misses) + '</span></td><td class="num">' + x.misses + '</td>' +
        '<td class="num"><b>' + w + '</b> <span class="muted small">(' + x.visible + ' / ' + x.cross + ' / ' + x.phantom + ')</span></td>' +
        '<td class="num">' + (x.hits + w ? pct(x.P) : '—') + '</td><td class="num">' + pct(x.R) + '</td><td class="num">' + pct(x.F1) + '</td><td class="num"><b>' + f2(x.viewer_cost) + '</b></td></tr>';
    });
    h += '</tbody></table></div><p class="small muted">Wrong pictures in brackets: source visible / different sound / no such sound. Lower viewer cost is better.</p>';
    var diffs = [['F1', 'F1', 'higher is better'], ['P', 'precision', 'higher is better'], ['R', 'recall', 'higher is better'],
      ['viewer_cost', 'viewer cost / clip', 'lower is better, so a negative difference is good']];
    function sd(x) { return (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(2); }
    h += '<div class="grid basediffs">';
    have.forEach(function (b) {
      var B = BS[b[0]];
      h += '<div><h4 class="subh">Ours − ' + b[2] + ' alone, with 95 % intervals</h4><div class="tw"><table class="results"><thead><tr><th>measure</th><th class="num">difference</th>' +
        '<th class="num">95 % interval</th><th>significant?</th></tr></thead><tbody>';
      diffs.forEach(function (d) {
        var v = (B.diff || {})[d[0]]; if (!v) return;
        var sig = v[1] > 0 || v[2] < 0;
        h += '<tr><td>' + d[1] + '<br><span class="muted small">' + d[2] + '</span></td><td class="num"><b>' + sd(v[0]) + '</b></td>' +
          '<td class="num nowrap">[' + sd(v[1]) + ', ' + sd(v[2]) + ']</td><td>' +
          (sig ? '<span class="tag hit">yes</span>' : '<span class="tag dup">no</span>') + '</td></tr>';
      });
      h += '</tbody></table></div></div>';
    });
    h += '</div><p class="small muted">Significant = the 95 % interval does not include 0.</p>';
    h += '<p class="sentence">Against an existing single-detector approach, our system finds more of the needed sounds and shows a higher share of correct pictures.</p>';
    host.innerHTML = h;
  }

  function countMiss(reason, systems) {
    var n = 0;
    CLIPS.forEach(function (c) {
      if (c.split !== 'DEV' && c.split !== 'TEST') return;
      (systems || ['ours', 'blind']).forEach(function (s) {
        var mm = c.derived.miss[s] || {};
        Object.keys(mm).forEach(function (i) { if (mm[i].reason === reason) n++; });
      });
    });
    return n;
  }

  function sensTable(host) {
    var S = D.sensitivity;
    if (!S) { host.innerHTML = '<p class="muted">Not in data.js: re-run build.py.</p>'; return; }
    var rows = [];
    ['DEV', 'TEST', 'DEV+TEST', 'sliceB'].forEach(function (g) {
      if (!S[g]) return;
      ['ours', 'blind', 'silence'].forEach(function (s) { rows.push({ g: g, s: s, x: S[g][s] }); });
    });
    function ar(a, b, f) { f = f || function (v) { return v; }; return f(a) + (a === b ? '' : ' → <b>' + f(b) + '</b>'); }
    var h = '<div class="tw"><table class="results"><thead><tr><th>split</th><th>system</th><th class="num">hits (onset rule → while playing)</th><th class="num">misses</th>' +
      '<th class="num">wrong pictures</th><th class="num">F1</th><th class="num">viewer cost / clip</th></tr></thead><tbody>';
    rows.forEach(function (r, k) {
      var o = r.x.onset, d = r.x.during;
      var first = k % 3 === 0;
      h += '<tr' + (r.s === 'ours' ? ' class="ours"' : '') + '><td>' + (first ? esc(r.g) + (r.g.indexOf('TEST') >= 0 ? ' ' + badge() : '') : '') + '</td><td>' + sysName(r.s) + '</td>' +
        '<td class="num">' + ar(o.hits, d.hits) + ' <span class="muted">of ' + (o.hits + o.misses) + '</span></td><td class="num">' + ar(o.misses, d.misses) + '</td>' +
        '<td class="num">' + ar(o.wrong, d.wrong) + '</td><td class="num">' + ar(o.F1, d.F1, pct) + '</td><td class="num">' + ar(o.cost, d.cost, f2) + '</td></tr>';
    });
    h += '</tbody></table></div>';
    var t = S.TEST;
    function order(k) { return ['ours', 'blind', 'silence'].sort(function (a, b) { return t[a][k].cost - t[b][k].cost; }).map(sysName).join(' < '); }
    if (t) h += '<p class="sentence">On TEST the relaxed rule adds ' + (t.ours.during.hits - t.ours.onset.hits) + ' hits to ours and ' + (t.blind.during.hits - t.blind.onset.hits) +
      ' to the pipeline without gate (each one also removes a wrong picture). Cost order with the scoring rule: <b>' + order('onset') + '</b>; with the relaxed rule: <b>' + order('during') + '</b>. ' +
      'See the <a href="#/mistakes/late">late or early pictures</a> to judge each case.</p>';
    host.innerHTML = h;
  }

  // ------------------------------------------------------------------ MISTAKES
  var M = { tab: 'miss', split: 'DEVTEST', cat: '', sys: 'ours', q: '', reason: '', wtype: '', pos: '' };
  function splitOk(sp) {
    if (M.split === 'ALL') return true;
    if (M.split === 'DEVTEST') return sp === 'DEV' || sp === 'TEST';
    return sp === M.split;
  }
  function baseClips() {
    return CLIPS.filter(function (c) { return splitOk(c.split) && (!M.cat || c.category === M.cat); });
  }
  function qOk() {
    var q = lc(M.q).trim();
    if (!q) return function () { return true; };
    return function () {
      for (var i = 0; i < arguments.length; i++) if (lc(arguments[i]).indexOf(q) >= 0) return true;
      return false;
    };
  }
  function systemsSel() { return M.sys === 'both' ? ['ours', 'blind'] : [M.sys]; }

  function missRows() {
    var ok = qOk(), rows = [];
    baseClips().forEach(function (c) {
      systemsSel().forEach(function (sys) {
        var mm = c.derived.miss[sys] || {};
        Object.keys(mm).forEach(function (i) {
          var s = c.sounds[+i];
          if (!ok(s.label, c.clip)) return;
          rows.push({ c: c, sys: sys, s: s, i: +i, r: mm[i] });
        });
      });
    });
    return rows;
  }
  function augFor(c, label, start) {
    var a = ours(c).augmentations || [];
    for (var i = 0; i < a.length; i++) if (a[i].event_label === label && Math.abs(a[i].start - start) < 0.01) return a[i];
    return null;
  }
  function missDetail(row) {
    var r = row.r, c = row.c, s = row.s, g = gateOf(c);
    if (r.reason === 'removed by the gate') {
      var lines = (r.gate || []).map(function (k) { return gateLine(g[k]); }).join('');
      return 'The pipeline without gate drew ' + esc(r.pic_label) + ' at ' + ft(r.pic_start) + ' s (' + sgn(r.pic_start - s.start) + ' s). ' + (lines || '<span class="muted">no gate stretch found near the onset</span>');
    }
    if (r.reason === 'timing') {
      return esc(sysName(r.pic_system)) + ' drew ' + esc(r.pic_label) + ' at ' + ft(r.pic_start) + ' s; the sound starts at ' + ft(s.start) + ' s (<b>' + sgn(r.late) +
        ' s</b>). A hit must start between ' + sgn(WIN[0]) + ' and ' + sgn(WIN[1]) + ' s.';
    }
    if (r.reason === 'detected, not drawn') {
      var a = augFor(c, r.event_label, r.event_start);
      return 'Detector heard ' + esc(r.event_label) + ' at ' + ft(r.event_start) + ' s (score ' + f2(r.confidence) + ').' +
        (a ? ' Not drawn because: <i>' + esc(a.reason) + '</i>' : '');
    }
    if (r.reason === 'taken by another sound') return 'A ' + esc(r.pic_label) + ' picture at ' + ft(r.pic_start) + ' s was matched to another sound of the same kind.';
    return '<span class="muted">No detector event of this kind of sound near ' + ft(s.start) + ' s.</span>';
  }

  function wrongRows() {
    var ok = qOk(), rows = [];
    baseClips().forEach(function (c) {
      systemsSel().forEach(function (sys) {
        var ps = c.systems[sys] ? c.systems[sys].pictures || [] : [];
        ps.forEach(function (p, j) {
          if (!isWrong(p.class)) return;
          if (!ok(p.label, c.clip)) return;
          rows.push({ c: c, sys: sys, p: p, j: j, k: pclass(p.class).k });
        });
      });
    });
    return rows;
  }
  function thereText(row) {
    var c = row.c, p = row.p;
    var idx = p.sound && p.sound.length ? p.sound : soundsAt(c, p.start).map(function (o) { return o.i; });
    var seen = {};
    var t = idx.filter(function (i) { if (seen[i]) return false; seen[i] = 1; return true; }).map(function (i) {
      var s = c.sounds[i];
      return esc(s.label) + ' ' + ft(s.start) + '–' + ft(s.end) + ' s ' + kindTag(s);
    });
    return t.length ? t.join('<br>') : '<span class="muted">no gold sound near ' + ft(p.start) + ' s</span>';
  }
  function gateForPic(row) {
    var c = row.c, g = gateOf(c);
    if (row.sys === 'ours') {
      var d = c.derived.ours_pics[row.j];
      if (!d || d.gate == null) return '<span class="muted">—</span>';
      return gateLine(g[d.gate]) + (d.gate_approx ? '<span class="muted small">(nearest stretch)</span>' : '');
    }
    var b = c.derived.blind_pics[row.j];
    if (!b) return '—';
    var line = b.gate != null ? gateLine(g[b.gate]) : '';
    return (b.in_ours ? '<b>ours showed it too</b>' : '<b>gate removed it in ours</b>') + line;   // row of the pipeline without gate
  }

  function gateRowsA() {
    var ok = qOk(), rows = [];
    baseClips().forEach(function (c) {
      var g = gateOf(c);
      c.sounds.forEach(function (s, i) {
        if (!(s.needed && s.importance >= 2)) return;
        var near = (c.derived.near_gate[String(i)] || []).filter(function (k) { return g[k].seen; });
        if (!near.length || !ok(s.label, c.clip)) return;
        rows.push({ c: c, s: s, i: i, g: near });
      });
    });
    return rows;
  }
  function gateRowsB() {
    var ok = qOk(), rows = [];
    baseClips().forEach(function (c) {
      (ours(c).pictures || []).forEach(function (p, j) {
        if (p.class !== 'wrong: source visible or obvious' || !ok(p.label, c.clip)) return;
        rows.push({ c: c, sys: 'ours', p: p, j: j });
      });
    });
    return rows;
  }

  var colPlaySound = { k: 'play', t: 'play', h: function (r) { return playBtn(r.c, r.s.start, r.sys === 'blind' ? 'blind' : 'ours'); }, v: function (r) { return r.s.start; } };
  var colPlayPic = { k: 'play', t: 'play', h: function (r) { return playBtn(r.c, r.p.start, r.sys === 'blind' ? 'blind' : 'ours'); }, v: function (r) { return r.p.start; } };
  var colSplit = { k: 'split', t: 'split', h: function (r) { return splitTag(r.c.split); }, v: function (r) { return r.c.split; } };
  var colClip = { k: 'clip', t: 'clip', h: function (r) { return clipLink(r.c); }, v: function (r) { return r.c.clip; }, cls: 'nowrap' };
  var colCat = { k: 'cat', t: 'clip type', h: function (r) { return CAT[r.c.category]; }, v: function (r) { return CAT[r.c.category]; }, cls: 'nowrap' };
  var colSys = { k: 'sys', t: 'system', h: function (r) { return sysName(r.sys); }, v: function (r) { return r.sys; } };

  // ---- late or early pictures (timing misses)
  var POS_TEXT = { early: 'early (before the window)', window: 'in the window', during: 'late, sound still playing', after: 'after the sound ended' };
  function posOf(t, s) {
    if (t < s.start + WIN[0]) return 'early';
    if (t <= s.start + WIN[1]) return 'window';
    if (t <= s.end) return 'during';
    return 'after';
  }
  function posTag(p) { return '<span class="pos ' + p + '">' + POS_TEXT[p] + '</span>'; }
  function miniTL(row) {
    var c = row.c, s = row.s, r = row.r, fam = c.derived.family || [];
    var dur = duration(c), W = 300, L = 52, R = 6;
    var X = function (t) { return L + Math.max(0, Math.min(dur, t)) / dur * (W - L - R); };
    var h = '', y = 2;
    function lab(t, yy) { h += '<text x="0" y="' + (yy + 10) + '" class="ink2" style="font-size:10px">' + t + '</text>'; }
    function bar(a, b, yy, hh, k, tipTxt, extra) {
      h += '<rect class="m k-' + k + '" x="' + X(a) + '" y="' + yy + '" width="' + Math.max(3, X(b) - X(a)) + '" height="' + hh + '" rx="2"' +
        (extra || '') + ' data-tip="' + esc(tipTxt) + '"></rect>';
    }
    // the hit window of this sound, across all rows
    var rowsH = 12 * 3 + 6 * 2 + 8;
    h += '<rect class="win" x="' + X(s.start + WIN[0]) + '" y="0" width="' + (X(s.start + WIN[1]) - X(s.start + WIN[0])) + '" height="' + rowsH + '"></rect>';
    lab('sound', y);
    c.sounds.forEach(function (g, k) {
      if (k !== row.i && fam[k] !== fam[row.i]) return;
      var kk = k === row.i ? 'needed' : soundKind(g);
      bar(g.start, Math.max(g.end, g.start + 0.1), y, 12, kk, '<b>' + esc(g.label) + '</b> ' + ft(g.start) + '–' + ft(g.end) + ' s' + (k === row.i ? ' (this sound)' : ' (' + KIND_TEXT[soundKind(g)] + ')'),
        k === row.i ? ' style="fill-opacity:.45"' : '');
    });
    y += 18;
    lab('pictures', y);
    (r.fam_pics || []).forEach(function (p) {
      var own = p.sys === row.sys;
      bar(p.start, Math.max(p.end, p.start + 0.1), y + (own ? 0 : 7), own ? 12 : 5, pclass(p.class).k,
        '<b>' + sysName(p.sys) + ': ' + esc(p.label) + '</b> ' + ft(p.start) + '–' + ft(p.end) + ' s<br>' + POS_TEXT[posOf(p.start, s)] + ' (' + sgn(p.start - s.start) + ' s)<br>' + esc(pclass(p.class).s));
    });
    y += 18;
    lab('detector', y);
    (r.fam_events || []).forEach(function (e) {
      bar(e.start, Math.max(e.end, e.start + 0.1), y + 3, 6, 'event', '<b>' + esc(e.label) + '</b> ' + ft(e.start) + '–' + ft(e.end) + ' s, score ' + f2(e.confidence));
    });
    y += 16;
    h += '<line class="axisl" x1="' + L + '" x2="' + (W - R) + '" y1="' + y + '" y2="' + y + '"></line>';
    var step = dur <= 12 ? 2 : dur <= 30 ? 5 : 10;
    for (var t = 0; t <= dur + 1e-6; t += step) h += '<text x="' + X(t) + '" y="' + (y + 11) + '" text-anchor="middle" class="muted-t" style="font-size:10px">' + t + '</text>';
    return '<svg class="tl mini" width="' + W + '" height="' + (y + 14) + '" viewBox="0 0 ' + W + ' ' + (y + 14) + '">' + h + '</svg>';
  }
  function renderLate(chips, ex, tb) {
    var rows = missRows().filter(function (r) { return r.r.reason === 'timing'; });
    rows.forEach(function (r) { r.pos = posOf(r.r.pic_start, r.s); });
    var cnt = {};
    rows.forEach(function (r) { cnt[r.pos] = (cnt[r.pos] || 0) + 1; });
    chips.innerHTML = '<div class="rule-note">The scoring rule counts a picture as correct only if it starts within 0.5 s before to 1 s after the sound\'s start. ' +
      'A picture that appears later while a long sound is still playing counts as a miss plus a wrong picture.</div>' +
      '<div class="reasons"><button type="button" class="chip' + (M.pos ? '' : ' on') + '" data-pos="">all <b>' + rows.length + '</b></button>' +
      ['early', 'during', 'after'].filter(function (p) { return cnt[p]; }).map(function (p) {
        return '<button type="button" class="chip' + (M.pos === p ? ' on' : '') + '" data-pos="' + p + '">' + POS_TEXT[p] + ' <b>' + cnt[p] + '</b></button>';
      }).join('') + '</div>';
    ex.innerHTML = '<div class="explain">Each row is a needed sound (importance 2–3) that had a picture of the right kind, but at the wrong moment. ' +
      'The small timeline shows the whole clip: the sound (and other sounds of the same kind), the shaded band = the hit window, the pictures ' +
      '(thin bars = the other system) and the detector. If a picture starts well after the sound\'s start but the detector shows a new burst there, ' +
      'the annotation may have merged two occurrences into one long sound. Chips count by the picture nearest the start.</div>';
    var shown = rows.filter(function (r) { return !M.pos || r.pos === M.pos; });
    var cols = [
      { k: 'play', t: 'sound start', h: function (r) { return playBtn(r.c, r.s.start, r.sys === 'blind' ? 'blind' : 'ours'); }, v: function (r) { return r.s.start; } },
      { k: 'clip', t: 'clip · system', h: function (r) { return splitTag(r.c.split) + ' ' + clipLink(r.c) + '<br><span class="small">' + sysName(r.sys) + ' · ' + CAT[r.c.category] + '</span>'; }, v: function (r) { return r.c.clip; } },
      { k: 'snd', t: 'sound', h: function (r) { return '<b>' + esc(r.s.label) + '</b><br><span class="small">' + ft(r.s.start) + '–' + ft(r.s.end) + ' s (' + ft(r.s.end - r.s.start) + ' s long)</span>'; }, v: function (r) { return r.s.label; } },
      { k: 'pos', t: 'nearest picture', h: function (r) { return posTag(r.pos) + '<br><span class="small">' + sgn(r.r.late) + ' s from the start</span>'; }, v: function (r) { return r.r.late; } },
      { k: 'pics', t: 'pictures of this kind (play at picture)', h: function (r) {
        var ps = (r.r.fam_pics || []).slice().sort(function (a, b) { return (a.sys === r.sys ? 0 : 1) - (b.sys === r.sys ? 0 : 1) || a.start - b.start; });
        return ps.map(function (p) {
          return '<div class="picline' + (p.sys === r.sys ? '' : ' other') + '">' + playBtn(r.c, p.start, p.sys === 'blind' ? 'blind' : 'ours') + ' ' +
            (p.sys === r.sys ? '' : '<span class="small">(' + sysName(p.sys) + ')</span> ') + esc(p.label) + ' ' + ft(p.start) + '–' + ft(p.end) + ' s ' + posTag(posOf(p.start, r.s)) + ' ' + classTag(p.class) + '</div>';
        }).join('') || '<span class="muted">—</span>';
      } },
      { k: 'tl', t: 'whole clip', h: miniTL }
    ];
    mkTable(tb, 'late', cols, shown);
    bindTips(tb);
    Array.prototype.forEach.call(chips.querySelectorAll('[data-pos]'), function (b) {
      b.addEventListener('click', function () { M.pos = b.getAttribute('data-pos'); renderMistakesBody(); });
    });
  }

  // ---- removed by the gate, grouped by family
  function renderGated(chips, ex, tb) {
    var ok = qOk(), rows = [];
    baseClips().forEach(function (c) {
      var mm = c.derived.miss.ours || {};
      Object.keys(mm).forEach(function (i) {
        if (mm[i].reason !== 'removed by the gate') return;
        var s = c.sounds[+i];
        if (!ok(s.label, c.clip)) return;
        rows.push({ c: c, sys: 'ours', s: s, i: +i, r: mm[i], fam: (c.derived.family || [])[+i] || s.label });
      });
    });
    var groups = {};
    rows.forEach(function (r) { (groups[r.fam] = groups[r.fam] || []).push(r); });
    var fams = Object.keys(groups).sort(function (a, b) { return groups[b].length - groups[a].length || (a < b ? -1 : 1); });
    chips.innerHTML = '<p class="small muted">Only ours has a gate, so the system filter is ignored here.</p><div class="reasons"><span class="chip">all <b>' + rows.length + '</b></span>' +
      fams.map(function (f) { return '<a class="chip" href="#" data-fam="' + esc(f) + '">' + esc(f) + ' <b>' + groups[f].length + '</b></a>'; }).join('') + '</div>';
    ex.innerHTML = '<div class="explain">A needed sound the gate withheld: the pipeline without gate drew a picture of it in time, ours did not, because the VLM said the source is visible ' +
      'in every stretch of that detected sound. The last column shows what the VLM said it saw in each stretch, and its three votes ' +
      '(name / a-b choice / description; “visible” = it said the source is on screen).</div>';
    if (!rows.length) { tb.innerHTML = '<p class="muted">Nothing matches these filters.</p>'; return; }
    tb.innerHTML = fams.map(function (f, k) { return '<h3 class="famhead" id="fam-' + k + '">' + esc(f) + ' <span class="muted">(' + groups[f].length + ')</span></h3><div id="famt-' + k + '"></div>'; }).join('');
    fams.forEach(function (f, k) {
      mkTable(document.getElementById('famt-' + k), 'gated-' + f, [
        { k: 'play', t: 'ours', h: function (r) { return playBtn(r.c, r.s.start, 'ours'); }, v: function (r) { return r.s.start; } },
        { k: 'playb', t: 'pipeline without gate', h: function (r) { return playBtn(r.c, r.r.pic_start, 'blind'); } },
        colSplit, colClip, colCat,
        { k: 'snd', t: 'sound', h: function (r) { return esc(r.s.label) + '<br><span class="small">' + ft(r.s.start) + '–' + ft(r.s.end) + ' s, imp. ' + r.s.importance + '</span>'; }, v: function (r) { return r.s.label; } },
        { k: 'bp', t: 'picture without gate', h: function (r) { return esc(r.r.pic_label) + ' at ' + ft(r.r.pic_start) + ' s (' + sgn(r.r.pic_start - r.s.start) + ' s)'; } },
        { k: 'vlm', t: 'what the VLM said it saw', h: function (r) { var g = gateOf(r.c); return (r.r.gate || []).map(function (x) { return gateLine(g[x]); }).join('') || '<span class="muted">no stretch found</span>'; } }
      ], groups[f], { nocount: 1 });
    });
    Array.prototype.forEach.call(chips.querySelectorAll('[data-fam]'), function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var k = fams.indexOf(a.getAttribute('data-fam'));
        var el = document.getElementById('fam-' + k); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  function renderMistakesBody() {
    var body = document.getElementById('mbody');
    var hasTest = CLIPS.some(function (c) { return c.split === 'TEST' && splitOk(c.split); });
    var h = hasTest ? '<p>' + badge() + ' <span class="small muted">These rows include TEST clips. Look and explain; do not change the system because of them.</span></p>' : '';
    body.innerHTML = h + '<div id="mchips"></div><div id="mexplain"></div><div id="mtable"></div><div id="mtable2"></div>';
    var chips = document.getElementById('mchips'), ex = document.getElementById('mexplain'), tb = document.getElementById('mtable');

    if (M.tab === 'miss') {
      var all = missRows();
      var cnt = {};
      all.forEach(function (r) { cnt[r.r.reason] = (cnt[r.r.reason] || 0) + 1; });
      var nd = M.reason === 'never detected';
      chips.innerHTML = '<button type="button" class="quick' + (nd ? ' on' : '') + '" data-reason="' + (nd ? '' : 'never detected') + '">' +
        (nd ? 'Showing only “never detected” (' + (cnt['never detected'] || 0) + ') — click to show all reasons' :
          'Inspect the “never detected” sounds one by one &rarr; <b>' + (cnt['never detected'] || 0) + '</b>') +
        '<small>' + (nd ? 'Play each one and read what each detector heard at its start.' :
          'The detector did not hear these near their start. The last column shows what it heard instead.') + '</small></button>' +
        '<div class="reasons"><button type="button" class="chip' + (M.reason ? '' : ' on') + '" data-reason="">all reasons <b>' + all.length + '</b></button>' +
        REASONS.filter(function (x) { return cnt[x]; }).map(function (x) {
          return '<button type="button" class="chip' + (M.reason === x ? ' on' : '') + '" data-reason="' + esc(x) + '">' + esc(x) + ' <b>' + cnt[x] + '</b></button>';
        }).join('') + '</div>';
      var rows = all.filter(function (r) { return !M.reason || r.r.reason === M.reason; });
      ex.innerHTML = '<div class="explain">' + (M.reason ? '<b>' + esc(M.reason) + ':</b> ' + esc(REASON_TEXT[M.reason]) :
        'A <b>missed sound</b> is a needed sound (off-screen, not obvious, importance 2–3) with no correct picture. The reason is checked in this order: ' +
        REASONS.slice(0, 4).map(function (x) { return '<b>' + esc(x) + '</b>'; }).join(' → ') + '. Click a reason to see only those rows.') + '</div>' + heardNote();
      var cols = [colPlaySound, colSplit, colClip, colCat];
      if (M.sys === 'both') cols.push(colSys);
      cols = cols.concat([
        { k: 'snd', t: 'sound', h: function (r) { return esc(r.s.label); }, v: function (r) { return r.s.label; } },
        { k: 'on', t: 'starts', num: 1, h: function (r) { return ft(r.s.start) + ' s'; }, v: function (r) { return r.s.start; } },
        { k: 'imp', t: 'imp.', num: 1, h: function (r) { return r.s.importance; }, v: function (r) { return r.s.importance; } },
        { k: 'why', t: 'reason', h: function (r) { return '<b>' + esc(r.r.reason) + '</b>'; }, v: function (r) { return REASONS.indexOf(r.r.reason); } },
        { k: 'det', t: 'what happened', h: missDetail },
        { k: 'heard', t: 'what the detectors heard (−0.5 to +1.0 s)', h: function (r) { return heardCell(r.c, r.i); }, cls: 'heardcol',
          v: function (r) { var x = r.s.heard && r.s.heard.BEATs; return x && x.own != null ? x.own : null; } }
      ]);
      mkTable(tb, 'miss', cols, rows);
    } else if (M.tab === 'wrong') {
      var allw = wrongRows();
      var wc = {};
      allw.forEach(function (r) { wc[r.k] = (wc[r.k] || 0) + 1; });
      chips.innerHTML = '<div class="reasons"><button type="button" class="chip' + (M.wtype ? '' : ' on') + '" data-wtype="">all types <b>' + allw.length + '</b></button>' +
        WRONG_KEYS.map(function (w) {
          return '<button type="button" class="chip' + (M.wtype === w[0] ? ' on' : '') + '" data-wtype="' + w[0] + '"><span class="sw" style="background:var(--c-' + w[0] + ')"></span>' + w[1] + ' <b>' + (wc[w[0]] || 0) + '</b></button>';
        }).join('') + '</div>';
      ex.innerHTML = '<div class="explain"><b>source visible</b>: a real sound, but you can see (or easily guess) its source — the gate should have stopped it. ' +
        '<b>different sound</b>: a real sound was there, but the picture shows another kind of sound (the detector named it wrong). ' +
        '<b>no such sound</b>: nothing was there at all. “What was there” lists the gold sounds at that moment.</div>';
      var roww = allw.filter(function (r) { return !M.wtype || r.k === M.wtype; });
      var colsw = [colPlayPic, colSplit, colClip, colCat];
      if (M.sys === 'both') colsw.push(colSys);
      colsw = colsw.concat([
        { k: 'lab', t: 'picture of', h: function (r) { return esc(r.p.label); }, v: function (r) { return r.p.label; } },
        { k: 'on', t: 'shown', num: 1, h: function (r) { return ft(r.p.start) + '–' + ft(r.p.end) + ' s'; }, v: function (r) { return r.p.start; }, cls: 'nowrap' },
        { k: 'type', t: 'type', h: function (r) { return classTag(r.p.class); }, v: function (r) { return r.k; } },
        { k: 'there', t: 'what was there', h: thereText },
        { k: 'gate', t: M.sys === 'ours' ? 'gate (VLM) at this stretch' : 'gate (VLM)', h: gateForPic }
      ]);
      mkTable(tb, 'wrong', colsw, roww);
    } else if (M.tab === 'late') {
      renderLate(chips, ex, tb);
    } else if (M.tab === 'gated') {
      renderGated(chips, ex, tb);
    } else {
      var ra = gateRowsA(), rb = gateRowsB();
      chips.innerHTML = '<p class="small muted">Gate errors are about ours only (the pipeline without gate has no gate). The system filter is ignored here.</p>';
      ex.innerHTML = '<h3 style="margin-top:10px">(a) Needed sound, but the VLM said “visible” (' + ra.length + ')</h3>' +
        '<div class="explain">The sound is off-screen, but at least one gate stretch of its kind near its start was judged “visible”. ' +
        'The picture is only withheld when every stretch of that detected sound says “visible”, so some of these still became hits — see the last column.</div>';
      mkTable(tb, 'gateA', [
        { k: 'play', t: 'play', h: function (r) { return playBtn(r.c, r.s.start, 'ours'); }, v: function (r) { return r.s.start; } },
        colSplit, colClip, colCat,
        { k: 'snd', t: 'sound', h: function (r) { return esc(r.s.label); }, v: function (r) { return r.s.label; } },
        { k: 'on', t: 'starts', num: 1, h: function (r) { return ft(r.s.start) + ' s'; }, v: function (r) { return r.s.start; } },
        { k: 'vlm', t: 'what the VLM said', h: function (r) { var g = gateOf(r.c); return r.g.map(function (k) { return gateLine(g[k]); }).join(''); } },
        { k: 'res', t: 'ours result', h: function (r) { return outcomeText(r.c, 'ours', r.i); }, v: function (r) { var x = ours(r.c).sounds[r.i]; return x ? x.outcome : ''; } },
        { k: 'bres', t: 'result without gate', h: function (r) { return outcomeText(r.c, 'blind', r.i); }, v: function (r) { var x = blind(r.c).sounds[r.i]; return x ? x.outcome : ''; } }
      ], ra);
      var t2 = document.getElementById('mtable2');
      t2.innerHTML = '<h3 style="margin-top:18px">(b) Visible or obvious sound, but the VLM said “not visible” (' + rb.length + ')</h3>' +
        '<div class="explain">Ours showed a picture of a sound whose source the annotator could see (or that was obvious). These are the “source visible” wrong pictures of ours.</div><div></div>';
      mkTable(t2.lastChild, 'gateB', [colPlayPic, colSplit, colClip, colCat,
        { k: 'lab', t: 'picture of', h: function (r) { return esc(r.p.label); }, v: function (r) { return r.p.label; } },
        { k: 'on', t: 'shown', num: 1, h: function (r) { return ft(r.p.start) + '–' + ft(r.p.end) + ' s'; }, v: function (r) { return r.p.start; }, cls: 'nowrap' },
        { k: 'there', t: 'the gold sound', h: thereText },
        { k: 'gate', t: 'what the VLM said', h: gateForPic }
      ], rb);
    }
    Array.prototype.forEach.call(chips.querySelectorAll('[data-reason]'), function (b) {
      b.addEventListener('click', function () { M.reason = b.getAttribute('data-reason'); renderMistakesBody(); });
    });
    Array.prototype.forEach.call(chips.querySelectorAll('[data-wtype]'), function (b) {
      b.addEventListener('click', function () { M.wtype = b.getAttribute('data-wtype'); renderMistakesBody(); });
    });
  }

  function splitSelect(val) {
    return '<select id="f-split">' + [['DEVTEST', 'DEV + TEST (109)'], ['DEV', 'DEV (49)'], ['TEST', 'TEST (60)'], ['sliceB', 'Slice B (30)'], ['ALL', 'all three (139)']].map(function (o) {
      return '<option value="' + o[0] + '"' + (val === o[0] ? ' selected' : '') + '>' + o[1] + '</option>';
    }).join('') + '</select>';
  }
  function catSelect(val) {
    return '<select id="f-cat"><option value="">all</option>' + CAT_ORDER.map(function (k) {
      return '<option value="' + k + '"' + (val === k ? ' selected' : '') + '>' + CAT[k] + '</option>';
    }).join('') + '</select>';
  }

  function renderMistakes() {
    var h = '<h2>Mistakes</h2><p class="lede">Every mistake, one row each. Press &#9654; to play the clip from the moment shown on the button. ' +
      'Click a column title to sort. Click a clip name to see the whole clip.</p>';
    h += '<div class="subtabs">' + [['miss', 'Missed sounds'], ['late', 'Late or early pictures'], ['gated', 'Removed by the gate'], ['wrong', 'Wrong pictures'], ['gate', 'Gate errors']].map(function (t) {
      return '<button type="button" data-tab="' + t[0] + '" class="' + (M.tab === t[0] ? 'on' : '') + '">' + t[1] + '</button>';
    }).join('') + '</div>';
    h += '<div class="filters"><label>split' + splitSelect(M.split) + '</label><label>clip type' + catSelect(M.cat) + '</label>' +
      '<label>system<select id="f-sys">' + [['ours', 'ours'], ['blind', 'pipeline without gate'], ['both', 'both']].map(function (o) {
        return '<option value="' + o[0] + '"' + (M.sys === o[0] ? ' selected' : '') + '>' + o[1] + '</option>';
      }).join('') + '</select></label>' +
      '<label>sound or clip name<input id="f-q" type="search" placeholder="e.g. dog, siren, vehicle" value="' + esc(M.q) + '"></label></div>';
    h += '<div id="mbody"></div>';
    app.innerHTML = h;
    Array.prototype.forEach.call(app.querySelectorAll('.subtabs button'), function (b) {
      b.addEventListener('click', function () {
        M.tab = b.getAttribute('data-tab');
        Array.prototype.forEach.call(app.querySelectorAll('.subtabs button'), function (x) { x.className = x === b ? 'on' : ''; });
        if (location.hash !== '#/mistakes/' + M.tab) { SKIP_ROUTE = true; location.hash = '#/mistakes/' + M.tab; }   // a link to this view, no re-render
        renderMistakesBody();
      });
    });
    document.getElementById('f-split').addEventListener('change', function (e) { M.split = e.target.value; renderMistakesBody(); });
    document.getElementById('f-cat').addEventListener('change', function (e) { M.cat = e.target.value; renderMistakesBody(); });
    document.getElementById('f-sys').addEventListener('change', function (e) { M.sys = e.target.value; renderMistakesBody(); });
    document.getElementById('f-q').addEventListener('input', function (e) { M.q = e.target.value; renderMistakesBody(); });
    renderMistakesBody();
  }

  // ------------------------------------------------------------------ CLIPS list
  var C = { split: 'DEVTEST', cat: '', q: '' };
  function renderClips() {
    var h = '<h2>Clips</h2><p class="lede">All clips with their scores. Click a row to open the clip: video, timeline, and every sound and picture.</p>';
    h += '<div class="filters"><label>split' + splitSelect(C.split) + '</label><label>clip type' + catSelect(C.cat) + '</label>' +
      '<label>clip or sound name<input id="f-q" type="search" placeholder="e.g. citywalk, siren" value="' + esc(C.q) + '"></label></div>';
    h += '<div id="cbadge"></div><div id="ctable"></div>';
    app.innerHTML = h;
    function draw() {
      var q = lc(C.q).trim();
      var rows = CLIPS.filter(function (c) {
        var sp = C.split === 'ALL' || (C.split === 'DEVTEST' ? (c.split === 'DEV' || c.split === 'TEST') : c.split === C.split);
        if (!sp || (C.cat && c.category !== C.cat)) return false;
        if (!q) return true;
        if (lc(c.clip).indexOf(q) >= 0) return true;
        return c.sounds.some(function (s) { return lc(s.label).indexOf(q) >= 0; });
      });
      document.getElementById('cbadge').innerHTML = rows.some(function (c) { return c.split === 'TEST'; }) ? '<p>' + badge() + '</p>' : '';
      mkTable(document.getElementById('ctable'), 'clips', [
        colSplit, colClip, colCat,
        { k: 'snd', t: 'sounds', num: 1, h: function (r) { return r.sounds.length; }, v: function (r) { return r.sounds.length; } },
        { k: 'need', t: 'needed', num: 1, h: function (r) { return r._needed; }, v: function (r) { return r._needed; } },
        { k: 'oh', t: 'ours hits', num: 1, h: function (r) { return r._o.h; }, v: function (r) { return r._o.h; } },
        { k: 'om', t: 'ours misses', num: 1, h: function (r) { return r._o.m; }, v: function (r) { return r._o.m; } },
        { k: 'ow', t: 'ours wrong', num: 1, h: function (r) { return r._o.w; }, v: function (r) { return r._o.w; } },
        { k: 'bh', t: 'without gate: hits', num: 1, h: function (r) { return r._b.h; }, v: function (r) { return r._b.h; } },
        { k: 'bm', t: 'without gate: misses', num: 1, h: function (r) { return r._b.m; }, v: function (r) { return r._b.m; } },
        { k: 'bw', t: 'without gate: wrong', num: 1, h: function (r) { return r._b.w; }, v: function (r) { return r._b.w; } }
      ].map(function (col) {
        if (col.k === 'split' || col.k === 'clip' || col.k === 'cat') {
          return { k: col.k, t: col.t, cls: col.cls, h: function (r) { return col.h({ c: r }); }, v: function (r) { return col.v({ c: r }); } };
        }
        return col;
      }), rows, { href: clipHref });
    }
    document.getElementById('f-split').addEventListener('change', function (e) { C.split = e.target.value; draw(); });
    document.getElementById('f-cat').addEventListener('change', function (e) { C.cat = e.target.value; draw(); });
    document.getElementById('f-q').addEventListener('input', function (e) { C.q = e.target.value; draw(); });
    draw();
  }

  // ------------------------------------------------------------------ video helper (missing file -> note)
  function setVideo(video, note, src, t) {
    note.hidden = true;
    note.innerHTML = '';
    video.onloadedmetadata = function () {
      note.hidden = true;
      if (t != null) { try { video.currentTime = Math.max(0, t); } catch (e) { /* ignore */ } }
    };
    video.onerror = function () {
      note.innerHTML = '<div><b>Video not ready yet.</b><br><code>' + esc(src) + '</code><br><span class="small muted">The videos are still being made. ' +
        'Put the file in that place and reload the page.</span></div>';
      note.hidden = false;
    };
    video.src = src;
    video.load();
  }
  function variantTabs(host, cur, onPick) {
    host.innerHTML = [['ours', 'ours (clean)'], ['debug', 'ours (debug)'], ['blind', 'pipeline without gate (label chips)']].map(function (v) {
      return '<button type="button" data-v="' + v[0] + '" class="' + (cur === v[0] ? 'on' : '') + '">' + v[1] + '</button>';
    }).join('');
    Array.prototype.forEach.call(host.querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(host.querySelectorAll('button'), function (x) { x.className = x === b ? 'on' : ''; });
        onPick(b.getAttribute('data-v'));
      });
    });
  }
  function curTime(video) { return video.readyState >= 1 && isFinite(video.currentTime) ? video.currentTime : null; }

  // ------------------------------------------------------------------ modal player (from Mistakes)
  var modal = document.getElementById('modal');
  var mv = document.getElementById('m-video'), mn = document.getElementById('m-note');
  function openPlayer(split, clip, t, kind) {
    var c = BYKEY[split + '/' + clip]; if (!c) return;
    document.getElementById('m-title').textContent = clip + ' · ' + ft(t) + ' s';
    document.getElementById('m-badge').innerHTML = split === 'TEST' ? badge() : '';
    document.getElementById('m-open').setAttribute('href', clipHref(c));
    document.getElementById('m-info').textContent = split + ' · ' + CAT[c.category] + ' · starts at the moment shown. ' +
      'Clean = what a viewer sees; debug = with labels; pipeline without gate = the same system with the visibility gate removed, shown with text label chips instead of pictures.';
    var pick = function (v) { setVideo(mv, mn, media(split, clip, v), curTime(mv) != null ? curTime(mv) : t); };
    variantTabs(document.getElementById('m-tabs'), kind || 'ours', pick);
    modal.hidden = false;
    setVideo(mv, mn, media(split, clip, kind || 'ours'), t);
  }
  function closePlayer() { modal.hidden = true; mv.pause(); mv.removeAttribute('src'); mv.load(); }
  document.getElementById('m-close').addEventListener('click', closePlayer);
  document.getElementById('m-open').addEventListener('click', function () { closePlayer(); });
  modal.addEventListener('click', function (e) { if (e.target === modal) closePlayer(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !modal.hidden) closePlayer(); });
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('button.play');
    if (b) { e.stopPropagation(); openPlayer(b.getAttribute('data-split'), b.getAttribute('data-clip'), parseFloat(b.getAttribute('data-t')), b.getAttribute('data-kind')); return; }
    var tr = e.target.closest && e.target.closest('tr[data-href]');
    if (tr && !e.target.closest('a,button')) location.hash = tr.getAttribute('data-href');
  });

  // ------------------------------------------------------------------ CLIP view
  function lanes(items, gap) {
    var ends = [];
    return items.map(function (it) {
      for (var l = 0; l < ends.length; l++) if (it.a >= ends[l] + gap) { ends[l] = it.b; return l; }
      ends.push(it.b); return ends.length - 1;
    });
  }
  // Clip timeline: fixed lanes, top to bottom, one time axis. opts.lanes picks the lanes (default: the clip view's set);
  // opts.a2i = audio-to-image windows [{k,start,end,src}] for the Audio-to-image tab; opts.onTime(t) is called as the video plays.
  var TL_KIND = { needed: 'needed — must be shown', visible: 'visible on screen', obvious: 'obvious without sound', imp1: 'needed, importance 1 (not scored)' };
  var TL_KEPT = { dwell: 'kept on screen: minimum 1.5 s', join: 'kept on screen: joined with a repeat of the same sound (less than 2 s apart)' };
  function tlLegend(showGate) {
    function sw(style) { return '<span class="sw" style="' + style + '"></span>'; }
    return '<div class="legend-row"><b>Sounds:</b>' +
      '<span>' + sw('background:var(--c-needed)') + 'needed — must be shown</span><span>' + sw('background:var(--c-visible)') + 'visible on screen</span>' +
      '<span>' + sw('background:var(--c-obvious)') + 'obvious</span><span><span class="sw dash"></span>importance 1</span>' +
      '<span>' + sw('background:var(--c-needed);opacity:.22') + 'hit window of a needed sound (−0.5 to +1.0 s; dashed line = its start)</span></div>' +
      '<div class="legend-row"><b>Pictures:</b>' +
      '<span>' + sw('background:var(--c-hit)') + 'correct</span><span>' + sw('background:var(--c-vis)') + 'wrong: source visible</span>' +
      '<span>' + sw('background:var(--c-diff)') + 'wrong: different sound</span><span>' + sw('background:var(--c-none)') + 'wrong: no such sound</span>' +
      '<span>' + sw('background:var(--c-dup)') + 'duplicate / don’t care</span>' +
      '<span><span class="sw hatch"></span>kept on screen after its sound ended</span></div>' +
      (showGate ? '<div class="legend-row"><b>Gate:</b><span>' + sw('background:var(--c-seen)') + 'VLM: visible</span>' +
        '<span><span class="sw" style="border:1.5px dashed var(--c-unseen)"></span>VLM: not visible</span></div>' : '');
  }
  function drawTimeline(host, c, video, opts) {
    opts = opts || {};
    var laneSet = opts.lanes || ['sounds', 'ours', 'blind', 'gate', 'events'];
    var W = Math.max(320, host.clientWidth || 700);
    var dur = opts.dur || duration(c);
    var L = 6, R = 10;
    var X = function (t) { return L + Math.max(0, Math.min(dur, t)) / dur * (W - L - R); };
    var secPerPx = dur / (W - L - R);
    var g = gateOf(c);
    var der = c.derived || {};
    var lanesOut = [];
    function lane(key, title, items, o) { if (laneSet.indexOf(key) >= 0) lanesOut.push({ key: key, title: title, items: items, o: o || {} }); }
    function fit(text, px) {
      var n = Math.floor((px - 6) / 6.4);
      if (n < 3) return '';
      return text.length > n ? text.slice(0, Math.max(1, n - 1)) + '…' : text;
    }

    if (opts.a2i) lane('a2i', 'Audio-to-image pictures (one per window, made from the sound alone)', opts.a2i.map(function (w) {
      return { a: w.start, b: w.end, img: w.src, k: 'a2i', t: w.start,
        tip: '<b>audio-to-image, window ' + esc(w.k) + '</b><br>' + ft(w.start) + '–' + ft(w.end) + ' s<br><img class="tipimg" src="' + esc(w.src) + '" alt="">' };
    }), { h: 0 });
    lane('sounds', 'Sounds in the clip (human labels)', c.sounds.map(function (s, i) {
      var k = soundKind(s);
      return { a: s.start, b: Math.max(s.end, s.start + 0.15), k: k, lab: s.label, t: s.start,
        tip: '<b>' + esc(s.label) + '</b> ' + ft(s.start) + '–' + ft(s.end) + ' s<br>' + TL_KIND[k] + ', importance ' + s.importance +
          '<br>ours: ' + outcomeText(c, 'ours', i) + '<br>without gate: ' + outcomeText(c, 'blind', i) };
    }), { h: 20 });
    function picItems(sys, sd, dd, withThumb) {
      return (sd.pictures || []).map(function (p, j) {
        var pc = pclass(p.class), d = (dd || [])[j] || {};
        var img = withThumb && d.aug != null ? picSrc(c, d.aug) : null;
        var kept = (d.kept || []).filter(function (x) { return x[2] === 'dwell' || x[2] === 'join'; });
        var head = '<b>' + sysName(sys) + ': ' + esc(p.label) + '</b><br>' + ft(p.start) + '–' + ft(p.end) + ' s on screen<br>' + esc(pc.s);
        return { a: p.start, b: Math.max(p.end, p.start + 0.15), k: pc.k, lab: p.label, t: p.start, img: img, kept: kept, head: head,
          tip: head + (kept.length ? '<br><span class="muted">hatched: ' + kept.map(function (x) { return TL_KEPT[x[2]] + ' (' + ft(x[0]) + '–' + ft(x[1]) + ' s)'; }).join('; ') + '</span>' : '') +
            (img ? '<br><img class="tipimg" src="' + esc(img) + '" alt="" onerror="this.remove()">' : '') };
      });
    }
    lane('ours', 'What ours showed', picItems('ours', ours(c), der.ours_pics, true), { h: 30 });
    lane('blind', 'What the pipeline without gate showed', picItems('blind', blind(c), der.blind_pics, false), { h: 20 });
    lane('gate', W < 520 ? 'Gate decision (ours only)' : 'Gate decision (per stretch of a detected sound; ours only)', g.map(function (e) {
      var a = augFor(c, e.label, e.start);
      var drawn = a ? !!a.augment : !g.every(function (x) { return x.label !== e.label || Math.abs(x.start - e.start) > 0.01 || x.seen; });
      var txt = (e.seen ? 'visible' : 'not visible') + ' → ' + (drawn ? 'drawn' : 'not drawn');
      return { a: e.stretch[0], b: Math.max(e.stretch[1], e.stretch[0] + 0.15), k: e.seen ? 'seen' : 'unseen', lab: e.label + ': ' + txt, t: e.stretch[0],
        tip: '<b>' + esc(e.label) + '</b> ' + ft(e.stretch[0]) + '–' + ft(e.stretch[1]) + ' s<br>VLM: <b>' + (e.seen ? 'visible' : 'not visible') + '</b> → ' + (drawn ? 'drawn' : 'not drawn') +
          '<br>the VLM saw “' + esc(e.named) + '”<br>' + esc(votes(e)) + (a && !a.augment ? '<br><span class="muted">' + esc(a.reason) + '</span>' : '') };
    }), { h: 18 });
    var evs = (ours(c).events || []).slice().sort(function (a, b) { return a.start - b.start; });
    if (opts.events) lane('events', 'Detector events', evs.map(function (e) {
      return { a: e.start, b: Math.max(e.end, e.start + 0.1), k: 'event', lab: e.label, t: e.start,
        tip: '<b>' + esc(e.label) + '</b> ' + ft(e.start) + '–' + ft(e.end) + ' s<br>detector score ' + f2(e.confidence) };
    }), { h: 12, small: 1 });

    var y = 0, titleH = 17, laneGap = 3, secGap = 8;
    var bodyTop = null, pieces = [], a2iRects = [];
    lanesOut.forEach(function (ln) {
      var items = ln.items.slice().sort(function (a, b) { return a.a - b.a; });
      var rh = ln.o.h;
      if (ln.key === 'a2i') {
        var ww = items.length ? (X(items[0].b) - X(items[0].a)) : 60;
        rh = Math.max(28, Math.min(84, ww - 2));
      }
      var lnIdx = ln.key === 'a2i' ? items.map(function () { return 0; }) : lanes(items, 0.05);
      var nl = Math.max(1, lnIdx.reduce(function (m, v) { return Math.max(m, v + 1); }, 0));
      var hh = nl * (rh + laneGap) - laneGap + 8;
      pieces.push('<text x="' + L + '" y="' + (y + 12) + '" class="lanet">' + esc(ln.title) + (items.length ? '' : ' — none') + '</text>');
      y += titleH;
      if (bodyTop == null) bodyTop = y;
      pieces.push('<rect class="sec" x="' + L + '" y="' + y + '" width="' + (W - L - R) + '" height="' + hh + '" rx="4"></rect>');
      items.forEach(function (it, i) {
        var x0 = X(it.a), x1 = X(it.b), yy = y + 4 + lnIdx[i] * (rh + laneGap);
        var w = Math.max(3, x1 - x0);
        if (it.k === 'a2i') {
          pieces.push('<image href="' + esc(it.img) + '" x="' + (x0 + 1) + '" y="' + yy + '" width="' + Math.max(1, w - 2) + '" height="' + rh + '" preserveAspectRatio="xMidYMid slice" onerror="this.remove()"></image>');
          pieces.push('<rect class="a2ibox" x="' + (x0 + 1) + '" y="' + yy + '" width="' + Math.max(1, w - 2) + '" height="' + rh + '" data-tip="' + esc(it.tip) + '"></rect>');
          a2iRects.push({ a: it.a, b: it.b, x: x0 + 1, y: yy, w: Math.max(1, w - 2), h: rh });
          return;
        }
        pieces.push('<rect class="m k-' + it.k + '" x="' + x0 + '" y="' + yy + '" width="' + w + '" height="' + rh + '" rx="3" data-tip="' + esc(it.tip) + '"></rect>');
        (it.kept || []).forEach(function (kp) {
          var kx0 = X(kp[0]), kx1 = X(kp[1]);
          if (kx1 - kx0 < 1) return;
          pieces.push('<rect class="kept" fill="url(#tlh-' + it.k + ')" x="' + kx0 + '" y="' + (yy + 1) + '" width="' + (kx1 - kx0) + '" height="' + (rh - 2) + '" data-tip="' +
            esc(it.head + '<br><b>' + TL_KEPT[kp[2]] + '</b> (' + ft(kp[0]) + '–' + ft(kp[1]) + ' s)') + '"></rect>');
        });
        var tx = x0 + 4;
        if (it.img && w > 16) {
          var ts = rh - 4;
          pieces.push('<image href="' + esc(it.img) + '" x="' + (x0 + 2) + '" y="' + (yy + 2) + '" width="' + ts + '" height="' + ts + '" preserveAspectRatio="xMidYMid slice" onerror="this.remove()" style="pointer-events:none"></image>');
          tx = x0 + ts + 6;
        }
        if (!ln.o.small) {
          var lab = fit(it.lab, x0 + w - tx);
          if (lab) pieces.push('<text class="lab" x="' + tx + '" y="' + (yy + rh / 2 + 4) + '">' + esc(lab) + '</text>');
        }
      });
      y += hh + secGap;
    });
    if (bodyTop == null) bodyTop = 0;
    var bodyBot = y - secGap;
    // hit windows and onset lines of needed sounds, across every lane (windows behind the bars, lines in front)
    var back = '', front = '';
    c.sounds.forEach(function (s) {
      if (!(s.needed && s.importance >= 2)) return;
      back += '<rect class="win" x="' + X(s.start + WIN[0]) + '" y="' + bodyTop + '" width="' + (X(s.start + WIN[1]) - X(s.start + WIN[0])) + '" height="' + (bodyBot - bodyTop) + '"></rect>';
      front += '<line class="onset" x1="' + X(s.start) + '" x2="' + X(s.start) + '" y1="' + bodyTop + '" y2="' + bodyBot + '"></line>';
    });
    // axis: a tick every second, a number every 1/2/5/10 s
    var ax = '';
    var yA = bodyBot + 4;
    var step = dur <= 20 ? 1 : dur <= 40 ? 2 : dur <= 100 ? 5 : 10;
    var minor = dur <= 100 ? 1 : 5;
    ax += '<line class="axisl" x1="' + L + '" x2="' + (W - R) + '" y1="' + yA + '" y2="' + yA + '"></line>';
    for (var t = 0; t <= dur + 1e-6; t += minor) {
      var major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      ax += '<line class="axisl" x1="' + X(t) + '" x2="' + X(t) + '" y1="' + yA + '" y2="' + (yA + (major ? 6 : 3)) + '"></line>';
      if (major) ax += '<text x="' + X(t) + '" y="' + (yA + 18) + '" text-anchor="middle" class="muted-t" style="font-size:11px">' + t + '</text>';
    }
    ax += '<text x="' + (W - R) + '" y="' + (yA + 32) + '" text-anchor="end" class="muted-t" style="font-size:11px">seconds · click anywhere to jump there</text>';
    var H = yA + 38;
    var defs = '<defs>' + ['hit', 'vis', 'diff', 'none', 'dup'].map(function (k) {
      return '<pattern id="tlh-' + k + '" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" class="hbg"></rect>' +
        '<line x1="1" y1="0" x2="1" y2="6" class="hs k-' + k + '"></line></pattern>';
    }).join('') + '</defs>';
    var hl = opts.a2i ? '<rect id="a2i-hl" class="a2ihl" x="-100" y="0" width="0" height="0"></rect>' : '';
    var ph = '<line class="playhead" x1="' + L + '" x2="' + L + '" y1="' + bodyTop + '" y2="' + yA + '"></line>';
    host.innerHTML = '<svg class="tl" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Timeline: sounds, pictures, gate">' +
      defs + back + pieces.join('') + front + ax + hl + ph + '</svg>';
    bindTips(host);
    var svg = host.firstChild;
    var phl = svg.querySelector('.playhead'), hlr = svg.querySelector('#a2i-hl');
    var lastA2i = -2;
    function upd(tForce) {
      var t = tForce != null ? tForce : (video.currentTime || 0);
      var x = X(t); phl.setAttribute('x1', x); phl.setAttribute('x2', x);
      if (hlr) {
        var k = -1;
        a2iRects.forEach(function (r, i) { if (t >= r.a && t < r.b) k = i; });
        if (k !== lastA2i) {
          lastA2i = k;
          if (k < 0) hlr.setAttribute('x', -100);
          else { var r = a2iRects[k]; hlr.setAttribute('x', r.x - 1.5); hlr.setAttribute('y', r.y - 1.5); hlr.setAttribute('width', r.w + 3); hlr.setAttribute('height', r.h + 3); }
        }
      }
      if (opts.onTime) opts.onTime(t);
    }
    svg.addEventListener('click', function (e) {
      var box = svg.getBoundingClientRect();
      var px = (e.clientX - box.left) * (W / box.width);
      var tt = Math.max(0, Math.min(dur, (px - L) * secPerPx));
      if (video.readyState >= 1) video.currentTime = tt;
      upd(tt);
    });
    video.ontimeupdate = function () { upd(); };
    video.onseeked = function () { upd(); };
    upd();
  }

  function renderClip(split, name) {
    var c = BYKEY[split + '/' + name];
    if (!c) { app.innerHTML = '<div class="warn">No clip called <code>' + esc(name) + '</code> in ' + esc(split) + '.</div>'; return; }
    var o = ours(c), b = blind(c), g = gateOf(c);
    var h = '<p><a href="#/clips">&larr; all clips</a></p><div class="clip-head">' + splitTag(c.split) + '<h2>' + esc(c.clip) + '</h2>' + (c.split === 'TEST' ? badge() : '') + '</div>';
    h += '<div class="clip-grid"><div><div class="vtabs" id="v-tabs"></div><div class="player"><video id="v" controls preload="metadata" playsinline></video><div class="vnote" id="v-note" hidden></div></div></div>';
    h += '<div class="card"><dl class="kv">' +
      '<dt>clip type</dt><dd>' + CAT[c.category] + '</dd>' +
      '<dt>length</dt><dd>' + ft(duration(c)) + ' s</dd>' +
      '<dt>gold sounds</dt><dd>' + c.sounds.length + ' (' + c._needed + ' needed, importance 2–3)</dd>' +
      '<dt>ours</dt><dd>' + c._o.h + ' hits · ' + c._o.m + ' misses · ' + c._o.w + ' wrong of ' + c._o.n + ' pictures</dd>' +
      '<dt>pipeline without gate</dt><dd>' + c._b.h + ' hits · ' + c._b.m + ' misses · ' + c._b.w + ' wrong of ' + c._b.n + ' pictures</dd>' +
      '<dt>gate</dt><dd>' + g.length + ' stretches judged (' + g.filter(function (x) { return x.seen; }).length + ' “visible”)</dd>' +
      '</dl><p class="small muted" style="margin-top:10px">Clean = the video a viewer sees (ours). Debug = the same with labels. Pipeline without gate = ' + esc(NOGATE_DEF) + ' Its video shows text label chips instead of pictures.</p></div></div>';
    h += '<h3 style="margin-top:16px">Timeline</h3><p class="small muted">Read it top to bottom: what was in the clip, what each system showed, and what the gate decided. ' +
      'A picture is correct only if it starts inside the shaded window of a needed sound. Hover a bar for details; click anywhere to jump the video there.</p>' + tlLegend(true);
    h += '<div class="card" style="padding:10px"><div id="tl"></div><button type="button" id="tl-ev" class="small">Show detector events (' + (o.events || []).length + ')</button></div>';
    h += '<h3 style="margin-top:18px">Gold sounds</h3>' + heardNote() + '<div id="t-snd"></div>';
    h += '<h3 style="margin-top:18px">Ours: pictures shown</h3><div id="t-op"></div>';
    h += '<h3 style="margin-top:18px">Pipeline without gate: pictures shown</h3><div id="t-bp"></div>';
    h += '<details><summary>Gate: every stretch the VLM judged (' + g.length + ')</summary><div id="t-gate"></div></details>';
    h += '<details><summary>Detector events (' + (o.events || []).length + ')</summary><div id="t-ev"></div></details>';
    h += '<details><summary>Augmentations (ours, ' + (o.augmentations || []).length + ')</summary><div id="t-aug"></div></details>';
    app.innerHTML = h;

    var video = document.getElementById('v'), note = document.getElementById('v-note');
    var kind = 'ours';
    variantTabs(document.getElementById('v-tabs'), kind, function (v) {
      var t = curTime(video);
      setVideo(video, note, media(c.split, c.clip, v), t);
    });
    var t0 = parseFloat((location.hash.split('?t=')[1] || ''));
    setVideo(video, note, media(c.split, c.clip, kind), isNaN(t0) ? null : t0);
    var tl = document.getElementById('tl'), tlOpts = { events: false };
    drawTimeline(tl, c, video, tlOpts);
    REDRAW.push(function () { if (document.body.contains(tl)) drawTimeline(tl, c, video, tlOpts); });
    document.getElementById('tl-ev').addEventListener('click', function (e) {
      tlOpts.events = !tlOpts.events;
      e.target.textContent = (tlOpts.events ? 'Hide' : 'Show') + ' detector events (' + (o.events || []).length + ')';
      drawTimeline(tl, c, video, tlOpts);
    });

    var sndRows = c.sounds.map(function (s, i) { return { c: c, s: s, i: i }; });
    mkTable(document.getElementById('t-snd'), 'c-snd', [
      { k: 'play', t: 'play', h: function (r) { return playBtnLocal(r.s.start); }, v: function (r) { return r.s.start; } },
      { k: 'lab', t: 'sound', h: function (r) { return esc(r.s.label); }, v: function (r) { return r.s.label; } },
      { k: 'time', t: 'time', num: 1, h: function (r) { return ft(r.s.start) + '–' + ft(r.s.end) + ' s'; }, v: function (r) { return r.s.start; }, cls: 'nowrap' },
      { k: 'kind', t: 'type', h: function (r) { return kindTag(r.s); }, v: function (r) { return soundKind(r.s); } },
      { k: 'imp', t: 'imp.', num: 1, h: function (r) { return r.s.importance; }, v: function (r) { return r.s.importance; } },
      { k: 'vis', t: 'visible / obvious', h: function (r) { return (r.s.visible ? 'yes' : 'no') + ' / ' + (r.s.obvious ? 'yes' : 'no'); } },
      { k: 'o', t: 'ours', h: function (r) { return outcomeText(c, 'ours', r.i); } },
      { k: 'b', t: 'without gate', h: function (r) { return outcomeText(c, 'blind', r.i); } },
      { k: 'g', t: 'gate near the start', h: function (r) { return (c.derived.near_gate[String(r.i)] || []).map(function (k) { return gateLine(g[k]); }).join('') || '<span class="muted">—</span>'; } },
      { k: 'heard', t: 'what the detectors heard (−0.5 to +1.0 s)', h: function (r) { return heardCell(c, r.i); }, cls: 'heardcol' }
    ], sndRows);

    var opRows = (o.pictures || []).map(function (p, j) { return { c: c, p: p, j: j, sys: 'ours', d: c.derived.ours_pics[j] || {} }; });
    mkTable(document.getElementById('t-op'), 'c-op', [
      { k: 'pic', t: 'picture', h: function (r) {
        if (r.d.aug == null) return '<span class="muted">—</span>';
        return '<div class="thumb"><img loading="lazy" alt="picture ' + r.d.aug + '" src="' + esc(picSrc(c, r.d.aug)) + '" onerror="this.parentNode.hidden=true"></div>';
      } },
      { k: 'play', t: 'play', h: function (r) { return playBtnLocal(r.p.start); }, v: function (r) { return r.p.start; } },
      { k: 'lab', t: 'picture of', h: function (r) { return esc(r.p.label); }, v: function (r) { return r.p.label; } },
      { k: 'time', t: 'shown', num: 1, h: function (r) { return ft(r.p.start) + '–' + ft(r.p.end) + ' s'; }, v: function (r) { return r.p.start; }, cls: 'nowrap' },
      { k: 'cls', t: 'class', h: function (r) { return classTag(r.p.class); }, v: function (r) { return pclass(r.p.class).k; } },
      { k: 'there', t: 'what was there', h: thereText },
      { k: 'gate', t: 'gate (VLM)', h: gateForPic },
      { k: 'aug', t: 'picture made from', h: function (r) {
        var a = r.d.aug != null ? (o.augmentations || [])[r.d.aug] : null;
        return a ? esc(a.subject || a.event_label) + (r.d.aug_approx ? ' <span class="muted small">(nearest)</span>' : '') + '<br><span class="muted small">#' + r.d.aug + ' · ' + esc(a.reason) + '</span>' : '—';
      } }
    ], opRows);

    var bpRows = (b.pictures || []).map(function (p, j) { return { c: c, p: p, j: j, sys: 'blind' }; });
    mkTable(document.getElementById('t-bp'), 'c-bp', [
      { k: 'play', t: 'play', h: function (r) { return playBtnLocal(r.p.start, 'blind'); }, v: function (r) { return r.p.start; } },
      { k: 'lab', t: 'picture of', h: function (r) { return esc(r.p.label); }, v: function (r) { return r.p.label; } },
      { k: 'time', t: 'shown', num: 1, h: function (r) { return ft(r.p.start) + '–' + ft(r.p.end) + ' s'; }, v: function (r) { return r.p.start; }, cls: 'nowrap' },
      { k: 'cls', t: 'class', h: function (r) { return classTag(r.p.class); }, v: function (r) { return pclass(r.p.class).k; } },
      { k: 'there', t: 'what was there', h: thereText },
      { k: 'gate', t: 'in ours?', h: gateForPic }
    ], bpRows);

    mkTable(document.getElementById('t-gate'), 'c-gate', [
      { k: 'play', t: 'play', h: function (r) { return playBtnLocal(r.stretch[0]); }, v: function (r) { return r.stretch[0]; } },
      { k: 'lab', t: 'sound', h: function (r) { return esc(r.label); }, v: function (r) { return r.label; } },
      { k: 'st', t: 'stretch', num: 1, h: function (r) { return ft(r.stretch[0]) + '–' + ft(r.stretch[1]) + ' s'; }, v: function (r) { return r.stretch[0]; }, cls: 'nowrap' },
      { k: 'v', t: 'verdict', h: function (r) { return verdict(r); }, v: function (r) { return r.seen ? 1 : 0; } },
      { k: 'n', t: 'VLM saw', h: function (r) { return esc(r.named); }, v: function (r) { return r.named; } },
      { k: 'votes', t: 'votes (name · a/b · description)', h: function (r) { return esc(votes(r)); } }
    ], g);
    mkTable(document.getElementById('t-ev'), 'c-ev', [
      { k: 'lab', t: 'label', h: function (r) { return esc(r.label); }, v: function (r) { return r.label; } },
      { k: 'st', t: 'time', num: 1, h: function (r) { return ft(r.start) + '–' + ft(r.end) + ' s'; }, v: function (r) { return r.start; }, cls: 'nowrap' },
      { k: 'cf', t: 'score', num: 1, h: function (r) { return f2(r.confidence); }, v: function (r) { return r.confidence; } }
    ], o.events || []);
    mkTable(document.getElementById('t-aug'), 'c-aug', [
      { k: 'i', t: '#', num: 1, h: function (r) { return r.i; }, v: function (r) { return r.i; } },
      { k: 'lab', t: 'sound', h: function (r) { return esc(r.a.event_label); }, v: function (r) { return r.a.event_label; } },
      { k: 'st', t: 'time', num: 1, h: function (r) { return ft(r.a.start) + '–' + ft(r.a.end) + ' s'; }, v: function (r) { return r.a.start; }, cls: 'nowrap' },
      { k: 'aug', t: 'drawn?', h: function (r) { return r.a.augment ? 'yes' : 'no'; }, v: function (r) { return r.a.augment ? 1 : 0; } },
      { k: 'sub', t: 'picture subject', h: function (r) { return esc(r.a.subject || ''); } },
      { k: 'why', t: 'reason', h: function (r) { return '<span class="small">' + esc(r.a.reason) + '</span>'; } }
    ], (o.augmentations || []).map(function (a, i) { return { a: a, i: i }; }));

    function playBtnLocal(t, v) {
      return '<button type="button" class="seek" data-t="' + t + '" data-v="' + (v || '') + '" title="Jump the video above to this moment">&#9654; ' + ft(t) + ' s</button>';
    }
    CUR = { c: c, video: video, note: note };
  }

  // one handler for the clip view's play buttons (they seek the video above instead of opening the pop-up)
  var CUR = null;
  document.addEventListener('click', function (e) {
    var sb = e.target.closest && e.target.closest('button.seek');
    if (!sb || !CUR || !document.body.contains(CUR.video)) return;
    var video = CUR.video, c = CUR.c;
    var t = parseFloat(sb.getAttribute('data-t'));
    var v = sb.getAttribute('data-v');
    if (v === 'blind' && (video.getAttribute('src') || '').indexOf('_blind.mp4') < 0) {
      Array.prototype.forEach.call(document.querySelectorAll('#v-tabs button'), function (x) { x.className = x.getAttribute('data-v') === 'blind' ? 'on' : ''; });
      setVideo(video, CUR.note, media(c.split, c.clip, 'blind'), t);
    } else if (video.readyState >= 1) {
      video.currentTime = Math.max(0, t);
      var pr = video.play(); if (pr && pr.catch) pr.catch(function () {});
    } else {
      setVideo(video, CUR.note, video.getAttribute('src'), t);
    }
    var pl = document.querySelector('.player'); if (pl) pl.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  // ------------------------------------------------------------------ AUDIO-TO-IMAGE (existing approach)
  // window.A2I from a2i.js (build.py wraps a2i.json): {model, window_s, note, clips: [{clip, split, video, windows: [{k, start, end, image}]}]}
  var A2I_SENTENCE = 'An audio-to-image model draws one picture per window from the sound alone — no timing, no labels, and no check whether the source is already on screen.';
  function a2iSrc(ac, w) {
    var im = w.image ? String(w.image).replace(/\\/g, '/') : '';
    if (im && !/^([a-z]+:|\/)/i.test(im)) return im.split('/').map(encodeURIComponent).join('/');   // a path relative to docs/inspector
    return 'media/A2I/' + encodeURIComponent(ac.clip) + '/' + encodeURIComponent(w.k) + '.png';
  }
  function renderA2I(name) {
    var A = window.A2I;
    var h = '<h2>Audio-to-image (existing approach)</h2><p class="lede">' + esc(A2I_SENTENCE) + '</p>';
    if (!A || !(A.clips || []).length) {
      app.innerHTML = h + '<div class="warn"><b>Not ready yet.</b> The audio-to-image pictures are still being made. When <code>docs/inspector/a2i.json</code> is there, ' +
        'run <code>python docs/inspector/build.py</code> and reload this page.</div>';
      return;
    }
    var clips = A.clips;
    var ac = clips.filter(function (x) { return x.clip === name; })[0] || clips[0];
    var c = BYKEY[(ac.split || 'TEST') + '/' + ac.clip];
    if (!c) CLIPS.forEach(function (x) { if (!c && x.clip === ac.clip) c = x; });
    h += '<div class="card"><dl class="kv"><dt>model</dt><dd><b>' + esc(A.model || '—') + '</b></dd>' +
      '<dt>window</dt><dd>' + esc(A.window_s != null ? A.window_s + ' s' : '—') + '</dd>' +
      (A.note ? '<dt>note</dt><dd>' + esc(A.note) + '</dd>' : '') + '</dl></div>';
    h += '<div class="reasons" style="margin-top:12px">' + clips.map(function (x) {
      return '<a class="chip' + (x === ac ? ' on' : '') + '" href="#/a2i/' + encodeURIComponent(x.clip) + '">' + esc(x.clip) + '</a>';
    }).join('') + '</div>';
    h += '<div class="clip-head">' + splitTag(ac.split || (c && c.split) || '') + '<h2>' + esc(ac.clip) + '</h2>' + ((ac.split || (c && c.split)) === 'TEST' ? badge() : '') +
      (c ? ' <a href="' + clipHref(c) + '">open the clip view</a>' : '') + '</div>';
    h += '<div class="clip-grid"><div><p class="small muted" style="margin:0 0 4px">Ours (the clean video a viewer sees)</p>' +
      '<div class="player"><video id="v" controls preload="metadata" playsinline></video><div class="vnote" id="v-note" hidden></div></div></div>' +
      '<div class="card"><p class="small muted" style="margin:0 0 6px">Audio-to-image picture for the moment the video is at</p>' +
      '<div class="a2inow"><img id="a2i-now" alt="audio-to-image picture for this window"><div class="vnote" id="a2i-miss" hidden>No picture for this moment.</div></div>' +
      '<p class="small" id="a2i-cap"></p></div></div>';
    h += '<h3 style="margin-top:16px">Timeline</h3><p class="small muted">Same time axis for both approaches. The top strip is the audio-to-image pictures; ' +
      'the highlighted one is where the video is now. Below: the sounds a human marked, and what ours showed. Click anywhere to jump the video there.</p>' + tlLegend(false);
    h += '<div class="card" style="padding:10px"><div id="tl"></div></div>';
    app.innerHTML = h;
    var video = document.getElementById('v'), note = document.getElementById('v-note');
    var wins = (ac.windows || []).slice().sort(function (a, b) { return a.start - b.start; }).map(function (w) {
      return { k: w.k, start: +w.start, end: +w.end, src: a2iSrc(ac, w) };
    });
    var img = document.getElementById('a2i-now'), miss = document.getElementById('a2i-miss'), cap = document.getElementById('a2i-cap');
    var cur = -2;
    function onTime(t) {
      var k = -1;
      wins.forEach(function (w, i) { if (t >= w.start && t < w.end) k = i; });
      if (k === cur) return;
      cur = k;
      if (k < 0) { img.removeAttribute('src'); img.hidden = true; miss.hidden = false; cap.textContent = ''; return; }
      img.hidden = false; miss.hidden = true;
      img.onerror = function () { img.hidden = true; miss.hidden = false; miss.textContent = 'Picture not ready yet: ' + wins[k].src; };
      img.src = wins[k].src;
      cap.textContent = 'window ' + wins[k].k + ': ' + ft(wins[k].start) + '–' + ft(wins[k].end) + ' s';
    }
    setVideo(video, note, c ? media(c.split, c.clip, 'ours') : (ac.video || ''), null);
    var tl = document.getElementById('tl');
    if (c) {
      var endW = wins.reduce(function (m, w) { return Math.max(m, w.end); }, 0);
      var opts = { lanes: ['a2i', 'sounds', 'ours'], a2i: wins, onTime: onTime, dur: Math.max(duration(c), endW) };
      drawTimeline(tl, c, video, opts);
      REDRAW.push(function () { if (document.body.contains(tl)) drawTimeline(tl, c, video, opts); });
    } else {
      tl.innerHTML = '<p class="muted">This clip is not in data.js, so there are no human labels or pictures of ours to compare.</p>';
      video.ontimeupdate = function () { onTime(video.currentTime || 0); };
      onTime(0);
    }
  }

  // ------------------------------------------------------------------ DEFINITIONS
  function renderDefs() {
    var h = '<h2>Definitions</h2><p class="lede">The words used on this page, in short. The full rules are in the thesis, chapter 4 (benchmark) and chapter 5 (results).</p><dl class="defs">';
    var defs = [
      ['Gold sound', 'A non-speech, non-music sound that the annotator marked in a clip: its name, when it starts and ends, and three ticks (visible, obvious, importance). Example: “Siren, 12.0–18.0 s”.'],
      ['Visible', 'You can see the thing making the sound on screen. Example: a dog barking in the middle of the picture.'],
      ['Obvious', 'The source is not on screen, but a viewer with no sound would still know the sound is happening now. Example: fireworks lighting up the sky, the bangs are off screen.'],
      ['Needed', 'A sound that is <em>neither visible nor obvious</em>. A deaf or hard-of-hearing viewer would miss it without help, so the system should show a picture. Example: a siren from a street you cannot see.'],
      ['Importance (1–3)', '1 = steady background noise of the place, no clear start (city hum). 2 = an event you can say in one sentence (a door slams). 3 = danger or a key story moment (a gunshot). Only needed sounds of importance 2–3 are scored; a needed sound of importance 1 is “don’t care”.'],
      ['Hit and the hit window', 'A picture is a <em>hit</em> when it shows the same family of sound (the class, a parent or a child in the AudioSet list; “Truck” fits “Vehicle”) and <em>starts</em> between 0.5 s before and 1.0 s after the sound starts. Example: a siren starts at 12.0 s; a siren picture at 12.8 s is a hit, one at 13.5 s is late (a miss). Each sound takes at most one picture.'],
      ['Miss', 'A needed sound (importance 2–3) with no hit.'],
      ['Wrong picture: source visible', 'The picture matches a real sound, but its source was visible or obvious. The gate should have stopped it. Example: a picture of a car horn while the car is on screen.'],
      ['Wrong picture: different sound', 'A real sound was there at that moment, but the picture shows another kind of sound. Usually the detector named it wrong. Example: a gunshot picture during an explosion.'],
      ['Wrong picture: no such sound', 'Nothing in the gold list was happening then. The detector heard something that is not there (or not in the gold list).'],
      ['Duplicate', 'A second picture for a sound that already has one. Not counted as wrong and not as a hit.'],
      ['Gate', 'Step 5 of the system. For each detected sound and each stretch of up to 5 s, six frames go to a vision-language model (VLM, Qwen3.8-27B) with three questions: name the source, an a/b choice (“you can see it happening” or not), and a description. The majority of the three votes gives the stretch verdict. A sound is silenced only if every stretch says “visible”.'],
      ['Ours', 'The full system: detector → gate → pictures. It draws a picture only when the VLM says the source is not visible.'],
      ['Pipeline without gate', 'The same detector stack, label filter, subject text and pictures as ours; only the visibility gate is removed — it draws every detected sound. Comparing ours with it shows what the gate adds. (Called “blind” in older notes and in file names.)'],
      ['Existing approach (Overview)', 'One detector alone and a picture for every sound it detects above one threshold, chosen on DEV. No gate, no second detector. Two are shown: PretrainedSED (CP-JKU, ICASSP 2025; the state-of-the-art frame-level detector on AudioSet-Strong) and PANNs (older, widely used).'],
      ['What the detectors heard', 'For each gold sound: the 5 highest scores of BEATs, FlexSED and PANNs from 0.5 s before to 1.0 s after the sound starts, and <em>own</em> = the highest score of the sound\'s own family. ' + esc(HEARD_NOTE) + ' The bar is the score the system needs to use a detection (BEATs 0.35, FlexSED 0.8; PANNs has no bar, it is only used as a veto).'],
      ['Silence', 'Shows nothing. It never shows a wrong picture but misses every needed sound.'],
      ['Precision, recall, F1', 'Precision = hits / all pictures shown. Recall = hits / needed sounds. F1 = one number that is high only when both are high (their harmonic mean).'],
      ['Viewer cost and β', 'A modelled price per clip: 4 for each missed needed sound plus β for each wrong picture, with β = 2 assumed. Lower is better. Example: 1 miss and 2 wrong pictures cost 4 + 2 × 2 = 8. No one has measured β for deaf viewers yet, so the thesis shows the whole curve over β in chapter 5 §5.5 (on TEST, ours is the cheapest for β between 0.39 and 2.56).'],
      ['DEV, TEST, Slice B', 'DEV (49 clips) was used to build and tune the system. TEST (60 clips) was only scored at the end; <em>look only, do not tune</em>. Slice B (30 AudioSet clips where speech or music covers the sound) is reported apart.'],
      ['Clip type', 'off-screen: every important sound is needed. mixed: at least one needed and one seen sound (importance 2–3). on-screen: sounds are there but none is needed. nothing to draw: no non-speech, non-music sound.'],
      ['Early / late picture (Mistakes → Late or early pictures)', 'Where a picture of the right kind starts, compared with the sound: <em>early</em> = more than 0.5 s before the start; <em>in the window</em> = a hit; <em>late, sound still playing</em> = more than 1 s after the start but before the sound ends; <em>after the sound ended</em>. Only the window counts. A late picture counts as a miss plus a wrong picture. The Overview has a sensitivity table where “late, sound still playing” also counts.'],
      ['Miss reasons (Mistakes page)', 'Checked in this order. <em>removed by the gate</em>: the pipeline without gate drew it in time, ours did not. <em>timing</em>: a picture of that family overlapped the sound but started outside the window. <em>detected, not drawn</em>: the detector heard it but no picture was made. <em>never detected</em>: the detector did not hear it. These are a diagnosis for explaining, not part of the score.']
    ];
    defs.forEach(function (d) { h += '<dt>' + d[0] + '</dt><dd>' + d[1] + '</dd>'; });
    h += '</dl><p class="small muted" style="margin-top:20px">' + esc(D.note || '') + '</p>';
    app.innerHTML = h;
  }

  // ------------------------------------------------------------------ router and theme
  var SKIP_ROUTE = false;
  var TABS = ['miss', 'late', 'gated', 'wrong', 'gate'];
  function route() {
    if (SKIP_ROUTE) { SKIP_ROUTE = false; return; }
    REDRAW = [];
    tip.style.display = 'none';
    var h = location.hash.replace(/^#\/?/, '');
    var parts = h.split('?')[0].split('/');
    var page = parts[0] || 'overview';
    Array.prototype.forEach.call(document.querySelectorAll('#nav a'), function (a) {
      var p = a.getAttribute('data-p');
      a.className = (p === page || (page === 'clip' && p === 'clips')) ? 'on' : '';
    });
    if (page === 'mistakes') {
      if (parts[1] === 'never') { M.tab = 'miss'; M.reason = 'never detected'; M.sys = 'ours'; }   // link from the Overview
      else if (TABS.indexOf(parts[1]) >= 0) M.tab = parts[1];
      renderMistakes();
    }
    else if (page === 'clips') renderClips();
    else if (page === 'clip') renderClip(decodeURIComponent(parts[1] || ''), decodeURIComponent(parts.slice(2).join('/') || ''));
    else if (page === 'a2i') renderA2I(decodeURIComponent(parts.slice(1).join('/') || ''));
    else if (page === 'defs') renderDefs();
    else renderOverview();
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', route);

  var themeBtn = document.getElementById('theme');
  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function syncThemeBtn() { themeBtn.textContent = isDark() ? 'Light' : 'Dark'; }
  themeBtn.addEventListener('click', function () {
    var t = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('inspector-theme', t); } catch (e) { /* private window: fine */ }
    syncThemeBtn();
  });
  syncThemeBtn();

  // expose a little for checking in the console
  window.INSPECTOR_PAGE = { pooled: pooled, funnel: funnel, checks: CHECKS, setVideo: setVideo };
  route();
})();

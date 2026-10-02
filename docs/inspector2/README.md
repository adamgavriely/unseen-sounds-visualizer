# Decision Inspector (inspector 2)

For each needed sound and each picture, the inspector shows where the decision happened, at the exact step. It names the
measured value and the bar it was compared with. Where a model was asked, it shows the exact question and the exact answer.
Examples: "dropped at DASM local veto: DASM max 0.21 < bar 0.35; Qwen3-Omni said yes, Audio Flamingo said 'engine'", or
"silenced by the visibility gate: Q-name answered 'church bell' (visible), Q-ab answered (a) visible, Q-desc answered visible
→ 3 of 3 seen".

Open `index.html` by double-click. It needs no server, since the data loads as a script (`data.js`). Videos load from
`media/<split>/<clip>.mp4`, next to the page.

Page tabs: **Overview**, **Build** (the current version in five plain parts and the version chain), **Tried** (ideas that did not ship and why), (where needed sounds are lost, step by step, and which steps the wrong pictures passed),
**Misses**, **Wrong pictures**, **Clips** (video + timeline + every candidate), **Steps** (what each step does, its bar,
its question).

## Data contract (`data.js` = `window.INSPECTOR2 = {...}`)

```
{
  "meta":  {"version": "D′", "arm": "SHIP8+MD3+WW5+SL", "built": "<ISO time>", "sample": false,
            "window": [-0.5, 1.0], "cost": "(4·miss + 2·wrong)/clips"},
  "sets":  {"DEV": {"clips": 71, "needed": 58, "hits": 29, "wrong": 15, "visible": 6, "cross": 7, "phantom": 2, "cost": 2.056},
            "TEST": {...}},
  "steps": [ {"id": "dasm_local", "stage": 4, "name": "DASM local veto", "plain": "one sentence",
              "bar": "DASM max ≥ 0.35, or both listeners name it", "model": "DASM",
              "question": null} , ... ]                       // pipeline order; ids are referenced below
  "clips": [ {
      "clip": "bell_miami", "split": "DEV", "dur": 15.0, "video": "media/DEV/bell_miami.mp4",
      "gold":  [ {"id": "g1", "label": "Bell", "start": 0.2, "end": 14.5, "needed": true, "visible": false,
                  "outcome": "miss",                          // hit | miss (needed only)
                  "lost_at": "gate",                          // step id where the LAST candidate for it died (misses)
                  "why": "one line, plain",                   // shown in lists
                  "cands": ["c3", "c4"] } ],                  // candidate ids in its window (same family)
      "pictures": [ {"id": "p1", "label": "Thunder", "start": 0.06, "end": 2.0, "verdict": "visible",
                     "gold": "g2", "cand": "c1"} ],           // verdict: hit | visible | cross | phantom
      "cands": [ {"id": "c3", "label": "Church bell", "start": 0.22, "end": 3.1, "origin": "beats|flexsed|rescue|...",
                  "fate": "dropped|drawn|merged", "at": "gate",
                  "trail": [ {"step": "beats_bar", "res": "pass", "value": "0.70", "bar": "≥ 0.175"},
                             {"step": "gate", "res": "drop",
                              "asks": [ {"q": "exact question text", "a": "exact answer", "vote": "seen"} ],
                              "note": "3 of 3 seen (family rule: same label)"} ] } ]
  } ]
}
```

`res` is one of `pass`, `drop`, `move`, `merge`, `relabel`, `rescue`, `skip`. Only steps the candidate reached are listed.
Steps that do not apply to it (for example a listener rescue on a strong span) are written as `skip` with a short note, or
left out.

## How the data is made

1. **Pipeline log.** Each decision point in the shipped pipeline calls `src/trail.py: decide(...)`, which records the step id,
   the result, the value, the bar and any question and answer. The trail goes to `trail.json` in the clip's work folder.
   This replaces the partial `onset_trace.json`, where several vetoes only print to stdout. It is added once the detector is
   frozen, so the scores cannot move. The parity check must reproduce the shipped DEV and TEST numbers exactly.
2. **Export.** `python docs/inspector2/export.py --arm <arm>` reads every clip's `trail.json`, the gold file and the scorer
   (`benchmark/gold/score_per_sound`), matches candidates to gold sounds, and writes `data.js`.
3. **Videos.** Rendered once per shipped version into `media/<split>/<clip>.mp4`. A clip is re-rendered only when its
   pictures change, using the display span signature (label, start, end) as the key.

`data.js` holds the real D′ data (159 clips), written on the cluster by `benchmark/gold/inspector_trail_export.py`
(see `docs/review/inspector_trail_ready.md`; parity exact). Videos: `../inspector/media/bysig/<split>/<clip>.<sig>.mp4`.
`content.js` (Build and Tried tabs) and `data_sample.js` (the old sample) are written by `sample_data.py`; regenerating
them never touches `data.js`. The Overview's "What this data does not show" table lists the known gaps in the log.

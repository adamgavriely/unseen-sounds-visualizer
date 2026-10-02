# Decision Inspector: trail data ready (D′, 2 Oct 2026)

**Data file:** `docs/inspector2/data.js` (`window.INSPECTOR2`, the contract in the docs/inspector2 README; 12.8 MB, 159 clips).
It sits where `export.py` writes by default. On branch `claude/project-thread-4rwrnp` a merge gives an add/add conflict on this
file: take this (main) version, because the branch copy is the sample.
Next to it: `data_parity.json` (parity result) and `data_media_sigs.json` (per-clip picture signature, which is also the video key).

**Videos:** `data.js` points each clip at `../inspector/media/bysig/<split>/<clip>.<hash>.mp4`. The hash is
sha1 of the clip's picture list [(label, start, end)], first 10 hex digits. A `.sig.json` sits next to each video. A clip
is re-rendered only when its signature changes, so older versions' videos stay. They are rendered by
`benchmark/gold/render_trail_media.py` (job `slurm/job_trail_media.sh`) and copied to `docs/inspector/media/bysig/`.

**Media status (2 Oct):** 159 videos (DEV 71, TEST 88), 350 MB, in `docs/inspector/media/bysig/{DEV,TEST}/`. The folder is
gitignored. All 159 passed the check that the compositor's display spans equal the clip's signature, and every `data.js`
video path resolves. The videos use the shipped display (`use_shipped`: a picture stays at most 1.0 s past the sound's
real end). The scoring harness leaves that limit off, so on 3 TEST clips the scored picture ends 0.04-0.38 s later than
in the video. Starts and labels are identical, and only starts are scored. The video key therefore uses the shipped
display (`SHIPPED_MAX_AFTER_END` in the exporter).

## How it was made
- Logging calls (`src/trail.py`, `src/trail_ctx.py`) sit at every D′ decision site of docs/inspector2/HOOKS.md. They only log.
  The DEV/TEST harness writes `trail_s4.json` + `trail.json` per clip. The exporter adds the stage-6 display records, which it
  takes from the scorer's own picture step, and writes `trail_full.json` per clip on the cluster.
- Arm `SHIP8+MD3+WW5+SL_trail` has D′'s flags (`config.use_shipped()`, tag detector-frozen-2026-10-02), with new output folders.
- **Parity: EXACT.** Merged DEV 29/58, 15 wrong (6/7/2), cost 2.056; merged TEST 24/65, 24 wrong (4/15/5), cost 2.409.
  Every clip's pictures are identical to the frozen run (0 of 71 DEV and 0 of 88 TEST clips differ).
- Exporter: `python benchmark/gold/inspector_trail_export.py --expect DEV=29/58/15/6/7/2/2.056 --expect TEST=24/65/24/4/15/5/2.409`
  (run on the cluster from ~/MscProj_tg).

## Counts
| set | misses | miss lost at a named step | never heard | timing / matching ("scorer") | wrong pictures | wrong with a full trail |
|---|---|---|---|---|---|---|
| DEV | 29 | 20 | 6 | 3 | 15 | 15 |
| TEST | 41 | 32 | 6 | 3 | 24 | 23 |

Where DEV misses are lost: band rescue 6, gate 5, dasm vote 2, family merge 2, mirror veto 2, scene margin 1, masked weak 1,
rescue once 1. TEST: continuation veto 5, band rescue 6, family merge 4, gate 3, mirror veto 3, dasm vote 2, beats extract 2,
and one each for dasm local veto, finelap veto, group, masked weak, dasm rescue, scene margin and k4a inventory.

## Decision points that still have no full reason
1. **Gate raw answers, reused stretches.** In the harness, a gate stretch that the stored gate run already judged reuses
   that run's verdict. For those the trail has the three votes and the named object, but not the raw describe answer or the
   a/b letters. 70 of 156 gate decisions have every raw answer, and 86 have votes only (the trail's note says so). The
   live pipeline (`pipeline.run`) stores all raw answers. Filling the 86 needs a separate re-ask job; that job would only log.
2. **DEPICT look-alike replies (L1/L2).** The cache stores only "any maker said yes twice", not the replies. The E1/E2
   event replies are shown.
3. **Never heard (6 DEV, 6 TEST).** No detector produced a span of the family near the onset, so no step decided anything.
   The detectors' sub-bar scores are not in the trail. The old inspector's `heard` field (top-5 per detector) covers this.
4. **Timing / matching (3 DEV, 3 TEST).** A same-family span existed, but no span starts within 1 s of the onset, or the
   picture is outside the −0.5…+1.0 s window. The note names the nearest span and its fate.
5. **One TEST wrong picture (cross)** has no candidate linked to it: no kept span of its family overlaps the picture in the
   exported candidates, so its page shows the picture without a trail.

# Pre-registration: detector round 12 — a corrected AudioSet cost ("v2"), all old cells re-scored, four new cells

*Written 2026-09-28, BEFORE any v2 number was computed and before any round-12 cell existed. Adam (28 Sept): "do
everything you think is good to improve our detector (within reason)"; he approved (a) counting runs of the same sound as
one event and (b) using the video benchmark's time window as tolerance. Harness: `benchmark/detector_round12.py`; results:
`benchmark/detector_round12.json`; jobs: `slurm/job_round12_*.sh`. Nothing on TEST (`test_bench`). Nothing on slice B
(see the note in Step 5).*

## Why a new cost, and why it is not fitted to outcomes
Nine rounds of detector ideas failed on the held-out 415 under the old AudioSet cost C. The setup audit
(`docs/history/analyses/setup_audit_2026-09-28.md`) found no bug in the audio, the time stamps or the harness, but it found that C does not
measure the task the product is judged on: (1) 10 gold labels were matched by display name, not by sound ID; (2) C used
the "lists" label filter, but the product draws with the "depictable" filter; (3) AudioSet-Strong labels every bark as its
own event, while our own gold rule is one row per continuous sound; (4) C had no time tolerance, while the video benchmark
accepts a picture that starts 0.5 s early to 1.0 s late. Under (3)+(4), a correct but slightly late span costs more than
showing nothing.

**v2 is defined after nine rounds and after the audit.** That is a risk: a cost chosen after seeing results can be bent
to favour an idea. To avoid that, every v2 choice below is copied from a rule the project already had before round 12
(the audit's bug fix, the shipped filter, the DEV gold annotation rule, the frozen video scorer). No choice is tuned, no
alternative is tried to see which one "works". The audit already published two parts separately on the 280 (depictable +
name fix 2.643; merged runs 2.050, both for the shipped spans); the full v2 combination and the time tolerance have never
been computed.

## The v2 cost (fixed here)
Per clip, in this order:
1. **Gold names by sound ID (MID).** Every gold label whose AudioSet-Strong display name differs from the ontology / BEATs
   name for the same MID is renamed to the ontology name, and its "consequential" flag is recomputed with the builders'
   own list (`benchmark/gold/audioset_slice.CONSEQUENTIAL`) — `setup_audit.strong_to_beats()` + `remap_clip()` (the
   audit's bug fix, check 2).
2. **Label filter = the shipped "depictable" filter** (`config.LABEL_FILTER = "depictable"`, the value `use_shipped()`
   runs with). A gold event is scored only if it is consequential AND the shipped filter would draw its label
   (`is_salient_nonspeech` under "depictable") AND it is not music. The same filter is applied to the predicted spans.
   This settles the audit's open questions by rule, not by choice: "Beep, bleep" maps to the family Alarm and is drawn, so
   it stays; "Rumble" is a texture and is never drawn, so it leaves.
3. **Runs merged.** The scored events are grouped by canonical family (`src.labels.canonical`); within a family, events
   are sorted by start and an event joins the previous one when the pause (its start − the previous end) is **≤ 2.0 s**.
   Source of the rule: our DEV/TEST gold annotation rule "one row for a continuous sound … split only when it stops for
   > 2–3 s" (`docs/history/daily_notes/HANDOFF_2026-09-21.md`, tagging rules) and its code form `config.MERGE_GAP = 2.0` ("2 s = the
   annotation rule 'one row per continuous sound, split at pauses > 2 s'"). The lower end of the rule's range (2 s) is
   used, as `MERGE_GAP` does. Merging is done after the filter (step 2), because the filter reads the raw label first.
   A merged event keeps every member label (a span matches it if it matches any member).
4. **Time tolerance = the video benchmark's hit window** (`benchmark/gold/score_per_sound.py`: `EARLY, LATE = 0.5, 1.0`).
   - **C-onset v2:** a merged event is hit iff some same-family predicted span **starts** in [onset − 0.5, onset + 1.0] s.
   - **C-overlap v2 (primary):** a merged event is hit iff some same-family span **overlaps it** (any positive overlap;
     the old minimum of 0.5 s or half the event is dropped) **or** starts in its hit window. So C-overlap v2 ≤ C-onset v2
     on every clip: C-overlap is the lenient "any timing within tolerance" view, C-onset the strict video rule.
   - **False span:** a predicted span is false iff it matches no gold event of its family: it neither overlaps nor starts
     in the hit window of any same-family gold event. The gold list for this test = the merged scored events + every other
     gold event of the clip (renamed by MID, any label, consequential or not), so a span inside a ≤ 2-s pause of a run is
     not false. Source: in `score_per_sound.score_clip` a picture whose start lies in a same-family onset window is never a
     false alarm (it is a hit, or a duplicate, which costs nothing). The old rule ("overlaps no same-family event") is
     reported beside it as a labelled sensitivity row for the baseline and for "show nothing".
5. **Family matching unchanged:** `audioset_detector_eval._same` (same label, same canonical family, or ancestor /
   descendant in `src/audioset_parents.json`, which keeps ONE parent per label — the frozen video scorer uses the same
   file). Limitation: 38 labels have more than one official parent; the audit measured the effect as small (C 3.036 →
   2.986, 5 false spans).
6. **Cost per clip = 4 × missed events + 2 × false spans**, as before; the set's C = the mean over clips. Clip lists are
   unchanged (`detector_round2.usable()`).
Bootstrap everywhere: paired over clips, 2000 draws, seed 0 (`detector_round8.boot8`: mean, 95 % CI, one-sided p = share of
draws with mean Δ ≥ 0). Verdict words in every table, against **ours** (the shipped stack) on the same clips: **better**
= upper 95 % CI of Δ < 0; **worse** = lower 95 % CI > 0; **same** = otherwise.

Known gap, disclosed: the harness spans have no occlusion onset refinement (the product's starts are ~0.2 s later, audit
check 4). The time window absorbs this; it is not corrected.

## Step 1 — does the shipped stack beat "show nothing" under v2? (the 280; a STOP rule)
Gate 0 first: the harness baseline reproduces round 5 span for span and the old numbers (C-overlap 3.036, C-onset 3.850,
207 false spans, 564 shown; `detector_round8.gate_set`). Then, on the 280 under v2: ours (the shipped stack) vs "show
nothing" (no spans). **Ours beats nothing iff the upper 95 % CI of ΔC-overlap v2 (ours − nothing) < 0.** If it does not,
the round **STOPS** here: the table is reported, and the report says plainly that this AudioSet harness cannot track the
task, and that the video benchmark (DEV/TEST gold, `score_per_sound`) stays the only test. v2 is not changed after this
number is seen. Reported rows: ours, nothing, ours with the old false-span rule, and each v2 part switched on one at a time
(MID names; + depictable; + merged runs; + window) so the reader sees what each part does.

## Step 2 — every earlier cell re-scored under v2 (the 280, CPU, saved caches; no refit)
Each cell is rebuilt with its own round's code and its frozen values read from that round's JSON:
- **Round 4:** "PANNs veto" (the stack before round 4, PANNs clip-max 0.05 on FlexSED-only spans) and "T1 end-trim". Round 4
  ran the trim on the PANNs stack; here it is applied to **the shipped stack** (self-veto), the only baseline now.
- **Round 5:** EAT-P, EAT-R, Dasheng-P, Dasheng-R (aed, display bar and b from `detector_round5.json → fit`).
- **Round 6:** DASM D1, D2 (g, v from `detector_round6.json → bars`).
- **Round 8:** I2 (6 cells), I3, I4, I5, I6, I7, I8, I9, I10, I1-i, I1-ii, I1-iii (I3 d, I5 d1, I1-iii m from
  `detector_round8.json → params`; VLM answers `round8_vlm.json`).
- **Round 9:** J1, J2.  **Round 10:** R0 (reference), R1–R7.  **Round 11:** A0 (reference), M1, M1m, M1b, M2, M3, M5, M7
  (on M2, as registered), C, W — the round-11 re-run caches exist for the 280 only.
Rounds 2/3 (old baselines, superseded) and 7 (SAM-Audio, stopped at QC, no cell) are not re-scored.
Table per cell: C-overlap v2, C-onset v2 (each with Δ vs ours, 95 % CI, verdict), recall (overlap / onset), false/min,
false spans, shown spans; ours and "show nothing" rows on top.

## Step 3 — four new cells (fixed here; all add FlexSED-only spans that must pass DASM agreement)
**DASM agreement** = round 10's R1 rule, frozen: DASM (`<WIN>/dasm_cache`, the 215 queries, 50 fps) scores the span's
family ≥ 0.359375 (= 0.575 × 0.5 / 0.8) at some frame in [start − 0.5, end + 0.5] (nearest frame if none); family columns
= same canonical family (`detector_round10.r1`, match=False) unless stated. Every added span is appended to ours; nothing
of ours is removed or moved.
- **N1 — weak-twin band candidates.** Round 10's band candidates (FlexSED 0.4 spans, low 0.4, min 0.5 s, peak < 0.8,
  BEATs self-veto) with one rule changed as in round 10's reach note: a BEATs 0.175 twin within 1 s discards the candidate
  only if that twin is shown (confidence ≥ 0.35); a weak twin (< 0.35) does not (`detector_round10.cands_opt(twin="shown")`).
  Kept iff DASM agrees. (N1 ⊇ R1.)
- **N2 — short-sound path.** Per FlexSED column, runs of consecutive frames ≥ 0.4 (the same 0.4 level as the band
  candidates, so N1 and N2 never share a run) that are **≥ 2 frames and shorter than 0.5 s** (at 25 fps: 2–12 frames; the
  shipped extractor drops them by its 0.5-s minimum) and whose peak is **≥ 0.5** (no upper bound: a short run above 0.8 is
  dropped by the shipped stack too). Each becomes a **1-s span centred on the peak frame**, [t_peak − 0.5, t_peak + 0.5],
  clipped to [0, clip end], confidence = the peak. Overlapping N2 spans of one family are merged into one (union). A span is
  kept iff (i) ours shows no span of the same canonical family within 1 s of it (so it adds a picture only where ours has
  none), (ii) it passes the BEATs self-veto (BEATs clip-max of the family ≥ 0.1218; a family BEATs lacks passes), and
  (iii) DASM agrees on the 1-s span. Ceiling, written before any number: the reach note found that even a 0.25-s minimum
  reached only 5 of the 22 short band events on the 280, and most short events are single barks and beeps inside runs that
  v2 now merges; N2's headroom is small.
- **N3 — the declared best combination: R6 + R7 + N1 + N2.** Order: (1) round 10's `stack10(i4="conf", i6=("conf",
  listed))` — the I4 parent spans and the I6 lowered-bar spans, each kept only if R1 OR R2 confirms them (as R6 / R7), both
  in one stack; (2) the N1 spans; (3) the N2 spans, with rule (i) read against the output of (1)+(2). An N1 or N2 span that
  overlaps a span of the same canonical family already in the output is dropped (no double picture). Nothing else changes.
- **N4 — ontology-aware vetoes, DASM-filtered.** Round 8's I9 (both vetoes read the other detector's clip-max over every
  label that matches the span's label by `E._same`) keeps a superset of ours. N4 = ours + those extra spans (I9's output
  minus ours, matched by label and time) that pass DASM agreement with **match=True columns** (DASM columns related by
  `E._same`, as R6). This applies to every extra span, whether FlexSED-only or BEATs-origin (a BEATs span rescued from the
  FlexSED clip veto), so one rule covers all additions; no matching DASM column → not kept.
- **Reference rows (never picked, never tested):** N1-all and N2-all (the same spans with no DASM test), so the filter's
  effect is visible. I9 unfiltered is the Step-2 row.

## Step 4 — picks on the 280 and the 415 test (v2)
- **Pool:** every Step-2 and Step-3 cell except the reference rows (R0, A0, N1-all, N2-all). A cell is **eligible** iff its
  C-overlap v2 AND its C-onset v2 are below ours on the 280. **At most 3 picks:** the eligible cells with the lowest
  C-overlap v2 (tie → lower C-onset v2 → the earlier round). A cell whose output equals ours on every clip is not eligible.
- **The 415 (`heldout`), each pick frozen:** passes iff the upper 95 % CI of ΔC-overlap v2 (pick − ours) < 0; **Holm**
  across the picks (one-sided p, family α = 0.025, step-down); a pick counts as passed only if it passes AND Holm rejects.
  Rows: ours, "show nothing", every pick; strata complex / random reported.
- **Disclosure:** the 415 is not fresh for every cell. Under the old C it was already scored for round 4's self-veto (now
  ours), EAT-R, DASM D1, I4, I6, I7, I1-ii and J2. Their 415 results under old C were seen. So for them the 415 is a weaker
  check, and the fresh set (Step 5) is the clean test.
- If a pick needs a cache the 415 does not have (round-11 re-run views), that cache is built first (GPU, cache only, no
  score), with the round's own code.

## Step 5 — only for picks that pass the 415 (outline fixed now; details as a dated amendment BEFORE any DEV number)
Adam's standing rule (via the lead, 28 Sept): **a candidate must be tested the same way the delivered system was tested.**
So the AudioSet v2 cost is only a SCREEN; the main test is the video benchmark:
1. **DEV check (main test).** The full pipeline path: stage 4 (the candidate's added spans rebuilt on each DEV clip from
   the pipeline's caches, DASM on the DEV `audio.wav`) → the stage-5 gate → the shipped display rules, scored with
   `benchmark/gold/score_per_sound.py` exactly as the delivered DEV tables (onset window [−0.5, +1.0], needed sounds rated
   2–3, viewer cost = 4 × misses + 2 × false alarms, wrong-picture types visible / cross / phantom, paired clip bootstrap
   2000 draws seed 0). **Ours is scored by the same code on the same clips, beside it**, for the gated system ("proposed")
   and the system without gate ("blind_a2i"). Adam's rule, made explicit: every candidate is tested on the SAME DEV videos,
   and our shipped detector is **re-run on those videos by the same code in the same job** (not only read from the old
   scored renders), so the comparison is paired and truthful. Baselines, as in `dev_candidates_check.py`: **B1** = the
   shipped stack re-run (BEATs self-veto 0.1218, PANNs off) = the primary baseline; **B0r** = the scored-render repro, which
   must give the scored pictures (gate D5; if D5 fails on a clip, that clip is reported and the check is read with care).
   Candidate vs B1 is reported per clip (pictures added / removed, hit / wrong changes). The machinery is copied from `benchmark/gold/dev_candidates_check.py` into a
   new file with its own work folder (that file and `data/work/devcand/*` belong to job 31330563; they are read, never
   edited). **Ship rule:** DEV hits do not drop AND DEV wrong pictures do not rise by more than 2 × the hits gained AND DEV
   viewer cost does not rise.
2. **Fresh set** (`R.use_set("fresh")`, 422 clips, never scored so far; `docs/history/preregistrations/prereg_fresh_confirm_set.md`): ONE final
   test of the single best DEV-passing candidate (the largest 415 effect among DEV passes), frozen, under v2: passes iff
   the upper 95 % CI of ΔC-overlap v2 < 0; ours and "show nothing" beside it. First: 4 cache folders × 422 `.npz` counted;
   the DASM cache for fresh is built (GPU) if the candidate needs it.
3. **Slice B** (30 video-gold clips): the lead asked whether it could be a second video check. `docs/history/preregistrations/prereg_v4.md` shows it
   was scored many times (the detector tables, the v4b4 / v4ab4 gold re-runs), and its numbers triggered the PSED
   re-comparison (2026-09-19). So it is **not clean**, and the round's brief says "nothing on slice B". It is **not run**
   in this round; at most it could be a report-only row, and only if Adam lifts the brief's line.
4. **Ship** only if the 415 + DEV ship rule + fresh all pass: the rule goes into `src/stage4_audio_event_detection` behind a
   config flag (default off), switched on in `config.use_shipped()`, with a short test. The frozen TEST table is not
   changed (a note only).

## What will not be done
No other gap, window, bar, filter or family rule; no refit of any old cell; no re-pick after the 415; nothing on TEST or
slice B; no edit of any running job's file.

## Result of Step 1 (2026-09-28, job 31330706, CPU; `benchmark/detector_round12.json → step1`) — **STOP**
Gate 0 passed (the harness gives round 5's spans on every clip; old C 3.036 / 3.850, 207 false spans, 564 shown). Check:
with every v2 part off, the new scorer equals `detector_round2.clip_cost` on all 280 clips.

The 280, ours (the shipped stack) and "show nothing" on the same clips, v2 parts switched on one at a time. Δ = ours −
nothing (paired clip bootstrap, 2000 draws, seed 0). Verdict = how ours compares with showing nothing.

| cost variant | ours C-overlap | ours C-onset | nothing | Δ C-overlap (95 % CI) | verdict | Δ C-onset (95 % CI) | verdict | ours false spans | events scored (clips) | ours recall overlap / onset |
|---|---|---|---|---|---|---|---|---|---|---|
| old C (all parts off) | 3.036 | 3.850 | 3.200 | −0.164 [−1.365, +0.800] | same | +0.650 [+0.121, +1.143] | worse | 207 | 224 (59) | 51.3 % / 25.9 % |
| + MID names | 3.064 | 3.879 | 3.286 | −0.221 [−1.436, +0.736] | same | +0.593 [+0.064, +1.086] | worse | 201 | 230 (60) | 50.4 % / 25.7 % |
| + depictable filter | 2.693 | 3.507 | 3.214 | −0.521 [−1.750, +0.414] | same | +0.293 [−0.214, +0.736] | same | 159 | 225 (56) | 51.6 % / 26.2 % |
| + runs merged (≤ 2 s) | 1.664 | 1.750 | 0.986 | +0.679 [+0.393, +0.971] | worse | +0.764 [+0.486, +1.043] | worse | 159 | 69 (56) | 46.4 % / 37.7 % |
| + window hit rule (old false rule) | 1.650 | 1.750 | 0.986 | +0.664 [+0.379, +0.950] | worse | +0.764 [+0.486, +1.043] | worse | 159 | 69 (56) | 47.8 % / 37.7 % |
| **v2** (+ window false rule) | **1.650** | **1.750** | **0.986** | **+0.664 [+0.379, +0.950]** | **worse** | +0.764 [+0.486, +1.043] | worse | 159 | 69 (56) | 47.8 % / 37.7 % |

**By the rule written above, the round stops here.** Under v2, ours does not beat "show nothing" on the 280; it is clearly
worse (upper CI > 0 is not even close: the whole CI is above 0). Steps 2–6 are not run: no old cell is re-scored, no new
cell is built, nothing goes to the 415, DEV or the fresh set, and nothing changes in `src/` or `config.py`.

What this says, in plain words:
- **This AudioSet harness cannot track our task.** Once runs of barks count as one sound (our own gold rule), the 280 has
  only 69 needed sounds in 56 of 280 clips. Showing nothing costs 4 × 69 / 280 = 0.986. Ours shows 159 false spans:
  2 × 159 / 280 = 1.136 on false spans alone, so **even with 100 % recall ours would lose to showing nothing**. A detector
  idea can only "win" on this harness by cutting false spans below 138, whatever it does for recall. That is not what the
  product is judged on.
- Even under the old C, ours never beat showing nothing on the 280 (C-overlap "same", CI −1.37 to +0.80; C-onset
  "worse"). The earlier rounds compared ideas with ours on a scale where ours itself was not better than silence.
- The time window changes little (recall 46.4 → 47.8 %; no false span is rescued by the window rule). The two parts that
  move the cost are the shipped filter (false spans 201 → 159) and the run merge (events 225 → 69).
- **The video benchmark (DEV/TEST gold, `score_per_sound`) stays the only test of the detector.** There, the shipped
  system does beat silence (`docs/history/preregistrations/prereg_v4.md`, v4b4 gold re-run on the 109 benchmark clips: ΔF1 vs silence +0.290
  [+0.199, +0.377]; `docs/history/analyses/GOLD_RERUN_2026-09-22.md`). AudioSet clips are
  mostly clips with few or no needed sounds and many other labelled sounds, so any shown span is more often "false" there.

Not done because of the STOP (and so not claimed): the Step-2 re-score, N1–N4, the 415, DEV, fresh, any ship.

Note on the window false-span rule (it rescued no false span, 159 → 159): this is expected, not a bug. A span is ≥ 0.5 s
long, so a match by the window alone needs a span that starts after a short gold event has ended and still within 1.0 s
of its onset; the in-run pauses are already covered by the merge. The STOP does not depend on it: ours needs ≤ 137 false
spans to break even with silence at 100 % recall.

## Handback — one item still open (2026-09-28)
**Cluster sync of the stage-4 code (the brief's "ALSO").**
- `config.py`: local and cluster copies are already identical (md5 `360446c1…` on both) — nothing to do.
- `src/stage4_audio_event_detection/__init__.py`: the cluster copy lacks only the 14-line BEATs self-veto block (diff
  checked against `git show HEAD:`). **Not copied yet**, because job 31330563 (`devcand_s`, PENDING at handback) imports
  this module (`_extract_events`, `_refine_onsets_cam`) when it starts, and the rule is to copy only when no job imports it.
- **Condition:** `squeue -u adamg` shows no `devcand*` job and no other job that imports stage 4 (pipeline / protocol /
  devcand / detector jobs).
- **Commands (from P:\MscProj, Git Bash):**
  ```
  git show HEAD:src/stage4_audio_event_detection/__init__.py | tr -d '\r' > /tmp/s4.py
  md5sum /tmp/s4.py
  scp /tmp/s4.py adamg@slurm-login1.lnx.biu.ac.il:MscProj/src/stage4_audio_event_detection/__init__.py
  ssh adamg@slurm-login1.lnx.biu.ac.il 'md5sum ~/MscProj/src/stage4_audio_event_detection/__init__.py'   # must equal
  ```
Nothing else of round 12 is open: by the STOP rule, Steps 2–6 are not run.

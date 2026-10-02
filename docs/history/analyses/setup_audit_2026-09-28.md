# Setup audit of the stage-4 detector tests, 2026-09-28

Question: nine rounds of detector ideas failed on the held-out 415. Is something in the test setup broken?
This audit looks for bugs, not new ideas. Code: `benchmark/setup_audit.py`. All numbers: `benchmark/setup_audit.json`.
Scores are on the 280 only. The 415 and the fresh set were not re-scored. The fresh set had audio and gold checks only.
Nothing on DEV/TEST. Cluster jobs: 31330529 (CPU), 31330487 and 31330530 (GPU; they ran on L4, which is enough for BEATs).

## TLDR

- **The audio is aligned, the BEATs time stamps are right, and the harness equals the real stage-4 code.** None of
  these explains the failed rounds.
- **One real gold bug, small.** The AudioSet-Strong label file uses different names than BEATs for 10 labels in our
  gold (for example "Glass shatter" vs "Shatter"). Those glass and car-horn events are never counted as consequential.
  BEATs' correct spans on them count as false. Fixing it moves the 280 baseline C from 3.036 to 3.064.
- **The big effects are in how the cost is defined, not in the code.**
  1. The shipped label filter ("depictable") gives C 2.664 instead of 3.036 on the 280 (false spans 207 → 165).
  2. AudioSet-Strong counts each bark as its own event. 152 of the 224 consequential events on the 280 are parts of such runs.
     Three clips hold 39 of the 63 onset misses. Merging runs with the project's own rule (split at pauses over 2 s) gives C 2.050.
  3. The overlap rule has no time tolerance. A perfect detector that is 0.5 s late scores C 3.86. Showing nothing scores 3.20.
     38 % of consequential events are shorter than 0.5 s.
  4. Showing nothing is almost as good as the shipped system: C 3.20 vs 3.036 on the 280. On the 415, nothing costs
     4 × 171 / 415 = 1.65 and the shipped system costs 1.928. This is plain arithmetic from counts already published.
- **Deployment bug (not in the harness):** the cluster copy of `src/stage4_audio_event_detection/__init__.py` has no
  BEATs self-veto. `use_shipped()` sets PANNS_VETO to 0. So any cluster pipeline run under `use_shipped()` puts no
  veto on FlexSED-only spans.

## Checks

| # | check | result |
|---|---|---|
| 1 | time alignment (audio vs gold) | **OK** |
| 2 | gold mapping (MID → name, consequential) | **PROBLEM** (small): 10 renamed labels; the consequential rule is uneven |
| 3 | scorer sanity | **OK** as code. Two cost properties to decide on (no time tolerance; runs) |
| 4 | harness vs real pipeline | **OK** (40/40 clips identical). Known filter gap quantified. Cluster file missing the self-veto |
| 5 | BEATs window stamps | **OK** (stamp = window end − 0.5 s, as designed) |
| 6 | other (duplicates, short, silent, rate) | **OK**. One muted clip in the 415, one in fresh, none in the 280 |

### 1. Time alignment: OK

- One `fetch()` in `benchmark/gold/audioset_slice.py` downloads slice B, the 280, the 415 and fresh. It has used
  `--download-sections *start-start+10 --force-keyframes-at-cuts` since its first commit. The cut is re-encoded, so it
  is exact. Every audio stream starts at 0.000 s. Full clips are 10.00 s long.
- Sharp events (bark, gunshot, slam, knock, glass …, no other event starting within 0.5 s): 102 events in 47 clips of
  the 280. The spectral-flux peak comes **+0.04 s** after the gold onset (median). This is normal, because the flux
  peaks just after the attack. 39 % are within 0.1 s; random times give 10 %. The 415 and fresh give the same (+0.04 s).
- There is no clip-level shift. Where one event in a clip has a large lag, the other events in the same clip do not
  agree. So the cause is another loud sound nearby, not a moved clip. I did not re-download: the task's rule was
  "only if offsets are systematic", and they are not.
- The BEATs frame curves agree (check 5). Span ends come +1.7 s after gold ends, exactly as the stamp formula says.
- Residual, not a finding: a second method (all labelled onsets of a clip against the flux curve, lag grid ±3 s) flags
  30 of 161 confident clips with |lag| > 0.5 s. Most have only 2-5 onsets, several sit at the edge of the grid
  (−2.96, +2.98, +2.93 s), and where a clip has two or more sharp events, their lags disagree, which a real
  shift would not do. Only a re-download of those clips with a cross-correlation would settle them for certain; I did
  not run it, because the task allowed re-downloads only for a systematic offset. List: `align.calib.clip_lag_all_onsets`.
- Short clips: 24/280 (and 34/415, 24/fresh) are shorter than 9.9 s because the YouTube video ends early. The gold
  stops at the same point, except one clip in the 415, which runs 0.5 s past the audio.

### 2. Gold mapping: PROBLEM (small)

- `mid_to_display_name.tsv` (the strong set) gives 44 MIDs a different name from the ontology and BEATs
  (`src/audioset_mid_names.json`). 10 of them are in the 280 or the 415 gold. On the 280: Tire squeal, skidding 8; Ducks, geese,
  waterfowl 7; Glass shatter 4; Vehicle horn … toot 2; and 4 more. On the 415: Glass shatter 14; Glass chink, clink 14;
  Ducks 13; car horn 3; and 4 more.
- Effect 1: `CONSEQUENTIAL` has "Shatter" and "Vehicle horn, car horn, honking". The gold names do not match them, so
  **4 + 2 events on the 280 (14 + 3 on the 415) are never consequential**.
  Example: rW1nuTy0qTU_0 has a glass shatter that the gold does not count.
- Effect 2: `_same()` cannot link "Shatter" to "Glass shatter" (the names differ and there is no ontology link). A
  correct BEATs span is therefore scored as false. Test: if the gold events themselves are fed in under BEATs' names,
  C = 0.093, not 0 (13 false spans in 8 clips).
- 32 strong-only labels have no ontology entry; 6 of them are in our gold (Paper rustling 8, Stomp 8, Unknown sound 6
  on the 280; Tap dance 8 on the 415 …). No prediction can ever match them. None of them is consequential.
- The consequential rule is applied the same way everywhere (0 stale flags). But the family map sweeps in doubtful
  labels. On the 280: **Beep, bleep → Alarm 26 of 224**, Growling → Dog 9, Rumble → Thunder 5 (Rumble is never drawn
  in the product), Hiss → Cat 2. Also, Machine gun, Fusillade and Cap gun map to the family "Gunshot". That family is
  not in the list (only "Gunshot, gunfire" is), so they are not consequential.
- Few clips carry the cost. The 224 scored events sit in only 59 of the 280 clips. The top 10 clips hold 56 % (one
  clip has 26). On the 415: 171 events in 61 clips, top 10 hold 45 %.

### 3. Scorer sanity: OK as code, with two cost properties

| predictions (280) | C-overlap | recall | false spans |
|---|---|---|---|
| gold events themselves | 0.000 | 100 % | 0 |
| gold, in BEATs' names | 0.093 | 100 % | 13 (the naming bug) |
| gold shifted +0.5 s / −0.5 s | 3.864 / 4.057 | 63 % / 61 % | 377 / 392 |
| gold shifted +1.5 s | 5.650 | 54 % | 585 |
| gold, one family up | 0.414 | 87 % | 0 (generic parents such as "Domestic animals" are filtered, by design) |
| gold, first non-generic parent | 0.000 | 100 % | 0 |
| empty | 3.200 | 0 % | 0 |
| random spans | 5.900 | 6 % | 404 |

- **No time tolerance.** A short sound shown 0.5 s late costs 6 (a miss plus a false span). Not showing it costs 4.
  The shipped spans are long (2-s window), so they are hit less by this. A sharper detector is punished more.
- **Multi-parent labels.** `src/audioset_parents.json` keeps the official *first* parent for all 38 multi-parent
  labels (checked). 356 gold events on the 280 have such a label on their path (Tap 113, Tick 69, Chirp 45 …). If all
  parents are used in `_same`, the 280 C changes from 3.036 to 2.986 (dC −0.05 [−0.13, 0.00], 5 fewer false spans).

### 4. Harness vs the real stage-4 code: OK

- Method: the committed local `__init__.py` (with the self-veto) was loaded from a copy; the cluster `src/` was not
  touched. It ran under `config.use_shipped()`, with the live BEATs model on GPU and the FlexSED caches. I used 20
  random clips of the 280, plus 20 clips that have consequential events.
- The spans before onset refinement are **identical in 40/40 clips** (labels, starts, ends, count). Live BEATs vs the
  cache: max difference 0.0014 (fp16 storage). The time stamps are identical.
- The occlusion refinement (ONSET_CAM + ONSET_MONOTONE) moved 71 starts, all **later** (median about +0.2 s), and
  never earlier. On these clips it changed no cost (both sides scored after `use_shipped()`, so with the "depictable"
  filter). So the harness onsets are about 0.2 s earlier than the product's.
  This is small.
- The known filter gap, now measured. The harness uses "lists"; the product uses "depictable". On the 280:
  C 3.036 → **2.664** (dC −0.37 [−0.55, −0.22]), false spans 207 → 165, recall 51.3 → 52.5 %.
- **Cluster file:** `~/MscProj/src/stage4_audio_event_detection/__init__.py` lacks the 14-line self-veto block. The
  other 17 files checked are the same as the local copy (config.py, src/labels.py, src/types.py, beats_infer.py,
  flexsed_infer.py, the three ontology JSONs, detector_round2/4/5, audioset_stage4_report, audioset_detector_eval,
  detector_audit, and the three gold JSONs). All 1117 clip files are byte-identical (md5).

### 5. BEATs window stamps: OK

- In `beats_infer.infer_beats`, times = (window end) − 0.5 s. So a stamp t scores the audio [t − 1.5, t + 0.5].
  The caches agree: the first stamp is 0.0, the step is 0.25 s (the tail window gives one shorter last step), and
  FlexSED runs at 25 fps.
- Measured on 11 isolated events: the first stamp ≥ 0.35 comes 0.32 s before the gold onset (median). The span ends
  1.71 s after the gold end. This matches the formula. A centre stamp would give about −1.0 / +1.25 s. An end stamp
  would give about 0 / +2.25 s. So there is no off-by-half-window error.
- **The early-onset explanation holds.** 63 events are hit by overlap but missed by onset. 54 of them lie inside a
  span that already covers an earlier sound of the same family. The median pause to that sound is 0.32 s, and BEATs
  stays ≥ 0.175 through the pause in 48 of 50 cases. The twin rule moved only 5 starts. A 2-s window cannot split
  barks that are 0.3 s apart. Three clips hold 39 of the 63. The other 9: 5 are late (the gold starts at 0.0 s and
  BEATs starts later), and 4 have another cause.

### 6. Other: OK

- No duplicate clips and no shared YouTube ids, inside a set or across the 280, the 415 and fresh.
- All audio is AAC at 44.1 or 48 kHz (ffmpeg resamples to 16 kHz mono, the same as the pipeline). 5/280 clips are mono.
- Every clip has gold events. 221/280 have no scored consequential event, so these clips can only add cost.
- Muted audio under labelled sound: the 415 has JZx7Ac_sTD8_270000 (digital silence under "Music"; no consequential
  event, so no cost effect). Fresh has YpOHemscGCk_0. The 280 has none.
- The gold JSON says `duration: 10.0` for the short clips, so false/min is about 1 % too low. C is not affected.

## What a fix would change (280 only, shipped spans unchanged)

| scoring variant | C-overlap | C-onset | recall | false spans | consequential events |
|---|---|---|---|---|---|
| baseline (as all rounds) | 3.036 | 3.850 | 51.3 % | 207 | 224 |
| gold names by MID (bug fix) | 3.064 | 3.879 | 50.4 % | 201 | 230 |
| all ontology parents | 2.986 | 3.800 | 51.8 % | 202 | 224 |
| both | 3.014 | 3.829 | 50.9 % | 196 | 230 |
| "depictable" filter (as shipped) | 2.664 | 3.479 | 52.5 % | 165 | 219 |
| depictable + both fixes | **2.643** | 3.457 | 52.0 % | 154 | 225 |
| runs merged at ≤ 2 s (project gold rule) | 2.050 | 2.136 | 43.7 % | 207 | 71 |

The bug fixes alone change C by less than the bootstrap noise. They do not explain nine failures. The choices in how
the cost is defined (filter, runs, time tolerance, "nothing" as a floor) matter much more. They also punish any idea
that adds spans or makes sharper ones.

## Actions to decide

1. **Map gold names by MID at load time** (a bug fix; a small change). Recommendation: yes, before round 10.
2. **Score with the shipped "depictable" filter.** Recommendation: yes; it is the product's own rule.
3. **Use all ontology parents in `_same`** (and in the two vetoes, where the audit found the same exact-name problem).
   Recommendation: yes; small.
4. **Count runs as one event (merge ≤ 2 s) or keep one event per bark?** Recommendation: merge, because it is the
   project's own gold rule. It changes C a lot, so decide it before any new round.
5. **Add a time tolerance to the overlap rule** (for example a 0.5-s collar for short events)? Right now a slightly
   late but correct span costs more than showing nothing.
6. **Copy the committed `stage4/__init__.py` to the cluster** when no job imports it.
7. Review the consequential list: is "Beep, bleep" (26 events) really consequential? Is "Rumble"?

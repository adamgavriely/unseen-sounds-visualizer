# Tagger clips: selection rule (written 2026-09-30, before Adam tags any of the new clips)

Clips for Adam's own tagging in `tagger/` (the source of DEV2/TEST2, split by `sha256(stem) % 2`, commit 733dba9).
Nothing below reads Adam's tags or any gold label of DEV, TEST, TEST2, the 415 or the fresh set.

## Keep rule (every clip in the tool)
- **At least 2 BEATs-detected sounds:** `benchmark/gold/tagger_beats_suggest.py` (BEATs alone, scored stage-4 path,
  display bar, salient non-speech non-music labels), families grouped by `labels.same_source`; keep iff n_sounds >= 2.
  Applied on 2026-09-30 to the 82 untagged clips d001–d100: 44 pass; d002, d081, d084 are kept anyway (Adam had
  already added rows); 35 moved to `data/input/tagger_removed/` (`benchmark/gold/tagger_refresh.py`).
- Suggestions in the tool = BEATs only (hidden until asked; the export records use).

## New clips (named d101, d102, ... in the order of `tagger_refresh.py`; the tool is topped up to 100)
1. **Web clips** (`benchmark/gold/tagger_new_sources.json`): yt-dlp search by scene type, one 15–20 s cut per upload,
   the delegation QC (loudness, motion, no still image, not music-dominated), n_sounds >= 2. About half from scenes
   where sources are often off screen (city walks, street markets, busy streets, transport/stations), the rest spread
   over nature/farm, water, home, work/events, incidents and film/TV. Chosen by scene type, never by what a detector
   hears beyond n_sounds >= 2.
2. **AudioSet-Strong clips** (Adam, 2026-09-30: human strong labels are useful). Evaluation split only, from the pool
   left after excluding every YouTube id in the fresh set's exclusion union, the 500 ids drawn for the 415, the 500 ids
   drawn for the fresh set, and any id anywhere under `benchmark/`, `data/`, `docs/`, `tagger/`. The fresh set itself
   (422 clips) is NOT used: it is a sealed confirmation set (`docs/prereg_fresh_confirm_set.md`). Keep a candidate iff
   its strong labels have >= 2 distinct depictable non-speech, non-music event types, it is not a still image, and
   BEATs n_sounds >= 2. Order: `random.Random(20260930)` shuffle of the eligible pool. At most 30 in the tool.
   The strong labels stay private (`benchmark/gold/tagger_audioset_sources.json`) and are used only to check Adam's
   tagging quality, never shown in the tool.
3. Web and AudioSet clips are interleaved when numbered, so the names don't reveal the source.

## Amendment 1 (Adam, 2026-09-30 00:40 UTC, before any new clip is in the tool)
"I don't want street scene preference. Use the AudioSet-Strong sets, videos I did not use already, with 3+ sounds."
- New clips come from **AudioSet-Strong only** (point 2); the web sourcing (point 1) is stopped and none of its clips
  go into the tool.
- Eligible AudioSet clips need **>= 3** distinct depictable non-speech, non-music strong-labelled event types (grouped
  with `labels.same_source`), plus BEATs n_sounds >= 2 as before. Same pool, exclusions, seed and order; the fresh set
  stays untouched.
- Target: 53 clips (a001–a053), so the tool lists 100. `tagger_refresh.py` uses MAX_AUDIOSET = 53 and no web clips.
- Disclosure: eval split only, so BEATs/PANNs (trained on AudioSet train) see unseen clips; DASM's checkpoint was picked
  on AudioSet-Strong eval (`docs/prereg_round6_dasm.md`), so these clips slightly favour DASM.

## Amendment 2 (Adam, 2026-09-30 00:47 UTC, before any new clip is in the tool): "I don't care if BEATs detected"
- New AudioSet clips no longer need BEATs n_sounds >= 2; the keep rule is >= 3 labelled sound types + download,
  still-image and silence checks. BEATs still runs on them, for the suggestions only.

## Amendment 3 (Adam, 2026-09-30 00:47 UTC, before any new clip is in the tool): "50 clips with 3+ sounds and some interesting ones like explosions"
- 50 AudioSet clips (a001–a050), each >= 3 labelled sound types. Up to 25 must contain a strong-labelled "striking"
  event (Explosion, Gunshot, Fireworks, Burst/pop, Boom, Siren, alarms incl. smoke/fire/car alarm, Glass/shatter,
  Screaming, Crying, Baby cry, Thunder, Crash/Smash, car/air horn, Doorbell, Telephone ringing, Dog bark/growl,
  Breaking; descendants included). Striking clips first in the seed order until 25, then the rest of the eligible
  pool in the same order. Chosen from labels only, never from detector output.

## Amendment 4 (Adam, 2026-09-30 02:23 UTC)
The AudioSet-Strong labels ARE shown in the tool, as "other annotators' tags" behind their own hidden button
(`benchmark/gold/tagger_annot.py`; the export records `annotators_shown_at` / `from_annotators`). They are not used as a
quality check of Adam's tags ("we just want a tagged dataset; AudioSet-Strong contains errors"). This replaces the
"never shown in the tool" line of point 2.

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

# Pre-registration: a fresh AudioSet-Strong confirmation set for stage-4 detector tests

*Written 2026-09-28, after the id list was drawn (`--dry`) and BEFORE any download, cache or score. Download approved
by Adam on 2026-09-28. Builder: `benchmark/gold/audioset_fresh.py`; list: `benchmark/gold/audioset_fresh.json`;
set name in code: `fresh` (`R.use_set("fresh")`).*

## Why
The held-out 415 (`benchmark/gold/audioset_heldout.py`, amendment 24) has now been read by six detector rounds. From
now on a new detector idea is picked on the 280, checked on the 415, and a candidate that passes the 415 gets ONE final
check on this fresh set.

## Who may look
Nobody computes or looks at any detector result on this set (no cost, no recall, no false alarms, no per-clip output)
until a candidate has passed the 415. The build only reports clip counts, download failures and cache completeness.

## Selection rule (the 415's recipe, new seed)
- Source: AudioSet-Strong **evaluation** split only (`data/audioset_strong_labels/audioset_eval_strong.tsv`,
  16,996 clips). No train-split clip.
- **300 COMPLEX + 200 RANDOM, seed 20260928** (Python `random.Random(20260928)`; same draw order as the 415's builder:
  shuffle the complex pool and take 300, then shuffle the rest of the pool and take 200).
  - complex = Speech or Music covers >= 50 % of the 10-s clip AND at least one labelled non-speech, non-music event
    lies >= half under Speech/Music (`audioset_heldout.is_complex`, unchanged);
  - random = drawn from the remaining pool (complex clips can appear here too).
- The list is fixed by this file. Clips gone from YouTube are listed as missing, never replaced. The 415 kept 415 of
  500 (83 %); the same attrition gives about 400 usable clips.

## Exclusions (at the YouTube-id level: an eval clip is out if its 11-character YouTube id appears in any source)
| source | eval YouTube ids excluded |
|---|---|
| the 415 builder's list: the 280 fit set + missing, slice B + missing, every gold clip named by a segment id | 470 |
| all 500 ids drawn for the 415 (415 fetched + 85 gone) | 500 |
| repo scan: any eval YouTube id in any text file or file name under `benchmark/` and `data/` (DEV/TEST split, pilots, caches, `round6_wavcaps_ids.json`, the UnAV-100 list in `data/work/unav/`, ...); label TSVs skipped; sealed / key / rate_confirm files not opened | 2,401 |
| gold clips named `as_<class>_<first 8 chars of YouTube id>` (`scripts/source_audioset.py`): any eval id with that prefix | 16 (over-matches, harmless) |
| **union** | **2,402** |

Pool after exclusion: 14,594 clips; complex pool 4,297. Drawn: 300 complex + 200 random (54 of the random are complex
too). Independent check on the cluster (2026-09-28): `grep -F` of the 500 YouTube ids over all 90,066 text files
(json/jsonl/tsv/csv/txt/md/log/py/html/yaml/sh, < 100 MB) under `~/MscProj/benchmark` and `data`, label TSVs and
sealed / key / rate_confirm files skipped, plus every file and folder NAME there: **0 hits**.

**Id list fingerprint:** sha256 of the 500 sorted segment ids joined by `\n` =
`8534b8eeafa582c77fa22223c8a6bd99a5dc5fb288b9facaca76a93152b1bcbf` (also stored as `ids_sha256` in the JSON).

## Download and caches (same as the 415)
- `python benchmark/gold/audioset_fresh.py` locally: yt-dlp from YouTube (the public AudioSet source), the labelled
  10 s only, <= 480p mp4, into `data/input/audioset_fresh/`; log `data/audioset_strong_labels/fresh_download.log`.
- Cluster caches under `benchmark/audioset_fresh_windows/` and `data/work/flexsed_fresh/`: BEATs (2-s windows,
  0.25-s hop), PANNs, PE-A-Frame, FlexSED (215 family queries) — the four caches the shipped stack reads
  (`detector_round2.usable`). Candidate-model caches (EAT, Dasheng, DASM, ...) are not built now.

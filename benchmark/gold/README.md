# benchmark/gold — labels, scorer and scoring harness

This folder holds the human labels, the per-sound scorer, and the harness that scored every version of the system.
Most of its ~500 files are per-round experiment scripts and their result files; they stay here because the
pipeline imports some of them and the development record cites them by path. The files that matter for the final
system:

| purpose | files |
|---|---|
| **Labels** | `annotations/gold_AG.json` (all per-sound labels), `dev_stems.txt`, `test_stems.txt`, `tagger_split.json` |
| **Scorer** | `score_per_sound.py` (matching rules, cost), `holm_table.py`, `dev_candidates_check.py` (bootstrap) |
| **Harness** (used by the pipeline too) | `round13_dev.py` (variants; the final system is `SHIP8+MD3+WW5+SL`), `tagger_prep.py` (imported by `src/listener_prep.py`), `merged_dev.py`, `test_vs_ship8.py`, `final_test.py`, `parity_check.py` |
| **Decision trail** | `inspector_trail_export.py`, `render_trail_media.py` → `docs/inspector2/` |
| **Analyses cited in the report** | `ceiling_ship7.md`, `visible_weight_sweep.md`, `final_test_ship8.{json,md}`, `holm_test_final_v33_test_bench.json`, `ledger_*.json` |
| **Labelling tool** | `index.html`, `tool_template.html` (guideline) |
| Everything else | per-round experiment scripts and results (`expect*`, `nameall*`, `weakwitness*`, …), described in `docs/history/preregistrations/prereg_round13_detector_push.md` |

Short codes in file names (SHIP8, WW5, …) are explained in the report, Appendix D.

---

# Gold set (human annotation of the 100 test clips)

Why: the thesis has one annotator (Adam) with a single clip-level label; every automatic
reference failed at the visibility question. A human-annotated gold set — per sound: heard,
source visible, covered by speech/music, rough time; per clip: one sentence "what a hearing
viewer gets that a deaf viewer misses" — is what any comparison of systems (old or new
models) will be scored against.

## How to annotate

1. Open `benchmark/gold/index.html` in a browser (double-click; it plays the clips from
   `data/input/benchmark/`, so it must be run from a checkout that has the clips).
2. Type your initials at the top. Read the twelve-line guideline on the right.
3. Per clip: the detector's guesses are pre-filled (label + rough time). Listen once; delete
   what you do not hear (×), add what it missed (`A`); tick "source visible" and "covered by
   speech/music" as they apply. Write the sentence, or press "nothing beyond the picture".
   Save and next (`N`). The pre-fill is disclosed in the thesis as a possible anchoring bias.
4. Stop any time; press **Export JSON** and put the file in `benchmark/gold/annotations/`.
   Progress is also kept in the browser under your initials.

About 2–3 minutes per clip. Start with 10–20 clips, compare between annotators, adjust the
guideline once, then continue to all 100.

Two more fields per sound, both pre-filled: **obvious from picture** (does the frame alone
already make it clear the sound is happening now? default = the system's "visible" verdict
on the benchmark page, unticked on the AudioSet-Strong page) and **importance** 1/2/3
(1 = background texture such as traffic hum or rain; 2 = context such as dishes, footsteps,
birds; 3 = safety or plot such as siren, alarm, glass breaking, baby crying, doorbell, phone).
The default importance comes from keywords in the label (`importance_of` in `build_tool.py`);
a sound added by hand starts as not obvious, importance 2. `merge.py` writes per clip a
`sounds` list with the majority `obvious` and the median `importance` per label.

## Merge and agreement

    python benchmark/gold/merge.py

prints Cohen's κ on "picture due" between annotators and against Adam's original labels,
and writes `benchmark/gold/gold_set.json` (majority verdicts; disputed clips flagged).

## Slice B: AudioSet-Strong clips

A second page whose pre-fill comes from human labels (AudioSet-Strong event spans), not the
detector. `benchmark/gold/audioset_slice.py` writes `benchmark/gold/audioset_slice.json`
and the clips to `data/input/audioset_strong/<segment_id>.mp4`; then

    python benchmark/gold/build_tool.py --slice audioset

writes `benchmark/gold/index_audioset.html` (clips without an mp4 on disk are skipped).
Open it as above. Labels and times are already human; only **source visible**, **covered by
speech/music** (pre-ticked from the Speech/Music spans), the **picture-due** box and the
**sentence** need annotating. Nothing is pre-ticked as due and the sentence starts empty.
Exports carry `"tag": "audioset_strong"`; `merge.py` keeps these clips in `gold_set.json`
with `"slice": "audioset_strong"` and leaves them out of the κ against Adam's labels.

## Rebuild the tool (after changing the template or the clip list)

    python benchmark/gold/build_tool.py

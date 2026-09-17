# Gold set (human annotation of the 100 test clips)

Why: the thesis has one annotator (Adam) with a single clip-level label; every automatic
reference failed at the visibility question. A human-annotated gold set — per sound: heard,
source visible, covered by speech/music, rough time; per clip: one sentence "what a hearing
viewer gets that a deaf viewer misses" — is what any comparison of systems (old or new
models) will be scored against.

## How to annotate

1. Open `benchmark/gold/index.html` in a browser (double-click; it plays the clips from
   `data/input/benchmark/`, so it must be run from a checkout that has the clips).
2. Type your initials at the top. Read the ten-line guideline on the right.
3. Per clip: listen once; add every noticeable non-speech, non-music sound; tick "source
   visible" and "covered by speech/music" as they apply; stamp start/end with `[` and `]`;
   only then, if you want, open "what the detector heard". Write the sentence, or press
   "nothing beyond the picture". Save and next (`N`).
4. Stop any time; press **Export JSON** and put the file in `benchmark/gold/annotations/`.
   Progress is also kept in the browser under your initials.

About 2–3 minutes per clip. Start with 10–20 clips, compare between annotators, adjust the
guideline once, then continue to all 100.

## Merge and agreement

    python benchmark/gold/merge.py

prints Cohen's κ on "picture due" between annotators and against Adam's original labels,
and writes `benchmark/gold/gold_set.json` (majority verdicts; disputed clips flagged).

## Rebuild the tool (after changing the template or the clip list)

    python benchmark/gold/build_tool.py

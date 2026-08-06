# Stage 4 — Audio Event Detection

**Role:** Detect and classify **non-speech** sounds (ambient, environmental, acoustic events) with
time boundaries. This is the core signal the whole system augments.

**Input:** `audio.wav` from Stage 1.
**Output:** list of `{label, confidence, start, end}` events over the AudioSet ontology (527 classes).

**Candidate models:** PANNs / CNN14 (fast, light, good first implementation) → BEATs tagger
(quality) → CRNN/frame-level (ATST-Frame-style) only if precise onset/offset timing is needed.

**Status:** Not implemented (research phase). New component — no legacy code.
**TODO (Adam):** check licences of specific checkpoints — some BEATs/DCASE checkpoints are
research-only.

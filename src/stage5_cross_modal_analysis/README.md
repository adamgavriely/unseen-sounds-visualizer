# Stage 5 — Cross-Modal Semantic Analysis  *(the intellectual core)*

**Role:** Fuse scene context (Stage 2), speech (Stage 3), and audio events (Stage 4) into a decision:
**which audio cues are semantically meaningful AND not already conveyed by the video**, and how to
depict each. This "reason about the gap between what is heard and what is seen" step is the project's
main contribution and the axis of comparison against baselines that ignore the visual stream.

**Input:** Stage 2/3/4 outputs.
**Output:** per-selected-event augmentation spec — `{event, worth_showing, reason, image_prompt,
timing}`.

**Candidate models:** Qwen3 / Llama 3.1 (8B local-quant, 14B+ on uni GPU or hosted API).

**Status:** Stub wired into the pipeline — `plan_augmentations()` runs a transparent rule-based gate
(salience threshold + "stay silent if source visible"); the localization signal + grounded LLM are
TODO (see `docs/project_notes.tex` sec:stage5). New component — no legacy code.
**Note:** conceptually related to the old `planner.py` (an LLM deciding what to visualize), but the
decision criterion is now *visual-gap-aware*, not speech-content-based. Study `planner.py` in
archive for prompt-structure ideas only.

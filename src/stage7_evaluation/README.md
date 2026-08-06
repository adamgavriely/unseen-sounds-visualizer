# Stage 7 — Evaluation

**Role:** Measure whether the generated augmentations communicate the intended audio semantics.

**Automatic protocol:** generate augmentation → VLM *describes* the augmented scene → compare against
a semantic reference derived from the original multimodal input (LLM) → independent **LLM judge**
scores semantic consistency. Complement with CLIP/CLAP relevance and a **gating-accuracy** metric
(did we correctly withhold augmentation when the sound was already visible, and add it when not?).

**Baselines:** (1) direct audio→image (Sound2Scene); (2) audio captioning only; (3) proposed method.

**Input:** benchmark clips + system outputs.
**Output:** per-clip scores, aggregate tables, baseline comparison.

**Status:** Not implemented (research phase). New component — no legacy code.
**TODO (Adam):** decide whether to run the optional human study; if yes, check department
ethics/IRB requirements and timeline early.

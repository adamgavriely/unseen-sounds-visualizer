# Stage 2 — Video Understanding

**Role:** Analyze the *visual* stream to determine what is already visible to the viewer. This is the
input to cross-modal gating (Stage 5): we only augment sound whose source/meaning is **not** already
on screen.

**Input:** video clip (sampled frames).
**Output:** structured scene description — visible objects/agents, setting, on-screen actions, and
(ideally) whether likely sound sources are visible.

**Candidate models:** Qwen2.5-VL (7B quality / 3B local), InternVL3, LLaVA.

**Status:** Stub wired into the pipeline — `analyze_video()` returns an empty `SceneContext`; the
VLM call is TODO. New component — no legacy code.
**Note:** the same VLM family is reused as the *describer* in Stage 7 evaluation — use separate
prompts/instances so a model never grades its own output.

# Shared context for the three topic panels (27 Sept 2026) — read this first

**Project.** MSc thesis (Adam, Bar-Ilan). A deaf or hard-of-hearing viewer misses sounds that are heard but not seen. The
system detects non-speech sounds in a video, decides whether each sound's source is visible, and — only when it is not —
shows a picture of the sound in a side panel beside the video, timed to the sound. No model is trained; pretrained
models are chained. Deadline: deliver by **3 October 2026** (6 days). Compute: BIU cluster, H200/A100 GPUs, 125 GB disk
free; the login node has internet. Adam's own time is scarce (he is the only annotator so far).

**Pipeline (shipped, `config.use_shipped()`):** ffmpeg → Whisper (speech, never drawn) → OWLv2 (scene context) → sound
detection: BEATs (bar 0.35) ∪ FlexSED (text-queried, bar 0.8) + FlexSED veto 0.3 + PANNs veto 0.05, onset clamp, no length
cap → drawable-label filter → **visibility gate**: Qwen3.8-27B, 6 frames per ≤ 5-s stretch, 3 questions, majority →
picture subject (Qwen3.8 + word guards) → picture (FLUX.1-schnell shipped; Qwen-Image-2512 tested) → side panel.

**Data.** Gold: 139 clips, one annotator, per sound: label, onset/offset, visible?, obvious?, importance 1–3. "Needed" =
importance ≥ 2, not visible, not obvious. DEV 49 clips (development), TEST 60 (held out; the final table is scored and
**closed** — no further TEST numbers), slice B 30 (AudioSet-Strong clips). Also 280 AudioSet-Strong calibration clips with
human frame labels (no visibility labels), and 50 never-annotated clips used for blind picture ratings.

**Headline results (final TEST table, 60 clips, vs "draw every detected sound", paired bootstrap, Holm):**
ΔF1 +0.059 [−0.030, +0.144] (null); precision +0.137, wrong pictures/clip −0.52, viewer cost −0.83, clean-clip accuracy
+0.22 (all significant after Holm); on clips whose sound is off screen, cost vs showing nothing −3.08 (significant).
Pictures (blind human glance test, 54 sounds): old generator 14/54 recognised, Qwen-Image-2512 26/54 (+22 pts, sig.);
one new-generator picture showed a wrong object.

**Full records:** `docs/history/preregistrations/prereg_v4.md` (amendments 1–23), `docs/history/daily_notes/LEDGER_2026-09-26.md` (everything tried, valid/discarded),
`docs/history/analyses/GOLD_RERUN_2026-09-22.md`, `docs/history/earlier_drafts/thesis/ch3_method.md`, `docs/history/earlier_drafts/thesis/ch5_results.md`, supervisor page
`docs/supervisor/supervisor_2026-09-26/index.html`. Repository `P:\MscProj` — read anything, change nothing, do not ssh.

**House rules.** Every choice is made by a test with a bar written before it runs; a shared stage is chosen by the BLIND
system's own numbers, never by the gated system's gain; negative results are reported; no TEST reads.

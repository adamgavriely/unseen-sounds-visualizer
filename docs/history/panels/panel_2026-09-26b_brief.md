# Panel 2026-09-26b — final week: audit everything, push each part to its maximum (round 1 brief)

Adam (the student): *"I plan to finish and deliver in 1 week maximum. I want to be at the peak performance of this
project."* He will show a supervisor page (mostly images, videos, graphs; minimal text) with what we did, what we
tried, why we dropped or selected things, good / bad / complicated examples, and questions to ask.

You are one of six reviewers, each owning one part (below). Repository `P:\MscProj` — read anything, change
nothing, do not ssh to the cluster. Key records: `docs/history/plans/PLAN.md`, `docs/history/preregistrations/prereg_v4.md` (amendments 1–21),
`docs/history/analyses/GOLD_RERUN_2026-09-22.md`, `docs/NIGHT_REPORT_2026-09-{18,20,24,25}.md`, `docs/history/daily_notes/night_report_2026-09-14.md`,
`docs/history/plans/failure_catalogue.md`, `docs/history/analyses/test_second_look.md`, `docs/history/preregistrations/picture_v3_prereg.md`,
`docs/history/preregistrations/freeze_picture_setup_2026-09-25.md`, `docs/panel_2026-09-25_*.md`, `docs/history/panels/panel_2026-09-26_plan.md` (signed
5/5 yesterday: reporting hierarchy, Holm families), `config.py`, `src/`, `benchmark/`.

## Where things stand (one paragraph)
Training-free pipeline: audio detectors (BEATs ∪ FlexSED 0.8, FlexSED veto 0.3, PANNs veto 0.05, onset rule, no
cap) → label filter → visibility gate (Qwen3.8-27B, 6 frames per 5 s, majority of 3 votes) → picture subject
(Qwen3.8) → picture (FLUX shipped; Qwen-Image-2512 + guarded scene-kind text + templates + word cards frozen, awaiting
one human confirmation) → side panel beside the video. Gold: one annotator, 139 clips (DEV 49, TEST 60, slice B 30).
Significant on TEST (23 Sep look): vs blind ΔP +0.143, Δcost +0.73; vs silence on 13 off-screen clips +3.08. ΔF1 vs
blind null (structural; MDE ≈ 0.13). Pictures: blind human round 2, new generator +22 points significant. Three
automatic picture checkers failed. A final like-for-like TEST table (ours / blind / ungated text at the shipped
config) was rendered today and is scored once under amendment 21. Caption baseline is ungated (confound, fixed in
reporting). Time left: 7 days; compute plentiful (H200/A100); Adam's time scarce.

## Parts
- **F1 audio** — extraction, speech (Whisper/Granite), detectors (BEATs, FlexSED, PANNs, PSED, FLAM, CLAP…), label
  filter, family map, onset rule, ends / cap.
- **F2 video + gate** — scene/objects (OWLv2, SAM 3, Qwen VL), visibility votes, kinship rule, AV sync attempts.
- **F3 pictures** — subject text (DEPICT prompts, V3/V3.1, guards), generators (FLUX, Qwen-Image-2512/2.1, others
  tried), templates, word cards, the side-panel display and video output.
- **F4 evaluation** — gold set and annotation, per-sound metric, judges (Mistral, Gemma direct, per-picture, verifier,
  pairwise), human ratings, statistics, TEST reads, what the final results table must contain.
- **F5 history auditor** — everything tried since August across all parts; mark each VALID, DISCARDED (wrong data,
  bug, broken test — say which) or SUPERSEDED.
- **F6 supervisor page + delivery** — the page storyboard, the examples, the graphs, the questions for the supervisor,
  the 7-day delivery plan.

## What to return (F1–F4)
1. **Ledger of your part**: every approach tried → result → kept / dropped → why. Mark results from wrong data,
   bugs or broken tests as DISCARDED with the reason (they must not appear as evidence).
2. **Current choice vs SOTA**: a table — stage · model in use · SOTA alternatives considered/tried · why ours
   (evidence, licence, VRAM, measured result). Name any stronger model not yet tried and whether it is worth trying
   this week.
3. **≤ 3 improvements for the final week**, ranked: what, expected effect on which number, cost (GPU hours, Adam
   minutes), the decision rule written before it runs, and whether it risks TEST or a sealed thing. Say "none worth
   it" if that is the honest answer.
4. **Examples for the supervisor page**: 2–4 clip names from your part (clearly good / clearly bad / complicated),
   what each shows in one line, and where its media lives (paths you found).

**F5** returns one ledger table for the whole project (date · part · what · result · status · reason), plus the five
things most worth telling the supervisor. **F6** returns: page storyboard (sections in order, each = one visual + ≤ 15
words), the graphs to draw (data source for each), 5–7 questions for the supervisor, and a day-by-day 7-day plan.

Be concrete; cite files. Keep it under ~900 words (F5 may be longer).

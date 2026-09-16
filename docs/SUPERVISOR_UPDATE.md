# Supervisor update — Visual Augmentation of Audio Semantics for Accessibility

*Adam Gavriely, 16 September 2026. First status update since the project started. Two pages.
Full detail: `docs/EXECUTIVE_SUMMARY.pdf` (14 pages, plain language) and the thesis draft
`docs/report/report.pdf` (19 pages).*

## 1. Where the project stands, in five sentences

The system is built and evaluated: a no-training pipeline of off-the-shelf models that, for
each ambient sound in a video, decides whether its source is visible on screen and shows a
picture beside the video only when it is not (the "gate"). On a 274-clip benchmark I
labelled (100 test clips, four scenarios), scored by an automatic judge on a 0–4 scale, the
gated system **ties** the simplest alternative — draw a picture for every sound, never look
at the video — at 3.17 vs 3.16, and beats a text-caption baseline (+0.28). The gate does
what it should where nothing is missing (+0.26) and loses where a picture is due (−0.24)
because the **sound detector**, not the visibility decision, misses sounds buried under
speech or music. Seven pre-registered attempts to fix the detector without training and
three attempts to build a fully automatic evaluation reference all failed their declared
bars; these are reported as findings. The thesis draft is complete; I plan to freeze all
numbers on 17 Sept and submit on 18 Sept.

## 2. What exists

- **Pipeline** (7 stages): FFmpeg → OWLv2 → Whisper (context only) → BEATs sound detector →
  Qwen2.5-VL-7B (visibility by three differently worded votes; picture phrase; duplicate
  removal via the AudioSet ontology) → FLUX.1 image generation → side panel. `python main.py <video>`.
- **Benchmark**: 274 clips, four scenarios (unseen / mixed / seen / no ambient); dev split
  (174) for settings, test (100) touched once for the headline. My own re-labelling of 60
  clips agrees with itself on 78% (κ = 0.60); nobody else has labelled.
- **Evaluation protocol**: a VLM describes what the panel conveys; a judge LLM scores it
  against a reference sentence; two baselines (blind audio-to-image, audio caption).
- **Outside check**: DCASE 2025 Task 3 (human on/off-screen labels): our visibility decision
  agrees 66.7% (±6); it keeps 97% of off-screen sounds and silences only 35% of on-screen ones —
  it errs on the safe side.
- **Documents**: thesis draft (19 pp), executive summary (14 pp), `LIMITATIONS.md`, a failure
  catalogue, three demo videos, all scripts reproducible from the repository without a GPU
  for the headline numbers.

## 3. The main findings (all in the report)

1. **Tie with the blind baseline** (95% CI −0.15 to +0.18). A perfect gate would score
   3.62; both systems sit ~0.45 below it, in opposite places: ours where a picture is due
   (withheld pictures), blind where none is due (redundant pictures).
2. **The detector is the bottleneck.** Of 11 wrongly withheld test pictures, 7 are sounds
   that never reached the detector's bar under speech/music; ~80% of redundant pictures on
   dev are detector "phantoms" (a whale at a Christmas market). Seven fixes (threshold
   sweep, CLAP, PANNs agreement, persistence rule, 32B VLM, audio-LLM veto, source
   separation) all failed pre-declared bars: with one clip-level AudioSet tagger, phantoms
   and buried sounds are one dial.
3. **The evaluation's reference matters.** The proposal's reference was written by the
   system's own models and was circular (it scored a correct silence 0). A human-grounded
   reference is the headline. Three automatic references (independent models; with-sound
   vs muted description; gap-closing score) all failed at the same point: today's open
   audio-visual models cannot tell whether a sound's source is in view better than chance.
4. **Cost next to benefit.** The gate keeps the panel on 47% of a clip vs 60% for blind
   and removes 30% of the panel time wasted where nothing was due.
5. **The judge's asymmetry** (a withheld picture costs ~4, a redundant one ~1) comes from
   the rubric, not from deaf viewers; a judge-free cost-sensitivity view prefers the gate
   once a redundant picture counts ≥ 40% of a missing one (exploratory).

## 4. Questions I need to ask you

1. **Contribution framing.** Is this acceptable as the thesis contribution: a working
   training-free pipeline + a benchmark and protocol + the measured negative findings
   (detector bottleneck; automatic references fail at visibility) — rather than a system
   that beats the baseline? The report is written that way.
2. **Second annotator.** The biggest hole an examiner will see is one annotator. Can two or
   three people (lab members, students) spend **one hour** each labelling 60 test clips
   ("is the sound's source visible?") on a web page I will prepare? Does that need ethics
   approval? If yes: by when could they do it?
3. **User study.** The proposal specified automatic evaluation only. Should I add a small
   "muted-clip" check with hearing people (not a DHH study) as validation, or state clearly
   that helpfulness to DHH viewers is untested and leave it to future work?
4. **"No training" rule and future work.** May the future-work section propose a small
   trained detector head (with a declared bar), or must future work also stay training-free?
5. **Report format.** Any required structure, length limit, language (Hebrew abstract?),
   title page, or submission channel? The draft is 19 pages in article style.
6. **Benchmark release.** May I release the labels + source IDs (not the videos) with the
   code? Any licence or copyright concern with the clips?
7. **AI assistance disclosure.** I used an AI coding assistant throughout (code, drafting,
   and an independent AI "reviewer" for every decision, all logged in git). How should this
   be acknowledged in the thesis?
8. **Deadline.** Is 18 Sept the hard date, or is there room for the annotator step (which
   would move the number freeze by one day)?

## 5. What I need from you to continue

- Decisions on 1, 2, 4, 8 by tomorrow (17 Sept) — they determine what goes into the freeze.
- If you can read one thing before we talk: `docs/EXECUTIVE_SUMMARY.pdf`, sections 1–3 and 9
  (about 10 minutes).
- Names of any people who could label for an hour (question 2).

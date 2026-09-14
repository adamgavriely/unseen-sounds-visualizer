# Executive summary — Visual Augmentation of Audio Semantics for Accessibility

*For someone taking over this project today (2026-09-14). Everything here is in plain
language; the detailed record is in `docs/project_notes.tex` (the lab notebook),
`docs/literature_review.tex` (the background), `docs/report/report.pdf` (the thesis draft)
and `LIMITATIONS.md`.*

---

## 1. The problem in one paragraph

A deaf viewer watching a video gets the speech through subtitles. They do not get the rest of
the soundtrack: a siren behind the camera, a dog barking in the next room, glass breaking off
screen, a crowd gasping. Subtitle guidelines actually tell captioners to *skip* any sound the
picture already shows, and in practice most ambient sound is never captioned at all. This
project builds a system that watches a video, hears its ambient sounds, decides for each one
whether its source is **already visible on screen**, and — only when it is not — shows a small
picture of that sound next to the video for as long as it lasts. The picture is the missing
information; the "only when not visible" rule (we call it the **gate**) is the whole idea.

## 2. Background: what exists and why it is not enough

- **Subtitles / SDH** carry speech well and non-speech sound badly and inconsistently.
  Sign-language interpretation is richer but rare and cannot be automated.
- **Sound-awareness tools for deaf users** (HomeSound, SoundWatch, ProtoSound) detect ~20
  everyday sounds — knocks, alarms, a baby crying — and users prefer icons/pictures to text.
  They target the real world, not media.
- **Audio event detection** is mature: models trained on AudioSet (527 sound classes) can name
  what is in a soundtrack. We use **BEATs**; PANNs was the first choice and was replaced.
- **Vision-language models (VLMs)** can answer open questions about video frames — the only
  way to decide "is the thing making this sound visible?" without a fixed list of objects.
- **Audio-to-image generation** (Sound2Scene etc.) turns sound into a picture blindly, without
  looking at the video. Those systems are our baseline: "draw everything you hear".
- Nobody had combined: detect ambient sound → reason about what is heard but not seen →
  show a picture for accessibility. That combination is the contribution.

Constraints set by the supervisor: **no training** (glue existing open models), **minimum
human involvement** (every decision made by a model, not a hand-written rule), finish this
week.

## 3. What the system does (the seven stages)

| stage | what it does | model |
|---|---|---|
| 1 | extract audio | FFmpeg |
| 2 | a cheap whole-clip object pass (defers to stage 5) | OWLv2 |
| 3 | transcribe speech — used only as *context* for the gate, never drawn | Whisper |
| 4 | detect ambient sounds with start/end times | BEATs (527 AudioSet labels) |
| 5 | **the reasoning**: for each sound, is its source visible? what kind of sound is it? what should the picture show? which sounds are the same source? | Qwen2.5-VL-7B |
| 6 | draw the picture and compose the side panel | FLUX.1-schnell |
| 7 | evaluate | see §6 |

Key behaviours, all decided by models at run time:
- **Visibility** is asked per sound, per 5-second stretch, with three differently phrased
  questions to the VLM (name the source; is it visibly happening, asked in both option
  orders; describe the frames), majority vote. "Visible" means *seen making the sound*: a
  baby in its mother's arms is not visibly crying; a woman with her head back is visibly
  laughing.
- **One continuous sound → one continuous picture**, held at least 1.5 s; the panel is empty
  when nothing is heard.
- **The picture shows the event** ("glass shattering", "audience clapping"), not the object,
  built from the detector's label, the frames, and a short place ("kitchen"), never from the
  dialogue and never from example sentences.
- **One picture per source**: the AudioSet ontology (the detector's own taxonomy) merges
  "Laughter" with "Giggle"; no similarity threshold.
- **Speech as context**: if people are reacting to a sound ("whoa, look at that"), that sound
  gets priority — but it can never override visibility.

## 4. How it was built, and what was learned

The system was developed on demo clips, and almost every early fix was wrong in a way that
only showed on the next clip. The lessons are the most reusable part of the project:

1. **Fixed vocabularies fail.** A hand-written list of 30 "visible concepts" could never mark
   laughter or a telephone as visible. Every such table was replaced by a question to a model.
2. **Ask for a name, not a yes.** A yes/no question to a VLM collects agreement; a name can be
   checked. Every two-way question is asked in both orderings because the 7B model prefers
   option (b) 22 times out of 23.
3. **Prefer reading a structured answer to inferring one.** The AudioSet ontology shipped with
   the detector solved deduplication after three cleverer attempts failed.
4. **Thresholds tuned on one clip are invalid on the next.** Since 2026-09-13: knobs are set
   once on a development split; every failure goes into a catalogue
   (`docs/failure_catalogue.md`); a change needs three cases of one pattern.
5. **Evaluation code produces plausible wrong numbers.** An empty result scored 2/4; a
   similarity returned 0.0 for everything after a library upgrade; an audio model received no
   audio and answered "none" 100 times. Each was found by reading logs; each is now guarded.

## 5. The benchmark

274 video clips, hand-labelled by Adam into four scenarios (25 of each in the 100-clip test
set): **unseen** (an ambient sound whose source is off screen — a picture is due), **mixed**,
**seen** (source on screen — no picture due), **no ambient** (speech/music only — no picture
due). The proposal asked for 300; usable clips were only 15–20% of candidates from four
sourcing strategies, reported as a finding. Label stability: κ = 0.60 on re-labelling.
The remaining 174 labelled clips are the development split.

External check: on DCASE 2025 Task 3 (30k clips with on/off-screen labels), our visibility
question agrees 66.7%: it almost never silences an off-screen sound (97%) but silences only
35% of on-screen ones — it is timid in the safe direction.

## 6. How it is evaluated (this matters more than it sounds)

Per clip: the system's output is described by a VLM ("a viewer would learn that…"); a
**reference** sentence says what a hearing viewer gets that a deaf viewer misses; an
independent judge (Mistral-7B) scores the match 0–4. Showing nothing when something was
missing = 0; showing nothing when nothing was missing = 4. Two baselines: **blind** (draw every
sound, never look) and **caption** (text instead of a picture).

Three versions of the reference exist, and none is neutral:
- **model-derived** (as in the proposal): written by the system's own models from its own
  detections → circular, favours the system that repeats the detector.
- **human-grounded**: the same, corrected by Adam's label → the only one that can score the
  gate, but needs labels, so a deployed system could not use it.
- **independent** (built 2026-09-14): audio LM + CLAP verification + a different VLM + a
  different writer, none of which saw the system → removes the circularity, but its own
  "is the source visible?" decision is at chance (55%), so it rewards showing.

The rubric is asymmetric: a wrongly withheld picture costs 4 points, a redundant picture
about 1. A gate must be right ~80% of the time when it stays silent just to break even.

## 7. Results (v3 = the current system, 100 clips, 0–4 scale)

| reference | gated | blind | caption | gated − blind |
|---|---|---|---|---|
| model-derived | 2.64 | 3.04 | 2.79 | −0.40 [−0.66, −0.16] |
| **human-grounded** | **3.17** | 3.16 | 2.89 | **+0.01** (tie; win 21 / tie 64 / loss 15) |

Split by scenario (grounded): where **no picture is due** the gate wins, 3.56 vs 3.30
(+0.26, CI excludes 0); where **a picture is due** it loses, 2.78 vs 3.02 (−0.24). A perfect
gate would score 3.25. The gate beats the caption baseline (+0.28). In words: *the system does
the right thing when nothing is missing, and still sometimes stays silent when it should not*
— mostly because the VLM named a source that was on screen but not visibly making the sound.

Other measured results: the previous configuration (v2) lost clearly (−0.24 to −0.70); the
onset fix was measured on 255 DCASE gold onsets — hysteresis cut the timing error from
0.43 s to 0.31 s, the occlusion refinement added nothing; generation was compared with stock
photo retrieval (retrieval +0.17, but 55% of photos had restrictive licences).

## 8. What is running / what is next

- **Running now**: a development-split sweep of the gate's silence rule using the rubric's
  own costs (a withheld picture costs 4× a redundant one). If it picks a different setting,
  the affected test clips are re-run as **v4** and reported next to v3, never instead of it.
- **Open decisions for Adam**: apply the v4 setting (yes/no once the sweep is in); keep or
  drop the occlusion onset step (no measurable effect); whether to run the independent
  reference for v3 given it cannot score the gate.
- **Report**: `docs/report/report.pdf` is a full 13-page draft with one placeholder (v4).
- **Future work** (stated in the report): open-vocabulary audio model as a second opinion
  where AudioSet has no word (e.g. a phone alert); source separation in front of the
  detector; a second annotator; a study with deaf viewers.

## 9. Where everything is

- Code: `src/` (stages), `config.py` (all knobs, commented), `main.py` (run on one video).
- Benchmark and results: `benchmark/` — `tags.json` (labels), `protocol_results_<tag>.json`
  (scores), `paired_stats.json`; `python scripts/paired_stats.py v3 v3_grounded` prints the
  headline tables.
- Cluster (BIU Slurm): `slurm/*.sh` jobs; `bash slurm/sync_data.sh --code-only` uploads code;
  `slurm/RUNBOOK.md`. Long jobs resume from stamps — resubmit, never restart.
- Documents: `docs/project_notes.tex` (everything, dated), `docs/failure_catalogue.md`,
  `docs/plan_robustness.md` (the process rules), `docs/night_report_2026-09-14.md`,
  `LIMITATIONS.md`, `docs/report/`.

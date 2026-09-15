# Executive summary — Visual Augmentation of Audio Semantics for Accessibility

*For someone stepping into this project today (updated 2026-09-15, evening). Written in
plain language; every technical word is explained the first time it appears. Every number
here comes from a file in the repository. The full record is in `docs/project_notes.tex`
(the dated lab notebook), `docs/report/report.pdf` (the thesis draft, 15 pages) and
`LIMITATIONS.md` (what the numbers do not support).*

---

## 1. The problem

A deaf viewer gets a video's speech through subtitles, but not the rest of the soundtrack:
a siren behind the camera, a dog barking in the next room, glass breaking off screen.
Subtitle guidelines even tell captioners to *skip* sounds the picture already shows, and in
practice most ambient sound is never captioned at all.

This project built a system that listens to a video, and for each ambient sound decides
whether the thing making it is **already visible on screen**. If it is not, the system shows
a small picture of that sound next to the video for as long as the sound lasts. If it is
(you can see the dog barking), it shows nothing. That "show only when not visible" rule is
called the **gate**. It is the whole idea, and the thing the evaluation tests.

Rules set by the supervisor: **no training** — the system is built by connecting existing,
freely available AI models; **minimum human involvement** — decisions are made by models,
not by hand-written rules; finish within the week.

## 2. Background: what already exists

- **Subtitles** carry speech well and ambient sound badly. **Sign language** is richer but
  rare and cannot be automated.
- **Sound-awareness apps for deaf users** (HomeSound, SoundWatch) recognise ~20 everyday
  sounds in the home — a knock, an alarm, a crying baby — and users prefer pictures to text.
  They are for real life, not for watching video.
- **Sound detectors**: AI models trained on *AudioSet* (a Google collection of 2 million
  YouTube clips labelled with 527 kinds of sound) can name what is in a soundtrack. We use one
  called **BEATs**.
- **Vision-language models (VLMs)**: AI models you can show pictures to and ask questions in
  words ("what is making this sound?"). We use **Qwen2.5-VL-7B** ("7B" = seven billion
  parameters, a mid-size model).
- **Audio-to-image systems** (e.g. Sound2Scene) turn a sound directly into a picture, without
  looking at the video. They are our main comparison: "draw everything you hear".
- We found no existing system that does all three: detect ambient sound → decide whether it
  is heard-but-not-seen → show a picture. That combination, plus a way to test it, is the
  contribution.

## 3. How the system works (seven stages, code in `src/`)

| stage | what it does | model |
|---|---|---|
| 1 | pulls the audio out of the video | FFmpeg |
| 2 | quick list of objects seen anywhere in the clip (stage 5 makes the real decision) | OWLv2 |
| 3 | writes down any speech — used only as a hint, never drawn | Whisper |
| 4 | **detects** ambient sounds and when they start and stop | BEATs |
| 5 | **reasons**: is the source visible? what should the picture show? which sounds are one source? | Qwen2.5-VL-7B |
| 6 | draws the picture and places it beside the video | FLUX.1 (an image generator) |
| 7 | evaluates the result | see §6 |

### The important behaviours (all decided by models at run time; settings in `config.py`)

**Detection (stage 4).** BEATs scores every 2-second slice of audio, sliding 0.25 s at a
time, giving each of the 527 sound kinds a confidence between 0 and 1. A sound counts if its
confidence reaches **0.35** (the "bar"). Its start is stamped where the confidence begins to
rise, not where it crosses the bar, so the picture is not late (this cut timing error from
0.43 s to 0.31 s on an outside dataset). Speech and music are never shown.

**The visibility decision (stage 5, the gate).** For each detected sound:
1. Its duration is cut into pieces of at most 5 seconds. For each piece, 6 video frames are
   taken, from 1 s before to 1 s after it.
2. The VLM is asked three differently worded questions about those frames:
   - *Name it*: "What in these frames is making the sound of X, or nothing?" If it names
     something, a follow-up checks common sense: "Does a <that> make an X sound?" — so a
     cowboy hat cannot be the source of a vehicle sound.
   - *Is it happening*: "Can you SEE X happening — the source in frame and visibly making that
     sound?" Two options, asked twice with the options swapped; counted only if both answers
     agree (the model otherwise tends to pick the second option whatever it says).
   - *Describe*: "Describe what is in these frames." If the description names the source, or
     common sense says what it describes makes that sound, that counts as a vote.
3. Majority of the three votes = "visible" for that piece. The sound is silenced only if it is
   visible in **every** piece; otherwise the picture stays for the whole sound (one sound,
   one continuous picture — no flicker).

"Visible" means *seen making the sound*: a baby held in its mother's arms is not visibly
crying; a fire-alarm box on a wall is not visibly ringing; a woman with her head back is
visibly laughing.

**What the picture shows.** The VLM writes a 2–5-word phrase describing the *event* — "glass
shattering", "audience clapping" — from the detector's label, the frames (a window or a
bottle?) and a short place name ("kitchen"). Dialogue is never used (it once leaked into
pictures). A second question checks the phrase points back to the right sound. FLUX draws it
on a white background; the panel holds at most three pictures at once, each for at least
1.5 s.

**One picture per sound source.** The detector often gives one sound several names (Laughter,
Giggle, Snicker). AudioSet's own family tree (Giggle is a kind of Laughter) merges them; a
text-similarity check catches paraphrases the tree cannot; and when two names have exactly
the same start and end (a crying baby heard as "Sheep" and "Baby cry"), the VLM looks at the
frames and picks one.

**Speech as a hint.** If people on the soundtrack react to a sound ("whoa, what's that?"),
that sound gets priority when the panel is full. It never overrides the visibility decision.

## 4. Lessons from building it

Every early fix was made on the clip in front of us and broke on the next one. The rules that
came out of that:
1. **Lists fail.** A hand-written list of 30 "visible objects" could never mark laughter as
   visible. Every list became a question to a model.
2. **Ask for a name, not a yes.** A yes/no question to a VLM gets agreement, not evidence.
   A name can be checked.
3. **Read what a model already gives you.** Deduplication was solved by AudioSet's family
   tree after three cleverer ideas failed.
4. **Set thresholds once, on separate clips.** Since 2026-09-13: settings are chosen on a
   *development split* (clips not used for the final score), every failure is logged
   (`docs/failure_catalogue.md`), and a change needs three cases of one pattern.
5. **Evaluation code produces plausible wrong numbers.** An empty result scored 2/4; a
   similarity returned 0.0 for everything after a library update; an audio model was never
   given its audio and answered "none" 100 times. Each was found by reading logs.
6. **Declare the pass bar before running.** Every experiment this week wrote its pass/fail
   rule into the code before the job was submitted (see §8).

## 5. The benchmark (`benchmark/tags.json`)

**274 short clips**, labelled by Adam into four scenarios: **unseen** (an ambient sound whose
source is off screen — a picture is due), **mixed** (some sources visible, some not — a
picture is due), **seen** (the source is on screen — no picture due), **no ambient** (speech or
music only — no picture due). The labelling rule is the one professional captioners use: "does
the picture already tell the viewer this sound is happening?"

**Test set**: 100 clips, 25 per scenario — every reported score uses these. **Development
split**: the other 174 clips, used only for choosing settings. (All 25 mixed clips ended up in
the test set; this mattered once, §8.)

**Label reliability**: Adam re-labelled 60 clips blind and agreed with himself on 78%
(κ = 0.60, "moderate"); the disagreement is mostly on the "unseen" clips, which are exactly the
ones the gate is judged on. One annotator; a second would make this firmer.

**Outside check — DCASE 2025.** A public research dataset of 30,000 five-second indoor clips
in which humans marked, for every sound, when it starts and whether its source is inside the
camera's view. We use it as ground truth we did not make ourselves: on 258 of its sounds our
visibility decision agrees 66.7% — it keeps 97% of off-screen sounds but silences only 35% of
on-screen ones (it errs on the safe side: better a redundant picture than a missing one).

## 6. How the system is scored

There is no user study; scoring is automatic (as the proposal specified). For each clip and
each system:
1. The system produces its output — pictures beside the video, or nothing.
2. A VLM looks at the result and writes what a viewer would learn from the pictures.
3. A **reference sentence** states what a hearing viewer gets from the soundtrack that a deaf
   viewer would miss — or "nothing beyond the picture" if nothing is missing.
4. A **judge** (a text model, Mistral-7B) scores how well (2) matches (3), from 0 to 4.
   Two cases are decided by code: showing nothing when something was missing = **0**;
   showing nothing when nothing was missing = **4**.

**Two baselines** are scored the same way: **blind** — draw a picture for every sound heard,
never look at the video; **caption** — write the sound's name as text instead of a picture.

**Who writes the reference sentence matters.** Three versions:
- *model-derived* (the proposal's): written by the system's own models from its own
  detections. Found to be circular — it rewards whichever system repeats the detector.
- *human-grounded*: the same, corrected by Adam's labels (on seen / no-ambient clips it
  becomes "nothing beyond the picture"). **This is the one the headline uses.**
- *independent*: written by a different set of models that never saw our system. It fixes
  the circularity, but its own guess at "is the source visible?" is no better than a coin
  flip (55% agreement with the labels), so it cannot score the gate. Reported, not used.

**The score has a built-in asymmetry.** A picture wrongly withheld scores 0 where it could
have scored 4 — a loss of ~4. A redundant picture (source already visible) scores ~3 where
silence scores 4 — a loss of ~1. So the gate must be right about 80% of the time when it
stays silent just to break even. This shapes every result below.

**Two things the judge cannot see** (found 2026-09-15): if the detector hears *nothing* on a
clip where a picture was due, the reference itself says nothing is missing and *every*
system gets 4 — so a detector miss is invisible to the score. And the score cannot say
whether a picture was shown at the right moment; only that it was shown.

## 7. Results (v3 = the current system; 100 test clips; score 0–4)

| reference | gated (ours) | blind | caption | ours − blind |
|---|---|---|---|---|
| model-derived | 2.64 | **3.04** | 2.79 | −0.40 (clearly behind) |
| **human-grounded** | **3.17** | 3.16 | 2.89 | **+0.01 — a tie** (21 clips won, 64 tied, 15 lost) |

Split by whether a picture was due (human-grounded):
- **no picture due** (50 clips): ours 3.56 vs blind 3.30 → **+0.26, we win** — we correctly
  stay quiet, blind draws a redundant picture.
- **picture due** (50 clips): ours 2.78 vs blind 3.02 → **−0.24, we lose** — sometimes we
  stay quiet when we should not.

In words: *the system does the right thing when nothing is missing, and still sometimes stays
silent when something is.* A perfect gate would score 3.25. We beat the caption baseline
(+0.28). The earlier version (v2, one week older) lost clearly under every reference; the
week's changes closed the gap to a tie.

**A second view of the same result.** Because the score's 4-to-1 asymmetry comes from the
rubric and not from deaf viewers, we also computed a simpler, judge-free score: per clip, did
the gate open when a picture was due and stay shut when not? Ours: 39 hits, 11 misses, 28
redundant. Blind: 43 hits, 7 misses, 39 redundant. If a redundant picture is treated as
costing a quarter of a miss (the judge's implicit rate), the two tie; if it costs 40% or
more, ours is ahead. So: *the gate is preferred as soon as a redundant picture is counted as
more than about 40% as bad as a missing one.* Whether real viewers feel it that way is an open
question — no user study.

**Where the remaining errors come from** (read from saved decisions, no new runs):
- Of the 11 test clips where we stayed silent and should not have: **7 are the detector** —
  the ambient sound was under loud speech or music and never reached the 0.35 bar; 4 are
  debatable labels (cars visible on screen, traffic noise labelled off-screen); 0 are timing.
- Of the redundant pictures on the development clips, **about 80% are "phantoms"** — sounds
  the detector reports confidently that are not there at all (a whale at a Christmas market,
  roaring lions at a helicopter landing). The gate is right that nothing visible makes them;
  the picture is a detector mistake.

So the **detector** is the bottleneck on both sides, more than the visibility decision.

## 8. What was tried against the detector, and why it stopped

Each attempt had a pass/fail rule written down before it ran, and was reviewed by an
independent AI reviewer ("Fable"). None passed; all are reported honestly.

| attempt | idea | result |
|---|---|---|
| threshold sweep | try 16 gate settings on the development clips | the current setting was already best |
| CLAP second opinion | a second audio model (CLAP) must agree with BEATs | CLAP recognises only 50% of real sounds at the loosest useful setting — unusable |
| PANNs agreement | a second AudioSet detector must agree | removed 24 of 77 phantoms (needed 40); both detectors share the same confusions |
| persistence + rank rule | count a sound only if it stays among the top-3 for ≥1 s, instead of one peak | same trade-off curve as the current rule, just shifted |
| bigger VLM (32B) for the gate | four times larger vision model | recognises visible sources better (67% vs 35%) but wrongly silences 25% of off-screen sounds (was 3%) — worse under the score |
| audio language model | ask Qwen2-Audio "which of these sounds do you hear?" | small test passed (6/10 phantoms vetoed, 9/10 real kept); full run removed only 25 of 77 |
| source separation (Demucs) | remove speech before detecting | assessed, not run: removes speech only, and 4 of the 7 masked misses are under music |

Conclusion for the thesis: with today's off-the-shelf sound detectors, invented sounds and
missed quiet sounds are two ends of one dial; no filter on top fixes both.

## 9. Decisions taken and still open

Decided: keep `ONSET_CAM` (a timing refinement measured to make no difference; v3 was
evaluated with it on). Skip the independent reference for v3 (it cannot score the gate).
Stop detector experiments (§8).

Open for Adam: read the report; decide on submission polish (bibliography entries marked
`[verify]`, two figures — pipeline diagram and a per-clip example, three demo videos
re-rendered with v3).

Future work (in the report): a detector-only benchmark on the labelled clips; a speech-aware
detector measured against it; an audio model with an open vocabulary for sounds AudioSet has
no name for ("phone alert"); a second annotator; a study with deaf viewers.

## 10. Where everything is

- Run one video: `python main.py <video>`; settings in `config.py`.
- Scores: `benchmark/protocol_results_<tag>.json`; `python scripts/paired_stats.py v3
  v3_grounded` prints the tables; `scripts/cost_sensitivity.py` the judge-free view.
- Experiment records: `benchmark/gate_setting.json`, `clap_setting.json`,
  `panns_agree.json`, `persist_rank_setting.json`, `eval_dcase_visibility_32b.json`,
  `audio_llm_eval.json`.
- Cluster (BIU Slurm, VPN via F5 at access.biu.ac.il): `bash slurm/sync_data.sh --code-only`
  uploads code; jobs are `slurm/*.sh`; the main experiment is `sbatch slurm/job_v3.sh`
  (~10 h; resumable — resubmit, never restart). A Hugging Face token in `~/.bashrc` on the
  cluster is needed for the gated models.
- Documents: `docs/report/` (thesis), `docs/project_notes.tex` (lab notebook),
  `docs/failure_catalogue.md`, `docs/plan_robustness.md` (process rules), `LIMITATIONS.md`.

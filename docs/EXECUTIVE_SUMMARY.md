# Executive summary — Visual Augmentation of Audio Semantics for Accessibility

*For someone stepping into this project today (updated 2026-09-15, night). Plain
language; every technical word is explained the first time it appears, and the recurring ones
are collected in the glossary at the end (§13). Every number here comes from a file in the
repository. The full record is in `docs/project_notes.tex` (the dated lab notebook),
`docs/report/report.pdf` (the thesis draft, 16 pages) and `LIMITATIONS.md` (what the numbers
do not support).*

---

## 1. The problem, in one minute

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

## 2. The headline result, in three sentences

On 100 test clips, scored 0–4 by an automatic judge, our gated system scores **3.17** and the
simplest alternative — "draw a picture for every sound you hear, never look at the video"
(the **blind** baseline) — scores **3.16**: a tie. We win on the 50 clips where no picture was
needed (we stay quiet, blind draws something redundant) and lose on the 50 where one was
needed (we sometimes stay quiet when we should not). The reason we lose is mostly not the
gate but the **sound detector** underneath it. Seven attempts to improve that detector all
failed; each had its pass/fail rule written down before it ran. That they all failed is the
thesis's second finding.

## 3. The story of the project (what happened, in order)

- **Early August.** Project pivoted to its current form: pictures for heard-but-not-seen
  sounds. Proposal written with a seven-stage pipeline and an automatic evaluation.
- **Up to 8 Sept.** First working prototype: a sound detector (PANNs), a vision-language
  model to decide visibility, an image generator (SDXL), a side panel. Adam labelled a
  274-clip benchmark. Many hand fixes made on one clip broke on the next (lessons in §7).
- **8–9 Sept.** Better object finder (OWLv2); evaluation split into "describe" and "judge"
  passes; FLUX replaces SDXL for pictures; an experiment showed a *stock photo* beats a
  generated picture by +0.17 points on the 0–4 score, but 55% of the photos had restrictive licences, so generation
  stayed.
- **12 Sept.** BEATs replaces PANNs as the detector (fixed 11 of 14 logged complaints).
- **13 Sept.** Process rules adopted: clips split into a development set (for choosing
  settings) and a test set (for the final score); every failure logged; a pass bar declared
  before every experiment. First outside check on the DCASE dataset (visibility decision
  66.7% right). Version **v3** of the system sent to run on the university's GPU computers (the
  "cluster"); the run took 10 hours.
- **14 Sept.** The previous version (**v2**) loses to blind under every way of scoring
  (−0.70 at worst). Found that the proposal's reference sentences were written by the
  system's own models — circular. Built a fully independent reference; found that on the key question — is the source
  visible? — it agrees with Adam's labels only 55% of the time, a coin flip, so it cannot
  score the gate. **v3 lands:
  a tie (+0.01).** A sweep of 16 gate settings on the development clips shows the current
  one is already best. Thesis draft written.
- **15 Sept.** Seven detector improvements tried, all failed their declared bars (§10). A
  second, judge-free way of scoring (cost-sensitivity, §9) shows the gate comes out ahead if one unneeded picture counts as at least 40% as bad as
  one missing picture. Two things the judge
  cannot see were found (§8). Figures, bibliography, and this summary finished. Evening:
  separation-before-detection measured and failed; a with-sound / without-sound
  reference (Adam's idea) pre-registered, piloted on dev, and failed (§8); a "gap-closing"
  score that asks no visibility question pre-registered and failed its sanity check (§8).

## 4. Background: what already exists

- **Subtitles** carry speech well and ambient sound badly. **Sign language** is richer but
  rare and cannot be automated.
- **Sound-awareness apps for deaf users** (HomeSound, SoundWatch) recognise ~20 everyday
  sounds in the home — a knock, an alarm, a crying baby — and users prefer pictures to text.
  They are for real life, not for watching video.
- **Sound detectors**: AI models trained on *AudioSet* (a Google collection of 2 million
  YouTube clips labelled with 527 kinds of sound) can name what is in a soundtrack. We use one
  called **BEATs**.
- **Vision-language models (VLMs)**: AI models you can show pictures to and ask questions in
  words ("what is making this sound?"). We use **Qwen2.5-VL-7B** (mid-size; runs on one GPU).
- **Audio-to-image systems** (e.g. Sound2Scene) turn a sound directly into a picture, without
  looking at the video. They are our main comparison: "draw everything you hear".
- We found no existing system that does all three: detect ambient sound → decide whether it
  is heard-but-not-seen → show a picture. That combination, plus a way to test it, is the
  contribution.

## 5. How the system works (seven stages, code in `src/`)

| stage | what it does | model |
|---|---|---|
| 1 | pulls the audio out of the video | FFmpeg |
| 2 | quick list of objects seen anywhere in the clip (stage 5 makes the real decision) | OWLv2 |
| 3 | writes down any speech — used only as a hint, never drawn | Whisper |
| 4 | **detects** ambient sounds and when they start and stop | BEATs |
| 5 | **reasons**: is the source visible? what should the picture show? which sounds are one source? | Qwen2.5-VL-7B |
| 6 | draws the picture and places it beside the video | FLUX.1 (an image generator) |
| 7 | evaluates the result | see §8 |

### The important behaviours (all decided by models at run time; settings in `config.py`)

**Detection (stage 4).** BEATs looks at the audio two seconds at a time, moving forward a
quarter of a second per step, and gives each of the 527 sound kinds a confidence between 0
and 1. A sound counts if its confidence reaches **0.35** (the "bar"). Two timing refinements:
the start is stamped where the confidence *begins to rise*, not where it crosses the bar
("hysteresis"), so the picture is not late — this cut timing error from 0.43 s to 0.31 s on
an outside dataset; and a second timing step ("occlusion", which handles a
sound briefly hidden by a louder one) was also kept: it made no measurable difference, but
v3 was scored with it on. Speech and music are never shown.

**The visibility decision (stage 5, the gate).** For each detected sound:
1. Its duration is cut into pieces of at most 5 seconds (a "stretch"). For each stretch, 6
   video frames are taken, from 1 s before to 1 s after it.
2. The VLM is asked three differently worded questions about those frames (three "votes"):
   - *Name it*: "What in these frames is making the sound of X, or nothing?" If it names
     something, a follow-up checks common sense: "Does a <that> make an X sound?" — so a
     cowboy hat cannot be the source of a vehicle sound.
   - *Is it happening*: "Can you SEE X happening — the source in frame and visibly making that
     sound?" Two options, asked twice with the options swapped; counted only if both answers
     agree (otherwise the model tends to pick whichever option is listed second, regardless of the
     frames).
   - *Describe*: "Describe what is in these frames." If the description names the source, or
     common sense says what it describes makes that sound, that counts as a vote.
3. Majority of the three votes = "visible" for that stretch. The sound is silenced only if it
   is visible in **every** stretch; otherwise the picture stays for the whole sound (one
   sound, one continuous picture — no flicker).

"Visible" means *seen making the sound*: a baby held in its mother's arms is not visibly
crying; a fire-alarm box on a wall is not visibly ringing; a woman with her head back is
visibly laughing.

**What the picture shows.** The VLM writes a 2–5-word phrase describing the *event* — "glass
shattering", "audience clapping" — from the detector's label, the frames (a window or a
bottle?) and a short place name ("kitchen"). Dialogue is never used (it once leaked into
pictures). A second question checks the phrase points back to the right sound. FLUX draws it
on a white background; the panel holds at most three pictures at once, each for at least
1.5 s.

**One picture per sound source ("dedup").** The detector often gives one sound several
names (Laughter, Giggle, Snicker). AudioSet's own family tree (Giggle is a kind of Laughter)
merges them; a check on word meaning catches names for the same sound that the family tree does not
list; and when two
names have exactly the same start and end (a crying baby heard as "Sheep" and "Baby cry"),
the VLM looks at the frames and picks one.

**Speech as a hint.** If people on the soundtrack react to a sound ("whoa, what's that?"),
that sound gets priority when the panel is full. It never overrides the visibility decision.

## 6. One clip through all seven stages

Test clip `ly_ambulance_(siren)_-yPSgCn.mp4` (18 s): a street filmed from a car; an
ambulance siren is heard but no ambulance is ever in frame. This is example (c) in the
report's Figure 2.

1. **Audio out** (FFmpeg): an 18-second wav.
2. **Objects seen** (OWLv2): road, van, cars, train — a list for later.
3. **Speech** (Whisper): none.
4. **Detect** (BEATs): "Siren" confidence rises above 0.35 almost at once and stays there —
   highest confidence 0.84, heard from 0.0 to 17.8 s. "Vehicle" is also heard (peak 0.61).
5. **Reason** (Qwen2.5-VL, for "Siren"): the 18 s become four stretches. In every one,
   *Name it* answers "nothing", *Is it happening* answers "no", and only *Describe* counts
   as a yes (the description mentions vehicles that could carry a siren). One vote of three
   → not visible, in all four stretches → **show a picture for the whole sound**. Phrase
   written: "ambulance siren blaring". ("Vehicle" was visible in three stretches but not the
   fourth, so it also gets a picture.)
6. **Draw** (FLUX): an ambulance with lights on, white background, in the side panel for the
   whole clip.
7. **Score** (see §8): the reference says a hearing viewer gets "the siren and vehicle
   sounds" that a deaf viewer would miss. A VLM looks at our panel and writes "an ambulance
   with its lights and siren activated"; the judge gives **4**. Blind also drew pictures but
   its VLM description talked about brake lights and traffic, not a siren: **3**. The
   text-caption baseline ("The soundtrack contains: Siren, Vehicle") got **2**.

## 7. Lessons from building it

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
   (`docs/failure_catalogue.md`), and a setting is changed only after the same kind of failure appears on three clips.
5. **Evaluation code produces plausible wrong numbers.** An empty result scored 2/4; a
   similarity returned 0.0 for everything after a library update; an audio model was never
   given its audio and answered "none" 100 times. Each was found by reading logs.
6. **Declare the pass bar before running.** Every experiment this week wrote its pass/fail
   rule into the code before the job was submitted (see §10).

## 8. How a clip is scored (and why the score is lopsided)

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

**Two real clips show the lopsidedness.**
- *Police car on screen* (`un_police_car_siren_WEROVWDp.mp4`, Figure 2a): the siren's source
  is visible, so the reference says "nothing beyond the picture". We stay silent → **4**.
  Blind draws a police car anyway → **3**. Being right here gains us **1 point**.
- *New York street* (`ambient_citywalk_nyc_2627.mp4`, Figure 2b): cars are visible, but the
  labelled sound is off-screen traffic, so the reference says a picture is due. Our gate saw
  the cars and stayed silent → **0**. Blind drew traffic → **3**. Being wrong here costs us
  **3–4 points**.

So a wrong silence costs about four times what a redundant picture costs. To come out even, the gate has to
be right in about 4 of every 5 cases where it stays silent. This shapes every
result below. Whether deaf viewers feel the same 4-to-1 is unknown (§9 looks at other rates).

**Who writes the reference sentence matters.** Three versions:
- *model-derived* (the proposal's): written by the system's own models from its own
  detections. Found to be circular — it rewards whichever system repeats the detector. On
  the police-car clip above, this reference said the siren was *missing* (because the
  detector heard one) and gave our correct silence a **0**.
- *human-grounded*: the same, corrected by Adam's labels (on seen / no-ambient clips it
  becomes "nothing beyond the picture"). **This is the one the headline uses.** On the
  police-car clip it gives the correct silence its **4**.
- *independent*: written by a different set of models that never saw our system. It fixes
  the circularity, but its own guess at "is the source visible?" is no better than a coin
  flip (55% agreement with the labels), so it cannot score the gate. Reported, not used.
- *with sound / without sound* (Adam's idea, 15 Sept evening): one audio-visual model
  describes the clip with the soundtrack and again muted, on the same frames; what it
  "hears" and did not already see is the reference. Rules and pass bar written down and
  committed before any code (`docs/prereg_av_reference.md`); tried on the 156 dev clips
  with Qwen2.5-Omni. Its "is anything missing?" decision scored **49.5%** (chance is 50%;
  bar was 67.7%), and with the sound off it still claimed to hear something on 37% of
  clips. The model calls off-screen sources "seen and heard" (a helicopter overhead, a
  church bell behind a pointing hand). Same lesson as the independent reference: today's
  open models cannot tell whether a sound's source is in view. Reported as a failed pilot.
- *gap closing* (Adam's second idea, same night): no visibility question at all — the model
  answers four fixed questions (what is happening / anything dangerous / mood / anything
  out of view) with sound, muted, and muted on slightly shifted frames; a system would be
  scored by how much its panel moves the muted answers toward the hearing ones. Pre-registered
  with a sanity check on 60 dev clips (`docs/prereg_gap_closing.md`): the answers change as
  much from shifting the frames as from adding the sound, and no more on clips with an
  ambient sound than without (AUROC 0.53, bar 0.75). **Failed**; not run on test. Same
  lesson: with today's open audio-visual models, no automatic "hearing viewer" is reliable
  enough to score against.

**Two things the judge cannot see** (found 2026-09-15): if the detector hears *nothing* on a
clip where a picture was due, the reference itself says nothing is missing and *every*
system gets 4 — so a detector miss is invisible to the score. And the score cannot say
whether a picture was shown at the right moment; only that it was shown.

**How reliable is the judge?** A second, different judge model gave a score within one
point of Mistral's on 97% of clips and reproduced the gated-vs-blind gap — but it moved the caption
baseline by +0.61, so any ranking involving the text baseline depends on which judge you ask.

## 9. Results (v3 = the current system; 100 test clips; score 0–4)

| reference | gated (ours) | blind | caption | ours − blind |
|---|---|---|---|---|
| model-derived | 2.64 | **3.04** | 2.79 | −0.40 (clearly behind) |
| **human-grounded** | **3.17** | 3.16 | 2.89 | **+0.01 — a tie** (21 clips won, 64 tied, 15 lost) |

Statistically, the true difference could be anywhere between −0.15 and +0.18 points (a 95%
confidence interval): the data cannot tell the two apart. An average over 25 clips is
uncertain by about ±0.2 points, so small differences below are indications, not proof.

Split by whether a picture was due (human-grounded):
- **no picture due** (50 clips): ours 3.56 vs blind 3.30 → **+0.26, we win** — we correctly
  stay quiet, blind draws a redundant picture.
- **picture due** (50 clips): ours 2.78 vs blind 3.02 → **−0.24, we lose** — sometimes we
  stay quiet when we should not.

In words: *the system does the right thing when nothing is missing, and still sometimes stays
silent when something is.* A perfect gate (using Adam's labels as the gate) would score
3.25 — the "oracle" ceiling. We beat the caption baseline (+0.28). The earlier version (v2,
one week older) lost clearly under every reference; the week's changes closed the gap to a
tie.

**A second view of the same result (cost-sensitivity).** Because the 4-to-1 lopsidedness
comes from the scoring rubric and not from deaf viewers, we also computed a simpler,
judge-free score: per clip, did the gate open when a picture was due and stay shut when
not? Ours: 39 hits, 11 misses, 28 redundant. Blind: 43 hits, 7 misses, 39 redundant. If a
redundant picture is treated as costing a quarter of a miss ((the rate built into the judge's scoring, §8)), the
two tie; if it costs 40% or more, ours is ahead. So: *the gate is preferred as soon as a
redundant picture is counted as more than about 40% as bad as a missing one.* Whether real
viewers feel it that way is an open question — no user study.

**Where the remaining errors come from** ((found by reading the decisions saved during the test run; nothing was re-run)):
- Of the 11 test clips where we stayed silent and should not have: **7 are the detector** —
  the ambient sound was under loud speech or music and never reached the 0.35 bar (a "masked
  miss"); 4 are debatable labels (cars visible on screen, traffic noise labelled off-screen,
  like Figure 2b); 0 are timing.
- Of the redundant pictures on the development clips, **about 80% are "phantoms"** — sounds
  the detector reports confidently that are not there at all (a whale at a Christmas market,
  roaring lions at a helicopter landing). The gate is right that nothing visible makes them;
  the picture is a detector mistake.

So the **detector** is the bottleneck on both sides, more than the visibility decision.

**Outside check — DCASE 2025.** A public research dataset of 30,000 five-second indoor clips
in which humans marked, for every sound, when it starts and whether its source is inside the
camera's view. We use it as ground truth we did not make ourselves: on 258 of its sounds our
visibility decision matches the human mark 66.7% of the time: it correctly shows a picture
for 97% of off-screen sounds, but correctly stays silent for only 35% of on-screen ones (it errs on the safe side: better a redundant picture than a missing one).

## 10. What was tried against the detector, and why it stopped

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
| source separation (Demucs) | remove voices, drums and bass before detecting; detect on what is left | ran 15 Sept evening: the detector got *worse* — found fewer of the buried sounds (9.5% → 4.8%), lost two thirds of the clear ones, and lost 17 of 23 real dev detections; it was trained on mixed audio and does not understand separated audio |

Example of a phantom the second opinions could not remove: "whale vocalization" at a
Christmas market (confidence 0.36). The same phantom appears on other crowd clips (a crowd
of people, an aquarium walk), so the detector seems to hear crowd murmur as whale song. The
second detector (PANNs) makes the same confusion, so asking it to agree does not help; and a
confidence bar high enough to remove the whale also removes real quiet sounds elsewhere.

Conclusion for the thesis: with today's off-the-shelf sound detectors, one setting controls both
problems — raising the bar removes invented sounds but also loses real quiet ones — and no
filter on top fixes both.

## 11. The benchmark (`benchmark/tags.json`)

**274 short clips**, labelled by Adam into four scenarios: **unseen** (an ambient sound whose
source is off screen — a picture is due), **mixed** (some sources visible, some not — a
picture is due), **seen** (the source is on screen — no picture due), **no ambient** (speech or
music only — no picture due). The labelling rule is the one professional captioners use: "does
the picture already tell the viewer this sound is happening?"

**Test set**: 100 clips, 25 per scenario — every reported score uses these. **Development
split**: the other 174 clips, used only for choosing settings. (All 25 mixed clips ended up in
the test set; this mattered once, §10.)

**Label reliability**: Adam re-labelled 60 clips without seeing his first labels and agreed
with himself on 78% of them (on the standard agreement scale this counts as "moderate"); the disagreement is mostly on the "unseen"
clips, which are exactly the ones the gate is judged on. One annotator; a second would make
this firmer.

## 12. Decisions taken, what is open, and how to reproduce the numbers

**Decided:** keep the occlusion timing refinement (measured to make no difference; v3 was
evaluated with it on). Skip the independent reference for v3 (it cannot score the gate).
Stop detector experiments (§10). Bibliography verified, two figures added, demos re-rendered (15 Sept).

**Demo videos** (re-rendered with v3 on 15 Sept, Fable's choice): the three Figure 2
clips, in `data/output/demos_v3/` (clean) and `data/output/demos_v3_debug/` (the phrase and
every raw detection with the gate's verdict printed under the panel, so an empty panel reads
as "sound heard, source judged visible" rather than "nothing ran"). Not in git (videos);
`sbatch slurm/job_demos_v3.sh` re-creates them in ~12 minutes.

**Open for Adam:** read the report; final proofread.

**Future work** (in the report): a detector-only benchmark on the labelled clips; a
speech-aware detector measured against it; an audio model with an open vocabulary for sounds
AudioSet has no name for ("phone alert"); a second annotator; a study with deaf viewers.

**Reproduce the headline without a GPU** (the per-clip results are already in the
repository):

```
python scripts/paired_stats.py v3 v3_grounded      # 3.17 vs 3.16, the scenario split
python scripts/cost_sensitivity.py                  # 39/11/28 vs 43/7/39, crossover r = 0.40
```

**Run everything again** (GPU cluster, ~10 h): `bash slurm/sync_data.sh --code-only` uploads
the code; `sbatch slurm/job_v3.sh` runs the 100 test clips through the system and the two
baselines and scores them (resumable: if it stops, resubmit the same job, never start over).
Run one video locally: `python main.py <video>`; settings in `config.py`.

**Where everything is.**
- Scores: `benchmark/protocol_results_<tag>.json` (one line per clip and system, with the
  reference, the description and the judge's reason).
- Experiment records: `benchmark/gate_setting.json`, `clap_setting.json`,
  `panns_agree.json`, `persist_rank_setting.json`, `eval_dcase_visibility_32b.json`,
  `audio_llm_eval.json`.
- Cluster (BIU Slurm, VPN via F5 at access.biu.ac.il): jobs are `slurm/*.sh`. Some models require accepting a licence on Hugging Face (unrelated to our gate); the
  access token for them lives in `~/.bashrc` on the cluster.
- Documents: `docs/report/` (thesis), `docs/project_notes.tex` (lab notebook),
  `docs/failure_catalogue.md`, `docs/plan_robustness.md` (process rules), `LIMITATIONS.md`.

## 13. Glossary

- **gate** — the rule "show a picture only if the sound's source is not visible". The idea
  under test.
- **gated / ours / proposed** — our system. **blind** — draw a picture for every sound heard,
  never look at the video. **caption** — write the sound's name as text.
- **bar (threshold)** — the confidence (0.35) a sound must reach to count as detected.
- **phantom** — a sound the detector reports that is not there (a whale at a market).
- **masked miss** — a real sound the detector does not reach the bar on because speech or
  music is louder.
- **onset** — the moment a sound starts. **hysteresis** — stamping the onset where the
  confidence begins to rise, not where it crosses the bar.
- **stretch** — a piece of at most 5 s of one sound; the visibility question is asked per
  stretch. **vote** — one of the three differently worded visibility questions.
- **dedup** — merging several detector names for one sound source into one picture.
- **scenario** — Adam's label for a clip: unseen / mixed / seen / no ambient.
- **dev (development) split** — the 174 clips used to choose settings. **test set** — the
  100 clips every reported score uses.
- **reference (sentence)** — what a hearing viewer would get that a deaf viewer misses; the
  thing the judge compares against. Three versions: model-derived, human-grounded,
  independent.
- **judge** — the text model that scores a system's output against the reference, 0–4.
- **oracle** — the score if the gate used Adam's labels directly (3.25); the ceiling.
- **cost-sensitivity / r** — the judge-free view: how bad a redundant picture is, as a
  fraction of a missing one. The gate wins from r = 0.40.
- **DCASE 2025** — a public dataset with human marks for when each sound starts and whether
  its source is in view; our outside check.
- **VLM** — vision-language model, a model you can show pictures to and question in words.
- **AudioSet** — Google's 527-kind sound vocabulary; BEATs, PANNs and CLAP all learned from it.

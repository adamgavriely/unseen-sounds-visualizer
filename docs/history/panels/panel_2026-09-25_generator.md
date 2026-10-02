# Panel, 2026-09-25 — better pictures: the prompt AND the generator (2 reviewers, 3 rounds)

Adam, after rating 172 pictures blind: *"most images are very bad in my opinion. If we have a bad image
generator we need to improve the prompt AND the generator model."* Reviewers: G1 (generators, SOTA, compute)
and G2 (visual communication, prompts, DHH viewers). Read-only; the repo is `P:\MscProj`.

## What the pictures are for

A small panel beside a video. When a sound happens whose source is off screen, a picture of that source
appears for a deaf or hard-of-hearing viewer, who glances at it for about a second. It must say *what is
making the sound* with no caption. It must never suggest something that is not there.

## The measured facts (Adam's blind round 2, 54 sounds on 50 never-annotated clips)

Adam saw each picture with no name and typed what he thought was making a sound.

    arm                                          right     vague   wrong   can't tell
    today: FLUX.1-schnell, 4 steps, today's text  14/54     8       3       28
    N0:    Qwen-Image-2512, 50 steps, same text   26/54    12       5       11
    N:     Qwen-Image-2512 + V3 text              32/54     8       3       11

The generator swap gave most of the gain (+22 points, significant); the new text gave +11 (borderline).
Qwen-Image still fails about 40 %. Where N failed:
* **can't tell (11):** whoosh x2, thunder x2, car alarm x2, smash x2, train horn, rain on a surface,
  cupboard opening — sounds with no obvious single object, or an object that does not show its sound.
* **vague (8):** a chirping bird read as "bird" (x5), church bell as "bell", rain on a window.
* **wrong (3):** sheep read as goat, a sighing woman read as sneezing, and a Thunk drawn as a door (a false
  message: the audio never said door).
Earlier round (DEV, family name shown): FLUX ~30 % readable vs Qwen-Image ~48 % pooled over seeds; rater +
seed noise 25-30 %.

## The current setup

* Subject text: Qwen3.8-27B writes 3-6 words (`src/stage5_cross_modal_analysis/reason.py`, DEPICT_PROMPT_V3
  and V3.1 / PICTURE_SCENE: the whole source caught making this sound, no place; the scene may add a
  qualifier).
* Image prompt: `subject + ", plain white background, clearly visible"` (`stage6_visual_augmentation.plain_prompt`);
  six earlier rounds of art direction ("icon", "line art", "sticker", "flat cartoon") all made it worse.
* Negative prompt: scenery, landscape, room, street, buildings, sky background, text, letters, watermark.
* Qwen-Image-2512, 1024 px, 50 steps, true CFG 4.0, seed fixed; blank guard redraws a white picture.
* Shown at about 384 px in a side panel, for the length of the sound.

## Constraints

* Open weights, run on the BIU cluster: H200 141 GB / A100 80 GB / L40S 48 GB, 4-hour jobs, **no internet
  on compute nodes** (weights downloaded on the login node first; ~38 GB free in home, more on request).
* Licence must allow research use in an MSc thesis.
* A paid H100 pod exists only for what BIU cannot do, and only with Adam's OK.
* ~20 s per picture is acceptable (offline pipeline); a clip has 1-4 pictures.
* No training (the project is training-free); inference-time techniques are allowed (prompt rewriting,
  best-of-N with a checker, editing models, LoRA-free control).

## Round 1 questions (at most ten lines each; cite model cards / papers with links)

1. **Generator.** Which open text-to-image models (as of September 2026) are the strongest for *one
   recognisable object doing one thing, clean background, at a glance*? Name 2-3 to test against
   Qwen-Image-2512, each with licence, VRAM, speed, and why. (G1 leads; G2 comments.)
2. **Prompt.** What should the image prompt say (and the subject text) so that a picture of a *sound*
   reads at a glance — e.g. visible sound cues (motion lines, an open beak, a ringing bell's vibration
   marks), style (photo vs clean illustration vs pictogram), framing? Earlier art-direction attempts failed
   on FLUX — is that still true for stronger models? (G2 leads; G1 comments.)
3. **The hard sounds** (whoosh, thunder, car alarm, smash, rain on a surface, train horn): what should be
   drawn, or should some sounds get a different treatment (an icon, a text card)?
4. **How to test it** so Adam's time is spent well: arms, clips (the 50 already rated are now picture-DEV;
   slice B is reserved for V3.1), seeds, and a pass bar written before any picture is drawn.

---

# Round 2 — merged plan (sign or amend)

**Agreed in round 1**
* The pictures are mostly *legible but do not stand for a sound* (G2): the content is clear (a sword, a
  beacon, a cloud, a horn) but Adam cannot name a sound from it. So fix **what is drawn**, and test
  stronger generators — not style.
* **Sourceless sounds** (Whoosh, Thunk, Smash/crash with no object, Bang): **no generated picture**; a fixed
  card. Both of you agree; you differ on the form (G2: onomatopoeia burst [WHOOSH] as in DCMP captioning;
  G1: a hand-picked pictogram per family from a CC icon set).
* Benchmarks exclude models, they do not rank them; Adam's blind naming is the bar.

**Proposed plan (GP-1)**
1. **Screening round of generators**, identical V3.1 text, `negative_for` where supported, `seed_of(item)`:
   N-control (Qwen-Image-2512) / HiDream-O1-Image (full, MIT) / Qwen-Image-2.1 (research licence, RGBA) /
   Mage-Flow (MIT). FLUX.2 [dev] only if ~90 GB extra disk is granted. (G2 also named Qwen-Image 2.0 and
   Ideogram 4.0 — G1, please confirm or exclude.)
2. **Image-prompt rules** (G2), added to `plain_prompt`, as one arm on the winning generator: the whole thing,
   fully in frame; caught mid-act, with the act's visible effect; one large subject filling the picture;
   plain white background. Then one arm with LLM expansion to ~40 words under V3.1's no-new-noun guard (G1).
3. **Hard sounds, a list rule on the label, never the model:** (a) label names a thing → generated with the
   rules; (b) label is a sound with a canonical maker (thunder, rain on a surface, car alarm, train horn,
   church bell) → a fixed template subject per recurring label (G2) *or* a pictogram (G1) — decide which;
   (c) sourceless → the fixed card. Count how many DEV and TEST sounds each group covers first.
4. **Rule 2 widened** (G2): a picture that clearly shows a thing the audio never established is a false
   message whatever Adam types (the sword pictures escaped because he typed "can't tell"). Checked against
   the subject text before Adam's file is opened.
5. **Glance fidelity** (G2): each picture shown at 384 px for 1.5 s, then hidden, then the answer box.
6. **Which clips.** Correction to G1: Adam has **not** seen the key; he saw only the pictures. But he has seen
   pictures of these 54 sounds, so a re-rating is not fully naive. Slice B (30 clips, 43 sounds) is reserved
   for V3.1 and not yet rated. The unannotated pool has only seen/no-ambient clips left.
7. **Pass bar, written before drawing:** a generator is adopted only if right ≥ control + 8/54, wrong ≤
   control's, zero false messages under the widened rule, Adam's repeats ≥ 80 %.

## Round 2 questions (at most eight lines each)
1. Sign or amend GP-1 points 1-7.
2. Point 3(b): template subject or pictogram, and point 3(c): onomatopoeia card or pictogram — one answer
   each, with the reason a supervisor would accept.
3. Point 6: which clips for the screening round, and which for confirmation, given the thin pool?

---

# Round 3 — final plan GP-2 (sign, or name the one point you cannot sign)

**Settled in round 2:** Qwen-Image 2.0 (never published) and Ideogram 4.0 (JSON-prompt model, fp8 only) are
excluded. Candidates: **HiDream-O1-Image** (MIT, 35 GB), **Qwen-Image-2.1** (research licence, 33 GB + its
18 GB official prompt rewriter), **Mage-Flow** (MIT, 17 GB; Microsoft repo gated — needs Adam to accept the
terms; the ungated mirror is ComfyUI layout). Control: Qwen-Image-2512. Ties keep 2512. Group (b) sounds
(thunder, rain on a surface, car alarm, train horn, church bell, glass) get a **fixed template subject per
label**, same generator (both). LLM expansion must pass the list and person guards *after* rewriting; any
new noun → fall back to the short prompt, never retry (both); on Qwen-Image-2.1 the rewriter is its own
PE-T2I model (G1).

**GP-2, the plan**
1. **Compute gate first** (no rating): each candidate loads on one A100/H200, draws ≤ 20 s at 1024 px,
   accepts a negative prompt (or the missing negative is recorded). Downloaded one at a time on the login
   node; **disk is short** (86 GB needed, 37 GB free) — Adam decides: request more, or run-then-delete.
2. **Screening, picture-DEV (the 54 rated sounds), one sitting**: control / HiDream / Qwen-Image-2.1 /
   Mage-Flow (if its terms are accepted), identical V3.1 text, **all arms at seed_of(item)+1** so the control
   is not the picture Adam already saw (G2), shuffled together, 10 repeats, **glance fidelity** (384 px,
   1.5 s, then the answer box).
3. **Prompt arms on the screening winner**: + image-prompt rules (whole thing in frame, mid-act with its
   visible effect, one large subject) / + guarded expansion; group (b) templates on.
4. **Group (c), sourceless (whoosh, thunk, smash with no object):** G2 says onomatopoeia card (the DCMP
   convention DHH viewers already read); G1 says a language-free pictogram (the panel's contract is "no
   text", and a word card is the text control). **Compromise: a comic burst glyph with the word inside it,
   as its own arm, counted separately and never as a picture win.** ~9 % of DEV sounds, ~20 % of gold
   human-rated off-screen sounds.
5. **Confirmation on fresh clips** (G1): run the shipped detector + gate over the 489 unannotated
   `unsorted` clips, take the first 30 by filename hash with ≥ 1 off-screen sound in group (a) or (b),
   **freeze the list in the prereg before any picture is drawn**; winner vs control, two seeds. The key is the
   detector's source, as in round 2, so no annotation is needed.
6. **Slice B stays for V3.1** (G1) — its pages are already published; V3.1 is re-run on the confirmed winner
   later if one wins, and the generator gain is otherwise reported as additive.
7. **Rule 2 widened** (G2): a picture clearly showing a thing the audio never established is a false message
   whatever Adam types; flagged first mechanically (the checker's logged OBJECT line vs the subject), then by
   eye, before Adam's file is opened.
8. **Pass bar** (written now): adopt only if right ≥ control + 8/54 on screening **and** the confirmation
   difference is positive with its CI excluding zero; wrong ≤ control's; zero false messages under the
   widened rule; repeats ≥ 80 % (if under, lengthen the reveal to 2.5 s, never remove it). The same bar for
   each prompt arm, against its own generator's control.

## Round 3 question (at most six lines)
Sign GP-2, or name the single point you cannot sign and the fix.

## Round 3 outcome — GP-2 signed by both, with one shared fix

Both reviewers signed GP-2 (G2 including the burst-with-word compromise, drawn once per word as a fixed asset).
Both flagged the same power problem: 30 confirmation clips cannot exclude zero for a real ~+10-15 point gain.
**Fix adopted: the confirmation set is 50 fresh `unsorted` clips** (same filename-hash rule, list frozen in the
prereg before any picture), one seed per item for both arms, a second seed only on group (b) sounds. G1:
if HiDream-O1-Image (full, 50 steps, pixel space) misses 20 s per picture at the compute gate, fall back to
Dev-2604 (28 steps, no negative prompt, recorded as such) rather than drop it.

**Slice B:** Adam (2026-09-25): "most of them are clearly trash" — the slice-B pages show V3.1 on the
generator being replaced, so rating them is dropped. V3.1 is judged in the confirmation round instead,
on the new generator. The few pass-2 answers already given are kept as they are and reported, not scored.

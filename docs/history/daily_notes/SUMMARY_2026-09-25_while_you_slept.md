# While you slept — simple summary (25 Sept)

## What your round-2 ratings showed
- Pictures from the new picture model were understood **much more often**: today 14/54, same text with the
  new model 26/54, new text + new model 32/54. New vs today: **+33 points, significant**; most of it (+22) is
  the model.
- You were consistent: 9 of 10 repeated pictures got the same answer.
- The new text (V3) is **not adopted**: one picture showed a door for a sound that was only a "thud" — a false
  message, which our rule forbids.
- **Usable today, with one caveat:** "the new picture model alone beats today by +22 points, significant, blind,
  on fresh clips" — the false-message check was only run on N, not on this version, so it is not yet checked.

## Can an AI judge pictures instead of you? We tried three ways — all failed
- Asked "what is this?": said "can't tell" to almost everything (502 of 567).
- Told the sound and asked "would a viewer get it?": said yes to almost everything (missed 66 % of your "no").
- Shown two pictures, "which is clearer?": picked the first picture 187 of 190 times.
- So the picture result must come from a person. That is a finding worth one line in the thesis.

## The plan the five reviewers signed (3 rounds)
- **Scene:** the scene picks the *kind* of thing ("campfire flames" for fire in a forest), never a background —
  backgrounds made the model paint scenery and invent things.
- **The AI never picks what is drawn.** It may only describe how it looks; a word list blocks new objects.
- **Hard sounds:** thunder → a big lightning bolt, rain → drops on a window, etc.; sounds with no object
  (whoosh, thud) → a comic "WHOOSH" card instead of a made-up picture.
- **Your workload:** you rate **only one final sitting** (about 170 pictures, 30–40 min). I screen the
  versions myself, blind, and label that honestly.
- 50 fresh clips for that final sitting are already chosen and frozen, before any picture exists.

## What I did tonight
- **Picked 50 fresh clips for your final sitting** — by a fixed random order, before any picture existed
  (81 sounds). Nothing was tuned on them.
- **Tried a newer picture model (Qwen-Image-2.1).** It runs fine but was **not better** in my blind check,
  and its licence is research-only — so we keep the current model (Qwen-Image-2512).
- **Tried prompt versions** (I looked at 324 pictures myself, without knowing which version drew which —
  I am a generous rater, so these numbers only choose, they prove nothing):
  | version | right, of 54 |
  |---|---|
  | current (new model + new text) | 39 |
  | + "the whole thing, caught making the sound" | 42 |
  | + that, + fixed pictures for thunder/rain/alarms/bells/glass/train horns, + word cards | **45** (fewest wrong) |
  | longer AI-written description (its word check rejected 70 % of them) | 41 |
- **Fixed things I found by checking every picture of the chosen version:** the car-alarm picture drew a
  *police* light bar (a false message) — fixed; a bug in my new word check (it turned "a bird singing" into
  "bird vocalization") — fixed before anything was rated.
- **Froze the final setup** (commit c0d1d1b) and drew your final sitting.
- **Did not build an "end the picture when the sound ends" rule:** the development clips have no case of a
  picture staying too long, so there is nothing honest to test it on. Written down as a limitation.
- Three AI "judges" of pictures all failed against your ratings (see above) — reported as a finding.

## Actions for you
1. Optional disk cleanup (frees ~95 GB; the disk is 97 % full). I do **not** recommend testing more picture models:
   the newer one gave no gain, and the reviewers agree the remaining failures are about *what* is drawn. All of
   these can be downloaded again (Qwen2.5-VL-7B is only the old default model, not used by the current pipeline):
   `ssh adamg@slurm-login1.lnx.biu.ac.il 'cd ~ && rm -rf .cache/huggingface/hub/models--HuggingFaceM4--Idefics3-8B-Llama3 .cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3 .cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct .cache/openflam && source miniconda3/etc/profile.d/conda.sh && conda env remove -n comfy -y && conda env remove -n psed -y && df -h ~ | tail -1'`
   (the safety system blocked me from deleting; package lists of the two environments are saved in `docs/envs/`).
2. Mage-Flow: nothing to do — there is no official download, so it is dropped.
3. **The one final rating sitting** (168 pictures, ~30–40 min; stop and resume any time; answers save online):
   https://claude.ai/artifact/3UV2qKvwA8rkhLudtrgVPf — press Show, the picture appears for 1.5 s like beside a
   video, type what is making the sound (or Can't tell). This is the only rating the plan needs from you.
   After it: I score it blind, then open the key; that number is the thesis's picture result.

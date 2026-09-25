# While you slept — simple summary (25 Sept)

## What your round-2 ratings showed
- Pictures from the new picture model were understood **much more often**: today 14/54, same text with the
  new model 26/54, new text + new model 32/54. New vs today: **+33 points, significant**; most of it (+22) is
  the model.
- You were consistent: 9 of 10 repeated pictures got the same answer.
- The new text (V3) is **not adopted**: one picture showed a door for a sound that was only a "thud" — a false
  message, which our rule forbids.
- **Usable today:** "the new picture model alone beats today by +22 points, significant, blind, on fresh
  clips".

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
- (pending — filled in as results arrive)

## Actions for you
1. Run the cleanup command (frees ~95 GB so two more picture models can be tested):
   `ssh adamg@slurm-login1.lnx.biu.ac.il 'cd ~ && rm -rf .cache/huggingface/hub/models--HuggingFaceM4--Idefics3-8B-Llama3 .cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3 .cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct .cache/openflam && source miniconda3/etc/profile.d/conda.sh && conda env remove -n comfy -y && conda env remove -n psed -y && df -h ~ | tail -1'`
   (the safety system blocked me from deleting; package lists of the two environments are saved in `docs/envs/`).
2. Mage-Flow: nothing to do — there is no official download, so it is dropped.
3. When ready: the one final rating sitting (link will be here).

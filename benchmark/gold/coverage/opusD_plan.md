# Opus D plan: new signals and hits (10 Oct)

Floor 0.19 = about 8 hits or 15 wrongs. All 19 v1.6 wrongs removed perfectly = 0.25. No veto aimed at one wrong type can clear the floor.

## (1) The 19 wrongs: which need a new signal

| group | n | best new signal | ceiling | verdict |
|---|---|---|---|---|
| visible (Train, Baby cry, Explosion/fireworks, Chink, Siren 0.2) | 5 | Synchformer (A/V sync) | 0.063 | cheap stack only |
| phantom (keyboard, Dog, Bee) | 3 | none (strong peaks, both listeners agree) | 0.038 | skip |
| Gunshot/Explosion/Fire siblings (1917, live_fire, clay_shoot, d103, d088) | 5 | sibling classifier | 0.065 | skip: sibling_drop / parent-on-dispute lost one hit per wrong; the gold itself calls gunfire "Fire" |
| right family, start outside the window (golf, d031, wolves x2, applause) | 5 | none: timing | - | skip: restart / shift / start-move all exhausted |
| d146 Rowboat for Water | 1 | - | 0.013 | skip |

Three of the remaining wrongs (d088 10.8, d103 1.8, d110 3.5) are the ones the dense-frame "change" check dropped without losing a hit. That check was unstable across frame count and wording. Synchformer (MIT, about 0.2 B parameters, under 1 GPU-hour) gives a fixed sync score instead.

## (2) Hits: 3 ways

1. **Residual ear.** Many never-heard sounds are hidden under a loud one: Aircraft under gunfire (1917, d103), Crow under shots (clay_shoot), Civil defense siren under storm, Tick-tock under a siren. Remove the loudest family's stem (AudioSep is already installed; SAM-Audio, Meta 2025, text-prompted, custom SAM licence) and run BEATs and FlexSED again on what is left. New candidates go through the normal chain. AudioSep failed as a *verifier* (AUROC 0.68). It has never been tried as an *ear*.
2. **Better inventory.** Use Qwen3-Omni-30B-A3B-Captioner (Apache-2.0, one H200 or A100-80) to list the clip's sounds. The listed families become FlexSED text queries at a lower bar, beside the K4A inventory.
3. **Bigger benchmark.** Gold 80 more clips from the 415 held-out clips. The floor drops from 0.19 to about 0.15. Without this, gains of 0.05 to 0.10 cannot be shown.

## (3) Ranked plan (DEV only; merged TEST scored once at the end)

| # | experiment | cost | pre-registered DEV bar | expected gain |
|---|---|---|---|---|
| 1 | residual ear (AudioSep first, SAM-Audio if it passes) | 4 A100-h, half a day | stage 1: residual hears >= 3 DEV needed sounds that are never heard or lost as masked_weak; stage 2: full chain, net hits >= +2 and added wrongs <= hits/2 | -0.03 to -0.09 |
| 2 | Captioner inventory into FlexSED | 3 H200-h | stage 1: needed families recalled per clip > K4A + 3; stage 2 as in #1 | -0.02 to -0.06 |
| 3 | Synchformer on short pictures (<= 2.5 s), drop only | 1 GPU-h | clip-grouped CV on DEV: >= 3 wrongs out, 0 hits lost, same bar in >= 4 of 5 folds | -0.03 to -0.05 |
| 4 | gold 80 held-out clips | about 5 h of Adam's time, 0 GPU | - | floor 0.19 to 0.15 |
| 5 | stack the passing parts of 1 to 3, score merged TEST once | 1 h | - | -0.05 to -0.15 together; one part alone stays under the floor |

Not worth doing:
- **Stereo SELD.** Film stereo is panned in the mix, and SELD models are trained on microphone arrays. The visible ceiling is 0.063 anyway.
- **Lip or face activity.** It could remove at most 2 wrongs.
- **CAV-MAE or ImageBind.** They ask the same visibility question that SigLIP, OWL and Omni-AV failed. ImageBind is CC-BY-NC.
- **Another SED backbone.** 25 detector rounds failed, and EAT and Dasheng were already tried.
- **Undoing any pipeline step.** Every step drops less than 1 real sound in 3. scene_margin (38 %) costs only 1 sound.
- **Timing shifts.**

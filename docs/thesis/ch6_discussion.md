# Chapter 6 — Discussion (draft, 27 Sept 2026)

*Every number below is taken from Chapter 5 (`docs/thesis/ch5_results.md`) or from the pre-registration record
(`docs/prereg_v4.md`), which name the committed result files. Items marked **[pending]** are filled when their run or
sitting finishes.*

## 6.1 What the results mean for a DHH viewer: restraint before recall

The main result is about restraint. The visibility gate does not find more sounds; it removes pictures that should
not be there. On TEST it removed 31 wrong pictures and 3 needed sounds; on DEV, 26 wrong pictures and 3 needed
sounds (§5.4). That is about ten wrong pictures removed for each needed sound lost, on both halves.

Whether this is a good trade depends on the price of a wrong picture. The modelled viewer cost puts a missed
sound at 4 and a wrong picture at β; a picture is then worth showing when the chance that it is right is above
β / (4 + β) (`docs/beta_specification.md` §1), one in three at β = 2. So β sets how ready the system should be to
draw.

Because β for DHH viewers is not known, the result is an interval (§5.5). **On TEST, the gated system is the
cheapest of the three systems for every β between 0.39 and 2.56** (DEV: 0.46 to 2.33). Below that range, drawing
every detected sound is better; above it, an empty panel is better. Two points limit the reading:

- **Beating silence at a low β is not the gate's achievement.** The blind arm itself costs less than silence below
  β ≈ 1.36. The gate's own contribution is the *lower* end of the interval.
- **Over all clips, the system does not beat an empty panel.** The TEST cost difference against silence is −0.23
  [−1.00, +0.47], not significant (§5.6). Clips with nothing to draw can only cost, and busy mixed clips are the
  weak category (ours 6.27 per clip, silence 5.33).

The gate earns its keep where a visible source can be silenced: on-screen clips (−1.83 [−2.83, −1.00] against
blind) and mixed clips (−1.73 [−3.73, −0.13]). On off-screen clips ours and blind cost the same (4.00), so the
large gain against silence there (−3.08) is a detection and picture result, not a gate result.

The literature gives a direction for β, not a number. SoundWatch users wanted speed for
urgent sounds and accuracy for non-urgent ones (Jain et al., 2020), and users judge any added visual first on
readability and low distraction (de Lacerda Pataca et al., 2024). For a danger sound, a miss is the worst outcome
and β is low; for atmosphere, a wrong picture is pure distraction and β is high. DHH viewers are also not one
audience: in an early study of enhanced captions, hard-of-hearing viewers liked them while several Deaf viewers
did not (Fels et al., 2007). So a single β mixes sounds and people; hence a curve, not a point, as in cost curves
(Drummond & Holte, 2006).

## 6.2 Why the detector is the binding constraint

The primary F1 result is null (+0.059 [−0.030, +0.144]). The evidence says the cause is the detector, not the gate.

**Oracle diagnostics.** When the detector is replaced by the annotator's own sound list, the same gate's F1 gain is
significant (+0.067 [+0.012, +0.117]); giving back only the missed sounds, while keeping every false alarm, already
makes it significant (+0.075 [+0.029, +0.119]) (§5.3). These come from the 22 Sep renders on 109 benchmark clips,
which include the TEST clips (exposure row 1, §5.15). On the final cost curve (§5.5), the gate given the gold sound list would cost
1.23 per clip at β = 2 on TEST (DEV 1.63), against 2.63 for the shipped system (amendment 21). The headroom lies in detection.

**Where needed sounds are lost.** Of 21 missed DEV sounds, 11 were never detected, 5 were late, 3 were silenced by
the gate and 2 were removed by the label filter (§5.7).

**Masking.** The missed sounds are mostly quiet sounds under speech or music. At every masked needed sound, BEATs'
top labels were Speech or Music (Chapter 3): a tagger of the whole mixture reports what dominates it. FlexSED,
queried by text one label at a time, scores these sounds far higher, but mostly below its shipped bar: at 0.8 it
recovers 1 of 7 masked DEV sounds, and none of the 31 masked consequential events on the 280 AudioSet-Strong clips.

The ceiling is known only roughly. On DEV-49 the caches reach 21 of 36 needed sounds at the shipped bars and 30 at
looser bars; 4 are below every detector (amendment 22). This is an oracle read on the sounds it would be scored
on, so no bar was chosen from it. (The panel's finer split, 22 / 8 / 3 of 33 depictable sounds, §5.8, is a
reviewer's cache read, not a committed result file.)

**Four detector questions, all negative (amendments 22–25).** Each had a rule written before its numbers; none
touched TEST (§5.8).

1. *Lower bars and time-aligned support (amendment 22).* On the 280, no cell raised onset recall while staying at
   or below the shipped 4.46 false spans per minute; lower FlexSED bars and support from a second detector bought at
   most 0.4 points.
2. *The cascade E (amendments 22–23).* Promoting weak BEATs spans when FlexSED or PANNs agrees raised onset recall
   on the 280 from 25.0 % to 46.4 %, at 6.90 false spans per minute. On DEV its blind arm found **no new needed
   sound** (17 hits against 17) and drew 80 wrong pictures against 50, so its modelled cost at β = 2 rose from
   3.59 to 4.82. Most of the 280's gain was earlier starts on sounds already found.
3. *A cost rule and a new corroborator, PE-A-Frame (amendment 24).* No cell beat the shipped stack on cost
   (C-overlap 3.071). The cascade paid about **2.4 false spans per gained onset** (114 more false spans for 48 more
   onsets) and about **6 per gained event**, while at β = 2 a gained sound pays for itself only at two false spans
   or fewer. PE-A-Frame, with an encoder independent of FlexSED's, was at chance on the candidate pool (AUROC 0.53
   [0.45, 0.61]), below PANNs' 0.64 on the same spans.
4. *An audio "listener" (amendment 25).* An audio language model (Qwen3-Omni) was asked whether each uncertain
   span's sound was present. It is a sane listener (yes to 84 % of asked families, 7 % of families absent from the
   clip), but it separated real needed sounds from false ones only weakly: AUROC 0.661 [0.525, 0.784] on 14 hit and
   96 false spans (PANNs 0.574 on the same spans). The pre-written gate (lower bound ≥ 0.70) failed.

No cell passed, so the new held-out set was never scored. The reading: the shipped
stack sits at the cost optimum of every training-free stack tried, and the sounds still missing are those that
closed-set taggers cannot hear under speech or music. The exchange rates were measured on the 280, a fit set, not
out-of-sample data.

## 6.3 Pictures: recognition, false messages, and what is still sealed

In the author-rater glance test (54 sounds from 50 never-annotated clips, each picture shown for 1.5 s at 384 px),
the shipped generator, FLUX.1-schnell, was recognised 14 of 54 times; Qwen-Image-2512 with the same text 26 times
(+0.22 [+0.07, +0.37]); and with the V3 text 32 times (+0.33 [+0.22, +0.46]) (§5.11). The rater was the author,
blind to the version but not independent of the project.

Recognition is not the only bar: a picture can be recognised and still say something false. The V3 text was
rejected because it invented an object (a thud drawn as a door), and the new generator showed one wrong object (a
ringing phone drawn as a desk bell). A DHH viewer cannot check a picture against the sound. The image-generation
panel ranked an invented object as the worst picture error, above drawing what is already visible
(`docs/panel3_topic2_rounds.md`).

The project therefore wrote a **false-message rule** before the final sitting
(`docs/freeze_picture_setup_2026-09-25.md`, rule 2). In plain words: a picture is a false message if it shows as the
maker of the sound an object that is not the source (or a label in the source's family chain, its short qualifier,
or a word on a closed list declared in advance), or if it shows a person for a sound that is not a human sound. The
frozen final setup, Qwen-Image-2512 with the V3.1 text, is judged as one package in a sealed sitting: 81 frozen
sounds on 50 frozen clips, final against today's pictures. It is adopted only if the paired interval of the
recognition gain excludes zero, the final arm has zero false messages, and repeated pictures get the same answer at
least 80 % of the time; a tie keeps today's setup. A gain, if found, is not credited to the generator alone.

**Sealed sitting result: [sealed sitting result pending].**

Two limits remain. The automatic judge scores pictures and text tags alike (−0.04 [−0.22, +0.14], §5.10), since it
reads a tag as the label itself, so there is no evidence here that a picture beats a word. And independent raters
are **[pending]**.

## 6.4 What the evaluation taught

**Automatic instruments failed where it mattered.** The proposal planned a model-describes, model-judges protocol.
Every automatic instrument tried failed its own bar (§5.13). References written by the system's own model were circular. An independent four-model
reference agreed with the human tag on 55 of 100 clips, chance on the silence decision. An audio-visual reference
reached 49.5 % balanced accuracy (bar 67.7 %); a gap-closing score, AUROC 0.53 (bar 0.75); six picture checkers
failed calibration against the human rater. "What should the viewer have been shown?" needs a human answer. Two
instruments survived: the per-sound scorer on human labels, and a secondary judge valid for ranking, not for
absolute quality.

**One annotator carries the gold.** Every score depends on one person's ears and sense of what matters. The guide
makes the rules explicit (for example, importance is rated as if the screen were black), but only a second
annotator can show how stable the labels are. The 30-clip second pass is planned; κ is **[pending]**.

**TEST was held out, with disclosed exposure.** The final table is the tenth exposure of the TEST clips, all dated
in §5.15, and 35 of the 60 TEST clips were in the split on which FlexSED's bar was chosen (on the clean DEV-49 the
bar misses one of its three adoption rules by 0.011). TEST is therefore never called "unseen".

**The cost measure was declared post hoc.** Its weights (miss 4, wrong picture 2) were written before the gold
labels existed, but the measure became a reported outcome only after F1 came out null (amendment 9). Holm's
correction deals with the number of rows; it cannot undo choosing a row after seeing the data. So the cost result
is stated as an interval over β, and the headline is the declared wrong-picture row, which replicates on DEV.

**The primary metric could not see the trade.** F1 weighs a removed wrong picture and a lost right one equally, and
the gated system draws a subset of the blind arm's pictures. With 60 clips the smallest detectable effect was about
+0.13 (§5.3). A future study should pre-register a primary
metric that matches the decision: a cost over a stated range of β, or wrong pictures beside recall. Operating curves
of this kind are already used in sound-event detection (Bilen et al., 2020).

## 6.5 Limitations

1. **One annotator** for all 139 clips; second-pass κ [pending].
2. **Pilot size.** 60 TEST clips; smallest detectable F1 effect about 0.13. TEST has more nothing-to-draw clips than
   DEV (20 against 8).
3. **No DHH viewer** has used the system; β is assumed; DHH helpfulness is untested.
4. **The cost measure is post hoc.**
5. **Precision (+0.137) is TEST-only** (DEV Holm 0.080).
6. **TEST exposure and split overlap** (§5.15).
7. **Picture recognition** is author-rated; independent raters and the sealed sitting are [pending].
8. **Gate errors are "presence without source"**: a visible bell tower silences an off-screen bell (§5.9); balanced
   accuracy 0.62 on DEV gold sounds.
9. **Timing.** With no length cap, 4 of 15 TEST pictures outlast their sound by more than 2 s (§5.12).
10. **The detector ceiling** was measured on a fit set (the 280).

## 6.6 Future work

**A detector that hears under speech.** This is the fix the results point to: a detector trained on sounds mixed
under speech and music. The unscored held-out AudioSet-Strong set, weighted towards complex scenes, is ready as its
test set.

**The DHH user study, as designed** (`docs/beta_specification.md` §5, pre-registered, not run). Five to ten DHH
viewers watch the rendered side-panel output, with real pictures and real timing, and after each clip answer one
fixed question: "Did this panel help you, hurt you, or neither?". An ordinal regression, helped ~ a × (missed needed
sounds) + b × (wrong pictures), gives the measured price β̂ = 4b / a with a 95 % interval. The fit is repeated within
importance and clip category, because a β that changes with context is itself a finding. A hearing sound-off pilot was also designed (`docs/panel3_topic3_rounds.md`); it would use
DEV clips only and would be reported as a proxy, never as a DHH study.

**Benchmark v2.** At least 300 clips, as the proposal planned; at least two annotators with agreement reported; a β
measured with DHH viewers; a small sealed held-out split with a script that scores a submission.

**Picture plus word.** A one-word label under the picture is a common default (the renderer has the switch, off
today). The naming test cannot score it, since a word makes naming trivial; it needs another measure and DHH viewers.

**Audio-conditioned pictures, only with noun guards.** Free text from an audio model is the failure path that drew
a thud as a door. Audio should enter only as a closed question over sub-labels the detector already fired (for
example sheep, goat or unsure), with "unsure" falling back to the parent label and today's noun guards kept. A
report-only job of this kind was run on the 54 fresh-picture sounds (`listener_pictures.json`). Asked freely "what is
making this sound?", the listener named a different maker on 35 of 54 (a hen's cluck as "goats bleating", glass
shattering as "a sword"), which confirms that free audio text must never reach the prompt. Asked to choose among the
fired labels, it changed the label on 6 of the 8 sounds where a choice existed; three of the six made the noun vaguer
("siren" instead of a police car) and two might help (a ringtone for the phone once drawn as a desk bell). It is left
as future work.

## 6.7 Conclusion

This thesis defined a new accessibility task, off-screen sound visualisation, and built a training-free pipeline
for it. Its key step is a decision not to draw: a picture appears only when the viewer cannot see where a sound
comes from. On a pilot benchmark with one annotator, that decision halves wrong pictures at no detectable recall
price, and makes the gated system the cheapest option over a middle range of prices for a wrong picture. It does not
raise F1, and it has not been tested with DHH viewers. The limit is hearing, not seeing: the missed sounds are
masked by speech and music, and no training-free detector tried recovered them at a fair price. The benchmark, scorer,
cost curve and logged protocol let the next system, and the first DHH study, be measured by the same rules.

## References

Bilen, Ç., et al. (2020). A framework for the robust evaluation of sound event detection (PSDS). *ICASSP 2020*.
arXiv:1910.08440. *[title and authors to verify]*

de Lacerda Pataca, C., Hassan, S., Tinker, N., Peiris, R., & Huenerfauth, M. (2024). Caption Royale: Exploring the
design space of affective captions from the perspective of deaf and hard-of-hearing individuals. *CHI 2024*.
*[author list to verify]*

Drummond, C., & Holte, R. C. (2006). Cost curves: An improved method for visualizing classifier performance.
*Machine Learning*. *[to verify]*

Fels, D. I., Lee, D. G., Branje, C., & Hornburg, M. (2007). Emotive captioning. *ACM Computers in Entertainment*.
*[to verify]*

Jain, D., et al. (2020). SoundWatch: Smartwatch-based deep learning approaches to support sound awareness for DHH
users. *ASSETS 2020*.

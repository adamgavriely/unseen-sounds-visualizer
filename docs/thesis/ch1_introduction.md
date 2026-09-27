# Chapter 1 — Introduction (draft, 27 Sept 2026)

*Every number in this chapter is taken from a committed file. Most come from Chapter 5
(`docs/thesis/ch5_results.md`), which names the result file beside each table. Items marked **[pending]** are
filled when their run or sitting finishes.*

## 1.1 The problem: sounds that are heard but not seen

A film or a street video tells its story with two channels: the picture and the sound. A hearing viewer uses both
without effort. When a siren starts behind the camera, the viewer knows that something is happening off screen,
even though nothing in the picture changes. A deaf or hard-of-hearing (DHH) viewer gets only the picture. For this
viewer, a sound whose source is not visible is simply lost.

The usual tool for this gap is captions. Subtitles for the deaf and hard of hearing (SDH) write sounds in words,
for example "[siren wailing]". Captions carry speech well, but they carry non-speech sound poorly. Most non-speech
sounds on YouTube are not captioned at all, and background and environmental sounds are the least covered group
(May et al., 2024). When a sound is captioned, the short label often loses what the sound means for the scene.
An EEG study found that captions of non-verbal sounds tend to cause extra attention work rather than an emotional
response (Gerber-Morón et al., 2020). In a study with 168 DHH participants, users asked for more than a label. They
wanted the sound's emotional tone, where it comes from, and why it matters to the story, and above all they
wanted choice (May et al., 2025).

Pictures are one way to carry this information. Prior work shows that DHH users can read sound icons and graphics
quickly. In Beyond Subtitles, DHH viewers asked for the *important* non-speech sounds to be shown, and noted that
graphics are evocative but can be ambiguous or distracting (Alonzo et al., 2022). Sound-awareness
systems such as SoundWatch show that users want to filter sounds by importance, not see every sound (Jain et al.,
2020; Findlater et al., 2019). Users judge any added visual first on readability and low distraction (de Lacerda
Pataca et al., 2024). Sound words such as "swish" do not work for everyone, because a person who has never heard
the sound cannot decode the word (CapTune, 2025). A picture of the thing that makes the sound needs no knowledge of
how it sounds.

This thesis asks whether such pictures can be chosen and shown **automatically**, for recorded video, without any
model training.

## 1.2 Why "only when the source is not visible"

A picture beside the video costs the viewer attention. A picture of a sound whose source is already on screen adds
no information: the viewer can see the dog that barks. So the system should show a picture only when the picture
tells the viewer something new.

Professional captioners already follow this rule by hand. The DCMP Captioning Key asks captioners to name the source
of a sound effect unless the source is clearly seen on screen, and not to caption a sound that the viewer can infer
from the picture (DCMP, n.d.). The visibility gate in this thesis automates that editorial rule. It is the one
design choice that makes this task different from "turn every sound into a picture". Audio-to-image systems such
as Sound2Scene (Sung-Bin et al., 2023) draw what they hear and never look at the video. Sound-source localisation
work asks *where* in the frame a sound comes from (Senocak et al., 2023), which assumes that the source is visible.
Our question is the opposite one: is the source missing from the frame?

## 1.3 The task: off-screen sound visualisation

We call the task **off-screen sound visualisation**. It is defined as follows.

- **Input:** a short video (10–30 s) with its soundtrack.
- **Output:** the same video with a side panel. While a non-speech sound is heard *and its source cannot be seen*,
  the panel shows a picture of that sound's source, timed to the sound. When no such sound is heard, the panel is
  empty.
- **Not in scope:** speech (captions already handle it) and music. The picture does not replace captions or sign
  language; it sits beside them.

A system for this task must **hear** the sound and its timing, **decide** whether the source is visible, and
**draw** a picture that a viewer recognises at a glance. A missed sound is lost; a wrong picture is a false message.

## 1.4 Research questions and where each is answered

The project proposal (6 Aug 2026) listed five research questions. The table maps each one to the place in the
thesis where it is answered, and says how fully.

| # | research question (proposal) | short answer | where |
|---|---|---|---|
| RQ1 | Which types of audio information contribute most to scene understanding when presented visually? | Answered as a design rule, not by viewers. A sound is *needed* when it matters (importance 2 or 3: an event you can say in one sentence, or danger or a key moment), and its source is neither visible nor obvious. The steady noise of a place (importance 1) is not scored. Whether viewers agree is untested. | Ch. 4; §5.6; Ch. 6 |
| RQ2 | What level of semantic granularity should be used? | A picture counts when it shows the right sound *family* (for example any dog sound for a bark); root-level words such as "sound" or "vehicle" never count. Pictures name the specific source. In an author-rater glance test, a newer generator was recognised 26 of 54 times against 14 of 54. | §3.2; Ch. 4; §5.11 |
| RQ3 | Can generated visual augmentations improve accessibility beyond subtitles? | **Not answered.** No DHH viewer has used the system. The study that would answer it is designed, not run. The automatic judge scores pictures and text tags alike, and it cannot tell them apart. | §5.10; §5.14; Ch. 6 |
| RQ4 | How should such systems be evaluated? | Per sound, against human labels, with an onset window; a modelled viewer cost over a range of prices for a wrong picture; rules written before each run. The proposal's automatic protocol (a vision-language model describes, a language model judges) was tried and failed: its references were circular or at chance. | Ch. 4; §5.5; §5.13 |
| RQ5 | How does the pipeline compare with audio-to-visual generation approaches? | Partly. A direct audio-to-image model was not run. The *blind* arm stands in for it: the same detector and generator, drawing every detected sound without looking at the video. The audio-captioning baseline became the text-tags arm. | §5.3–5.4; §5.10; Ch. 6 |

For RQ5, the image-generation review panel (`docs/panel3_topic2_rounds.md`) judged that audio-conditioned image
models draw the whole scene rather than the one source, with no way to forbid a wrong object. The blind arm
isolates the part this thesis adds: the decision *not* to draw.

## 1.5 Contributions

The thesis makes five contributions. Each is stated at the strength the evidence supports.

1. **A training-free pipeline for the task.** Seven stages chain public pretrained models: speech recognition
   (speech is never drawn), sound-event detection (BEATs and FlexSED, with two cross-model vetoes), a visibility
   gate (a vision-language model, Qwen3.8-27B, asked three questions about six frames per stretch of up to 5 s), a
   subject step, an image generator, and a timed side panel (Chapter 3). No component is trained or fine-tuned.

2. **A pilot benchmark, a per-sound metric and a protocol for the new task.** The benchmark has 139 clips (49
   DEV, 60 TEST, 30 slice B), labelled by one annotator. Each sound has a label, an onset, a "visible" tick, an
   "obvious" tick and an importance rating. The scorer credits a picture only if it shows the right family and
   starts within [−0.5, +1.0] s of the sound's onset; a picture of a visible sound counts as wrong. We call this
   *pilot benchmark v1*: the protocol is the contribution, and the numbers are provisional. The proposal aimed at
   about 300 clips; that target moves to a second version (Chapter 6).

3. **A visibility gate that halves wrong pictures at no detectable recall price.** On 60 held-out TEST clips, the
   gate cuts wrong pictures per clip by −0.52 [−0.80, −0.28] (Holm p < 0.001), about half of the 0.93 that the
   blind arm shows. The same result replicates on DEV (−0.53). The recall change is −0.07 [−0.15, 0.00]. "No
   detectable price" is not "free": the interval touches zero, and the point estimate is three needed sounds lost
   (§5.4). Under a modelled viewer cost, where a missed sound costs 4 and a wrong picture costs β, the gated system
   is the cheapest of the three systems for every β between 0.39 and 2.56 on TEST (0.46 to 2.33 on DEV) (§5.5).

4. **The detector ceiling.** The sounds the system misses are mostly sounds masked by speech or music. The
   pre-registered detector rounds of amendments 22–25 asked whether a training-free detector can find them without
   adding more wrong pictures than they are worth. No stack tried could. The best route to more recall paid about
   2.4 false spans per gained onset (about 6 per gained event), while at β = 2 a gained sound pays for itself only
   at two false spans or fewer. A second text-queried detector (PE-A-Frame) was at chance (AUROC 0.53 [0.45,
   0.61]), and an audio language model used as a listener reached AUROC 0.661 [0.525, 0.784], below its
   pre-written bar of 0.70 (§5.8). No detector with public weights that hears under speech was found.

5. **A pre-registered, fully logged evaluation.** Every choice was made by a test with a bar written before the
   run (`docs/prereg_v4.md`, amendments 1–25). Negative results are reported, including every automatic
   instrument that failed its own calibration bar (§5.13). The TEST set is *held out, with disclosed
   exposure*: all ten exposures are dated in an appendix log, together with a split overlap found late (§5.15).

## 1.6 What this thesis does not claim

- **It does not claim an F1 gain.** The pre-registered primary metric, F1 against drawing every detected sound,
  shows no significant difference: +0.059 [−0.030, +0.144], p = 0.18. With 60 clips the smallest effect the design
  could detect was about +0.13 (§5.3).
- **It does not claim that the system helps DHH viewers.** No DHH viewer has used it. The modelled viewer cost is a
  model, and its price β is assumed, not measured. The cost measure was also promoted to a reported outcome after
  F1 came out null, so it is a post-hoc result (§5.2).
- **It rests on one annotator.** A 30-clip second pass is planned; its agreement (κ) is **[pending]**. Picture
  recognition was rated by the author, blind to the version but not independent of the project; independent raters
  are **[pending]**.
- **It does not claim that precision improved in general.** The precision gain (+0.137) is significant on TEST
  only and does not replicate on DEV.
- **It does not claim to beat showing nothing over all clips.** Across all TEST clips, the modelled cost against
  an empty panel is −0.23 [−1.00, +0.47], not significant (§5.6).

## 1.7 Chapter map

- **Chapter 2 — Related work**: sound awareness and captioning for DHH viewers, sound-event detection, source
  localisation and "is the source visible?", image generation from sound or text, and judges for generated content.
- **Chapter 3 — Method**: the seven stages, why each model was chosen, and what the system does not do.
- **Chapter 4 — Benchmark and evaluation protocol**: clips, annotation protocol, splits, reliability, the
  per-sound metric, the modelled viewer cost and the statistics.
- **Chapter 5 — Results**: the primary metric, the secondary family, the cost curve, and the analyses that
  explain them.
- **Chapter 6 — Discussion**: meaning for a DHH viewer, the detector limit, evaluation lessons, limitations and
  future work.

## References

Alonzo, O., Shin, H. V., & Li, D. (2022). Beyond Subtitles: Captioning and visualizing non-speech sounds to
improve accessibility of user-generated videos. *ASSETS 2022*.

CapTune (2025). CapTune: Adapting non-speech captions with anchored generative models. arXiv:2508.19971.
*[authors to verify]*

DCMP — Described and Captioned Media Program (n.d.). *Captioning Key: Guidelines and preferred techniques* (sound
effects, music, non-speech information). https://dcmp.org/learn/captioningkey

de Lacerda Pataca, C., Hassan, S., Tinker, N., Peiris, R., & Huenerfauth, M. (2024). Caption Royale: Exploring the
design space of affective captions from the perspective of deaf and hard-of-hearing individuals. *CHI 2024*.
*[author list to verify]*

Findlater, L., Chinh, B., Jain, D., Froehlich, J., & Kushalnagar, R. (2019). Deaf and hard-of-hearing
individuals' preferences for wearable and mobile sound awareness technologies. *CHI 2019*. *[author order to
verify]*

Gerber-Morón, V., et al. (2020). Limitations of standard accessible captioning of sounds and music for DHH people:
An EEG study. *Frontiers in Integrative Neuroscience*. *[authors to verify]*

Jain, D., et al. (2020). SoundWatch: Smartwatch-based deep learning approaches to support sound awareness for DHH
users. *ASSETS 2020*.

May, L., et al. (2024). Unspoken Sound: Identifying trends in non-speech audio captioning on YouTube.
https://sinc-lab.com/files/may2024unspoken.pdf

May, L., et al. (2025). "Choices? That's the dream": Challenges and opportunities in non-speech information
closed-captioning. *[venue to verify]*

Senocak, A., et al. (2023). Sound source localization is all about cross-modal alignment. *ICCV 2023*.
arXiv:2309.10724.

Sung-Bin, K., et al. (2023). Sound to visual scene generation by audio-to-visual latent alignment (Sound2Scene).
*CVPR 2023*. arXiv:2306.11504.

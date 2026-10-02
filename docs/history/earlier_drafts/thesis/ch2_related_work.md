# Chapter 2 — Related work (draft, 27 Sept 2026)

*This chapter places the thesis in five research areas. It reuses and updates `docs/history/earlier_drafts/literature_review.tex`: the
accessibility argument is kept; every sentence about "what we use" now follows the shipped system of Chapter 3.
Model names in the text refer to that system (for example BEATs ∪ FlexSED as the detector, Qwen3.8-27B as the
visibility gate, FLUX.1-schnell as the shipped generator, with Qwen-Image-2512 with the V3.1 text as the frozen final setup that is
adopted only if the sealed sitting confirms it, Gemma-4-31B as the secondary judge). Results of this project are
only pointed to (Chapter 5); they are not repeated here.*

## 2.1 Sound awareness and captioning for deaf and hard-of-hearing viewers

### Captioning practice

Captions for deaf and hard-of-hearing (DHH) viewers write speech as text. Good captioning practice also writes
non-speech information: sound effects, music and the manner of speaking. The DCMP Captioning Key is a widely used
style guide for this work (DCMP, n.d.). It asks the captioner to include the source of a sound effect, but says
that "the source may be omitted if it can be clearly seen onscreen". It also asks for background sound effects to
be captioned only when they are essential to the plot. This practice is the starting point of this thesis. A
professional captioner already judges, by hand, what the picture shows and so need not be repeated in words. The
visibility gate of Chapter 3 is an automatic version of a similar judgement, applied to whole sounds.

In practice, much non-speech sound is never captioned. May et al. (2024) built a dataset of over 715,000 YouTube
videos and found that non-speech captions are rare: only 5–7 % of the videos had any hand-written non-speech caption
beyond the few tags that automatic captioning adds, and environmental sounds appeared in only 2–3 %. A later review
and survey (May et al., 2025) combined 36 papers with an online survey of 168 DHH participants and interviews with
15 DHH viewers and 5 professional captioners. DHH viewers asked for more than a label. Many wanted objective
details such as the type and timing of a sound, and most wanted the sounds that matter to the story or set its
mood. Their preferences differed widely, and they asked for choice. Captions that are present can
still lose meaning. In an EEG study, Revuelta et al. (2020) found that standard captions of sounds and music lose
emotional information, and that captions of music raise attention work more than emotional processing.

Recent systems try to make non-speech captions richer. CapTune lets caption authors define safe changes and lets
DHH viewers adjust the level of detail, the expressiveness and the way a sound is written (Huang et al., 2025).
In its studies, preferences for sound words (onomatopoeia such as "swish") varied, and one Deaf participant in the
viewer evaluation noted that the phonetic side of such words may not be understood. This is an argument for showing
the *source* of a sound, which needs no knowledge of how the sound is heard.

### Sound visualisation studies

A second line of work shows sounds as graphics, mostly for the viewer's own surroundings. Matthews et al. (2006)
designed and tested peripheral visual displays of ambient sounds for deaf users. Jain et al. (2015) tested
head-mounted display visualisations that show where a sound comes from, with 24 DHH participants. Findlater et al.
(2019) surveyed 201 DHH people about wearable and mobile sound awareness. Most were highly interested in knowing
about sounds, but this interest depended on how they communicate (sign, oral or both). Jain et al. (2019) studied
sound awareness in the home. SoundWatch classified sounds on a smartwatch and alerted the user (Jain et al., 2020);
its users welcomed the idea but raised concerns about misclassifications and delay. For urgent sounds they wanted
speed; for other sounds they wanted accuracy, so as not to be disturbed without need.

Closest to this thesis, Beyond Subtitles studied text and graphic captions for non-speech sounds in user-generated
video (Alonzo et al., 2022). Formative interviews with 11 DHH participants shaped an authoring tool built on top of
automatic sound event detection, which 10 hearing video creators then tested. The DHH participants wanted the
*important* non-speech sounds included, and they gave criteria both for choosing sounds and for when text or a
graphic suits a sound. Caption Royale explored styled (affective) captions in
three studies with 39 DHH participants; readability, minimal distraction, intuitiveness and emotional clarity were
the main reasons behind their choices (de Lacerda Pataca et al., 2024). A picture panel adds a third place for the
eyes, next to the video and the captions. These findings argue for few, well-chosen pictures.

### Icons versus words

Should the panel show a picture, a word, or both? Caption practice is word-first (DCMP, n.d.). Many
sound-awareness systems show an icon together with a label. Classic work on user interfaces found that novice users
first learned a program better with text labels, alone or beside icons, than with icons alone; the gap mostly
closed in a later session, and users rated icons with labels the easiest to use (Wiedenbeck, 1999). This thesis
shows a picture without a word.
There are two reasons. First, the evaluation asks whether a picture alone is recognised at a glance; a word would
make that test trivial, because the viewer would read instead of recognise. Second, sound words may not be
understood by some Deaf viewers (Huang et al., 2025). Adding the word is therefore a disclosed display choice (it exists in the code
and is switched off), not a tested claim. All of the work above is either about the user's own surroundings or
about captions chosen by an author. None of it decides *automatically*, for recorded video, which sounds the
picture already shows.

## 2.2 Sound event detection

### Closed-set taggers

Sound event detection (SED) asks which sounds occur in a recording and when. Most modern detectors are trained on
AudioSet, about two million 10-second YouTube clips labelled with classes from a 632-class ontology (Gemmeke et
al., 2017); the released labels, and so the taggers trained on them, use 527 of these classes.
AudioSet labels are "weak": they say that a sound occurs somewhere in the clip, not when. AudioSet-Strong added
labels with a resolution of about 0.1 s for part of the data, including a strongly labelled evaluation set
(Hershey et al., 2021). Chapter 4 uses this evaluation set for calibration and for a held-out detector test.

PANNs are convolutional networks trained on AudioSet (Kong et al., 2020). BEATs is a transformer pre-trained with
an acoustic tokenizer and fine-tuned on AudioSet (Chen et al., 2023). Both are *closed-set* taggers: they can only
name the 527 classes, and they can give one score per class per time frame. In this project BEATs is the base
detector and PANNs is used as a veto (Chapter 3). A closed-set tagger has a known weakness for this task. When a
quiet sound sits under speech or music, the tagger's top labels at that moment are Speech or Music. Chapter 5
(§5.8) shows that most sounds the system misses are of this kind.

### Text-queried (open-vocabulary) detectors

Contrastive language–audio pre-training (CLAP) learns a shared space for audio and text, so a sound can be scored
against any text description (Elizalde et al., 2023). This makes *open-vocabulary* detection possible: the
detector is asked about one sound name at a time. FLAM adds a frame-wise objective, so a text query can be
localised in time (Y. Wu et al., 2025). FlexSED builds on a pre-trained self-supervised audio model and the CLAP text
encoder, and gives a frame-level score for a free-text query; it was trained and tested on AudioSet-Strong (Hai et
al., 2025). PE-AV is a family of audio–video–text encoders trained on about 100 million audio–video pairs with
synthetic captions (Vyas et al., 2025); its frame-level audio model (PE-A-Frame) is also text-queried but uses a
different encoder from FlexSED. Large audio–language models go one step further and answer questions about a
sound in words. Qwen2-Audio (Chu et al., 2024) and Qwen3-Omni (Xu et al., 2025) are examples. This project tried
FLAM, FlexSED, PE-A-Frame and Qwen3-Omni; only FlexSED was adopted (Chapter 3, Chapter 5 §5.8).

### How detection is evaluated

SED is usually scored per event. A detected event matches a reference event if its label is right and its onset
(and sometimes its offset) lies within a tolerance window, called a *collar*. Precision, recall and F1 are then
counted over events. The `sed_eval` toolbox made these metrics standard (Mesaros et al., 2016). Event-based
scores depend on one fixed detection threshold. The Polyphonic Sound Detection Score (PSDS) removes this
dependence: it sweeps the threshold, draws an operating curve, and uses an intersection-based match instead of a
collar (Bilen et al., 2020). An analysis of the DCASE 2020 challenge showed that the collar-based event criterion
is stricter for some event lengths than for others, and that PSDS is more robust to how people label event edges
(Ferroni et al., 2021).

The per-sound metric of this thesis (Chapter 4) is an event F1 with an onset-only, asymmetric collar: a picture
may start up to 0.5 s before a sound and up to 1.0 s after it. The window is set by the viewer, not by the
detector: a picture of a glass shattering that appears 3 s late has lost its point. The modelled viewer cost of
Chapter 4 follows the same idea as PSDS, but in cost units. It is not reported at one weight; it is drawn as a
curve over the price of a wrong picture, like the cost curves of Drummond and Holte (2006).

## 2.3 Audio-visual source localisation and "is the source visible?"

Sound-source localisation asks *where* in a video frame a sound comes from. Early self-supervised work learned
this from the natural pairing of frames and audio (Senocak et al., 2018). Chen et al. (2021) introduced the VGG-SS
benchmark and a training method that mines hard negative regions inside the image. Later work argued that
localisation also needs cross-modal semantic understanding, for example of silent objects and off-screen sounds, and
trained localisation together with a cross-modal alignment task (Senocak et al., 2023). Still, the standard
benchmarks mainly test sounds whose source is visible in the image. They ask "where?", not "whether".

Two recent papers question this focus on visible sources. Juanola et al. (2025a) tested visual sound-source localisation models
with *negative audio*: silence, noise and off-screen sounds, where no object in the image makes the sound. Many
state-of-the-art models did not change their prediction much when the audio changed, and no single threshold
separated positive from negative cases. Juanola et al. (2025b) then trained with silence and noise as negatives,
proposed a metric for the trade-off between positive and negative pairs, and released an extended test set. This
line of work is the nearest field to the visibility gate of this thesis. It shows that "the source is not in the
frame" is hard for current models, and that it needs its own test data.

A second line separates on-screen from off-screen sound. Owens and Efros (2018) learned audio-visual features
that can split a soundtrack into on-screen and off-screen parts. AudioScope separates on-screen sounds in
unlabelled, in-the-wild video (Tzinis et al., 2021). These systems output audio. They do not decide whether a
viewer who cannot hear needs to be told about a sound.

This thesis inverts the localisation question and joins it to viewer need. The gate asks a vision–language model
whether the thing that makes a sound can be seen doing so. The benchmark (Chapter 4) labels, for every sound,
whether it is visible, whether a viewer would know it is happening without hearing it (obvious), and how much it
matters (importance). No released benchmark we found pairs per-sound timing with these three labels. Chapter 5
(§5.9) shows the typical failure: *presence without source*. A visible bell tower makes the gate silence an
off-screen bell. Localisation benchmarks with only visible sources cannot reveal this error.

## 2.4 Generating images from sound or from text

### Audio-conditioned generators

Several models draw an image directly from a sound. Sound2Scene maps audio into the latent space of a
pre-trained image generator and synthesises a scene that matches the sound (Sung-Bin et al., 2023). AudioToken
encodes audio into a new token for a text-to-image diffusion model, with few trainable parameters (Yariv et al.,
2023). SonicDiffusion conditions a pre-trained diffusion model on audio for generation and editing (Biner et al.,
2024). Seeing Sound builds training pairs for audio-to-image generation from visual data (Petermann & Kalayeh,
2025). AudioCanvas is trained on A2I-Set, a dataset of 323,000 audio–image–text triples (Ge et al., 2026).

These models answer "what does this soundtrack look like?". Two properties make them a poor fit here. First,
they condition on the sound of the whole clip, so they tend to draw the whole scene, not the one source that the
viewer cannot see. Second, they give no direct way to forbid an object. A text prompt can carry a noun guard and a
negative prompt; an audio embedding cannot. The project's image-generation panel reached the same view
(`docs/history/panels/panel3_topic2_rounds.md`). None of them looks at the video, so none can decide which sounds need a picture.

### Text-to-image generators

Text-to-image models have improved fast. FLUX.1 (Black Forest Labs, 2024) and Qwen-Image (C. Wu et al., 2025) are
open models with strong prompt following; Qwen-Image also supports a negative prompt. This thesis uses
text-to-image generation with a short, guarded prompt. The subject comes from the detector's label and a fixed
word list. The vision–language model may add at most a two-word kind qualifier, and a list of forbidden nouns
blocks siblings and common false objects. The video frames never condition the picture: the gate has just decided
that the source is *not* in those frames, so a frame could only copy a visible object or invent one. Free text
from a model was tried and rejected: it invented an object (a thud drawn as a door; Chapter 5 §5.11).

### Why the proposal's audio-to-image baseline was dropped

The project proposal planned a direct audio-to-image model as a baseline. It was not run. It would change two
things at once, the decision (it draws everything) and the renderer (a different, weaker image model). A gain over
it could not be credited to the gate. The *blind* arm replaces it: the same detector, label filter and generator,
drawing every detected sound without looking at the video. The difference between the gated and the blind arm is
then the gate alone (Chapter 5 §5.3–5.4). This follows the image-generation panel's judgement
(`docs/history/panels/panel3_topic2_rounds.md`) and the research-question map of Chapter 1 §1.4.

## 2.5 Evaluating generated content, and LLM or VLM judges

### Metrics validated against people

Audio captioning faced a similar evaluation problem. Early work scored audio captions with image-caption metrics.
Zhou et al. (2022) built two human benchmarks, AudioCaps-Eval and Clotho-Eval, from pairwise human judgements, and
found that the image-caption metrics agreed poorly with people. Their metric, FENSE, combines sentence embeddings
with an error detector that penalises broken sentences. CLAIR-A asks a large language model for a semantic
distance between two captions; on Clotho-Eval it was 5.8 % (relative) more accurate than FENSE (T.-H. Wu et al., 2025).
The lesson for this thesis is procedural: an automatic score earns trust only by agreeing with people on a
human-labelled set, and that check must be reported.

### Limits of LLM and VLM judges

Using a strong language model as a judge is now common. Zheng et al. (2023) showed that such judges can agree
well with people, but also that they have biases: they prefer the answer in a given position, they prefer longer
answers, and they may favour their own outputs. This project met the same limits. The secondary judge
(Gemma-4-31B) comes from a different model family than the gate and the subject model, to avoid self-preference.
It passed three trust checks against the annotator, but it cannot tell a picture from a text tag (Chapter 5
§5.10). A pairwise picture judge chose the first picture 187 of 190 times, a clear position bias, and it failed
its calibration bar with five other automatic picture checkers (§5.13). For this reason the thesis measures
picture recognition with people, in a short glance-naming test (§5.11), and treats the judge as "validated for
ranking, not absolute quality".

No automatic judge, and no hearing rater, can say what a DHH viewer gains. That needs a study with DHH viewers,
which this project designed and did not run (`docs/history/plans/beta_specification.md` §5).

## 2.6 The gap

Each area above solves part of the problem. Captioning practice already leaves out what the picture shows: the
source of a sound effect may be omitted when it is clearly seen, and background sounds are captioned only when
they are essential. DHH studies show that viewers want the important sounds, value low distraction, and have been
offered graphics as well as text. Sound detectors can name and time sounds, but closed-set taggers cannot hear quiet sounds under
speech or music. Localisation work asks where a visible source is, and only recently asks whether any source is
visible. Audio-to-image models draw a sound but never look at the video. Caption metrics show how to validate an
automatic score against people. No existing system or benchmark joins these parts for recorded video: *detect the
non-speech sounds, decide which of them the viewer cannot see, and show a timed picture only for those*. This
thesis defines that task (off-screen sound visualisation), gives a training-free pipeline for it (Chapter 3), a
pilot benchmark and per-sound metric (Chapter 4), and a pre-registered evaluation of it (Chapter 5).

## References

All works cited in this chapter are listed in the shared reference list, `docs/history/earlier_drafts/thesis/references.md`.

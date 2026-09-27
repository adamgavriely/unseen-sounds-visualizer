# Chapter 2 — Related work (draft, 27 Sept 2026)

*This chapter places the thesis in five research areas. It reuses and updates `docs/literature_review.tex`: the
accessibility argument is kept; every sentence about "what we use" now follows the shipped system of Chapter 3.
Model names in the text refer to that system (for example BEATs ∪ FlexSED as the detector, Qwen3.8-27B as the
visibility gate, FLUX.1-schnell as the generator, Gemma-4-31B as the secondary judge). Results of this project are
only pointed to (Chapter 5); they are not repeated here.*

## 2.1 Sound awareness and captioning for deaf and hard-of-hearing viewers

### Captioning practice

Captions for deaf and hard-of-hearing (DHH) viewers write speech as text. Good captioning practice also writes
non-speech information: sound effects, music and the manner of speaking. The DCMP Captioning Key is a widely used
style guide for this work (DCMP, n.d.). It asks the captioner to name the source of a sound effect, "unless the
source is clearly seen on screen", and it says that a sound the viewer can infer from the picture does not need a
caption. This rule is the starting point of this thesis. A professional captioner already decides, by hand, which
sounds the picture fails to show. The visibility gate of Chapter 3 is an automatic version of that decision.

In practice, much non-speech sound is never captioned. May et al. (2024) built a dataset of non-speech captions on
YouTube and found that many non-speech elements are left out, although they are often needed to follow the story.
A later review and survey (May et al., 2025) combined 36 papers with an online survey of 168 DHH participants and
interviews with 5 professional captioners. DHH viewers asked for more than a label: for example the emotional tone
of a sound, where it comes from, and why it matters. They also asked for choice. Captions that are present can
still lose meaning. In an EEG study, Revuelta et al. (2020) found that standard captions of sounds and music lose
emotional information, and that captions of music raise attention work more than emotional processing.

Recent systems try to make non-speech captions richer. CapTune lets caption authors define safe changes and lets
DHH viewers adjust the level of detail, the expressiveness and the way a sound is written (Huang et al., 2025).
Its formative work reports that sound words (onomatopoeia such as "swish") do not help every Deaf viewer, because a
person who has never heard a sound cannot decode its spelling. This is an argument for showing the *source* of a
sound, which needs no knowledge of how the sound is heard.

### Sound visualisation studies

A second line of work shows sounds as graphics, mostly for the viewer's own surroundings. Matthews et al. (2006)
designed and tested peripheral visual displays of ambient sounds for deaf users. Jain et al. (2015) tested
head-mounted display visualisations that show where a sound comes from, with 24 DHH participants. Findlater et al.
(2019) surveyed 201 DHH people about wearable and mobile sound awareness. Most were highly interested in knowing
about sounds, but this interest depended on how they communicate (sign, oral or both). Jain et al. (2019) studied
sound awareness in the home. SoundWatch classified sounds on a smartwatch and alerted the user (Jain et al., 2020);
its users welcomed the idea but raised concerns about misclassifications and delay. For urgent sounds they wanted
speed; for other sounds they wanted accuracy, so as not to be disturbed without need.

Closest to this thesis, Beyond Subtitles interviewed 11 DHH viewers and hearing video creators about text and
graphic captions for non-speech sounds in user-generated video (Alonzo et al., 2022). The authors built an
authoring tool on top of automatic sound event detection. Their participants wanted the *important* non-speech
sounds represented. They also found that the right form depends on the sound, the genre and the audience, and that
graphics are evocative but can be ambiguous or distracting. Caption Royale explored styled (affective) captions in
three studies with 39 DHH participants; readability, minimal distraction, intuitiveness and emotional clarity were
the main reasons behind their choices (de Lacerda Pataca et al., 2024). A picture panel adds a third place for the
eyes, next to the video and the captions. These findings argue for few, well-chosen pictures.

### Icons versus words

Should the panel show a picture, a word, or both? Caption practice is word-first (DCMP, n.d.). Many
sound-awareness systems show an icon together with a label. Classic work on user interfaces found that icons with
text labels were easier to learn than icons alone (Wiedenbeck, 1999). This thesis shows a picture without a word.
There are two reasons. First, the evaluation asks whether a picture alone is recognised at a glance; a word would
make that test trivial, because the viewer would read instead of recognise. Second, onomatopoeia excludes some
Deaf viewers (Huang et al., 2025). Adding the word is therefore a disclosed display choice (it exists in the code
and is switched off), not a tested claim. All of the work above is either about the user's own surroundings or
about captions chosen by an author. None of it decides *automatically*, for recorded video, which sounds the
picture already shows.

## 2.2 Sound event detection

### Closed-set taggers

Sound event detection (SED) asks which sounds occur in a recording and when. Most modern detectors are trained on
AudioSet, about two million 10-second YouTube clips labelled with a 527-class ontology (Gemmeke et al., 2017).
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
sound in words. Qwen2-Audio (Chu et al., 2024) and Qwen3-Omni (Qwen Team, 2025) are examples. This project tried
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
benchmark and a training method that mines hard negative regions inside the image. Later methods improved the
alignment between audio and visual features (Senocak et al., 2023). All of these methods, and their benchmarks,
assume that the source is in the frame. They ask "where?", not "whether".

Two recent papers question this assumption. Juanola et al. (2025a) tested visual sound-source localisation models
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
(`docs/panel3_topic2_rounds.md`). None of them looks at the video, so none can decide which sounds need a picture.

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
(`docs/panel3_topic2_rounds.md`) and the research-question map of Chapter 1 §1.4.

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
which this project designed and did not run (`docs/beta_specification.md` §5).

## 2.6 The gap

Each area above solves part of the problem. Captioning practice states the rule: describe a sound only if the
picture does not show it. DHH studies show that viewers want the important sounds, dislike distraction, and can
use pictures. Sound detectors can name and time sounds, but closed-set taggers cannot hear quiet sounds under
speech or music. Localisation work asks where a visible source is, and only recently asks whether any source is
visible. Audio-to-image models draw a sound but never look at the video. Caption metrics show how to validate an
automatic score against people. No existing system or benchmark joins these parts for recorded video: *detect the
non-speech sounds, decide which of them the viewer cannot see, and show a timed picture only for those*. This
thesis defines that task (off-screen sound visualisation), gives a training-free pipeline for it (Chapter 3), a
pilot benchmark and per-sound metric (Chapter 4), and a pre-registered evaluation of it (Chapter 5).

## References

Alonzo, O., Shin, H. V., & Li, D. (2022). Beyond Subtitles: Captioning and visualizing non-speech sounds to
improve accessibility of user-generated videos. In *Proceedings of ASSETS 2022*. https://doi.org/10.1145/3517428.3544808

Bilen, Ç., Ferroni, G., Tuveri, F., Azcarreta, J., & Krstulović, S. (2020). A framework for the robust evaluation
of sound event detection. In *Proceedings of ICASSP 2020*. arXiv:1910.08440

Biner, B. C., Sofian, F. M., Karakaş, U. B., Ceylan, D., Erdem, E., & Erdem, A. (2024). SonicDiffusion:
Audio-driven image generation and editing with pretrained diffusion models. arXiv:2405.00878

Black Forest Labs (2024). FLUX.1 [model release]. https://blackforestlabs.ai/

Chen, H., Xie, W., Afouras, T., Nagrani, A., Vedaldi, A., & Zisserman, A. (2021). Localizing visual sounds the
hard way. In *Proceedings of CVPR 2021*. arXiv:2104.02691

Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., & Wei, F. (2023). BEATs: Audio pre-training with
acoustic tokenizers. In *Proceedings of ICML 2023*. arXiv:2212.09058

Chu, Y., et al. (2024). Qwen2-Audio technical report. arXiv:2407.10759

DCMP — Described and Captioned Media Program (n.d.). *Captioning Key: Guidelines and preferred techniques*.
https://dcmp.org/learn/captioningkey

de Lacerda Pataca, C., Hassan, S., Tinker, N., Peiris, R. L., & Huenerfauth, M. (2024). Caption Royale: Exploring
the design space of affective captions from the perspective of deaf and hard-of-hearing individuals. In
*Proceedings of CHI 2024*. https://doi.org/10.1145/3613904.3642258

Drummond, C., & Holte, R. C. (2006). Cost curves: An improved method for visualizing classifier performance.
*Machine Learning, 65*(1), 95–130.

Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). CLAP: Learning audio concepts from natural language
supervision. In *Proceedings of ICASSP 2023*.

Ferroni, G., Turpault, N., Azcarreta, J., Tuveri, F., Serizel, R., Bilen, Ç., & Krstulović, S. (2021). Improving
sound event detection metrics: Insights from DCASE 2020. In *Proceedings of ICASSP 2021*. arXiv:2010.13648

Findlater, L., Chinh, B., Jain, D., Froehlich, J., Kushalnagar, R., & Lin, A. C. (2019). Deaf and hard-of-hearing
individuals' preferences for wearable and mobile sound awareness technologies. In *Proceedings of CHI 2019*.
https://doi.org/10.1145/3290605.3300276

Ge, D., Liu, S., Gong, C., Zhang, X.-L., Zhang, C., & Li, X. (2026). Towards expressive and faithful audio-to-image
generation: A unified multimodal dataset and synthesis framework (AudioCanvas). In *Proceedings of ACM Multimedia
2026*. arXiv:2608.09529

Gemmeke, J. F., Ellis, D. P. W., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., & Ritter, M.
(2017). Audio Set: An ontology and human-labeled dataset for audio events. In *Proceedings of ICASSP 2017*.

Hai, J., Wang, H., Guo, W., & Elhilali, M. (2025). FlexSED: Towards open-vocabulary sound event detection. In
*Proceedings of WASPAA 2025*. arXiv:2509.18606

Hershey, S., Ellis, D. P. W., Fonseca, E., Jansen, A., Liu, C., Moore, R. C., & Plakal, M. (2021). The benefit of
temporally-strong labels in audio event classification. In *Proceedings of ICASSP 2021*. arXiv:2105.07031

Huang, J. Z., de Lacerda Pataca, C., Wu, L.-Y., & Jain, D. (2025). CapTune: Adapting non-speech captions with
anchored generative models. In *Proceedings of ASSETS 2025*. arXiv:2508.19971

Jain, D., Findlater, L., Gilkeson, J., Holland, B., Duraiswami, R., Zotkin, D., Vogler, C., & Froehlich, J. E.
(2015). Head-mounted display visualizations to support sound awareness for the deaf and hard of hearing. In
*Proceedings of CHI 2015*. https://doi.org/10.1145/2702123.2702393

Jain, D., Lin, A., Guttman, R., Amalachandran, M., Zeng, A., Findlater, L., & Froehlich, J. (2019). Exploring
sound awareness in the home for people who are deaf or hard of hearing. In *Proceedings of CHI 2019*.

Jain, D., Ngo, H., Patel, P., Goodman, S., Findlater, L., & Froehlich, J. (2020). SoundWatch: Exploring
smartwatch-based deep learning approaches to support sound awareness for deaf and hard of hearing users. In
*Proceedings of ASSETS 2020*. https://doi.org/10.1145/3373625.3416991

Juanola, X., Haro, G., & Fuentes, M. (2025a). A critical assessment of visual sound source localization models
including negative audio. In *Proceedings of ICASSP 2025*. arXiv:2410.01020

Juanola, X., Morais, G., Fuentes, M., & Haro, G. (2025b). Learning from silence and noise for visual sound source
localization. In *Proceedings of BMVC 2025*. arXiv:2508.21761

Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio
neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing,
28*, 2880–2894. arXiv:1912.10211

Matthews, T., Fong, J., Ho-Ching, F. W.-L., & Mankoff, J. (2006). Evaluating non-speech sound visualizations for
the deaf. *Behaviour & Information Technology, 25*(4), 333–351.

May, L., Ohshiro, K., Dang, K., Sridhar, S., Pai, J., Fuentes, M., Lee, S., & Cartwright, M. (2024). Unspoken
Sound: Identifying trends in non-speech audio captioning on YouTube. In *Proceedings of CHI 2024*.
https://doi.org/10.1145/3613904.3642162

May, L., et al. (2025). "Choices? That's the dream": Challenges and opportunities in non-speech information
closed-captioning. *Frontiers in Computer Science*. https://doi.org/10.3389/fcomp.2025.1575176

Mesaros, A., Heittola, T., & Virtanen, T. (2016). Metrics for polyphonic sound event detection. *Applied
Sciences, 6*(6), 162.

Owens, A., & Efros, A. A. (2018). Audio-visual scene analysis with self-supervised multisensory features. In
*Proceedings of ECCV 2018*.

Petermann, D., & Kalayeh, M. M. (2025). Seeing Sound: Assembling sounds from visuals for audio-to-image
generation. arXiv:2501.05413

Qwen Team (2025). Qwen3-Omni technical report. arXiv:2509.17765

Revuelta, P., Ortiz, T., Lucía, M. J., Ruiz, B., & Sánchez-Pena, J. M. (2020). Limitations of standard accessible
captioning of sounds and music for deaf and hard of hearing people: An EEG study. *Frontiers in Integrative
Neuroscience, 14*, 1. https://doi.org/10.3389/fnint.2020.00001

Senocak, A., Oh, T.-H., Kim, J., Yang, M.-H., & Kweon, I. S. (2018). Learning to localize sound source in visual
scenes. In *Proceedings of CVPR 2018*.

Senocak, A., Ryu, H., Kim, J., Oh, T.-H., Pfister, H., & Chung, J. S. (2023). Sound source localization is all
about cross-modal alignment. In *Proceedings of ICCV 2023*. arXiv:2309.10724

Sung-Bin, K., Senocak, A., Ha, H., Owens, A., & Oh, T.-H. (2023). Sound to visual scene generation by
audio-to-visual latent alignment. In *Proceedings of CVPR 2023*. arXiv:2306.11504

Tzinis, E., Wisdom, S., Jansen, A., Hershey, S., Remez, T., Ellis, D. P. W., & Hershey, J. R. (2021). Into the
wild with AudioScope: Unsupervised audio-visual separation of on-screen sounds. In *Proceedings of ICLR 2021*.

Vyas, A., Chang, H.-J., Yang, C.-F., Huang, P.-Y., Gao, L., et al. (2025). Pushing the frontier of audiovisual perception with large-scale multimodal correspondence learning
(PE-AV). arXiv:2512.19687

Wiedenbeck, S. (1999). The use of icons and labels in an end user application program: An empirical study of
learning and retention. *Behaviour & Information Technology, 18*(2), 68–82.

Wu, C., et al. (2025). Qwen-Image technical report. arXiv:2508.02324

Wu, T.-H., Gonzalez, J. E., Darrell, T., & Chan, D. M. (2025). CLAIR-A: Leveraging large language models to
judge audio captions. In *Proceedings of ASRU 2025*. arXiv:2409.12962

Wu, Y., Tsirigotis, C., Chen, K., Huang, C.-Z. A., Courville, A., Nieto, O., Seetharaman, P., & Salamon, J.
(2025). FLAM: Frame-wise language-audio modeling. In *Proceedings of ICML 2025*. arXiv:2505.05335

Yariv, G., Gat, I., Wolf, L., Adi, Y., & Schwartz, I. (2023). AudioToken: Adaptation of text-conditioned diffusion
models for audio-to-image generation. In *Proceedings of Interspeech 2023*. arXiv:2305.13050

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In
*Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks)*.

Zhou, Z., Zhang, Z., Xu, X., Xie, Z., Wu, M., & Zhu, K. Q. (2022). Can audio captions be evaluated with image
caption metrics? In *Proceedings of ICASSP 2022*.

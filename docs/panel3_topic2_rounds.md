# Topic 2 — Image generation: panel rounds (27 Sept 2026)

Brief: docs/panel3_topic2_*.md · context: docs/panel3_common_context.md

## T2-b — round 1 (DHH accessibility, visual communication)

**Q1 Audio + frames.**
1. The frame cannot show the source by construction (gate fires only when not visible): a frame as style/appearance
   reference (IP-Adapter, editing) can copy only the scene or a *visible* object — scene look lowers figure-ground contrast;
   a visible object is the worst false message. Keep the ≤ 2-word kind qualifier as the only frame contribution.
2. Free audio-LLM description = the V3 failure path (thud → door). Ledger has a valid negative for Qwen2-Audio as second
   opinion (25/77 vs bar 40, 2026-09-15). Do not reopen as free text.
3. The one honest 6-day use of audio: a **closed question over sub-labels the detector already fired** (sheep / goat /
   unsure; phone / bell / unsure). Guard-compatible, targets sibling and vague-label errors; blind-arm bar on picture-DEV
   first. "Unsure" → parent label, never a guess.
4. Audio-conditioned generators (AudioToken, SonicDiffusion, Seeing Sound, AudioCanvas) sit on SD1.4/2.1/3 trained on
   scene frames: would lose the +22-pt Qwen-Image gain and draw scenery. Future work only.
5. Risk order: invented object > drawn-what-is-visible > slower glance read. Any addition scored under widened rule 2 first.

**Q2 What a picture should be.**
1. Name the source, not the sound (DCMP Captioning Key: sound captions state the source, omitted when visible — the gate is
   that rule automated).
2. No onomatopoeia as primary channel (CapTune formative work: Deaf viewers cannot decode phonetic sound-words; Beyond
   Subtitles: graphics evocative but ambiguous). Imagery first; word cards only where no maker exists — as frozen.
3. Icons succeed when the set is small and learned (Fortnite, SoundVizVR, SoundSign). Open label space → photo-real single
   object for group a; pictograms defensible for template group b. Frozen a/b/c split matches the literature — say so in ch3.
4. Picture + word is the literature default for novices (Wiedenbeck 1999). `SHOW_LABELS` exists (stage6 l.746, off), but a
   word defeats the glance test (reading, not recognising) — cannot count as a picture win.
5. Testable this week: a picture+label arm on picture-DEV with a different measure (forced choice vs sibling distractor or
   time-to-correct), reported separately. Not this week: DHH viewers, eye tracking, beta study.
6. Split attention: panel is a third fixation target; one object, no internal text, never longer than the sound.

**Q3 Setup.** Keep the frozen setup (Qwen-Image-2512, V3.1, white ground, templates b, cards c); don't reopen before the
sealed 81-sound confirmation. **Renderer finding:** `_opacity` (stage6 l.695) fades the panel by confidence,
`0.45 + 0.55·min(1, conf/0.6)`, over a near-black base — at bar 0.35 alpha 0.77, so white-ground pictures render grey, and
the glance test rated full-opacity pictures. Recommend alpha = 1 for any shown picture (display fix, disclose). No art
direction, no text in pictures.

**Q4 Judging without Adam.** Glance naming is hearing-neutral: add 2–3 hearing raters to the sealed 81-sound sheet by an
amendment written before the seal opens, alongside Adam; report pooled + Adam-only. Keep rule: assistant screen (κ 0.46)
may exclude, only blind human naming adopts. Closed-question arm: score only changed pictures (~20 sounds), paired, same
384 px / 1.5 s. No new automatic checker.

**Q5 Don't.** Touch the frozen setup, add arms or change prompts before the sealed confirmation is scored. No free audio-LLM
text in prompts; no text/onomatopoeia inside pictures.

Sources: Beyond Subtitles ASSETS 2022; DCMP Captioning Key; Wiedenbeck 1999; CapTune arXiv 2508.19971; Unspoken Sound CHI
2024; Seeing Sound arXiv 2501.05413; SonicDiffusion arXiv 2405.00878; AudioToken (GitHub).

## T2-a — round 1 (generative-model expert)

**Q1 Audio + frames.**
- Start from failure data: of 172 round-2 answers, misses an audio description could fix are ~5 (Thunk → door/punch
  P115/P142; phone bell → reception bell P166; siren/train horn → megaphone P072/P089). Sheep → goat ×3, goose → duck ×2,
  sigh → sneeze/crying, ~20 vague "bird/bell/train" are rendering/recognition failures (detector already said Pigeon, Church
  bell, Railroad car). "Can't tell" block (whoosh, thunder, car alarm, smash, rain) is now templates/cards. **Ceiling for
  audio-side text gain ≈ 5/172** — say so in the thesis.
- Audio-LLM caption worth one job: Qwen3-Omni-30B-A3B-Captioner (Apache-2.0, ~65 GB bf16, one H200/A100-80, audio-only, no
  text prompt, ≤ 30 s). Cannot be asked the constrained RESOLVE question → only post-hoc list guards on free text (weaker);
  captions the whole mix → feed the sound's span ±1 s, never the clip.
- **Proposed: report-only, 0 Adam, ~1 GPU-hour** on the 54 picture-DEV sounds, pre-written counts: (i) captions naming the
  detector's family, (ii) naming a different maker (phantom rate), (iii) which of the 5 text-fixable misses would change
  under the guard. Thesis paragraph, not a pipeline change.
- Audio-conditioned image models (AudioToken, Sound2Scene, AudioCanvas/A2I-Set 2026): condition on the mix embedding, draw
  the scene not the one source on white; weaker bases; no negative prompt or noun guard. No.
- Frame as reference/editing/IP-Adapter (Qwen-Image-Edit-2511 etc.): structurally wrong — the gate ruled the source is not
  in the frames; can only copy a visible object or invent one. Scene look also slows the glance (panel reads as added info
  because it looks unlike the video). Frames' honest contribution = the ≤ 2-word qualifier.

**Q2 What a picture should be.** Literature default is picture + one word (DCMP names the source; Jain CHI 2015/2019;
SoundWatch/HomeSound use icon + label). Own data: 20/172 vague answers are what one word resolves. Cost: language-bound,
~250 ms reading, wrong label + wrong picture = doubled false message. Photo vs icon already decided (whole thing mid-act,
white; drawn assets only for cards/templates). **Not testable with the naming test** (word makes naming trivial); make
picture+word a disclosed display decision (SHOW_LABELS False today; sealed sitting stays label-free); DHH testing = future.

**Q3 Setup.** Keep Qwen-Image-2512 (Apache-2.0, ~57 GB, ~20 s/picture H200, true CFG 4, negative prompt) + frozen V3.1 +
rules tail + templates/cards. New fact: 125 GB free disk → FLUX.2-dev (~64 GB, non-commercial licence, fine for thesis) no
longer disk-blocked; HiDream-O1-Image (MIT, 8B, May 2026) is the strongest untested candidate (negative-prompt support
unknown). Adopting either needs a full Adam sitting under the signed bar → not this week. If GPU idle: HiDream-O1 on the
54 picture-DEV sounds as a report-only column (assistant screen, κ 0.46, selection evidence only). Keep short subject +
rules tail; guarded expand_prompt is the only long-prompt arm worth keeping. No best-of-N (no calibrated checker).

**Q4 Judging.** No fourth instrument (six failed). Cheap and honest: (a) second annotator also glance-rates the 81
confirmation pictures under the sealed protocol (κ with Adam, 0 Adam minutes); (b) micro-sittings ~25 pictures / 10 min,
one pre-registered question each; (c) mechanical rule-2 pass (OBJECT line vs subject) before any human look. Use Adam's 172
answers as calibration for text-side changes: count only sounds whose drawn noun changed.

**Q5 Don't.** No new arm (audio caption, frame reference, generator, label) touches the frozen setup before the sealed
81-sound sitting is scored. Corollary: never condition a picture on pixels the gate said do not contain the source.

**Actions it proposes:** one report-only H200 job for the audio-caption counts; whether annotator 2 also glance-rates the
81 pictures; HiDream-O1 report-only column now or after submission.

Sources: Qwen3-Omni Captioner, Qwen-Image-Edit-2511, FLUX.2-dev (+licence), HiDream-O1-Image (HF); AudioCanvas arXiv
2608.09529; AudioToken arXiv 2305.13050; DCMP non-speech information; Jain CHI 2015, CHI 2019.

## T2-c — round 1 (picture-claim skeptic)

Read unsealed confirm_set_frozen.json and answer sheets; no sealed file opened.

**Flags on the sealed sitting — write down now, before any answer exists.**
1. Rule-2 (false message) pass is by eye with the arm known — write the criterion as one sentence first: *"A false message
   is an object shown as the maker that is not the source, its chain, the RESOLVE qualifier, or a declared surface/effect
   word (window pane, gravel, pieces, splash, flash, smoke)."* Cover all 80 FINAL pictures in the sitting; a GLM non-flag
   clears nothing.
2. V3.1 text was never blind-rated by Adam (assistant screen only, κ 0.46); sitting scores FINAL:A0 only → cannot split
   generator from text; only the composite is clean. "+22 pts, generator alone" (round 2) carries N0:37 (desk bell) — say
   "not clean" whenever quoted.
3. Claim wording: answers are scored vs the detector's source, not gold (sheet #0 Sheep/Bleat on a car clip — real or false
   alarm unknowable). Thesis: "shows the detector's source recognisably", never "shows what sounded".
4. The 81 are not independent: 50 clips, 22 with > 1 sound; 11/81 bare "Vehicle"; 6/81 same Shatter template; 47/81 source
   = family (specificity judged on 34). Declare clip-cluster bootstrap sensitivity + source≠family split now.
5. Sheet leniency: round-2 sheet listed "goat" wrong for Sheep; confirm sheet (source Bleat, under Sheep and Goat) lists
   "goat" correct for 2 sounds — disclose.
6. Missing bar: GP-2 had "wrong ≤ control's"; freeze does not. Report wrong per arm anyway (round 2: N0 5 vs today 3).
7. Glance enforcement: repo doesn't show the 1.5-s hide / no re-reveal is enforced — state it. 8 repeats at 80 % is coarse;
   report class-by-class agreement.
8. Ties — checked, good: no source has a sibling within the 0.8 tie margin; near pairs are chain pairs.

**Q1.** Free audio-LLM text = a new noun channel (sword, door, desk bell); captioners invent the object. Honest uses only:
(a) tie-break among fired siblings; (b) ≤ 2-word qualifier under the names_forbidden guard. Rating-free pilot (1 GPU-h):
audio-LLM on the 54 picture-DEV sounds, count main noun as fired source / unfired sibling / off-ontology — a risk number,
future work. Frames as reference: structurally wrong (source not in frame by the gate's verdict); style transfer lowers
contrast. None enters the claim.

**Q2.** Glance test measures naming the detector's label, not viewer benefit; picture-only is harder than a deployed panel.
Captioning practice word-first (DCMP; Gallaudet: description + onomatopoeia). DHH literature (Matthews 2006; Jain CHI 2015;
Findlater CHI 2019, n = 201) centres on identity — verify wording. Practical answer picture + word; word arm excluded from
the sitting by design — say so. Limitation line: "one hearing rater, picture alone, naming task at 1.5 s, no DHH viewer".

**Q3.** Run nothing; any swap reopens the freeze. Quote licence/VRAM from model cards.

**Q4.** No human, no new claim. (a) P4's second naive rater on the same confirm cards after Adam's file is saved, arm key
closed; κ reported, never a gate. (b) Rule-2 only: OWLv2 over FINAL pictures with each sound's forbidden-noun list
(siblings + door, phone, bell, police car, person) as a flag list for the by-eye pass; sanity: N:39 and N0:37 must flag,
N:13, N0:14, N0:29 must not; declared before, run after answers saved. Also: no-person check on non-human sounds, OCR for
drawn text, near-duplicate detection.

**Q5 Don't.** Reopen the sealed sitting: no redraw, seed change, template rewording, extra arm, or re-reading rule 2 after
seeing a picture.

**Actions it proposes:** (1) one-sentence rule-2 criterion into the freeze doc now; (2) declare clip-cluster bootstrap +
wrong-per-arm + source≠family split before answers; (3) second naive rater after Adam's sitting; (4) OWLv2 flag pass after
answers saved; (5) one-hour audio-LLM noun pilot on the 54.

Sources: Matthews et al. 2006; Jain CHI 2015; Findlater CHI 2019; DCMP Captioning Key.


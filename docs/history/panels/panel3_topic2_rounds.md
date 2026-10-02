# Topic 2 — Image generation: panel rounds (27 Sept 2026)

Brief: docs/panel3_topic2_*.md · context: docs/history/panels/panel3_common_context.md

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

---
# Round 2

## T2-b — round 2

(1) **Concedes the closed-question arm** (ceiling ≈ 5/172, still costs a sitting). One report-only job: **Qwen3-Omni-30B-
A3B-Captioner** on the 54 picture-DEV sound spans ±1 s, 0 Adam, ~1 GPU-h; pre-written counts: main noun = fired source /
unfired sibling / off-ontology, + which of the 5 text-fixable misses would change under the guard. Thesis paragraph +
future work only. Frames: ≤ 2-word qualifier only (all agree).
(2) **Opacity fix confirmed display-only.** `_opacity` stage6 l.695–698, applied l.751 (`img.putalpha`, picture cell) and
l.729 (chip mode); confidence = detector framewise max (stage4 l.121) → alpha 0.77 at 0.35 over base (16,18,24), white →
~grey 200. Sealed sitting never uses this path (`rating_screen.py` l.48–61 loads `{arm}/{i:02d}.png` → RGB → 384 px). Fix:
`CONFIDENCE_FADE = False` in config.py gating l.751. Caveat: the closed TEST panels the Gemma judge saw had the fade — apply
after the confirmation is scored, never re-render/re-score, disclose in ch3.
(3) **Drops the forced-choice picture+word arm**; disclosed display decision (SHOW_LABELS False, config l.156); limitation
line as T2-c; forced-choice design → future work.
(4) **Second raters: converges on T2-c** — one naive rater after Adam's file is saved, arm key closed, κ, never a gate; two
if available.
(5) **Adopts rule-2 sentence** (surface/effect word list declared per template before the seal opens; no-person corollary)
**and OWLv2 flag pass** (flag list only, declared before, run after save, sanity cases; non-flag clears nothing).
**Agreed:** frozen setup untouched until scored; frames = qualifier; one Captioner count job; picture+word = display
decision; one naive rater after save; rule-2 sentence + OWLv2 flags; opacity fix post-scoring, disclosed.
**Open:** (a) opacity applied before submission (demo renders) or limitation; (b) 1.5-s hide / no re-reveal enforcement
unverified in repo — coordinator to confirm from the artifact source before Adam sits; (c) "+22 pts not clean" wording.

## T2-a — round 2

(1) One report-only job: **Qwen3-Omni-30B-A3B-Instruct** (audio + text; Captioner dropped — no prompt), 54 picture-DEV
spans ±1 s, two prompts: (A) free "what is making this sound?" → main noun fired source / unfired sibling / off-ontology;
(B) closed question over the sound's fired list (subjects_V31.json, picture_bench.py l.222) + "unsure" → parent. First count
how many of the 54 have > 1 fired sub-label; if ≈ 0, (B) is moot. Targets: phone bell P166, siren P089, train horn P072 +
bare-family sounds (Thunk now a card). Future-work bar: (B) earns an Adam micro-sitting only if it changes the drawn noun on
≥ 3 pre-listed sounds with zero names_forbidden hits.
(2) `_opacity`: judge reads image_path PNGs (judge_direct.py l.82); sitting rates JPEGs from PNGs (rating_screen.py l.61);
nothing scored consumes the composited panel → alpha = 1 touches nothing scored; one switch, own commit, pictures
byte-identical to freeze commit 3995769. Demo videos were rendered faded → thesis footnote; re-render demos only.
(3) Picture + word = disclosed display decision, not scored; if deployed, state the confound (picture + word vs word).
(4) Adopt T2-c's pre-sitting items with fixes: declared-word list must cover template nouns (window pane, storm cloud,
tower, locomotive, headlights); wrong-per-arm reported, not gated; cluster bootstrap + source≠family = sensitivity rows;
glance hide lives in the built artifact page → export/commit that HTML. Post-sitting: second naive rater(s) by amendment
written now; OWLv2 flag pass yes (flag list for a human pass adopts nothing).
(5) HiDream-O1 after submission.
**Agreed:** frames never condition a picture; no free audio-LLM text in prompts; frozen setup untouched; picture + word =
display decision; no new checker; second naive rater; HiDream after submission; alpha = 1 disclosed. **Open:** (B)'s
population; who the raters are; who closes the declared-word list.

## T2-c — round 2

Confirmed opacity finding (stage6 l.695–753) and ledger negative (l.61). (1) One job, **Qwen3-Omni Instruct**, 54 sounds,
span ±1 s, 0 Adam, no picture; counts: (i) free caption main noun = fired source / family / unfired sibling / off-ontology;
(ii) closed question over fired sub-labels: agreement with choose_source + how many of the ~5 misses change. Bar for "worth a
future round": closed question changes ≥ 5/54 sources and 0 to an unfired kind. "≤ 5/172" is post hoc (upper bound in that
sample); closed question = second attempt at the 09-15 negative, reported as such. (2) Alpha = 1 legitimate and needed;
compositing is downstream of the frozen setup; disclose it was a deliberate confidence channel, list pictures it touched;
Gemma rows + demos saw grey cards — state, don't re-render TEST; commit after the sitting is scored. (3) Raters: amendment
written now, raters act after Adam's file is saved and before the arm key opens; Adam-only primary; naive raters; template
pictures reported split. (4) Annotator 2 as one of the raters (gold first if time competes). (5) Draft amendment for
docs/history/preregistrations/freeze_picture_setup_2026-09-25.md:

    Amendment, 2026-09-27 (panel 3, topic 2), before any FINAL picture or answer is seen.
    1. Rule 2: a FINAL picture is a false message if it shows as the maker an object that is not the source, its chain,
       the RESOLVE qualifier, or a declared surface/effect word (window pane, gravel, pieces, splash, flash, smoke, drops),
       or a person outside Human sounds. Applied by eye by [name] to all 80 FINAL pictures in the sitting (1 of 81 is a
       burst card), arm known, Adam's file unopened; the GLM OBJECT line is a flag list only — a non-flag clears nothing; a
       per-picture yes/no with the object named is committed before the file is opened.
    2. Sensitivities, reported beside the primary and never replacing it: clip-cluster bootstrap over the 50 clips; the
       split source != family (34) / source == family (47); wrong count per arm.
    3. Glance: each card is shown at 384 px for 1.5 s, then hidden and not shown again; the report states whether this
       held for every card.
    4. Disclosed: the confirm sheet counts "goat" correct for the 2 Bleat sources; round 2's Sheep source did not.
    5. Additional raters, secondary: after Adam's answers are saved and before the arm key opens, N = [1-3, names] hearing
       raters with no pipeline knowledge rate the same cards on the same page; kappa with Adam and pooled; never a bar;
       Adam-only stays the primary; template pictures reported split.
    6. Display: stage6 `_opacity` fades a shown picture to alpha 0.77 at bar 0.35 over near-black; the sitting rates
       full-opacity pictures. Alpha is set to 1 for any drawn picture after the sitting is scored (disclosed display
       change); the frozen TEST judge rows and demos stay as rendered.

**Agreed:** frames = qualifier only; no audio-conditioned generator; no free audio-LLM text; one Omni job; alpha = 1 after
scoring; raters by amendment now, secondary; annotator 2 among them; nothing touches the sealed sitting; HiDream after.
**Open:** who does the by-eye rule-2 pass; rater count/names; Omni job before or after the sitting.

---
# Round 3 (final)

## T2-b — round 3

(a) Omni bar: Instruct, both prompts; count > 1-fired-sub-label sounds first. Bar = T2-c's: closed question changes the
drawn noun on ≥ 5/54 and 0 to an unfired or names_forbidden kind; "second attempt at the 09-15 negative". Run before the
sitting (0 Adam). (b) Declared-word list closed from gen_screen.py TEMPLATES l.28–39: lightning bolt, storm cloud, rain
drops, window pane, parked car, headlights, indicator lights, locomotive, horn, church bell, tower, glass window, pieces;
+ gravel, splash, flash, smoke, drops. Coordinator closes it; nothing added after a picture is seen. (c) Raters: N = 2
target, 1 minimum, hearing, naive; annotator 2 only if gold doesn't compete. (d) Opacity: CONFIDENCE_FADE = False, own
commit after scoring; demos re-rendered; TEST judge rows untouched. (e) ch5 wording: "Qwen-Image-2512 with the shipped
text: 26/54 recognised vs 14/54 for FLUX.1-schnell (+22 points, paired, significant); one of the 26 showed a wrong object
(phone → desk bell). The sealed confirmation scores the frozen setup (generator + V3.1 text) as one composite; the gain is
not attributed to the generator alone." (f) By-eye rule-2 pass: annotator 2 if available, else the assistant, named.

Plan: 1 commit amendment + export/check rating page HTML (1 h coordinator). 2 Omni job (~1 H200-h). 3 mechanical rule-2 +
OWLv2 flags (< 1 GPU-h). 4 Adam's sitting (~1.5 h). 5 by-eye pass (~1 h). 6 naive raters (~1 h each). 7 open key, score
primary + sensitivities. 8 own commit: CONFIDENCE_FADE, demos, ch3/ch5 wording, Omni paragraph. HiDream + picture+word →
future work. **Signs.** Condition: if the 1.5-s hide is not enforced, fix the page before step 4 and record it.

## T2-a — round 3

1. Amendment committed before any FINAL picture/answer is seen: T2-c's six points with edits — (i) declared words cover
   TEMPLATES nouns (storm cloud, window pane, locomotive, tower, headlights, indicator lights) but **not TEMPLATE_NEG**
   nouns (roof light, light bar, police car, beacon) — those stay false messages; (ii) "wrong count per arm: reported,
   never a bar"; (iii) "enforced by the built artifact page; its HTML exported to the repo before the sitting". By-eye pass
   by the assistant, arm known, per-picture yes/no committed before Adam's file is opened.
2. Export glance page HTML, confirm 1.5-s hide / no re-reveal. 3. Adam's sitting (~45 min). 4. GLM OBJECT + OWLv2 flags
   with anchors, after Adam saves (~0.5 GPU-h). 5. Extra raters N = 2 (annotator 2 + one naive), κ + pooled, Adam-only
   primary, templates split. 6. Score and unseal by frozen commands. 7. Omni Instruct job, prompts A + B; if < 5 sounds
   have > 1 fired sub-label, only A runs; bar T2-c's (≥ 5/54, 0 unfired); P166/P089/P072 named, not a bar (~1–2 H200-h).
8. Opacity after scoring, pictures byte-identical to 3995769; thesis states TEST judge rows and demos were faded. 9.
   HiDream-O1 / FLUX.2-dev: future-work table.

**Signs. Dissents** only: declared-word list must never include TEMPLATE_NEG nouns.

## T2-c — round 3

**Final amendment text** (for docs/history/preregistrations/freeze_picture_setup_2026-09-25.md; template nouns from gen_screen.py TEMPLATES):

    Amendment, 2026-09-27 (panel 3, topic 2), written before any FINAL picture or answer is seen. Code unchanged (3995769).
    1. Rule 2. A FINAL picture is a false message if it shows as the maker an object that is not the source, its chain, the
       RESOLVE qualifier, or a declared word, or if it shows a person for a sound outside Human sounds. Declared words —
       general: pieces, splash, flash, smoke, sparks, dust, drops; per template: Thunder/Thunderstorm: lightning bolt, storm
       cloud; Rain/Rain on surface: drops, window pane; Car alarm: parked car, headlights, indicator lights; Train horn:
       locomotive; Church bell: bell, tower; Shatter/Smash: glass window, pieces. This list is closed now; nothing is added
       after a picture is seen.
    2. The pass: by eye, by [NAME], on all 80 FINAL pictures in the sitting (1 of 81 is a burst card), arm known, Adam's
       file unopened; the GLM OBJECT line and an OWLv2 pass over each sound's forbidden nouns are flag lists only (a
       non-flag clears nothing); a per-picture yes/no with the object named is committed before Adam's file is opened.
    3. Report-only rows beside the primary, never replacing it: clip-cluster bootstrap over the 50 clips; the split
       source != family (34) / source == family (47); wrong count per arm; template pictures separately.
    4. Glance: each card is shown at 384 px for 1.5 s, then hidden and never re-shown. The built page's HTML is committed
       before Adam sits; the report states whether this held for every card.
    5. Disclosed: the confirm sheet counts "goat" correct for the 2 Bleat sources (Bleat sits under Sheep and Goat);
       round 2's Sheep source did not.
    6. Additional raters, secondary: after Adam's answers are saved and before the arm key is opened, [N = 1-3: NAMES]
       hearing raters with no pipeline knowledge and no prior sight of any picture rate the same cards on the same page;
       reported as kappa with Adam and as a pooled rate; never a bar; Adam-only stays the primary.
    7. Display: stage6 `_opacity` fades a shown picture to alpha 0.77 at bar 0.35 over near-black; the sitting rates
       full-opacity pictures. After the sitting is scored, alpha is set to 1 for any drawn picture (one switch, own
       commit); the frozen TEST rows are never re-rendered; demo videos may be re-rendered and the change is stated in ch3.

**Omni bar (T2-c's final):** Qwen3-Omni-30B-A3B-Instruct, prompts A (free) + B (closed over fired sub-labels + "unsure" →
parent); pre-count B's population. B earns a future Adam micro-sitting only if it changes the drawn noun on ≥ 3 pre-listed
sounds (P166, P089, P072 + bare-family) **and** 0 names_forbidden hits across all 54. "Second attempt at the 09-15
negative"; "≤ 5/172" post hoc.

Plan (Adam / GPU): 1 commit amendment, Adam fills names (5 min). 2 export + commit rating-page HTML, check hide. 3 Adam
rates 168 cards (~40 min). 4 OWLv2 + by-eye pass, per-picture log (~10 GPU-min). 5 open, score, sensitivities. 6 naive
raters (~40 min each; annotator 2 may be one). 7 alpha = 1 own commit, demos only (≤ 1 GPU-h). 8 Omni job (~1 H200-h).
9 HiDream after submission.

**Signs**, condition: [NAME] for the rule-2 pass should not default to the assistant that selected the setup — Adam or
annotator 2 preferred; if the assistant, the per-picture log is mandatory.

---
# Topic 2 — outcome (coordinator)

**Signed by all three.** Frozen setup untouched until the sealed sitting is scored. Amendment = T2-c's final text above
(TEMPLATE_NEG nouns stay false messages — T2-a; the list as written already excludes them). Remaining small split: Omni
bar — T2-a and T2-b chose "≥ 5/54 changed, 0 to an unfired kind", T2-c's final says "≥ 3 pre-listed changed, 0 forbidden";
coordinator takes the majority (≥ 5/54) — report-only either way. Rule-2 pass by annotator 2 if available, else Adam
(~20 min), else the assistant with a mandatory per-picture log.


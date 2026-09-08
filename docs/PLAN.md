# Five-day plan to a defensible submission

Written 2026-09-08. The benchmark is frozen (256 labelled clips), no human study,
GPU access working. What remains is the proposal's own deliverable list (sec 11):
**experimental evaluation** and a **technical report** framed around it.

Standing rule for these five days: **no more clip sourcing and no more tagging.**
Every hour there is an hour not spent on the deliverables.

---

## What is already done

| Deliverable (proposal sec 11) | State |
|---|---|
| Source code | done -- 7-stage pipeline, runs on CPU and GPU |
| Curated benchmark dataset | done -- 256 labelled clips, 3 scenarios |
| Automatic evaluation protocol | **built**, never run |
| Experimental evaluation | **not started** |
| Technical report | 26 pp, framed around the wrong metric |
| Prototype system | works; needs watchable demo output |

Component result already secured: four visibility backends (CLIP, SigLIP, OWLv2,
Qwen2.5-VL) score within ~1 point of each other on the gating decision. Since these
span contrastive embedding, sigmoid embedding and open-vocabulary detection, the
uniformity is the finding -- the models answer *object presence* where the task needs
*event visibility*. That analysis is finished; it does not need more runs.

---

## Day 1 -- prove the protocol discriminates

The whole plan rests on the judge producing meaningful scores. Find out immediately.

**On the cluster**
```bash
cd ~/MscProj && sed -i 's/\r$//' slurm/*.sh
sbatch --export=LIMIT=12,GEN=retrieve slurm/job_protocol.sh
```
`GEN=retrieve` keeps the pilot fast; SDXL comes later.

**Check, in `benchmark/protocol_results.json`:**
1. Do scores vary, or is everything 4 (or 0)? A constant judge is a broken judge.
2. Read five `why` fields. Do they refer to the actual sound?
3. Does `audio_caption` differ from `proposed`? If all three systems score identically
   the protocol is not measuring the systems.

**If it fails:** the fix is prompt wording in `src/stage7_evaluation/protocol.py`
(the three prompts are module constants for exactly this reason). Budget the whole
day for one or two iterations. Do not scale up until scores discriminate.

---

## Day 2 -- the main experiment

**On the cluster, in the morning**
```bash
sbatch --export=LIMIT=100 slurm/job_protocol.sh      # ~6-8 h, SDXL generation
```
100 clips x 3 systems. That gives roughly +/-10% on each mean, which is enough to
separate the systems; 256 clips would cost two days and buy precision the argument
does not need.

**In parallel, same day**
```bash
python -m benchmark.select_demo --write 12
sbatch slurm/job_diffusion.sh                        # ~1-2 h, demo renders
```

**Meanwhile (no GPU needed):** restructure the report skeleton -- move gating
accuracy out of the headline and into a component-analysis section, leave the
results chapter empty for Day 3.

---

## Day 3 -- results and demos

1. Pull everything back: `bash slurm/fetch_results.sh` (from the PC).
2. Watch the 12 demo videos. Discard any where the panel is on the whole time --
   `select_demo.py` ranks against that, but confirm by eye.
3. Write the results chapter around the one table that matters:

   | system | mean judge score | % scoring >= 3 |
   |---|---|---|
   | proposed (gated) | | |
   | blind audio-to-image | | |
   | audio captioning | | |

   Plus the same table broken down by the three proposal scenarios.

4. **The interpretation is decided by the data, and both directions are publishable:**
   - proposed > blind: the gate earns its place, measured on output rather than on an
     intermediate label. This is the positive result.
   - proposed ~ blind: selective augmentation does not improve semantic delivery.
     State it plainly; combined with the four-backend uniformity it becomes a
     coherent negative result about off-the-shelf cross-modal grounding.

---

## Day 4 -- write

Full pass over `docs/project_notes.tex`, in this order of importance:

1. **Results chapter** (Day 3's table + the per-scenario breakdown).
2. **Component analysis**: the four-backend comparison, the trivial baselines, and
   the object-vs-event finding with the annotator quotes.
3. **Method**: the protocol itself -- the five steps, the separate judge, and why the
   describer never sees the reference. This is a named contribution (sec 11) and
   deserves its own section.
4. **Limitations, stated before an examiner finds them:**
   - single annotator, kappa 0.60, instability concentrated in the positive class;
   - sourcing yield 15-20% across four independent strategies, hence 256 clips
     rather than 300 -- report the scarcity as a dataset finding;
   - part of the queue was model-enriched; the first 209 tags predate any filter and
     are the unbiased sample;
   - the judge is a single 7B LLM, not a human panel.

---

## Day 5 -- package

1. Recompile `project_notes.pdf` and `literature_review.pdf`.
2. Assemble the advisor package: the results table, 2-3 demo videos, the two PDFs.
3. `benchmark/ATTRIBUTIONS.md` complete for every clip source.
4. README: how to run the pipeline on one video, and how to reproduce the evaluation.

**Questions for the advisor**, prepared in advance:
- Is a rigorous negative result on cross-modal gating acceptable as the core
  contribution, given the proposal framed the work as a feasibility study?
- 256 clips rather than ~300: acceptable, given measured yield?
- Should the judge be a stronger or second model, or a small human panel?
- Is a DHH user study wanted at all, and if so does the ethics timeline allow it?

---

## Risks, and what to do about each

| Risk | Signal | Response |
|---|---|---|
| Judge does not discriminate | Day 1 scores all equal | rewrite the judge prompt; if still flat, score with CLIP image-text similarity instead and report the judge as a limitation |
| SDXL too slow for 300 runs | Day 2 job hits the 12 h wall | it requeues and resumes; failing that, `GEN=retrieve` for the baselines and SDXL only for demos |
| Qwen2.5-VL will not fit beside Mistral | OOM on Day 1 | run describe and judge as two sequential passes over the same clips |
| Nothing works by Day 3 | | fall back to the component results already secured: four backends, trivial baselines, object-vs-event, kappa. That alone is a thesis, just a thinner one |

## Explicitly out of scope
More sourcing, more tagging, tagger features, DHH user study, TempoTokens /
audio-to-video, icon-based rendering, stereo direction cues.


---

# If there are 8 days, not 5

The extra three days should NOT go on more clips. Yield is 15-20% and the benchmark
is not the weak part -- the *methodology* is. Three known holes can each be closed
with GPU time and no additional annotation, and each removes an objection an examiner
would otherwise raise.

## Day 6 -- make the evaluation itself defensible

**a) Judge reliability.** One 7B model deciding every score is the protocol's softest
point. Re-run the judging step only (descriptions and references are already cached)
with a second judge from a different family, and report agreement:

```bash
sbatch --export=JUDGE2=Qwen/Qwen3-8B slurm/job_judge_agreement.sh
```

Report Spearman correlation and exact-match rate between the two judges. High
agreement means the scores are a property of the augmentations, not of one model's
quirks. Low agreement is equally worth reporting -- it bounds how much any
LLM-judged number in this area should be trusted, which is a contribution to
proposal RQ4 ("how should such systems be evaluated?").

**b) Fix tuning on the test set.** Every threshold reported so far was chosen by
sweeping on the same 256 clips it is evaluated on. Split the benchmark 50/50, tune on
the first half, report on the second, and state both numbers. Cheap, and it removes a
guaranteed question.

## Day 7 -- ablations that answer "was it the components?"

All GPU-only, no human time. Each isolates one stage so the report can say which
choices mattered:

| ablation | question it settles | cost |
|---|---|---|
| SDXL vs Openverse retrieval | does *generating* the image beat *retrieving* one? | ~2 h |
| BEATs vs PANNs (Stage 4) | is the detector the bottleneck? (measured: no -- an oracle detector moved accuracy 49.3% -> 49.1%) | ~2 h |
| gate on vs off, same generator | isolates the gate's contribution to the OUTPUT score | free, already in the protocol |

The SDXL-vs-retrieval one is the most interesting: the proposal names diffusion as the
generation method, and nobody has checked whether it actually helps a DHH viewer more
than a stock photograph does.

## Day 8 -- qualitative analysis, then buffer

1. **Worked examples for the report.** Three clips where the system clearly helps,
   three where it clearly fails, each with the frame, the augmentation, the reference
   and the judge's reasoning. Examiners remember these far longer than tables.
2. **Failure taxonomy** from the protocol's low-scoring cases -- group them (wrong
   source depicted, right source but uninformative image, augmented something already
   visible, missed the salient sound) and count each.
3. Buffer. Something will have slipped.

## Optional, if a labmate can spare two hours

**Inter-annotator agreement.** A second person labels 30 clips; report Cohen's kappa
against Adam's labels. This is annotation reliability, not a user study -- no ethics
process, no DHH participants. It is the single highest-value number still missing,
because it separates "the model is bad" from "the task is not yet well-defined", and
it strengthens contribution (a). Either outcome helps: high kappa validates the task,
low kappa explains the 50% gate and makes the task-definition finding rigorous.

## Still out of scope, even with 8 days
More sourcing, more tagging by Adam, DHH user study, audio-to-video generation.

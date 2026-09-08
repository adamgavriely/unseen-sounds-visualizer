# Two-week plan to submission

Rewritten 2026-09-08 against the real repo state. Supersedes the earlier 5/8-day drafts.

## Where the project actually stands

| Proposal deliverable (sec 11) | State |
|---|---|
| Source code | done -- 7 stages, runs on CPU and GPU |
| Curated benchmark | done -- 274 labels, **frozen** |
| Automatic evaluation protocol (sec 6.1) | **built, never successfully run** <- the blocker |
| Experimental evaluation vs baselines (sec 7) | blocked on the above |
| Technical report | 26 pp, framed around the wrong metric |
| Prototype system | done -- 13 demo videos already rendered |

Component results already secured, and they do not need re-running:

* four Stage-2 backends within ~2 points of each other (CLIP 49.3, SigLIP 49.6,
  OWLv2 50.4, Qwen2.5-VL 51.2 accuracy) -- architecture does not decide this task;
* event-level AUROC 0.647 (SigLIP) / 0.616 (OWLv2) -- weak but above chance;
* trivial baselines: always-augment 26.6% acc / 42.1 F1, always-silent 73.4% / 0 F1;
* object presence != event visibility, evidenced by the annotator's own words;
* label stability kappa = 0.60, instability concentrated in the positive class.

**Standing rules.** No more sourcing. No more tagging. No more chasing gating accuracy.
The missing deliverable is the protocol and the baseline comparison built on it.

---

# Week 1 -- get the numbers

## Day 1 (today) -- unblock the protocol

The pilot job vanished with no log, so nothing is known about why. Diagnose it
interactively, where errors print live instead of disappearing:

```bash
sacct -j 28267981 --format=JobID,JobName%16,State,ExitCode,Elapsed,Reason%30

srun --partition=L4-4h --gres=gpu:1 --mem=32G --time=0:30:00 --pty bash
cd ~/MscProj && source ~/miniconda3/etc/profile.d/conda.sh && conda activate msproj
sed -i 's/\r$//' slurm/*.sh
python -m benchmark.run_protocol --limit 2 --systems proposed
```

Cause found (2026-09-08): nothing was wrong with the code -- the job was queued on
`H200-12h`, whose only node shows `mixed-` (draining), so it pended indefinitely.
`sinfo` showed 8 idle nodes on `generic`; the job now defaults there.

**The protocol runs in two passes** so the describing VLM (~16 GB) and the judging
LLM (~15 GB) are never resident together, since they do not co-fit on a 24 GB card
and the large-memory partitions are often draining:
`--phase describe` caches references and descriptions, frees the VLM, then
`--phase judge` loads the judge alone and scores the cache. A second benefit: the
Day-6 judge-agreement run needs no vision work at all.

**Done when** two clips produce a reference, a description and a score.

## Day 2 -- prove the judge discriminates

```bash
sbatch --export=LIMIT=12,GEN=retrieve slurm/job_protocol.sh
```

Then read `benchmark/protocol_results.json` and check three things:

1. scores **vary** -- all 4s or all 0s means a broken judge, not a good system;
2. the `why` text names the actual sound rather than praising generically;
3. `audio_caption` scores **differently** from `proposed` -- if all three systems tie,
   the protocol is not measuring the systems.

If any fail, the fix is prompt wording in `src/stage7_evaluation/protocol.py`, where
the three prompts are module constants for exactly this reason. Iterate here; do not
scale up on a judge that cannot discriminate.

## Day 3 -- the main experiment

```bash
sbatch --export=LIMIT=100 slurm/job_protocol.sh     # 100 clips x 3 systems, ~6-8 h
```

100 clips gives roughly +/-10% per mean, enough to separate three systems. 274 clips
costs two days for precision the argument does not need.

## Day 4 -- results

Fetch (`bash slurm/fetch_results.sh`), then build the table the thesis turns on:

| system | mean judge score | % scoring >= 3 |
|---|---|---|
| proposed (gated) | | |
| blind audio-to-image | | |
| audio captioning | | |

plus the same broken down by the three proposal sec 5.1 scenarios. Both directions
are publishable: *proposed > blind* means the gate earns its place measured on
output, not on an intermediate label; *proposed ~ blind* is a coherent negative
result alongside the four-backend uniformity.

## Day 5 -- watch the demos, fix what looks wrong

13 videos are already in `data/output/`. Watch them. Any where the panel is on for
the whole clip, or the image is unreadable, is a presentation bug worth an hour --
this is the artefact an advisor reacts to. `benchmark/select_demo.py` ranks clips by
augmentation dynamics if a fresh set is wanted.

---

# Week 2 -- make it defensible, then write

## Day 6 -- close two methodological holes

**Judge reliability.** One 7B model deciding every score is the protocol's softest
point. Because the run is split into two passes, re-scoring costs no vision work:

```bash
python -m benchmark.run_protocol --phase judge --judge Qwen/Qwen3-8B --tag judge2
```

Report Spearman correlation plus exact-match rate between the two judges. High
agreement means the scores are a property of the augmentations; low agreement bounds
how far any LLM-judged number in this area can be trusted -- which is itself an answer
to RQ "how should such systems be evaluated?".

**Stop tuning on the test set.** Every threshold so far was swept on the same clips it
is scored on. Split 50/50, tune on one half, report the other, state both numbers.

## Day 7 -- ablations (GPU only, no annotation)

| ablation | question it settles |
|---|---|
| SDXL vs Openverse retrieval | does *generating* the image beat *retrieving* one? The proposal names diffusion; nobody has checked it helps |
| BEATs vs PANNs | is the detector the bottleneck? (already measured: no -- an oracle detector moved accuracy 49.3 -> 49.1) |
| gate on/off, same generator | the gate's contribution to the OUTPUT score, not to an intermediate label |

## Days 8-9 -- write

1. **Results chapter** -- the Day 4 table and the per-scenario breakdown.
2. **Method** -- the protocol as a named contribution: the five steps, the separate
   judge, why the describer never sees the reference.
3. **Component analysis** -- four backends, trivial baselines, AUROC, and the
   object-vs-event finding with the annotator quotes.
4. **Limitations, stated before an examiner finds them:** single annotator with
   kappa 0.60; 15-20% sourcing yield hence 274 clips not 300 (report the scarcity as
   a dataset finding); part of the queue was model-enriched while the first 209 tags
   predate any filter and are the unbiased sample; a single 7B judge, not a panel.

## Day 10 -- package

Recompile both PDFs, complete `benchmark/ATTRIBUTIONS.md`, README with how to run the
pipeline on one video and how to reproduce the evaluation, and assemble the advisor
package: results table, 2-3 demo videos, the two PDFs.

---

## Risks and responses

| Risk | Signal | Response |
|---|---|---|
| Judge does not discriminate | Day 2 scores all equal | rewrite the judge prompt; if still flat, fall back to CLIP image-text similarity as the metric and report the LLM judge as a limitation |
| Models will not co-fit on the GPU | OOM | already handled -- the run is two sequential passes, one model resident at a time |
| SDXL too slow for 300 runs | Day 3 hits the wall | the job requeues and resumes; else `GEN=retrieve` for baselines, SDXL for demos only |
| Nothing works by Day 5 | | fall back to the secured component results -- four backends, trivial baselines, AUROC, object-vs-event, kappa. A thinner thesis, but a complete one |

## Out of scope for both weeks
More sourcing, more tagging, tagger features, DHH user study, audio-to-video
generation, icon rendering, stereo direction cues.

CONTEXT (do not read other files). Adam Gavriely's MSc (Bar-Ilan): a no-training pipeline that shows a picture beside a video for ambient sounds whose source is not visible (the "gate"); baselines: blind (picture for every sound, never looks) and caption (sound name as text). Open-weight models only; deadline this week (today 2026-09-15, late night). Headline: automatic judge, 100 test clips, gated 3.17 vs blind 3.16, tie.

THE CONCEPT WE WANT TO IMPROVE: "measure how much a picture beside a MUTED video gives a deaf viewer back" (Adam's idea, pre-registered tonight as a score, docs/history/preregistrations/prereg_gap_closing.md):
- One audio-visual model (Qwen2.5-Omni-7B, the only one that loads here) answers four fixed questions -- (1) what is happening? (2) is anything dangerous or urgent? (3) what is the mood? (4) is anything happening that you cannot see? -- three times: HEARING (frames + soundtrack), DEAF (same frames, muted), DEAF WITH PANEL (frames + a system's side panel, muted); one run per system.
- Gain = how much closer the with-panel answers get to the hearing answers than the muted answers were (MiniLM cosine on answer text; LLM pairwise judge secondary). Cost = seconds the panel is on. Two axes, because gain cannot see redundancy.
- A kill criterion is running right now on 60 dev clips: the hearing-vs-deaf gap must be larger on clips with an ambient sound than on clips with none (AUROC >= 0.75) and larger than the wording noise from re-sampling frames. Results in ~1.5 h.

KNOWN WEAK SPOTS (evidence from tonight's earlier pilot with the same model):
1. The model sometimes does not hear: kitchen smoke alarm with sound -> "food being cooked, food being stirred". Or hears the wrong thing: office fire alarm -> "gunshots ring out, glass shatters". It does hear sirens well: "car drives by" muted -> "emergency vehicle siren" with sound.
2. Wording noise: two runs describe the same scene with different words; a free "list of events" differed by 2-3 items on every clip. Fixed questions are the current fix; unverified.
3. Gain cannot see redundancy: a picture of an ambulance next to a visible ambulance still moves the answers toward "siren", so blind >= gated by construction on this axis. The gate's benefit only appears on the cost axis.
4. A siren picture conveys "danger" instantly to a person; a text answer may under-state that.
5. Nothing about timing (a picture 2 s late), trust (one wrong picture and viewers ignore the panel), or attention load.
6. It is still a model pretending to be a deaf viewer; no human has validated it. Adam will not annotate more himself; 2-3 hearing friends watching muted clips for an hour is possible.

FAILED BEFORE (do not re-propose): asking any model "is the source visible?" (three designs, all at chance ~50-55%); subtracting a with-sound event list from a muted event list (chance, wording noise); a second audio detector or audio-LLM to veto the sound detector (all failed declared bars); source separation before detection (made the detector worse).

RESOURCES: BIU Slurm GPU (L4 24 GB / H200), Omni ~1 clip/min per run; 100 test clips have rendered outputs for gated/blind/caption; 156 dev clips have no renders (~6 GPU-h to render all three). Process rules: bars declared before runs; validation on dev; test touched once (twice at most, stated).

DIRECTIONS TO INVENT AROUND:
A. Better "hearing" and "deaf" viewers (two ears: audio-only model + audio-visual model must agree; the with-panel viewer sees the picture as a picture, not a caption of it).
B. Urgency weighting: danger/urgency changes count more than mood changes (Adam's siren point).
C. Redundancy made visible without asking the visibility question (e.g., does the panel change the answers at all? if the muted viewer already says "ambulance", a panel that adds "ambulance" changes nothing -> zero gain, zero cost? or count panel-on time only when gain is zero?).
D. Timing: ask the questions at three time points so a late picture scores less.
E. The smallest human anchor (2-3 hearing people, muted clips) that would make the automatic number credible.

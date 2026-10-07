# Step 5 tier 1: ears on DEV

| ear | bar | burst AUROC (good vs not) | needed sounds with a span starting in the onset window |
|---|---|---|---|
| pipeline BEATs (reference) | 0.175 | 0.710 | see note |
| atstf_orig | 0.05 | 0.837 | 34 / 58 |
| beatsS_orig | 0.05 | 0.836 | 36 / 58 |

Reference AUROCs on the same bursts (Step 4 features): BEATs 0.710, FlexSED 0.809, DASM 0.885.

Notes (original checkpoints, before any fine-tuning):
- Both calibrated bars sit at the low edge of the declared grid (0.05): frame F1 on the 415 held-out keeps rising as the
  bar drops, so the grid edge is the bar. Kept as declared.
- Pipeline reference for the recall column, from the decision trail (any candidate of that origin, any length, starting
  in the onset window): BEATs 38 / 58, FlexSED (incl. band) 27 / 58, DASM 11 / 58, all ears together 48 / 58.

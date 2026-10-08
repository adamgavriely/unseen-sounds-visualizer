# Step 5 tier 1: ears on DEV

| ear | bar | burst AUROC (good vs not) | needed sounds with a span starting in the onset window |
|---|---|---|---|
| pipeline BEATs (reference) | 0.175 | 0.710 | see note |
| atstf_orig | 0.05 | 0.837 | 34 / 58 |
| beatsS_orig | 0.05 | 0.836 | 36 / 58 |
| atstf_ft_s0 | 0.05 | 0.859 | 29 / 58 |

Reference AUROCs on the same bursts (Step 4 features): BEATs 0.710, FlexSED 0.809, DASM 0.885.

## Verdict (as pre-registered)

Fine-tuned ATST-F (seed 0, best epoch 2 by held-out mixture loss; log ft_atstf_s0_log.json): burst AUROC 0.859 (up from
0.837 original, 0.710 pipeline BEATs) but needed sounds heard 29 / 58 (down from 34 original, 36 BEATs-strong; pipeline
BEATs candidates 38). Tier 2 needed both numbers above both references: **FAILED, tier 2 not run.** The mixture loss
rose after epoch 2 (overfitting to the mixtures). All three frame ears sit at the low edge of the declared bar grid.

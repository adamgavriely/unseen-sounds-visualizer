# benchmark/gold — labels, scorer and scoring harness

This folder holds the human per-sound labels, the scorer, and the harness that replays the final system on the
cached model outputs of each clip. Short codes in file and variant names (for example `SHIP8+MD3+WW5+SL`, the final
system) are explained in Appendix D of the report.

| purpose | files |
|---|---|
| **Labels** | `annotations/gold_AG.json` (all per-sound labels), `dev_stems.txt`, `test_stems.txt`, `tagger_split.json`, `split.json`; labelling guideline `tool_template.html` |
| **Scorer** | `score_per_sound.py` (matching rules, cost), `dev_candidates_check.py` (bootstrap), `holm_table.py` (Holm correction) |
| **Report tables** | `final_vs_baselines.py` → `final_vs_baselines.json` (final system and baselines); `final_vs_baselines_extra.py` → `final_vs_baselines_extra.json` (F1 intervals, danger sounds) |
| **Harness** | `round13_dev.py` (variants), `merged_dev.py`, `final_test.py`, `r13_test_final.py`, `parity_check.py`, `tagger_prep.py` (builds the per-clip model inputs; also used by `src/listener_prep.py`) |
| **Decision trail** | `inspector_trail_export.py`, `inspector_data.py` → `docs/inspector2/` |
| **Cached model answers** | `*_listener*.json` (listener answers per split), `panns_fw/` (PANNs frame scores), `grp/` (grouping answers), other `*.json` read by the modules above |
| **Analyses cited in the report** | `ceiling_ship7.md`, `visible_weight_sweep.md`, `holm_test_final_v33_test_bench.json`, `logit_gate_gold.json`, `depictable_vocab.json` |

The remaining Python modules (`*_screen.py`, `listener_*.py`, `flexsed_*.py`, …) hold rules and helpers that the
pipeline and the harness import.

## Reproducing the numbers

The reported numbers are stored here: `final_vs_baselines.json`, `final_vs_baselines_extra.json` and
`../../docs/inspector2/data_parity.json` (configuration check). Appendix C of the report gives the source file of
every number.

The scripts that made them read the stage-5 outputs of every benchmark clip (under `data/work/`), which are not in
the repository. They are rebuilt on the cluster from the video clips, and the clips are not redistributed: their
names are in `dev_stems.txt`, `dev2_stems.txt`, `test_stems.txt` and `test2_stems.txt` (the `tg_d*` clips go in
`data/input/tagger_set/`). With the clips in place, `tagger_prep.py` builds the model inputs and outputs (the order of
the steps is in `slurm/run_best.sh`), and then:

```
python benchmark/gold/final_vs_baselines.py                 # Table 3: final system and baselines
python benchmark/gold/final_vs_baselines_extra.py           # F1 intervals, danger sounds
python benchmark/gold/parity_check.py SHIP8+MD3+WW5+SL      # configuration check
python benchmark/gold/inspector_trail_export.py \
   --expect DEV=29/58/15/6/7/2/2.056 --expect TEST=24/65/24/4/15/5/2.409
```

From a plain clone these scripts stop at the first missing clip or output.

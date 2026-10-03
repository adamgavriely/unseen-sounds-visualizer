# benchmark/gold — labels, scorer and harness

This folder holds the human per-sound labels, the scorer, the harness that replays the final system on the cached
model outputs of each clip, and the listener harness that prepares the model inputs of a new clip. Short codes in
file and variant names (for example `SHIP8+MD3+WW5+SL`, the final system) are explained in Appendix D of the report.
The research screens, the cached model answers of the DEV/TEST splits and the other study outputs are in release
v1.2.0.

| purpose | files |
|---|---|
| **Labels** | `annotations/gold_AG.json` (all per-sound labels), `dev_stems.txt`, `dev2_stems.txt`, `test_stems.txt`, `test2_stems.txt`, `tagger_split.json`, `split.json`, `judge100.txt` (the development-set rule), `audioset_slice.json` (the AudioSet-Strong clips); labelling guideline `tool_template.html` |
| **Scorer** | `score_per_sound.py` (matching rules, cost, clip subsets), `dev_candidates_check.py` (stage-4/5 replay helpers, bootstrap, DASM step), `holm_table.py` (Holm correction) |
| **Report tables** | `final_vs_baselines.py` → `final_vs_baselines.json` (final system and baselines); `final_vs_baselines_extra.py` → `final_vs_baselines_extra.json` (F1 intervals, danger sounds) |
| **Scoring harness** | `round13_dev.py` (variants, stage 4/5 through the pipeline code), `merged_dev.py`, `r13_test_prep.py`, `r13_test_final.py` → `r13_test_final.{json,md}`, `final_test.py` → `final_test.json`, `parity_check.py` |
| **Listener harness** | `tagger_prep.py`, `flexsed_run.py`, `dev_listener.py`, `test_listener.py`, `listener_variants.py`, `listener_afnext.py`, `dasm_rescue.py`, `listener_p1v4.py`, `finelap_screen.py` (order below) |
| **Picture prompts** | `gen_screen.py` (picture templates and seeds; imported by `src/stage6_visual_augmentation/` and `comfyui_nodes/run_frozen.py`) |
| **Decision trail** | `inspector_trail_export.py`, `inspector_data.py` → `docs/inspector2/` |
| **Model inputs and analyses cited in the report** | `depictable_vocab.json` (the 215 sound families), `flexsed_extra_queries.json` (extra FlexSED queries of the `FLEXSED_EXTRA` option), `ceiling_ship7.md`, `visible_weight_sweep.md`, `holm_test_final_v33_test_bench.json`, `logit_gate_gold.json` |

## Listener harness: what runs for a new clip

`main.py` calls `src/listener_prep.py` (and `slurm/run_best.sh` runs the same chain on a folder). For a clip with no
prepared answers it runs, each as its own process, on a one-clip split named `live_<clip>`:

1. `src/stage4_audio_event_detection/dasm_infer.py`, once per machine: DASM's MGA-CLAP text embeddings of the 215
   families → `data/work/dasm_text_queries.pt` (skipped when the file exists).
2. `tagger_prep.py` steps `flexsed` (`flexsed_run.py`), `render` (`../run_protocol.py`, stages 1–6 with
   `config.use_scored()`), `wav16`, `beats`, `panns`, `stage4` and `stage5` with arm `B0r` (`round13_dev.py`),
   `lpool` (`test_listener.py`, `dev_listener.py`), `qwen` (Qwen3-Omni yes/no and variants: `dev_listener.py`,
   `test_listener.py`, `listener_variants.py`), `afn` (Audio Flamingo Next: `listener_afnext.py`) and `dasm`
   (`dev_candidates_check.dasm()` with `dasm_infer.py`).
3. `dasm_rescue.py pool` and `dasm_rescue.py listen`: DASM-only spans and both listeners' answers on them.
4. `listener_p1v4.py`: Qwen V4 on the P1 cuts.
5. `finelap_screen.py split`: FineLAP frame scores, in the FineLAP Python environment (`FINELAP_PYTHON`).

The answers land in `benchmark/gold/live_<clip>_listener*.json` and `data/work/` and are read by stage 4.

## Reproducing the numbers

The reported numbers are stored here: `final_vs_baselines.json`, `final_vs_baselines_extra.json` and
`../../docs/inspector2/data_parity.json` (configuration check). Appendix C of the report gives the source file of
every number.

The scripts that made them read the stage-5 outputs of every benchmark clip (under `data/work/`) and the cached model
answers of the DEV/TEST splits (listener answers, PANNs frame scores, grouping answers). Neither is in this
repository: the cached answers are in release v1.2.0 (the cluster caches), and the stage-5 outputs are rebuilt on
the cluster from the video clips, which are not redistributed. The clip names are in `dev_stems.txt`,
`dev2_stems.txt`, `test_stems.txt` and `test2_stems.txt` (the `tg_d*` clips go in `data/input/tagger_set/`). With the
clips and the release caches in place, `tagger_prep.py` builds the model inputs and outputs (the order of the steps
is in `slurm/run_best.sh`), and then:

```
python benchmark/gold/final_vs_baselines.py                 # Table 3: final system and baselines
python benchmark/gold/final_vs_baselines_extra.py           # F1 intervals, danger sounds
python benchmark/gold/parity_check.py SHIP8+MD3+WW5+SL      # configuration check
python benchmark/gold/inspector_trail_export.py \
   --expect DEV=29/58/15/6/7/2/2.056 --expect TEST=24/65/24/4/15/5/2.409
```

From a plain clone these scripts stop at the first missing clip or output.

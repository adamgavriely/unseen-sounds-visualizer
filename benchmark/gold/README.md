# benchmark/gold — labels, scorer and per-clip inputs

Human per-sound labels, the scorer, the scripts that produce the reported tables, the harness that replays the
final system on the cached model outputs of each clip, and the per-clip input builder (`clip_prep.py`) that
`main.py` uses for a new video. Short codes in variant names (`SHIP8+MD3+WW5+SL` = final system) are explained in
Appendix D of the report.
The research screens, the cached model answers of the DEV/TEST splits and the other study outputs are in release
v1.2.0.

| purpose | files |
|---|---|
| **Labels** | `annotations/gold_AG.json` (every per-sound label), `dev_stems.txt`, `test_stems.txt` (clip names of the two sets; the `tg_d*` lines are the second batch), `judge100.txt` (the rule for which development clips were scored), `audioset_slice.json` (the AudioSet-Strong clips) |
| **Scorer** | `score_per_sound.py` (matching rules, cost, clip subsets), `dev_candidates_check.py` (stage-4/5 replay helpers, bootstrap, DASM step), `holm_table.py` (Holm correction) |
| **Report tables** | `final_vs_baselines.py` → `final_vs_baselines.json` (final system and baselines); `final_vs_baselines_extra.py` → `final_vs_baselines_extra.json` (F1 intervals, danger sounds) |
| **Scoring harness** | `dev_harness.py` (variants, stage 4/5 through the pipeline code; first 49 development clips) and `merged_dev.py` (all 71); `test_harness_prep.py`, `test_harness.py` (first 59 test clips) and `final_test.py` → `final_test.json` (all 87), `parity_check.py`. The 49- and 59-clip harnesses ran on the first batch of clips; `merged_dev.py` and `final_test.py` give the reported 71/87 results |
| **Per-clip inputs** | `clip_prep.py`, `flexsed_run.py`, `dev_listener.py`, `test_listener.py`, `listener_variants.py`, `listener_afnext.py`, `dasm_rescue.py`, `listener_open_inventory.py`, `finelap_screen.py` (order below) |
| **Picture prompts** | `picture_templates.py` (picture templates and seeds; imported by `src/stage6_visual_augmentation/` and `comfyui_nodes/run_frozen.py`) |
| **Decision trail** | `inspector_trail_export.py`, `inspector_data.py` → `docs/decision_trail/` |
| **Model inputs and analyses cited in the report** | `depictable_vocab.json` (the 215 sound families), `flexsed_extra_queries.json` (extra FlexSED queries of the `FLEXSED_EXTRA` option), `ceiling_final/` (ceiling analysis of the final system: how many needed sounds any detector can find), `visible_weight_sweep.md` (cost of an on-screen picture), `holm_test_final_v33_test_bench.json` (September gate comparison on the test set, 60 clips), `logit_gate_gold.json` |
| **AudioSet side-by-side** (report Appendix E) | `fresh_side_by_side.py` → `fresh_side_by_side.json`; needs `benchmark/detector_round12.py` and `audioset_fresh.json` from release v1.2.0 |

## Per-clip inputs: what runs for a new clip

`main.py` calls `src/listener_prep.py`. For a clip with no
prepared answers it runs, each as its own process, on a one-clip split named `live_<clip>`:

1. `src/stage4_audio_event_detection/dasm_infer.py`, once per machine: DASM's MGA-CLAP text embeddings of the 215
   families → `data/work/dasm_text_queries.pt` (skipped when the file exists).
2. `clip_prep.py` steps `flexsed` (`flexsed_run.py`), `render` (`../run_protocol.py`, stages 1–6 with
   `config.use_scored()`), `wav16`, `beats`, `panns`, `stage4` and `stage5` with arm `B0r` (`dev_harness.py`),
   `lpool` (`test_listener.py`, `dev_listener.py`), `qwen` (Qwen3-Omni yes/no and variants: `dev_listener.py`,
   `test_listener.py`, `listener_variants.py`), `afn` (Audio Flamingo Next: `listener_afnext.py`) and `dasm`
   (`dev_candidates_check.dasm()` with `dasm_infer.py`).
3. `dasm_rescue.py pool` and `dasm_rescue.py listen`: DASM-only spans and both listeners' answers on them.
4. `listener_open_inventory.py`: Qwen V4 on the P1 cuts.
5. `finelap_screen.py split`: FineLAP frame scores, in the FineLAP Python environment (`FINELAP_PYTHON`).

The answers land in `benchmark/gold/live_<clip>_listener*.json` (ignored by git) and `data/work/` and are read by stage 4.

## Reproducing the numbers

The reported numbers are stored here: `final_vs_baselines.json`, `final_vs_baselines_extra.json` and
`../../docs/decision_trail/data_parity.json` (configuration check). Appendix C of the report gives the source file of
every number.

The scripts that made them read the stage-5 outputs of every benchmark clip (under `data/work/`) and the cached model
answers of the development and test sets (listener answers, PANNs frame scores, grouping answers). Both are in
release v1.2.0; the clips are not redistributed (sources: report Section 4). Clip names: `dev_stems.txt` (71) and
`test_stems.txt` (87); labels: `annotations/gold_AG.json`. With the clips in their input folders (the `tg_d*` clips
in `data/input/batch2/`; the other paths in `clip_prep.py` and `../run_protocol.py`) and the release caches in place,
`clip_prep.py` builds the model inputs and outputs (the step order is listed above), and then:

```
python benchmark/gold/final_vs_baselines.py                 # Table 3: final system and baselines
python benchmark/gold/final_vs_baselines_extra.py           # F1 intervals, danger sounds
python benchmark/gold/parity_check.py SHIP8+MD3+WW5+SL      # configuration check
python benchmark/gold/inspector_trail_export.py \
   --expect DEV=29/58/15/6/7/2/2.056 --expect TEST=24/65/23/4/15/4/2.414
```

From a plain clone these scripts stop at the first missing clip or output.

### Names in release v1.2.0

The caches and outputs in release v1.2.0 use the names of that release. Rename them (or read them under the new name)
as follows before running the scripts here:

| release v1.2.0 | this release |
|---|---|
| `benchmark/gold/tagger_prep.py` | `benchmark/gold/clip_prep.py` |
| `benchmark/tags.json` | `benchmark/scene_tags.json` |
| `benchmark/gold/dev2_stems.txt`, `test2_stems.txt` | the `tg_d*` lines of `dev_stems.txt`, `test_stems.txt` |
| `benchmark/gold/gen_screen.py` | `benchmark/gold/picture_templates.py` |
| `benchmark/gold/round13_dev.py`, `round13_dev.json` | `benchmark/gold/dev_harness.py`, `dev_harness.json` |
| `benchmark/gold/r13_test_prep.py` | `benchmark/gold/test_harness_prep.py` |
| `benchmark/gold/r13_test_final.py`, `.json`, `.md` | `benchmark/gold/test_harness.py`, `.json`, `.md` |
| `benchmark/gold/listener_p1v4.py` | `benchmark/gold/listener_open_inventory.py` |
| `benchmark/gold/{dev,test,dev2,test2}_listener_p1v4.json` | `benchmark/gold/{dev,test,dev2,test2}_listener_open_inventory.json` |
| `docs/inspector2/` | `docs/decision_trail/` |

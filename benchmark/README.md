# benchmark

Evaluation code and stored results. The reported per-sound results are made in `gold/`; the files at this level are
the detector and listener studies that set the final system's sound-detection rules, and the earlier model-judge
protocol. Result files name the final configuration `SHIP8+MD3+WW5+SL` (written D′ in some files); the short codes
are explained in Appendix D of the report.

| file | what it is | read or imported by |
|---|---|---|
| `run_protocol.py` | runs the system variants on the benchmark clips and the earlier model-judge protocol (stage 7) | `gold/tagger_prep.py`, `gold/round13_dev.py`, `gold/dev_candidates_check.py`, `comfyui_nodes/__init__.py` |
| `tags.json` | clip-level tags (source on screen or not) of the first benchmark | `run_protocol.py`, `slurm/sync_data.sh` |
| `listener_round.py`, `listener_round.json` | audio-listener study (Qwen3-Omni yes/no on candidate sounds): question, model and results | `gold/tagger_prep.py`, `gold/dev_listener.py`, `gold/test_listener.py`, `gold/listener_variants.py`, `gold/listener_afnext.py` |
| `audioset_detector_eval.py`, `audioset_detector_eval.json` | detector comparison (BEATs, PretrainedSED, FLAM) on AudioSet-Strong clips | `audioset_stage4_report.py` |
| `audioset_stage4_report.py`, `audioset_stage4_report.json` | the stage-4 rules replayed on AudioSet-Strong clips | the `detector_round*.py` scripts, `listener_round.py` |
| `detector_round2.py` … `detector_round10.py` and their `.json` | detector improvement rounds 2, 4, 5, 6, 8 and 10 (each imports the earlier ones) | each other, `gold/dev_candidates_check.py`; `detector_round6.json` holds the DASM bar used in `config.py` |
| `round5_eat_labels.csv`, `round6_wavcaps_ids.json`, `round10_paraphrases.json` | inputs of rounds 5, 6 and 10 | `detector_round5.py` (and `gold/tagens.py`), `detector_round6.py`, `detector_round10.py` |
| `detector_calib.json`, `psed_setting.json` | bars of the PretrainedSED detector variant | `src/stage4_audio_event_detection/psed_infer.py`, `audioset_detector_eval.py` |
| `flam_calibration.json` | bars of the FLAM detector variant | `src/stage4_audio_event_detection/flam_infer.py`, `audioset_detector_eval.py` |
| `eval_dcase_onset.json`, `eval_dcase_visibility.json` | onset-timing and on-screen-check results on DCASE clips (cited in Appendix C of the report; the scripts that made them are in release v1.2.0) | none |

- `gold/`: human per-sound labels, the per-sound scorer, the scoring harness and cached model answers; see
  [`gold/README.md`](gold/README.md).
- `results/`: outputs of the earlier model-judge protocol (`protocol_results.json`, `protocol_descriptions.json`),
  superseded by the per-sound scoring (report Appendix E).

The benchmark clips are not in the repository. Their source collections are listed in the "Data" section of the
report; the clip names are in `gold/*_stems.txt`.

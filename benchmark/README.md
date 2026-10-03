# benchmark

Evaluation code. The reported per-sound results are made in `gold/`; the files at this level are the earlier
model-judge protocol runner, the listener question it shares with the harness, and stored DCASE results. Result files
name the final configuration `SHIP8+MD3+WW5+SL` (written D′ in some files); the short codes are explained in
Appendix D of the report. The detector and listener study rounds (`detector_round*.py`, the AudioSet-Strong replays,
the listener study's own steps and results) are in release v1.2.0.

| file | what it is | read or imported by |
|---|---|---|
| `run_protocol.py` | runs the system variants on the benchmark clips (render phase) and the earlier model-judge protocol (stage 7) | `gold/tagger_prep.py`, `gold/round13_dev.py`, `gold/dev_candidates_check.py`, `comfyui_nodes/__init__.py` |
| `tags.json` | clip-level tags (source on screen or not) of the first benchmark | `run_protocol.py`, `slurm/sync_data.sh` |
| `listener_round.py` | the audio listener's model, yes/no question, 215-family vocabulary and AUROC helper (Qwen3-Omni) | `gold/tagger_prep.py`, `gold/dev_listener.py`, `gold/test_listener.py`, `gold/listener_variants.py`, `gold/listener_afnext.py` |
| `eval_dcase_onset.json`, `eval_dcase_visibility.json` | onset-timing and on-screen-check results on DCASE clips (cited in Appendix C of the report; the scripts that made them are in release v1.2.0) | none |

- `gold/`: human per-sound labels, the per-sound scorer, the scoring harness and the listener harness; see
  [`gold/README.md`](gold/README.md).

The benchmark clips are not in the repository. Their source collections are listed in the "Data" section of the
report; the clip names are in `gold/*_stems.txt`.

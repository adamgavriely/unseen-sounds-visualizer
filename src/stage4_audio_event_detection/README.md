# Stage 4 - Audio event detection

Finds non-speech sound events with start and end times over the AudioSet labels (`detect_events()` in `__init__.py`).
The final system uses BEATs (`beats_infer.py`, vendored model code in `beats/`, MIT) with FlexSED (`flexsed_infer.py`),
a PANNs clip veto, and the listener answers prepared by `src/listener_prep.py` (Qwen3-Omni, Audio Flamingo Next, DASM,
FineLAP). `dasm_infer.py` is the DASM frame scorer used by `src/listener_prep.py` and the scoring harness; run it once to compute DASM's text queries.
The detector variants that were tested and are not part of the final system (PretrainedSED, FLAM, SSLAM, a CLAP check)
are in release v1.2.0.

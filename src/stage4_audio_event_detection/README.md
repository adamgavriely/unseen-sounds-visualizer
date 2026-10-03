# Stage 4 - Audio event detection

Finds non-speech sound events with start and end times over the AudioSet labels (`detect_events()` in `__init__.py`).
The final system uses BEATs (`beats_infer.py`, vendored model code in `beats/`, MIT) with FlexSED (`flexsed_infer.py`),
a PANNs clip veto, and the listener answers prepared by `src/listener_prep.py` (Qwen3-Omni, Audio Flamingo Next, DASM,
FineLAP). `psed_infer.py`, `flam_infer.py` and `sslam_infer.py` are detector variants that were tested and are not
part of the final system; `clap_check.py` is an earlier CLAP second-opinion check that nothing imports now.

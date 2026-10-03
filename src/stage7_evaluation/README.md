# Stage 7 - Evaluation

Not part of the per-video pipeline; runs over the benchmark. `protocol.py` is the automatic protocol of the project
proposal (a vision model describes the output, a separate judge model scores it against a reference), driven by
`benchmark/run_protocol.py`; `independent_reference.py` builds that reference without the evaluated system's outputs.
`gating_accuracy()` in `__init__.py` scores the on-screen gate. The per-sound scorer used for the reported results is
`benchmark/gold/score_per_sound.py`.

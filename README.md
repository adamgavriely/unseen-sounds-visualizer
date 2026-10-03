# Visual Augmentation of Audio Semantics for Accessibility

MSc final project, Bar-Ilan University (Department of Computer Science), Adam Gavriely, 2026.

Subtitles carry speech, but a deaf or hard-of-hearing viewer still misses the other sounds of a video: a siren
behind the camera, a dog barking in the next room, glass breaking off screen. This system watches a video, detects
its non-speech, non-music sounds, decides for each one whether its source is already visible, and shows a generated
picture of the sound beside the video **only while an off-screen sound is heard**. No model is trained: the system
is a chain of open models joined by rules fixed on a development set.

The full description, the benchmark, the results and the limits are in the technical report,
[`docs/report/report.pdf`](docs/report/report.pdf).

## Results

![Hits and wrong pictures per system](docs/report/figures/results_bars.png)

Scored per sound against human labels. A hit is the right kind of sound with a picture starting between 0.5 s
before and 1 s after it; cost = (4 × missed + 2 × wrong pictures) / clips, lower is better. The baselines are those
of the project proposal.

| system | development set (71 clips) | test set (88 clips) |
|---|---|---|
| show nothing | cost 3.268 | cost 2.955 |
| direct audio-to-image (and audio captioning) | 34 / 58 found, 39 wrong, cost 2.451 | 26 / 65 found, 62 wrong, cost 3.182 |
| **final system** | **29 / 58 found, 15 wrong, cost 2.056** | **24 / 65 found, 24 wrong, cost 2.409** |

On the test set the final system is significantly cheaper than direct audio-to-image generation (−0.773 per clip,
95 % interval [−1.045, −0.500], p < 0.001) and than showing nothing (−0.545, [−1.091, −0.023], p = 0.044; p = 0.087
after a Holm correction over the five main tests). It finds 10 of the 13 danger sounds of the test set (sirens,
alarms, breaking glass, a crying baby). Every decision was made on the development set; the test set was only
scored. Limits are discussed in Section 9 of the report.

## How it works

1. **Audio and context**: extract the audio track, sample frames, detect visible objects and transcribe speech.
2. **Sound detection**: BEATs and FlexSED propose sound events; the Qwen3-Omni and Audio Flamingo Next listeners,
   DASM and FineLAP confirm or reject them.
3. **On-screen check**: Qwen3.8-27B looks at the frames and decides whether each sound's source is visible.
4. **Pictures**: Qwen-Image-2512 draws each off-screen sound; a vision model checks the picture and it is redrawn
   up to five times.
5. **Composition**: the pictures are shown beside the video while their sound plays.

## Installation

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

- Python 3.11 with PyTorch 2.5.1 and transformers 5.16.1; `requirements.txt` lists every package with the version
  used. FFmpeg must be on the path.
- One GPU with about 80 GB of memory (an NVIDIA H200 was used; two A100 80 GB cards also work). Qwen-Image alone
  needs about 57 GB in bf16. The models from Hugging Face are downloaded on first use; set `HF_TOKEN` for gated ones.
- The final system also needs these outside repositories and environments. Their locations are read from environment
  variables (see [`.env.example`](.env.example)); export them in the shell before running `main.py`.

| what | used for | variable |
|---|---|---|
| FlexSED repository | second sound detector | `FLEXSED_ROOT` (default `~/FlexSED`) |
| Transformer4SED repository ([github.com/cai525/Transformer4SED](https://github.com/cai525/Transformer4SED)) with the DASM weights and `third_parties/MGA-CLAP`, plus a local `bert-base-uncased` | DASM, the third listener | `T4SED_ROOT` (default `~/Transformer4SED`), `BERT_DIR` (default `~/bert-base-uncased`) |
| a second Python environment with transformers 4.51.3, for FineLAP ([huggingface.co/AndreasXi/FineLAP](https://huggingface.co/AndreasXi/FineLAP)) | FineLAP veto | `FINELAP_PYTHON` (default `~/venv_flap/bin/python`) |

  Only for variants that the final system does not use: PretrainedSED (`PSED_ROOT`), FLAM (the `openflam` package,
  in its own environment) and the EAT / SSLAM taggers.

## Usage

```
python main.py --input data/input/clip.mp4 --device cuda
```

This writes `data/output/clip_augmented.mp4` (the video with the picture panel) and the intermediate files under
`data/work/clip/`. The final system is the configuration `config.use_shipped()` in [`config.py`](config.py).
Before detection, `main.py` prepares the listener inputs of the clip on the spot (FlexSED, BEATs and PANNs scores,
the two audio-language listeners, DASM and FineLAP), with the same harness as the benchmark, once per clip.
Run time: about five minutes of GPU time (one NVIDIA H200) to prepare the listener inputs of a 15-s clip, plus the
picture step (about 20 s per picture try, up to five tries per picture). On a Slurm cluster, `slurm/run_best.sh`
runs the same system on a folder of clips (see [`slurm/README.md`](slurm/README.md)).

## Reproducing the reported numbers

The reported numbers are stored in the repository:
[`benchmark/gold/final_vs_baselines.json`](benchmark/gold/final_vs_baselines.json) (final system and baselines),
[`benchmark/gold/final_vs_baselines_extra.json`](benchmark/gold/final_vs_baselines_extra.json) (F1 intervals,
danger sounds) and [`docs/inspector2/data_parity.json`](docs/inspector2/data_parity.json) (configuration check).
Appendix C of the report gives the source file of every number.

Recomputing them from a clone is not possible as is. The scripts (for example
`python benchmark/gold/final_vs_baselines.py`) read the stage-5 outputs of every benchmark clip, and these are
built on the cluster from the video clips, which are not redistributed (their source collections are listed in
the "Data" section (Section 4) and the "Code and data availability" note of the report). The clip names are in
`benchmark/gold/dev_stems.txt`, `dev2_stems.txt`, `test_stems.txt` and `test2_stems.txt` (split rules in
`tagger_split.json` and `split.json`); the human labels are in `benchmark/gold/annotations/gold_AG.json`. With the
clips in place (`data/input/tagger_set/` for the `tg_d*` clips), the steps are those of `slurm/run_best.sh` and
[`benchmark/gold/README.md`](benchmark/gold/README.md).

## Repository layout

| path | content |
|---|---|
| `main.py`, `config.py` | entry point and configuration (`use_shipped()` = final system) |
| `src/` | the pipeline, one package per stage (`stage1_…` to `stage7_…`) |
| `benchmark/` | human labels, per-sound scorer, scoring harness and cached model answers |
| `tests/` | unit tests |
| `docs/report/` | the technical report (LaTeX sources, figures and PDF) |
| `docs/inspector2/` | decision trail: every decision of the final system on every clip, and its parity check |
| decision trail viewer | open `docs/inspector2/index.html` in a browser (no server needed); the clip videos are not in the repository, so the video box stays empty, but the trails, timelines and counts all work |
| `slurm/` | cluster job scripts for running the system on a folder of clips |
| `comfyui_nodes/` | optional ComfyUI interface ([`comfyui_nodes/README.md`](comfyui_nodes/README.md)) |
| `data/input/` | a small test clip |

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). The vendored BEATs code in `src/stage4_audio_event_detection/beats/` keeps its own licence (see the LICENSE file there). The video clips are not part of the repository.

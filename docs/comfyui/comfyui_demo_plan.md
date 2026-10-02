# ComfyUI demo — short plan (thesis artefact only, NOT a new pipeline)

Decided 2026-09-23: build the demo, do not migrate. `benchmark/run_protocol.py` stays the thing
that produces every number; the graph exists to be shown.

## Goal

One clip, seven boxes, live. The committee sees the pipeline as a graph, flips the gate off, and
watches the wrong pictures appear.

## Work, in order (about one day)

1. **Separate environment.** `conda create -n comfy`, install ComfyUI, start it with
   `python main.py --listen 127.0.0.1 --port 8188`. It will not share `msproj`'s torch pin.
2. **Seven thin nodes**, one per stage, in `comfyui_nodes/`. Each is a class with `INPUT_TYPES`,
   `RETURN_TYPES`, `FUNCTION` whose body calls the function we already have. No logic is rewritten.
   Custom types carry our own objects unchanged: `MEDIA`, `SCENE_CONTEXT`, `SEGMENTS`,
   `AUDIO_EVENTS`, `AUG_SPECS`, `VIDEO_OUT`.
3. **Two switches on the gate node**, because they are the demo: `gate_enabled` (on/off) and
   `panns_veto` (the threshold, 0 to 0.2). Everything else is a fixed widget.
4. **One saved workflow** (`demo.json`) and **one clip** chosen for the defence -- a street scene
   where the gate clearly matters, picked from the unseen category where precision is 0.85.
5. **Screenshot + a paragraph** for the system-architecture section of the thesis.

## Memory — the one thing that can stop it

Four unrelated heavy models (Qwen3.8-27B, FLUX, BEATs, FlexSED) in one server process is roughly
80 GB, where today they load and free stage by stage. Test this first, on one clip, before writing
any node beyond the first two. If it does not fit: the demo runs the gate and generation nodes only,
reading cached stage 1-4 outputs from a work directory, which still shows the switch that matters.

## Explicitly out of scope

  * running the benchmark through the graph;
  * anything that changes a number already in `docs/prereg_v4.md`;
  * custom nodes from the community ecosystem (no internet on compute nodes; we write our own seven).

## Not built, and why it stays that way

A full migration would make the graph the pipeline, and would then have to reproduce
`test_final_v30` exactly before any result could be trusted. That is a week of work and a real risk
to results that are already final. It is the right project for whoever continues this work, and the
wrong week to start it.

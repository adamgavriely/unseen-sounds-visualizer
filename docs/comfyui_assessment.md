# Would ComfyUI help this project? (research note, 2026-09-23 — nothing implemented)

Adam asked whether ComfyUI fits, how it works, and what it would cost in GPU terms. Short answer
first, then the evidence.

## Short answer

**ComfyUI cannot move a single number in the current results, and the reason is structural, not a
matter of quality.** The per-sound metric scores WHICH sound a picture names and WHEN it appears.
It never looks at the image. Every headline the project has — the viewer cost, the silence
crossover, the precision gain from the vetoes — would be bit-for-bit identical if the generator
produced perfect photographs or coloured rectangles.

It can move exactly one measured thing: the stage-7 LLM judge, which reads a description of the
picture. That is not the headline and is the weakest number in the project.

So the decision rule is: **ComfyUI is worth building only if our pictures actually fail to show
what they claim.** That has never been measured, so it is being measured now (see "The deciding
experiment" below). The threshold is declared before the result: adopt only at **> 20% depiction
failure on the labels the shipping configuration actually draws**.

## How it works

ComfyUI is a long-lived HTTP server, normally on port 8188, that executes a workflow given as JSON —
the same JSON the GUI exports as "Save (API Format)". Headless use is the normal use:

    POST /prompt          queue a workflow, returns a prompt id
    GET  /history/<id>    poll until it reports finished
    GET  /view?filename=  download the produced image
    ws://.../ws           the same, event-driven instead of polled

No browser is involved. A batch client is a few dozen lines: serialise the workflow, substitute the
prompt text per sound, queue, collect. Hundreds of jobs can be queued and harvested asynchronously,
which suits a SLURM array well.

## What it would cost us in GPU and operations

  * **VRAM.** FLUX.1-schnell is about 24 GB at FP16, ~12 GB at FP8, ~6.8 GB at Q4. Our current
    diffusers path already runs it, so this is not new — but a ComfyUI graph that adds ControlNet or
    IP-Adapter on top pushes 24–30 GB, which needs an A100/L40s/H200 rather than the 11 GB `generic`
    pool we have been using for the cheap passes.
  * **No internet on compute nodes.** This is the real cost. ComfyUI resolves custom nodes, models
    and Python dependencies at startup, and our compute nodes cannot reach the network. Every model,
    every custom node and its dependencies must be pre-staged under the home/workspace quota, with
    `ComfyUI/models/*` symlinked into the existing Hugging Face cache so nothing is stored twice.
  * **Environment conflict.** ComfyUI pins its own torch and wants xformers; our `msproj` env is
    pinned for BEATs, FlexSED and transformers 5. It needs a separate `comfy` conda env and a
    separate job script that starts `python main.py --listen 127.0.0.1 --port 8188` and talks to it
    over localhost inside the same allocation.
  * **Effort.** About a day to a working headless graph, most of it staging rather than code.

## What it would actually buy

  1. **Consistency between pictures.** IP-Adapter or a fixed seed/style graph would make the pictures
     of one clip look like one set rather than eight unrelated images. This is a real quality gap and
     the one a viewer would notice first.
  2. **Negative prompts and LoRAs** — cheap ways to stop the generator putting people in pictures of
     ambient sounds, which is a known failure of the current prompts.
  3. **Upscaling / refinement passes** for the demo and the defence.
  4. **Nothing on the metric.** Worth stating twice.

## The deciding experiment (running now)

`benchmark/gold/error_taxonomy.py --pictures` asks a VLM, of every picture the system showed,
"does this picture show {label}?" Two runs: the v4b4 configuration, which has 118 real FLUX images
on disk, and a fresh diffusion render of the SELECTED shipping cell, which is the honest target.

One caveat recorded in advance: for labels that are concrete nouns (Dog, Train, Bell) the question is
meaningful; for sound-verbs (Chink clink, Whack thwack, Steam) a VLM will say yes to almost any
image containing a hard object, so the two groups are reported separately and only the concrete-noun
number is used against the 20% threshold.

## Recommendation

Do not build it now. It is a defence-and-demo investment, not a thesis result, and the thesis result
is what is short. Revisit if the depiction check fails its threshold, or once the significance work
is finished and there is time to make the demo look like the system it is.

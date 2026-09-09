"""One time-boxed test: do 2-second generated clips convey the sound better than a still?

Two-frame cycles failed in both directions -- text2img gave the action but a different
subject each frame, img2img held the subject but never opened the beak, because
SDXL-Turbo ignores classifier-free guidance and so barely follows the prompt when
denoising from an init image. A text-to-video model is the right tool: it is conditioned
on the prompt AND keeps the subject stable across frames by construction.

AnimateDiff over SD1.5 is used rather than SVD. SVD is the obvious "animate my image"
choice but has NO text conditioning -- it adds drift and parallax, so it would pan across
a bird rather than open its beak, which is the one thing being tested.

This is a look-and-decide experiment, not a pipeline change. Every number in the report
was measured on stills, and Stage 7 describes a still with a VLM; adopting video would
mean a video-capable describer and re-running all 300 records.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

STYLE = ("simple flat illustration, clear simple shapes, few flat colours, plain white "
         "background, easy to understand at a glance, no text")

CLIPS = [
    ("bird_chirping",
     f"a bird opening and closing its beak, chirping, {STYLE}"),
    ("dog_barking",
     f"a dog barking, its mouth opening and closing, {STYLE}"),
    ("glass_shatter",
     f"a glass falling and shattering into pieces, {STYLE}"),
    ("footsteps",
     f"feet walking, one step after another, {STYLE}"),
]

NEGATIVE = ("photograph, photorealistic, text, letters, watermark, logo, scenery, "
            "background objects, clutter, blurry, distorted, extra limbs")


def main():
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_gif

    out = _ROOT / "data" / "output" / "short_video"
    out.mkdir(parents=True, exist_ok=True)

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        motion_adapter=adapter, torch_dtype=torch.float16)
    # the scheduler settings AnimateDiff expects; the defaults produce mush
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", clip_sample=False,
        timestep_spacing="linspace", steps_offset=1)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)

    for name, prompt in CLIPS:
        print(f"  {name}", flush=True)
        try:
            frames = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                          num_frames=16, guidance_scale=7.5,
                          num_inference_steps=25,
                          generator=torch.Generator("cuda").manual_seed(7)).frames[0]
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {e}", flush=True)
            continue
        export_to_gif(frames, str(out / f"{name}.gif"), fps=8)   # 16 frames @ 8fps = 2 s
        frames[0].save(out / f"{name}_first.png")
        frames[len(frames) // 2].save(out / f"{name}_mid.png")
        print(f"    -> {name}.gif ({len(frames)} frames)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

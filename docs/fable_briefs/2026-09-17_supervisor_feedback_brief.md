# Brief: the supervisor's feedback after the first meeting (2026-09-17)

CONTEXT. Adam Gavriely's MSc (Bar-Ilan): a training-free pipeline that shows a picture beside a video for ambient sounds whose source is not visible (the "gate"). State: system built (FFmpeg -> OWLv2 -> Whisper -> BEATs -> Qwen2.5-VL-7B -> FLUX.1-schnell -> side panel -> automatic judge), 274-clip benchmark labelled by Adam alone, headline tie with the blind baseline (3.17 vs 3.16 / 4), seven detector fixes and three automatic evaluation references failed pre-declared bars, thesis draft 19 pp, executive summary, LIMITATIONS, a meeting page with 35 v3 clips (clean + debug views). Submission had been planned for 18 Sept; the supervisor's feedback below implies a larger scope, and the new deadline is NOT yet known.

CLUSTER (BIU Slurm): L4 24 GB (8+2 cards, our workhorse), A100 40/80 GB (8), RTX Pro 6000 96 GB (8), RTX 6000 Ada 48 GB (2), L40S 48 GB (2), H200 141 GB (8), B200 192 GB (unusable: torch build lacks sm_100). Multi-GPU nodes exist (8 cards per node on the H200/A100/RTX Pro 6000 nodes). Env: transformers 5.16, torch 2.5.1+cu121, conda env "msproj"; ~70 GB free in the HF cache. MiniCPM-o could not load under transformers 5.16; Qwen2.5-Omni-7B does.

THE SUPERVISOR'S FIVE POINTS (verbatim intent):
1. "The models feel really old (BEATs, the Qwen version, FLUX.1). Use much more SOTA models, use the better GPUs (RTX 6000, A100 ...), don't limit to small GPUs, use more than one GPU at a time, and generate something better — perhaps in the future even video instead of an image."
2. "Use audio conditioning or video conditioning: take the existing video/audio and condition on it — like MiniMax-H3 + text + audio — and generate what is NOT seen in the video. Could be used as a baseline."
3. He sent https://x.com/xieenze_jr/status/2086379344134602793 (Enze Xie; MiniMax-H3 / Sol-Engine) and asked whether we can build working demonstrations as a ComfyUI workflow running locally on our GPUs.
4. "Compare audio-to-image as scoring: maybe generate an image from the audio alone, per second, and use this to compare / score what is missing."
5. "Get a gold set annotated (Adam or lab members) so we can compute scores on it."

FACTS FOUND TODAY ABOUT MINIMAX-H3 (web, 2026-08/09): open weights; omni-modal — takes text, images, video, audio in one context and generates video with native stereo audio, up to 2K and 15 s; it conditions on input audio rather than overwriting it; day-0 native support in ComfyUI (>= 0.30.0); full precision 123.6 GB, int8 variants ~42.5 GB, with offloading runs on 16-24 GB VRAM (slowly); "FL2VA" and "Ref2VA" weight variants exist (first-/last-frame-to-video-audio, reference-to-video-audio). Sol-H3 is an inference stack (5 s of video in 1.65 s on 8x B300). SANA-Video 2.0 is another fast video model from the same group.

WHAT THE THESIS ALREADY HAS THAT MATTERS HERE: the gate and its evaluation; per-clip saved decisions and pictures for 100 test clips; the finding that the detector, not the visibility decision, is the bottleneck; three failed automatic references (all at the visibility question); a cost axis (panel-on time); pre-registration discipline.

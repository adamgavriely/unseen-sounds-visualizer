# Cluster cleanup — run it yourself (the safety system blocks me from deleting)

Disk: 389 of 400 GB used (12 GB free). This frees about **160 GB**. Everything below is a downloaded model or an
environment for a tool we dropped; all of it can be downloaded again. Nothing the final system, the week plan or any
result needs is touched (kept: Qwen-Image-2512, Gemma-4-31B, Qwen3.8-27B, FLUX, OWLv2, BEATs, Whisper, all data and
results).

| what | size | why it can go |
|---|---|---|
| Qwen-Image-2.1 (+ its loose `hub/blobs` folder — checked: only 2.1 links into it) | 42 GB | dropped: no better, research-only licence |
| GLM-4.6V-Flash | 26 GB | failed picture checker; its sealed output is kept |
| Idefics3-8B | 22 GB | failed picture checker |
| Qwen2.5-VL-7B | 21 GB | replaced by Qwen3.8-27B |
| Mistral-7B | 19 GB | replaced judge |
| SAM 3 | 6.5 GB | dropped (lost to OWLv2) |
| Granite Speech | 6 GB | never adopted |
| conda envs `comfy`, `psed` | 16 GB | ComfyUI demo, PretrainedSED (dropped); package lists saved in `docs/envs/` |
| FLAM cache, conda package cache | 3 GB | FLAM dropped; cache |

Run this once, from your own terminal (make sure no job is running: `squeue -u adamg` is empty):

```bash
ssh adamg@slurm-login1.lnx.biu.ac.il 'cd ~/.cache/huggingface/hub && rm -rf models--Qwen--Qwen-Image-2.1 blobs models--HuggingFaceM4--Idefics3-8B-Llama3 models--Qwen--Qwen2.5-VL-7B-Instruct models--mistralai--Mistral-7B-Instruct-v0.3 models--zai-org--GLM-4.6V-Flash models--facebook--sam3 models--ibm-granite--granite-speech-4.1-2b && rm -rf ~/.cache/openflam && source ~/miniconda3/etc/profile.d/conda.sh && conda env remove -n comfy -y && conda env remove -n psed -y && conda clean -a -y && df -h ~ | tail -1'
```

Not deleted on purpose: `data/` (experiment records and renders — the evidence), the DCASE dataset (20 GB, could go
later if space is still needed).

"""Download every Hugging Face model the final system loads, so main.py can then run offline (HF_HUB_OFFLINE=1).

    python download_models.py            # into the default Hugging Face cache (HF_HOME)

Besides the models main.py names directly, this includes the ones the detectors load inside their own code
(CLAP for FlexSED, roberta-base for FineLAP, bert-base-uncased for DASM, a sentence model for the listener answers),
and applies two cache fixes needed with PyTorch 2.5.1 (see README, Installation):

- laion/clap-htsat-unfused publishes only pytorch_model.bin on its main revision, and transformers 5.x refuses to
  torch.load a .bin file with PyTorch < 2.6. The repository's own safetensors conversion (revision refs/pr/3, same
  weights) is placed in the main revision so FlexSED loads it.
- FineLAP's config names its text encoder "roberta-base", which now lives at FacebookAI/roberta-base; offline, the old
  name only resolves from a cache folder under that name, so the download is copied to it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = [
    # listeners and detectors (stages 4 and 5)
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "nvidia/audio-flamingo-next-hf",
    "CPF2/detect_any_sound",
    "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2",
    "AndreasXi/FineLAP",
    "laion/clap-htsat-unfused",                   # FlexSED's text encoder
    "FacebookAI/roberta-base",                    # FineLAP's text encoder
    "google-bert/bert-base-uncased",              # DASM's text encoder (BERT_DIR)
    "sentence-transformers/all-mpnet-base-v2",    # matches the listeners' free-text answers to labels
    # video understanding, speech, reasoning, pictures
    "google/owlv2-base-patch16-ensemble",
    "google/siglip-so400m-patch14-384",           # scene / setting classifier
    "Systran/faster-whisper-base",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen-Image-2512",
]


def snapshot_dir(repo: str) -> Path:
    return Path(snapshot_download(repo))


def main():
    for repo in MODELS:
        print(repo, "->", snapshot_download(repo), flush=True)

    # CLAP: the safetensors conversion next to the main revision's files
    main_dir = snapshot_dir("laion/clap-htsat-unfused")
    st = Path(hf_hub_download("laion/clap-htsat-unfused", "model.safetensors", revision="refs/pr/3"))
    target = main_dir / "model.safetensors"
    if not target.exists():
        shutil.copyfile(st.resolve(), target)
    (main_dir / "pytorch_model.bin").unlink(missing_ok=True)
    shutil.rmtree(main_dir.parent.parent / ".no_exist", ignore_errors=True)
    print("CLAP: model.safetensors placed in", main_dir, flush=True)

    # roberta-base under its old name, for FineLAP
    new = snapshot_dir("FacebookAI/roberta-base").parent.parent
    old = new.parent / "models--roberta-base"
    if not old.exists():
        shutil.copytree(new, old, symlinks=True)
    print("roberta-base: cache folder", old, flush=True)


if __name__ == "__main__":
    main()

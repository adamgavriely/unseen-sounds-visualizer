"""Print size, gating and licence of Hugging Face model repos (login node; token from the environment)."""
import os
import sys

from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN") or None)
for repo in sys.argv[1:]:
    try:
        info = api.model_info(repo, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings) / 1e9
        lic = info.card_data.get("license") if info.card_data else None
        print(f"{repo:34s} {total:6.1f} GB  gated={info.gated}  license={lic}")
    except Exception as e:
        print(f"{repo:34s} ERR {type(e).__name__}: {str(e)[:90]}")

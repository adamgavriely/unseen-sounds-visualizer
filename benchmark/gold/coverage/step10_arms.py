"""Step 10 (PREREG_step10_v4fix.md): arm SHIP8+MD3+WW5+SL+V4FIX = the frozen arm reading the fixed Qwen V4 caches
(~/MscProj_tg/scratch_cov/v4fix/<split>/). Run like step2_arms.py (dev from ~/MscProj_r13, dev2 from ~/MscProj_tg)."""
import os
import sys
from pathlib import Path

_ROOT = Path(os.getcwd())
sys.path.insert(0, str(_ROOT))
FIX = Path.home() / "MscProj_tg" / "scratch_cov" / "v4fix"
BASE_ARM, ARM = "SHIP8+MD3+WW5+SL", "SHIP8+MD3+WW5+SL+V4FIX"
KEYS = {"LISTENER_VCACHE": "{s}_listener_v.json", "RELABEL_P1V4": "{s}_listener_p1v4.json", "DASM_P4_CACHE": "{s}_listener_p4.json"}

try:
    from benchmark.gold import dev_harness as R
except ImportError:
    from benchmark.gold import round13_dev as R


def fixed(split):
    out = {k: str(FIX / split / v.format(s=split)) for k, v in KEYS.items()}
    for p in out.values():
        assert Path(p).exists(), p
    return out


part = sys.argv.pop(1)
if part == "dev":
    R.ARMS[ARM] = {**R.ARMS[BASE_ARM], **fixed("dev")}
    sys.argv[0] = "round13_dev"
    R.main()
else:
    R.ARMS[ARM] = dict(R.ARMS[BASE_ARM])
    os.environ["TG_ARMS"] = " ".join([ARM] + os.environ.get("TG_ARMS", "").split())
    try:
        from benchmark.gold import clip_prep as T
    except ImportError:
        from benchmark.gold import tagger_prep as T
    _conf = T.configure

    def configure(split):
        out = _conf(split)
        R.ARMS[ARM] = {**R.ARMS[ARM], **fixed(split)}      # after the per-split remap of the DEV file names
        return out

    T.configure = configure
    sys.argv[0] = "tagger_prep"
    T.main()

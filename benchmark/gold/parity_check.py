"""Parity check (1 Oct): the shipped config (config.use_shipped) must equal the scored base arm on every non-cache key.
Exit code 1 and a list when they differ.      python benchmark/gold/parity_check.py [arm]   (default SHIP8+MD3)"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config

ARM = sys.argv[1] if len(sys.argv) > 1 else "SHIP8+MD3"
SKIP = ("CACHE", "DIR", "_W", "PATH", "RELABEL_P1V4", "DASM_LOCAL_SCENE")      # per-split file locations, not behaviour
config.use_shipped()
from benchmark.gold import round13_dev as R
bad = []
for k, v in sorted(R.arm_cfg(ARM).items()):
    if any(s in k for s in SKIP):
        continue
    c = getattr(config, k, "<absent>")
    if c != v and not (isinstance(v, (int, float)) and isinstance(c, (int, float)) and abs(c - v) < 1e-9):
        bad.append(f"{k}: arm={v!r} shipped={c!r}")
print(f"parity {ARM}: {'OK' if not bad else 'MISMATCH'}")
for b in bad:
    print("  ", b)
sys.exit(1 if bad else 0)

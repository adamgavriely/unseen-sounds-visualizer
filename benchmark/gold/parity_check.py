"""Parity check: config.use_shipped() must equal the scored arm on every non-cache key.
python benchmark/gold/parity_check.py [arm] (default SHIP8+MD3+WW5+SL); exit 1 and a list when they differ."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config

ARM = sys.argv[1] if len(sys.argv) > 1 else "SHIP8+MD3+WW5+SL"
SKIP = ("CACHE", "DIR", "_W", "PATH", "RELABEL_P1V4", "DASM_LOCAL_SCENE")      # per-split file locations, not behaviour
V14 = ("VISIBILITY_RULE", "FLASH_RULE", "PICTURE_BAN", "HOLD_FLEXSED", "REPEAT_LOCK", "UNVERIFIABLE_BAN", "GATE_DOUBT_DASM",
       "COONSET_CONTEST", "WEAK_NO_RISE", "BEATS_NO_RISE")   # v1.4 + v1.5 + v1.6 additions on top of the scored variant (README)
config.use_shipped()
from benchmark.gold import dev_harness as R
bad = []
for k, v in sorted(R.arm_cfg(ARM).items()):
    if any(s in k for s in SKIP) or k in V14:
        continue
    c = getattr(config, k, "<absent>")
    if c != v and not (isinstance(v, (int, float)) and isinstance(c, (int, float)) and abs(c - v) < 1e-9):
        bad.append(f"{k}: arm={v!r} shipped={c!r}")
print(f"parity {ARM}: {'OK' if not bad else 'MISMATCH'} (v1.4 to v1.7 additions not compared: {', '.join(V14)})")
for b in bad:
    print("  ", b)
sys.exit(1 if bad else 0)

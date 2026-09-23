"""Start ComfyUI with the project's pipeline available, without upgrading the project's torch.

ComfyUI's mandatory `comfy_kitchen` registers custom torch operators whose signatures use PEP-585
builtin generics (`stride: list[int]`). `torch.library.infer_schema` only learned those in a later
release than the one this project is validated on, so it raises

    ValueError: Parameter stride has unsupported type list[int]

The project's torch is NOT upgraded to fix this -- every number in docs/prereg_v4.md was produced on
it, and the demo must not be the reason it changes. Instead the schema inference is taught the
builtin spellings, which are the same types under different names: `list[int]` IS `typing.List[int]`.
Nothing about how the operators execute is touched, and no project code is affected -- the patch
lives in the launcher, not in the pipeline.

    python comfyui_nodes/comfy_start.py [--port 8188] [--cpu]
"""
from __future__ import annotations

import os
import runpy
import sys
import typing
from pathlib import Path

_ROOT = Path(os.environ.get("MSCPROJ_ROOT", Path(__file__).resolve().parent.parent))
COMFY = Path(os.environ.get("COMFYUI_ROOT", Path.home() / "ComfyUI"))


def teach_torch_builtin_generics() -> str:
    """Make infer_schema accept list[int] wherever it already accepts typing.List[int]."""
    try:
        from torch._library import infer_schema as _is
    except Exception as e:                                  # torch too old to have it at all
        return f"skipped ({type(e).__name__})"
    table = getattr(_is, "SUPPORTED_PARAM_TYPES", None)
    if not isinstance(table, dict):
        return "skipped (no SUPPORTED_PARAM_TYPES)"
    added = 0
    for origin, builtin in ((typing.List, list), (typing.Tuple, tuple), (typing.Sequence, list)):
        for arg in (int, float, bool, str):
            for typ in (origin[arg], typing.Optional[origin[arg]]):
                if typ not in table:
                    continue
                # the builtin spelling of the same type, and its optional form
                try:
                    b = builtin[arg]
                except TypeError:
                    continue
                for cand in (b, typing.Optional[b], b | None):   # `list[int] | None` is a UnionType
                    try:
                        if cand not in table:
                            table[cand] = table[typ]
                            added += 1
                    except TypeError:
                        pass
    return f"added {added} builtin-generic spellings"


def main():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    deps = Path.home() / "comfy_deps"
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    print("[comfy_start] torch patch:", teach_torch_builtin_generics(), flush=True)
    os.environ.setdefault("MSCPROJ_ROOT", str(_ROOT))
    if not COMFY.exists():
        raise SystemExit(f"ComfyUI not found at {COMFY}; set COMFYUI_ROOT")
    os.chdir(COMFY)
    sys.path.insert(0, str(COMFY))
    sys.argv = [str(COMFY / "main.py")] + sys.argv[1:]
    runpy.run_path(str(COMFY / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

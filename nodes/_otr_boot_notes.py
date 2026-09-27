"""Boot-time notes that explain another pack's alarming-but-expected log lines.

Dependency-free, so `__init__.py` can call it at load and a test can call it
directly.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

#: The motion-module file suffixes AnimateDiff-Evolved lists (`.ckpt`,
#: `.safetensors`, `.pt`, `.pth`, `.bin`).
_MOTION_SUFFIXES = (".ckpt", ".safetensors", ".pt", ".pth", ".bin")


def _has_motion_module(folder: str) -> bool:
    try:
        return any(name.lower().endswith(_MOTION_SUFFIXES)
                   for name in os.listdir(folder))
    except OSError:
        return False


def ade_motion_module_note(custom_nodes_roots: Iterable[str],
                           models_dir: str) -> Optional[str]:
    """The note to print when AnimateDiff-Evolved is installed and has no
    motion module yet, else None.

    On a fresh install ADE logs, in red, "No motion models found. Please
    download one ..." at every boot until one exists. OTR fetches the module
    itself on the first AnimateDiff run, so the line is expected -- but a new
    user cannot know that (measured on a fresh 4060 portable, 2026-09-27).

    ORDER-INDEPENDENT on purpose: ComfyUI loads custom nodes in `os.listdir`
    order, so ADE may not have registered its `animatediff_models` category
    yet. This looks at the folders ADE reads (its own `models/` and
    `<models>/animatediff_models`) directly instead.
    """
    ade_dirs = []
    for root in custom_nodes_roots:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            lowered = name.lower()
            if "animatediff-evolved" in lowered and not lowered.endswith(".disabled"):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    ade_dirs.append(path)
    if not ade_dirs:
        return None
    folders = [os.path.join(d, "models") for d in ade_dirs]
    folders.append(os.path.join(models_dir, "animatediff_models"))
    if any(_has_motion_module(f) for f in folders):
        return None
    return ("[OldTimeRadio] AnimateDiff-Evolved reports 'No motion models found' "
            "until the first AnimateDiff run. That is expected: OTR downloads "
            "the motion module that run's lane needs, on that run.")

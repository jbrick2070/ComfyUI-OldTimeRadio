"""One OldTimeRadio pack per ComfyUI process.

ComfyUI imports every directory under custom_nodes. It skips names that
end with ``.disabled`` and ``__pycache__``; a ``.bak`` suffix is still a
live pack. Two copies (git clone ``ComfyUI-OldTimeRadio`` plus Manager
``comfyui-old-time-radio``, or a leftover backup folder) each ran
``__init__.py`` and each decorated ``GET /otr/latest_ledger``. aiohttp
then raised ``RuntimeError: method HEAD is already registered`` during
``add_routes`` and the server never came up.

``sys.modules`` is shared across those two different package names, so a
sentinel there is the one claim both copies can see.

A first-time registry install and a later Manager update are ONE folder.
``claim`` must return None in those cases so nodes and the HTTP route
load exactly as they did before this guard existed.
"""
from __future__ import annotations

import os
import sys
import types

PACK_GUARD = "_comfyui_old_time_radio_singleton"
PRESTARTUP_GUARD = "_comfyui_old_time_radio_prestartup_singleton"


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def claim(guard: str, loaded_from: str):
    """Return the first *other* folder's path if this copy is extra.

    None means this copy should load: first folder in the process, or
    the same folder imported again (Manager reload / same-path).
    """
    here = _norm(loaded_from)
    existing = sys.modules.get(guard)
    if existing is not None:
        first = getattr(existing, "loaded_from", "") or ""
        if first and _norm(first) == here:
            return None
        return first or "?"
    marker = types.ModuleType(guard)
    marker.loaded_from = loaded_from
    sys.modules[guard] = marker
    return None

"""ComfyUI entrypoint for the OTR upstream story lab v2."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Pytest imports this file as top-level ``__init__`` because the
    # custom-node folder name contains hyphens. ComfyUI uses the relative
    # import above.
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

#!/usr/bin/env python
"""Refresh the OpenRouter catalog cache used by the writer's slot-slug pickers.

The four-dropdown router (2026-06-01) populates openrouter_slot_a_model /
openrouter_slot_b_model from an on-disk cache so INPUT_TYPES never touches the
network. This script is the EXPLICIT operator action that fetches the live
OpenRouter /models list and atomically writes
``<repo>/models/openrouter_models.json``. It is the ONLY supported way to
refresh discovery -- nothing fetches at import or inside INPUT_TYPES.

A failed / offline fetch keeps the existing good cache and never raises (the
backend guarantees this); the script just reports it and exits non-zero so a
scheduled run can alert.

Usage (Windows, repo root, venv python):
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\otr_openrouter_refresh.py

OPENROUTER_API_KEY is read from the environment if present (recommended), but
the OpenRouter /models list is public, so a refresh works without a key too.
"""
from __future__ import annotations

import os
import sys

# Import the backend module standalone -- it pulls in no ComfyUI / torch deps
# at module scope, so a plain `import` works from a bare venv.
_NODES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes"
)
if _NODES not in sys.path:
    sys.path.insert(0, _NODES)

import _otr_openrouter_backend as orb  # noqa: E402


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "[otr-refresh] note: OPENROUTER_API_KEY is not set. The public "
            "/models list is still fetchable; set the key for account-scoped "
            "data and to enable remote runs."
        )
    catalog = orb.refresh_catalog_cache()
    meta = orb.catalog_meta()
    print(
        f"[otr-refresh] source={catalog.get('source')} "
        f"count={catalog.get('count')} "
        f"fetched_at={catalog.get('fetched_at')} "
        f"staleness={meta.get('staleness')}"
    )
    print(f"[otr-refresh] cache file: {orb._catalog_cache_path()}")
    if catalog.get("source") != "live":
        print(
            "[otr-refresh] WARNING: fetch did not return a live list; the "
            "previous cache (if any) was kept. Check the network / base URL."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

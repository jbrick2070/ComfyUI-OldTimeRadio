"""queue_smoke.py -- one-shot smoke queue for v2.0-beta verification.

POSTs `workflows/otr_scifi_16gb_full.json` to ComfyUI /prompt with the
30-word ultra-smoke configuration:

    target_words   = 30
    num_characters = 2
    target_length  = "30 words (smoke, 1 act)"

Returns the prompt_id on stdout. Polling lives in `smoke_watcher.py`.

Replaces the old supersoaker.py-based path. Built on `scripts/otr_api.py`,
which patches widgets BY NAME using live /object_info schemas (no fragile
positional WV_* indices).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from otr_api import (  # noqa: E402
    COMFYUI_URL,
    fetch_schemas,
    load_workflow,
    patch_widget_by_name,
    submit_prompt,
    workflow_to_api_prompt,
)

WORKFLOW_PATH = os.path.join(
    os.path.dirname(_HERE),
    "workflows",
    "otr_scifi_16gb_full.json",
)


def main() -> int:
    print(f"COMFYUI_URL = {COMFYUI_URL}", flush=True)
    print(f"workflow    = {WORKFLOW_PATH}", flush=True)

    print("Fetching /object_info schemas...", flush=True)
    schemas = fetch_schemas()

    print("Loading + patching workflow...", flush=True)
    wf = load_workflow(WORKFLOW_PATH)

    # Patch by NAME -- robust against future widget reorders.
    patch_widget_by_name(wf, 1, "target_words", 30, schemas)
    patch_widget_by_name(wf, 1, "num_characters", 2, schemas)
    patch_widget_by_name(wf, 1, "target_length", "30 words (smoke, 1 act)", schemas)

    api = workflow_to_api_prompt(wf, schemas)
    prompt_id = submit_prompt(api)
    print(f"QUEUED prompt_id={prompt_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

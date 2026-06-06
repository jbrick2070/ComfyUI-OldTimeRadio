"""_otr_chatterbox_smoke.py -- queue a 30-word full workflow with engine=chatterbox.

Scratch driver for the live chatterbox bug-hunt: the same 30/2/1 ultra-smoke
config as queue_smoke.py but flips node 81 (OTR_BatchCharacterVoices) engine ->
chatterbox (node 80 CastLock already voice_bank=default + auto_registry). Prints
the prompt_id on stdout. UTF-8, ASCII-only.
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
    os.path.dirname(_HERE), "workflows", "otr_scifi_16gb_full.json"
)


def main() -> int:
    print(f"COMFYUI_URL = {COMFYUI_URL}", flush=True)
    schemas = fetch_schemas()
    wf = load_workflow(WORKFLOW_PATH)
    patch_widget_by_name(wf, 1, "target_words", 30, schemas)
    patch_widget_by_name(wf, 1, "num_characters", 2, schemas)
    patch_widget_by_name(wf, 1, "act_count", "1", schemas)
    # the change under test: character voices on chatterbox (node 81), with the
    # cast-lock bank already at default + auto_registry (node 80).
    patch_widget_by_name(wf, 81, "engine", "chatterbox", schemas)
    patch_widget_by_name(wf, 80, "voice_bank", "default", schemas)
    api = workflow_to_api_prompt(wf, schemas)
    pid = submit_prompt(api)
    print(f"QUEUED prompt_id={pid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

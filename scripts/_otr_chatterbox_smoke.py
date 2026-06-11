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
    apply_profile_to_workflow,
    fetch_schemas,
    load_workflow,
    patch_creative,
    patch_widget_by_name,
    submit_prompt,
    workflow_to_api_prompt,
)

WORKFLOW_PATH = os.path.join(
    os.path.dirname(_HERE), "workflows", "otr_scifi_16gb_full.json"
)


def _profile_with_char_engine(engine: str) -> dict:
    """16gb_full with the char-voice slot overridden -- managed engine widgets
    ride the ONE applier (GATE B S2), never a hand patch."""
    import copy
    repo_root = os.path.dirname(_HERE)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from nodes._otr_shared.capability_profiles import load_profile
    profile = copy.deepcopy(load_profile("16gb_full"))
    profile["slot_overrides"]["char_voice_engine"] = engine
    return profile


def main() -> int:
    print(f"COMFYUI_URL = {COMFYUI_URL}", flush=True)
    schemas = fetch_schemas()
    wf = load_workflow(WORKFLOW_PATH)
    patch_creative(wf, 1, "target_words", 30, schemas)
    patch_creative(wf, 1, "num_characters", 2, schemas)
    patch_creative(wf, 1, "act_count", "1", schemas)
    # the change under test: character voices on chatterbox via the applier;
    # the cast-lock bank knob is unmanaged policy (node 80) and stays direct.
    wf = apply_profile_to_workflow(wf, _profile_with_char_engine("chatterbox"),
                                   schemas)
    patch_widget_by_name(wf, 80, "voice_bank", "default", schemas)
    api = workflow_to_api_prompt(wf, schemas)
    pid = submit_prompt(api)
    print(f"QUEUED prompt_id={pid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

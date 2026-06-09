"""queue_smoke.py -- one-shot smoke queue for v2.0-alpha bug-hunt loop.

POSTs `workflows/otr_scifi_16gb_full.json` to ComfyUI /prompt with the
30-word ultra-smoke configuration:

    target_words   = 30
    num_characters = 2
    act_count      = 1

Returns the prompt_id on stdout. Polling lives in `smoke_watcher.py`.

Built on `scripts/otr_api.py`, which patches widgets BY NAME using live
/object_info schemas (no fragile positional WV_* indices).

History note (2026-05-17 H bug-hunt iter 1): the original
`target_length = "30 words (smoke, 1 act)"` widget was retired in favor
of the numeric `act_count` widget. Smoke config now uses act_count=1.
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

    # Remote-LLM slot pickers: set to the first valid option from the live
    # schema (which is either the sentinel when disabled, or the catalog default
    # when OpenRouter / Comfy Credits are configured in this env).
    #
    # NOTE: /object_info schemas are keyed by node TYPE (e.g.
    # "OTR_LedgerScriptWriter"), NOT by node id. Resolve the node's type from
    # the workflow first, then index the schema by type -- mirroring what
    # patch_widget_by_name already does. (Prior revision keyed by str(node_id),
    # so the lookup always missed, _first_choice returned None, the patches
    # were skipped, and stale saved slugs reached /prompt.)
    def _node_type(node_id: int) -> str | None:
        for node in wf.get("nodes", []):
            if node.get("id") == node_id:
                return node.get("type")
        return None

    def _first_choice(node_id: int, widget_name: str) -> str | None:
        node_type = _node_type(node_id)
        node_schema = (schemas or {}).get(node_type) or {}
        inp = node_schema.get("input", {})
        for section in ("optional", "required"):
            spec = inp.get(section, {}).get(widget_name)
            if spec and isinstance(spec, (list, tuple)) and isinstance(spec[0], list):
                return spec[0][0]
        return None

    for _slot in (
        "openrouter_slot_a_model",
        "openrouter_slot_b_model",
        "comfy_slot_a_model",
        "comfy_slot_b_model",
    ):
        _val = _first_choice(1, _slot)
        if _val is not None:
            patch_widget_by_name(wf, 1, _slot, _val, schemas)

    # Patch by NAME -- robust against future widget reorders.
    patch_widget_by_name(wf, 1, "target_words", 30, schemas)
    patch_widget_by_name(wf, 1, "num_characters", 2, schemas)
    patch_widget_by_name(wf, 1, "act_count", "1", schemas)

    api = workflow_to_api_prompt(wf, schemas)
    prompt_id = submit_prompt(api)
    print(f"QUEUED prompt_id={prompt_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

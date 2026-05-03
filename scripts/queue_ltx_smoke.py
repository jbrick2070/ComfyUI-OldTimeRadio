"""queue_ltx_smoke.py -- queue the LTX-only fast smoke against a running
ComfyUI on http://127.0.0.1:8000 (override via COMFYUI_URL env var).

Bypasses the full LLM/audio/FLUX/HuMo cascade by feeding pre-baked
artifacts from a prior episode (ledger.json + procgen mp4 + HuMo clips
already on disk). Wallclock target: ~10 min.

Usage:
    python scripts\\queue_ltx_smoke.py
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
    submit_prompt,
    workflow_to_api_prompt,
)

WORKFLOW_PATH = os.path.join(
    os.path.dirname(_HERE),
    'workflows',
    'otr_ltx_smoke.json',
)


def main() -> int:
    print(f'COMFYUI_URL = {COMFYUI_URL}', flush=True)
    print(f'workflow    = {WORKFLOW_PATH}', flush=True)
    print('Fetching /object_info schemas...', flush=True)
    schemas = fetch_schemas()
    print('Loading workflow (no widget patches needed -- pre-baked)...', flush=True)
    wf = load_workflow(WORKFLOW_PATH)
    api = workflow_to_api_prompt(wf, schemas)
    prompt_id = submit_prompt(api)
    print(f'QUEUED prompt_id={prompt_id}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

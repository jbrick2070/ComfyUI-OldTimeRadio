"""soak_bug027_028.py -- one-shot acceptance soak for BUG-LOCAL-027 + BUG-LOCAL-028.

Submits ``workflows/otr_scifi_16gb_full.json`` to ComfyUI /prompt with the
exact widget configuration that REPRODUCED the BUG-027 dialogue wipe in
the 2026-05-03 EVENING soak (target_words=110, target_length="short (3 acts)",
num_characters=2, style="noir mystery", creativity="maximum chaos",
arc_enhancer ON, self_critique ON, open_close ON, optimization_profile
"Pro (Ultra Quality)", episode_title="Headless soak BUG-027/028"). After
the queue lands, prints the prompt_id and the acceptance signatures to
look for in ``otr_runtime.log``.

USAGE:
    1. Make sure ComfyUI Desktop is running on http://127.0.0.1:8000.
       Launch via ``scripts/run_comfyui.cmd`` if not.
    2. Verify the f1467a2 / 475c477 code is loaded (custom-node .py files
       are cached in sys.modules; a mid-process commit does NOT hot-reload).
       The script auto-checks /object_info for ``OTR_SaveToEpisodeWorkspace``
       (registered in commit f1467a2) and ABORTS if it's missing.
    3. Run::

           C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe ^
               C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio\\scripts\\soak_bug027_028.py

    4. Tail ``otr_runtime.log`` and grep for the acceptance signatures
       printed at the end of this script's output.

EXIT CODES:
    0 = queue submitted (does NOT wait for completion)
    1 = ComfyUI not reachable
    2 = ComfyUI loaded but pre-fix code (missing OTR_SaveToEpisodeWorkspace)
    3 = workflow patch / submit failed

BUG-LOCAL-027 + BUG-LOCAL-028 fix verification (2026-05-03 EVENING).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

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

# Widget config that reproduces BUG-027 in the 2026-05-03 EVENING run.
# Picked because every bug we just fixed needs to fire on this shape:
#   * target_length="short (3 acts)" -- not ULTRA_SMOKE, so BUG-005's
#     CHARACTER:/SCENE: enforcement does NOT short-circuit. The full
#     critique-and-revise loop runs.
#   * creativity="maximum chaos" -- temp=0.95, top_p=0.99. The high temp
#     is what made the revision LLM drop dialogue in the first place.
#   * num_characters=2 + ANNOUNCER -- gives 3 cast members, enough for
#     the per-character preservation gate to fire AND for the new total-
#     collapse gate to fire if the revision wipes them all.
WIDGET_PATCHES = {
    "episode_title":         "Headless Soak BUG-027/028",
    "target_words":          110,
    "num_characters":        2,
    "target_length":         "short (3 acts)",
    "style":                 "noir mystery",
    "creativity":            "maximum chaos",
    "arc_enhancer":          True,
    "self_critique":         True,
    "open_close":            True,
    "optimization_profile":  "Pro (Ultra Quality)",
    "include_act_breaks":    True,
}

NEW_NODE = "OTR_SaveToEpisodeWorkspace"

ACCEPTANCE_SIGNATURES_BUG_027 = [
    # If parser regex fix landed, the draft dict will be NON-EMPTY.
    "CRITIQUE: Character line counts - draft={'",
    # If revision wipes dialogue, the new total-collapse gate will REJECT
    # and fall back to draft. This line proves the gate fires on regression.
    "CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed",
    # If the gate accepts a healthy revision, this is the success line.
    "CRITIQUE: Revised script accepted",
    # Downstream confirmation: BatchBark sees dialogue.
    "[BatchBark] Found",  # follow-up word "0" = bug, ">=1" = fix held
]

ACCEPTANCE_SIGNATURES_BUG_028 = [
    # New save node fired and routed to per-episode workspace.
    "[SaveToEpisodeWorkspace] saving to per-episode dir:",
    # Radio bookend now lands per-episode (not in _legacy_stills).
    "radio still rendered + saved",  # follow-up path = otr/episodes/<ep>/stills/
    # HuMo's cast-still binding finds the new per-episode stills.
    "[BatchHumoRender] cast-still binding:",  # follow-up = "N/M cast members matched"
]


def _check_comfyui_loaded() -> int:
    """Verify ComfyUI is up AND has the f1467a2 code loaded.

    Returns 0 if all good. Otherwise prints a diagnostic and returns
    a nonzero exit code suitable for ``sys.exit``.
    """
    info_url = f"{COMFYUI_URL.rstrip('/')}/object_info"
    try:
        with urllib.request.urlopen(info_url, timeout=15) as resp:
            info = json.load(resp)
    except urllib.error.URLError as e:
        print(f"FAIL: ComfyUI not reachable at {info_url}: {e}", flush=True)
        print("HINT: launch via scripts/run_comfyui.cmd, wait for 'Server started',")
        print("      then re-run this script.")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: fetching {info_url}: {type(e).__name__}: {e}", flush=True)
        return 1

    if NEW_NODE not in info:
        print(f"FAIL: ComfyUI is up, but {NEW_NODE!r} is NOT registered.", flush=True)
        print("      ComfyUI is running pre-fix code. Custom-node .py files are")
        print("      cached in sys.modules; mid-process commits do not hot-reload.")
        print("HINT: quit ComfyUI Desktop completely (close window AND kill any")
        print("      lingering python.exe), re-launch via scripts/run_comfyui.cmd,")
        print("      then re-run this script.")
        return 2

    print(f"OK: ComfyUI on :8000 has {NEW_NODE!r} registered ({len(info)} total node types).")
    return 0


def _print_acceptance_signatures() -> None:
    print()
    print("=" * 72)
    print("BUG-LOCAL-027 acceptance signatures (grep otr_runtime.log for):")
    print("=" * 72)
    for sig in ACCEPTANCE_SIGNATURES_BUG_027:
        print(f"  {sig}")
    print()
    print("=" * 72)
    print("BUG-LOCAL-028 acceptance signatures (grep otr_runtime.log for):")
    print("=" * 72)
    for sig in ACCEPTANCE_SIGNATURES_BUG_028:
        print(f"  {sig}")
    print()
    print("=" * 72)
    print("After the run completes, check the per-episode workspace:")
    print("  C:\\Users\\jeffr\\Documents\\ComfyUI\\output\\otr\\episodes\\<ep>\\")
    print("Should contain:")
    print("  * stills/full_env_00001_.png .. full_env_NNNNN_.png  (BUG-028 fix)")
    print("  * stills/radio_bookend_<ep>.png                       (BUG-028 fix)")
    print("  * audio/<ep>.mp4 (master mix)                         (existing)")
    print("  * videos/l001.mp4 .. lNNN.mp4 (HuMo per-line clips)   (existing)")
    print("  * composited/<ep>.mp4 (final composite)                (existing)")
    print()


def main() -> int:
    print(f"COMFYUI_URL = {COMFYUI_URL}", flush=True)
    print(f"workflow    = {WORKFLOW_PATH}", flush=True)
    print()

    rc = _check_comfyui_loaded()
    if rc != 0:
        return rc

    print()
    print("Fetching /object_info schemas...", flush=True)
    schemas = fetch_schemas()

    print("Loading + patching workflow...", flush=True)
    wf = load_workflow(WORKFLOW_PATH)

    SCRIPT_WRITER_NODE_ID = 1  # OTR_LLMScriptWriter is node id 1 in this workflow
    for widget_name, value in WIDGET_PATCHES.items():
        try:
            patch_widget_by_name(wf, SCRIPT_WRITER_NODE_ID, widget_name, value, schemas)
            print(f"  patched: {widget_name} = {value!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: patch_widget_by_name({widget_name}={value!r}) failed: {e}",
                  flush=True)

    print()
    print("Converting to API prompt + submitting...", flush=True)
    try:
        api = workflow_to_api_prompt(wf, schemas)
        prompt_id = submit_prompt(api)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: submit failed: {type(e).__name__}: {e}", flush=True)
        return 3

    print(f"QUEUED prompt_id={prompt_id}", flush=True)
    _print_acceptance_signatures()
    return 0


if __name__ == "__main__":
    sys.exit(main())

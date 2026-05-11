"""scripts/fix_reviewer_node_layout.py

One-shot follow-up to wire_reviewer_into_workflow.py. Fixes three
visual / rendering issues that surfaced when Jeffrey loaded the
workflow in ComfyUI Desktop:

  1. The 4 passthrough inputs (script_text, script_json, news_used,
     estimated_minutes) were rendering widget editor boxes even
     though they had links. Strip the `widget` hint so ComfyUI
     renders them as pure link-bound sockets (matches the new
     `forceInput: True` declaration in the node's INPUT_TYPES).
  2. widgets_values carried 5 entries (one per input + model_id).
     With the 4 inputs forced to link-only, only model_id should
     have a widget. Reduce widgets_values to a single entry.
  3. Reviewer was placed at the midpoint between writer and critic,
     which visually OVERLAPPED the writer node. Move it ABOVE the
     writer (negative Y) so the canvas layout is clean.

Idempotent: if the reviewer node is already in the fixed shape,
exits without changes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflows" / "otr_scifi_16gb_full.json"

DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"

# Pure link-bound input names -- no widget editor, no widgets_values entry.
PURE_LINK_INPUTS = {
    "script_text", "script_json", "news_used", "estimated_minutes",
}


def main() -> int:
    if not WORKFLOW_PATH.exists():
        print(f"ERROR: {WORKFLOW_PATH} does not exist", file=sys.stderr)
        return 1

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    reviewer = next(
        (n for n in wf["nodes"]
         if n.get("type") == "OTR_LedgerScriptReviewer"),
        None,
    )
    if reviewer is None:
        print("ERROR: OTR_LedgerScriptReviewer node not found in workflow",
              file=sys.stderr)
        return 1

    writer = next(
        (n for n in wf["nodes"]
         if n.get("type") == "OTR_LedgerScriptWriter"),
        None,
    )
    if writer is None:
        print("ERROR: no OTR_LedgerScriptWriter node found", file=sys.stderr)
        return 1

    changed = False

    # ----- Fix 1: strip `widget` hint from the 4 passthrough inputs ----
    for inp in reviewer.get("inputs", []):
        name = inp.get("name", "")
        if name in PURE_LINK_INPUTS and "widget" in inp:
            del inp["widget"]
            changed = True
            print(f"  stripped widget hint from input {name!r}")

    # ----- Fix 2: reduce widgets_values to just [model_id_value] ------
    # Previously: [script_text, script_json, news_used, estimated_minutes,
    #              model_id]
    # Now: [model_id]
    existing = reviewer.get("widgets_values", [])
    if len(existing) != 1:
        # Preserve any user-set model_id; fall back to default.
        if len(existing) >= 5 and isinstance(existing[4], str) and existing[4]:
            model_id = existing[4]
        elif existing and isinstance(existing[-1], str) and existing[-1]:
            model_id = existing[-1]
        else:
            model_id = DEFAULT_MODEL_ID
        reviewer["widgets_values"] = [model_id]
        changed = True
        print(f"  trimmed widgets_values from {len(existing)} entries to "
              f"[{model_id!r}]")

    # ----- Fix 3: move the reviewer above the writer ------------------
    # Writer is at writer.pos. Place reviewer directly above the writer
    # with a clean Y gap so they don't visually overlap. Use the same X
    # as the writer so the reviewer sits in the same column.
    wpos = writer.get("pos", [50, 200])
    wsize = writer.get("size", [400, 320])
    desired_x = int(wpos[0])
    # 200px above the writer's top edge (writer top = wpos[1]).
    # Reviewer size is ~380x320 (defaults at creation); leave 60px
    # vertical breathing room.
    desired_y = int(wpos[1]) - 380
    current_pos = reviewer.get("pos", [0, 0])
    if list(current_pos) != [desired_x, desired_y]:
        reviewer["pos"] = [desired_x, desired_y]
        changed = True
        print(f"  moved reviewer to pos=[{desired_x}, {desired_y}] "
              f"(was {list(current_pos)})")

    if not changed:
        print("OK: reviewer node already in fixed shape; no-op.")
        return 0

    # ----- Atomic write ----------------------------------------------
    tmp_path = str(WORKFLOW_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, WORKFLOW_PATH)
    print("OK: workflow JSON updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

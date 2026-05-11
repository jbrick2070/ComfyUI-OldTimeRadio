"""scripts/strip_legacy_writer_widgets_from_workflow.py

One-shot follow-up to the post-Phase-3 cleanup pass.

The writer's INPUT_TYPES dropped four legacy v2.0 MVP no-op widgets:
  cleanup_model_id, self_critique, target_length, arc_enhancer

And moved `act_count` into the slot where `target_length` used to be.
Saved workflows must be re-aligned positionally so ComfyUI maps the
right values to the right widgets at load time.

Per the writer's INPUT_TYPES order, the writer's widgets are now:
  required:
    0  episode_title          STRING
    1  target_words           INT
    2  num_characters         INT
  optional:
    3  seed                   INT
       (ComfyUI auto-inserts a control_after_generate slot 4 here)
    5  model_id               COMBO
    6  custom_premise         STRING multiline
    7  include_act_breaks     BOOLEAN
    8  act_count              INT          (was target_length slot)
    9  style                  COMBO
   10  style_custom           STRING multiline
   11  creativity             COMBO
   12  optimization_profile   COMBO
   13  perfect_run_spacesaver BOOLEAN

= 14 widgets total (13 declared + 1 control_after_generate).

Pre-cleanup workflow had 17 values:
  0  episode_title        ""
  1  target_words         350
  2  num_characters       2
  3  seed                 0
  4  control_after_gen    "randomize"
  5  model_id             "mistralai/Mistral-Nemo-Instruct-2407"
  6  cleanup_model_id     "auto (use story model)"      <-- DROP
  7  custom_premise       ""
  8  include_act_breaks   True
  9  self_critique        True                          <-- DROP
 10  target_length        "short (3 acts)"              <-- REPLACE with act_count=0
 11  style                "let the story decide"
 12  style_custom         ""
 13  creativity           "balanced"
 14  arc_enhancer         True                          <-- DROP
 15  optimization_profile "Standard"
 16  perfect_run_spacesaver False

Result after cleanup (14 entries):
  0  ""
  1  350                     (preserved)
  2  2                       (preserved)
  3  0                       (preserved seed)
  4  "randomize"             (preserved control_after_generate)
  5  "mistralai/Mistral-Nemo-Instruct-2407"
  6  ""                      (preserved custom_premise)
  7  True                    (preserved include_act_breaks)
  8  0                       (act_count -- auto-derive sentinel, replaces slot 10)
  9  "let the story decide"  (preserved style)
 10  ""                      (preserved style_custom)
 11  "balanced"              (preserved creativity)
 12  "Standard"              (preserved optimization_profile)
 13  False                   (preserved perfect_run_spacesaver)

Idempotent: if widgets_values is already length 14, exits no-op.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflows" / "otr_scifi_16gb_full.json"


def main() -> int:
    if not WORKFLOW_PATH.exists():
        print(f"ERROR: {WORKFLOW_PATH} does not exist", file=sys.stderr)
        return 1

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    writer = next(
        (n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter"),
        None,
    )
    if writer is None:
        print("ERROR: no OTR_LedgerScriptWriter node found", file=sys.stderr)
        return 1

    values = list(writer.get("widgets_values", []))
    if len(values) == 14:
        print("OK: writer widgets_values already length 14; no-op.")
        return 0

    if len(values) != 17:
        print(
            f"WARNING: expected 17 widget values (legacy shape) or 14 "
            f"(post-cleanup); got {len(values)}. Refusing to remap.",
            file=sys.stderr,
        )
        return 1

    # Pre-cleanup positional layout, transformed into the new layout.
    new_values = [
        values[0],   # episode_title
        values[1],   # target_words
        values[2],   # num_characters
        values[3],   # seed
        values[4],   # control_after_generate
        values[5],   # model_id
        values[7],   # custom_premise (was 7; cleanup_model_id at 6 dropped)
        values[8],   # include_act_breaks (was 8)
        0,           # act_count (NEW slot replaces target_length; default 0)
        values[11],  # style (was 11)
        values[12],  # style_custom (was 12)
        values[13],  # creativity (was 13)
        values[15],  # optimization_profile (was 15; arc_enhancer at 14 dropped)
        values[16],  # perfect_run_spacesaver (was 16)
    ]

    writer["widgets_values"] = new_values
    print(f"OK: writer widgets_values trimmed 17 -> {len(new_values)}.")
    print(f"  Dropped: cleanup_model_id={values[6]!r}, "
          f"self_critique={values[9]!r}, target_length={values[10]!r}, "
          f"arc_enhancer={values[14]!r}.")
    print(f"  act_count inserted at slot 8 (was target_length) = 0 (auto-derive).")

    # Atomic write.
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

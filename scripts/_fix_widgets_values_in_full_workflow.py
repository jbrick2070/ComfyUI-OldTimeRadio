"""Fix widgets_values on OTR_BatchHumoRender + OTR_VideoComposite.

After three rounds of empirical mismatch-debugging from ComfyUI's
prompt validator (BUG-LOCAL-079 second pass), the actual rules are:

  1. Placeholders for widget-converted-to-link inputs DO stay in
     widgets_values. ComfyUI's runtime expects them at their slot;
     the link's runtime value overrides the placeholder. (Earlier
     "fix" that dropped them was wrong.)

  2. ComfyUI auto-inserts a ``control_after_generate`` companion
     widget after every INT widget named ``seed``. Its value is one
     of "randomize" / "fixed" / "increment" / "decrement". This
     companion eats one slot in widgets_values that the patcher
     never wrote, shifting every widget after seed by +1.

This patcher:
  - Restores the placeholders for converted inputs in
    OTR_VideoComposite (3 placeholders dropped earlier).
  - Inserts the missing ``control_after_generate`` companion after
    seed in OTR_BatchHumoRender.

Idempotent: re-running with the right widgets_values is no-op.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WF = REPO / "workflows" / "otr_scifi_16gb_full.json"


def _fix_node(node: dict) -> tuple[bool, str]:
    """Rebuild widgets_values from the canonical truth for the two
    OTR nodes we touched. Avoids fragile diff math against whatever
    earlier fixes produced -- just produce the known-correct shape.
    """
    ntype = node.get("type")
    inputs = node.get("inputs") or []
    current = node.get("widgets_values") or []

    if ntype == "OTR_BatchHumoRender":
        # Canonical widgets_values for BatchHumoRender (12 entries):
        #   [0] ledger_json placeholder ("" — link wins)
        #   [1] portraits_dir placeholder ("" — auto-resolve)
        #   [2] clip_length          7.0
        #   [3] max_clips            0
        #   [4] seed                 7
        #   [5] control_after_generate  "randomize"  (auto-companion to seed)
        #   [6] steps                6
        #   [7] cfg                  1.0
        #   [8] sampler_name         "uni_pc"
        #   [9] scheduler            "simple"
        #   [10] width                480
        #   [11] height               832
        target = ["", "", 7.0, 0, 7, "randomize", 6, 1.0, "uni_pc", "simple", 480, 832]
        if list(current) == target:
            return (False, "already canonical")
        node["widgets_values"] = target
        return (True, f"rebuilt to canonical 12-slot ({len(current)} -> 12)")

    if ntype == "OTR_VideoComposite":
        # Canonical widgets_values for VideoComposite (11 entries):
        #   [0] procgen_video_path placeholder ("" — link wins)
        #   [1] clips_dir placeholder ("" — link wins)
        #   [2] ledger_json placeholder ("" — link wins)
        #   [3] blend_mode    "addition"
        #   [4] blend_opacity 0.5
        #   [5] canvas_width  1920
        #   [6] canvas_height 1080
        #   [7] canvas_fps    25
        #   [8] humo_target_height 1080
        #   [9] fallback_clip_length 7.0
        #   [10] ffmpeg "ffmpeg"
        target = ["", "", "", "addition", 0.5, 1920, 1080, 25, 1080, 7.0, "ffmpeg"]
        if list(current) == target:
            return (False, "already canonical")
        node["widgets_values"] = target
        return (True, f"rebuilt to canonical 11-slot ({len(current)} -> 11)")

    return (False, f"untargeted node type {ntype}")


def main() -> int:
    if not WF.exists():
        print(f"FATAL: {WF} not found", file=sys.stderr)
        return 1
    with open(WF, "r", encoding="utf-8") as f:
        wf = json.load(f)

    targets = ("OTR_BatchHumoRender", "OTR_VideoComposite")
    changed_any = False
    for n in wf.get("nodes") or []:
        if n.get("type") not in targets:
            continue
        before = list(n.get("widgets_values") or [])
        changed, summary = _fix_node(n)
        if changed:
            after = list(n.get("widgets_values") or [])
            print(f"  {n['type']} (id={n['id']}): {summary}")
            print(f"    before ({len(before)}): {before}")
            print(f"    after  ({len(after)}): {after}")
            changed_any = True
        else:
            print(f"  {n['type']} (id={n['id']}): no-op ({summary})")

    if not changed_any:
        print("nothing to fix")
        return 0

    # Schema validation: every node must still have flags/order/mode
    req = {"flags", "order", "mode"}
    missing = [(n["id"], n.get("type"), k) for n in wf["nodes"]
               for k in req if k not in n]
    if missing:
        print(f"FATAL: schema check failed: {missing}", file=sys.stderr)
        return 4

    tmp = WF.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(WF)
    print(f"OK: wrote {WF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

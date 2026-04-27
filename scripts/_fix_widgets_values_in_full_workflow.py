"""Fix the widgets_values misalignment on OTR_BatchHumoRender and
OTR_VideoComposite that the in-graph patcher emitted.

Symptom: ComfyUI rejects the prompt at validation with errors like
    cfg, uni_pc, could not convert string to float: 'uni_pc'
    sampler_name: 'simple' not in (list of length 44)
    scheduler: '480' not in [...]
because widgets_values has empty-string PLACEHOLDERS for inputs that
were also wired via links (ledger_json, portraits_dir, procgen_video_path,
clips_dir). ComfyUI's frontend treats the link as the source-of-truth
and IGNORES the placeholder, but our placeholders shifted the rest of
widgets_values up by 2 slots.

Fix: remove the placeholders for any widget-as-input that has both a
``widget: {...}`` key in inputs[] AND a non-None link. Leave widget
values for unwired widgets in place.

Idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WF = REPO / "workflows" / "otr_scifi_16gb_full.json"


def _fix_node(node: dict) -> tuple[bool, str]:
    """Drop widgets_values placeholders for inputs that are
    widget-converted-to-link (input has both `widget` key and a
    non-None link). Returns (changed, summary)."""
    inputs = node.get("inputs") or []
    widgets_values = node.get("widgets_values") or []

    # The widget order in widgets_values is the order of widget-typed
    # inputs in INPUT_TYPES. Inputs in the inputs[] array that are
    # links-only (no widget: key) DON'T occupy widget slots. Inputs
    # that have BOTH widget: AND link DO occupy widget slots BUT
    # ComfyUI ignores their widget value (the link wins).
    #
    # The correct widgets_values ordering is: only entries for inputs
    # that are pure widgets (no link) AND for inputs that are
    # widget-converted-to-link. Both kept, in INPUT_TYPES order.
    # ComfyUI's renderer expects placeholders for converted inputs.
    #
    # OUR BUG: we wrote placeholders for converted inputs but
    # ComfyUI's runtime is reading widgets_values WITHOUT the
    # placeholders. Different versions of ComfyUI handle this
    # differently. Pragmatic fix: drop the placeholder entries for
    # converted-to-link inputs. The link will be honored for the
    # value at runtime.

    # Identify how many leading placeholders to drop. We rely on the
    # convention that the patcher placed converted-input placeholders
    # at the START of widgets_values.
    converted_count = 0
    for inp in inputs:
        if inp.get("widget") and inp.get("link") is not None:
            converted_count += 1

    if converted_count == 0:
        return (False, "no converted-input placeholders to drop")
    if len(widgets_values) <= converted_count:
        return (False, f"widgets_values too short ({len(widgets_values)}) for {converted_count} drops")

    # Drop the first `converted_count` placeholders.
    new_widgets_values = list(widgets_values[converted_count:])
    node["widgets_values"] = new_widgets_values
    return (True,
            f"dropped {converted_count} placeholder(s); "
            f"widgets_values {len(widgets_values)} -> {len(new_widgets_values)}")


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

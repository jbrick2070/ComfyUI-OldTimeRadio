"""scripts/normalize_workflow_widgets.py

================================================================
KNOWN OVER-AGGRESSIVE -- DO NOT RUN AUTONOMOUSLY (2026-05-17).
Bug-Bible candidate. The strategy below misclassifies ComfyUI's
auto-injected `control_after_generate` companion strings ("fixed"
/ "randomize" / "increment" / "decrement") as orphans whenever the
type spec for that slot is INT (the seed slot) and the value is a
string. The control_after_generate companion is a hidden widget
ComfyUI injects after every INT seed; it lives at widget index
seed_slot + 1 with a fixed string value, NOT a /object_info
declared widget, so the schema walk in this script cannot see it
and treats the value as type-incompatible-with-INT noise.

Empirical evidence: run against workflows/otr_canonical.json
on 2026-05-17 stripped 8 legitimate companion strings and broke 5
guardrail tests (test_writer_seed_control_is_fixed,
test_humo_seed_control_is_fixed, test_writer_both_slots_mistral_nemo,
test_writer_widget_migration_preserves_values,
test_writer_style_widget_saved_as_auto_sentinel). The committed
fix at Sprint H Commit A reverted the over-strip and dropped only
the single real orphan ("{}" at node 12 widgets[2]) by hand.

Needed hardening before this tool can be trusted:
  1. Detect and reserve seed_mode companion slots. After any
     "seed" widget of INPUT_TYPES kind INT, the next widget index
     is the companion -- skip it during the orphan walk.
  2. Detect socket-converted widgets and preserve their saved
     default value at the original slot index (today these often
     read as orphan "[]" / "{}" / "" matching nothing in the live
     schema's "widget" set).
  3. Add a smoke test that round-trips an unmodified workflow JSON
     through the tool and asserts no diff -- catches over-strip
     before any operator can ship it.

Use today as a READ-ONLY diagnostic with --dry-run. Inspect the
proposed changes manually; apply only the ones that are truly
orphan values. Never let CI or a loop driver execute this without
the three fixes above.
================================================================

Sprint H bug-hunt support tool. Realigns a ComfyUI editor-format workflow
JSON's per-node `widgets_values` arrays against the live `/object_info`
schemas, so stale orphan values from prior node revisions get removed.

Specific drift class this fixes (caught in Sprint H iter 1 on
v2.0-alpha @ dc0774b):

  - Orphan `"fixed"` strings inserted as ComfyUI's
    `control_after_generate` companion next to seed widgets, then
    left behind when the companion was removed from the node's
    INPUT_TYPES. Causes downstream values to misalign by +1 from
    that index.

  - Empty `"[]"` strings or empty `""` strings sitting where typed
    primitives (INT / FLOAT / COMBO) are expected.

  - Widget arrays shorter or longer than the live INPUT_TYPES
    declares.

Strategy:

  1. Load workflow JSON.
  2. Fetch /object_info from the running ComfyUI.
  3. For each node:
     a. Compute the expected widget_names list (required + optional,
        widget-backed only -- matches otr_api._ordered_widget_names).
     b. Determine "preserved" vs "stripped" mode by inspecting which
        widgets have been converted to input sockets.
     c. Walk widgets_values + expected widget_names side by side. At
        each position, validate the value's type against the spec. If
        the value is type-incompatible AND removing it (and shifting
        the tail down) produces a type-compatible alignment, treat the
        current value as an orphan and drop it.
     d. If the array is still longer than expected after orphan
        removal, trim from the end (these are extra trailing values
        with no slot to live in).
     e. If the array is shorter than expected, pad with the schema's
        documented default value where available, else None.
  4. Write a backup copy alongside the workflow file (suffix
     `.bak-normalize-<unix_ts>`).
  5. Write the normalized JSON in place.

Run via:

  & C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe ^
    scripts\\normalize_workflow_widgets.py ^
    --workflow workflows\\otr_canonical.json

Optional flags:
  --comfyui-url <url>   default: http://127.0.0.1:8000
  --dry-run             report changes without writing
  --no-backup           skip the .bak-normalize-* copy
  --verbose             per-node detail on every node, not just changed
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the validated primitives from otr_api.
from otr_api import (  # noqa: E402
    COMFYUI_URL,
    _is_widget_backed,
    _ordered_widget_names,
    fetch_schemas,
    load_workflow,
)


def _value_matches_spec(value: Any, spec: Any) -> bool:
    """Return True if `value` is type-compatible with the widget spec.

    Mirrors otr_api._validate_widget_value's checks but returns a bool
    instead of raising. Used by the orphan-detection walk.
    """
    if value is None:
        return True  # null = "use default" -- always tolerated

    type_def = (
        spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
    )

    # Dropdown / COMBO -- type_def is a list of choices.
    if isinstance(type_def, list):
        # Some custom nodes generate the choice list lazily and surface an
        # empty list at /object_info. Tolerate.
        if not type_def:
            return isinstance(value, (str, int, float, bool))
        return value in type_def

    if not isinstance(type_def, str):
        return True  # unknown spec shape -- don't reject

    t = type_def.upper()

    if t in ("BOOLEAN", "BOOL"):
        return isinstance(value, bool)
    if t == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "FLOAT":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "STRING":
        return isinstance(value, str)
    if t == "COMBO":
        return True  # COMBO with no inline choices -- dynamic enum
    return True


def _spec_for_widget(node_type: str, widget_name: str, schemas: dict) -> Any:
    """Return the spec for a (node_type, widget_name) pair, or None."""
    if node_type not in schemas:
        return None
    schema = schemas[node_type].get("input", {}) or {}
    required = schema.get("required", {}) or {}
    optional = schema.get("optional", {}) or {}
    if widget_name in required:
        return required[widget_name]
    if widget_name in optional:
        return optional[widget_name]
    return None


def _widget_default(spec: Any) -> Any:
    """Pull a default value out of a widget spec, or return None.

    Specs are usually (type_def, {"default": value, ...}). For dropdowns
    (type_def = list of choices), default to the first choice.
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) >= 2 and isinstance(spec[1], dict):
            if "default" in spec[1]:
                return spec[1]["default"]
        # Dropdown without an explicit default
        if len(spec) >= 1 and isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
    return None


def _linked_widget_names(node: dict, widget_names: list[str]) -> set[str]:
    """Names of widgets that have been converted to input sockets."""
    return {
        inp.get("name")
        for inp in node.get("inputs", []) or []
        if inp.get("link") is not None and inp.get("name") in widget_names
    }


def _drop_first_misalignment(
    wv: list[Any],
    widget_names: list[str],
    node_type: str,
    schemas: dict,
) -> tuple[list[Any], list[int]]:
    """Walk wv + widget_names side by side. At each position, if the
    value mismatches the spec AND removing this value makes everything
    downstream align, treat it as orphan and drop it. Returns the cleaned
    list plus the indices of dropped values.

    Bounded by len(wv). Drops at most (len(wv) - len(widget_names))
    orphans -- never less, never more.
    """
    excess = len(wv) - len(widget_names)
    if excess <= 0:
        return list(wv), []

    cleaned = list(wv)
    dropped: list[int] = []
    for _ in range(excess):
        drop_idx = None
        for i in range(min(len(cleaned), len(widget_names))):
            this_spec = _spec_for_widget(node_type, widget_names[i], schemas)
            if this_spec is None:
                continue
            if _value_matches_spec(cleaned[i], this_spec):
                continue
            # Mismatch at i. Try dropping i and see if i+1 onward aligns.
            tail = cleaned[i + 1:]
            tail_names = widget_names[i:i + len(tail)]
            if all(
                _value_matches_spec(
                    v,
                    _spec_for_widget(node_type, n, schemas),
                )
                for v, n in zip(tail, tail_names)
                if _spec_for_widget(node_type, n, schemas) is not None
            ):
                drop_idx = i
                break
        if drop_idx is None:
            # Could not localize the orphan -- bail and let trim-from-end
            # handle the excess.
            break
        dropped.append(drop_idx)
        cleaned.pop(drop_idx)
    return cleaned, dropped


def normalize_node(
    node: dict,
    schemas: dict,
    verbose: bool = False,
) -> dict | None:
    """Realign `node["widgets_values"]` against the live schema.

    Returns a dict with diagnostics if anything changed, else None.
    """
    node_type = node.get("type")
    node_id = node.get("id")
    if node_type not in schemas:
        return {
            "node_id": node_id,
            "node_type": node_type,
            "status": "SKIP_UNKNOWN_TYPE",
            "before_len": len(node.get("widgets_values") or []),
            "after_len": len(node.get("widgets_values") or []),
            "dropped_indices": [],
            "padded_count": 0,
        }

    try:
        widget_names = _ordered_widget_names(node_type, schemas)
    except KeyError:
        return None

    linked = _linked_widget_names(node, widget_names)
    visible_widget_names = [n for n in widget_names if n not in linked]

    wv_before = list(node.get("widgets_values") or [])

    # Preserved mode: array length == len(widget_names) (placeholder
    # slots for linked widgets). Stripped mode: == len(visible).
    # Decide which we're targeting based on whichever the live array is
    # closer to. Default to preserved when ambiguous.
    if abs(len(wv_before) - len(widget_names)) <= abs(
        len(wv_before) - len(visible_widget_names)
    ):
        target_names = widget_names
        mode = "preserved"
    else:
        target_names = visible_widget_names
        mode = "stripped"

    # Step 1: drop orphans where the type-mismatch points at an extra
    # value to remove.
    cleaned, dropped = _drop_first_misalignment(
        wv_before, target_names, node_type, schemas
    )

    # Step 2: trim trailing excess (orphans we couldn't localize).
    trimmed_count = 0
    while len(cleaned) > len(target_names):
        cleaned.pop()
        trimmed_count += 1

    # Step 3: pad with schema defaults where short.
    padded_count = 0
    while len(cleaned) < len(target_names):
        idx = len(cleaned)
        wname = target_names[idx]
        spec = _spec_for_widget(node_type, wname, schemas)
        cleaned.append(_widget_default(spec))
        padded_count += 1

    if (
        len(wv_before) == len(cleaned)
        and wv_before == cleaned
        and not dropped
        and trimmed_count == 0
        and padded_count == 0
    ):
        return None  # unchanged

    node["widgets_values"] = cleaned
    return {
        "node_id": node_id,
        "node_type": node_type,
        "mode": mode,
        "status": "NORMALIZED",
        "before_len": len(wv_before),
        "after_len": len(cleaned),
        "dropped_indices": dropped,
        "trimmed_from_end": trimmed_count,
        "padded_count": padded_count,
        "before": wv_before if verbose else None,
        "after": cleaned if verbose else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Realign ComfyUI workflow widgets against live /object_info."
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help="Path to workflow JSON file (editor / UI format).",
    )
    parser.add_argument(
        "--comfyui-url",
        default=COMFYUI_URL,
        help=f"ComfyUI URL (default: {COMFYUI_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing the file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak-normalize-<ts> copy.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print before/after widgets_values for every changed node.",
    )
    args = parser.parse_args()

    # Honor --comfyui-url even if env COMFYUI_URL is set differently.
    if args.comfyui_url:
        os.environ["COMFYUI_URL"] = args.comfyui_url

    workflow_path = Path(args.workflow).resolve()
    if not workflow_path.exists():
        print(f"FAIL workflow not found: {workflow_path}", file=sys.stderr)
        return 2

    print(f"workflow     = {workflow_path}")
    print(f"comfyui_url  = {os.environ.get('COMFYUI_URL', COMFYUI_URL)}")
    print(f"dry_run      = {args.dry_run}")
    print()

    print("Fetching /object_info schemas...")
    try:
        schemas = fetch_schemas()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL could not fetch schemas: {exc}", file=sys.stderr)
        return 2

    print("Loading workflow...")
    wf = load_workflow(str(workflow_path))

    print()
    print(f"Walking {len(wf.get('nodes', []))} nodes...")
    print()

    changes: list[dict] = []
    for node in wf.get("nodes", []):
        diag = normalize_node(node, schemas, verbose=args.verbose)
        if diag is not None:
            changes.append(diag)

    if not changes:
        print("No drift detected. Workflow is already aligned.")
        return 0

    print(f"{len(changes)} node(s) realigned:")
    print()
    for d in changes:
        nid = d["node_id"]
        ntype = d["node_type"]
        bl = d.get("before_len")
        al = d.get("after_len")
        mode = d.get("mode", "n/a")
        dropped = d.get("dropped_indices", [])
        trimmed = d.get("trimmed_from_end", 0)
        padded = d.get("padded_count", 0)
        status = d.get("status", "?")
        print(
            f"  [{nid:>3}] {ntype:<40s} {status:<14s} mode={mode:<9s} "
            f"len {bl}->{al}  dropped={dropped} trim_end={trimmed} pad={padded}"
        )
        if args.verbose and d.get("before") is not None:
            print(f"        before: {d['before']}")
            print(f"        after : {d['after']}")

    print()

    if args.dry_run:
        print("--dry-run: no file written.")
        return 0

    if not args.no_backup:
        ts = int(time.time())
        bak_path = workflow_path.with_suffix(
            workflow_path.suffix + f".bak-normalize-{ts}"
        )
        bak_path.write_bytes(workflow_path.read_bytes())
        print(f"Backup written: {bak_path}")

    # Write with stable 2-space indent + trailing newline -- matches
    # ComfyUI's own save format closely enough for git diff readability.
    serialized = json.dumps(wf, indent=2, ensure_ascii=False)
    workflow_path.write_text(serialized + "\n", encoding="utf-8")
    print(f"Wrote normalized workflow: {workflow_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

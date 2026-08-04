#!/usr/bin/env python
"""Audit workflow JSON files for ComfyUI zod-schema compliance.

ComfyUI's frontend uses a zod schema to validate workflow JSON before
the graph queues. Two requirements bite hand-edited workflows the most:

1. Top-level `id` field must be a UUID4 (8-4-4-4-12 hex), not an empty
   string and not a kebab-case slug. Documented in BUG-LOCAL-069 family.

2. Every `nodes[]` entry must carry `flags` (object), `order` (int),
   `mode` (int enum: 0=ALWAYS, 2=MUTE, 4=BYPASS). Documented as
   BUG-LOCAL-069. New nodes invented from scratch in Python often miss
   these.

Usage:
    python tools/audit_workflow_schema.py [--fix] [--workflows DIR]

    --fix       auto-rewrite invalid files in place
    --workflows DIR override the default workflows/ directory

Exit code is 0 if all files pass, 1 if any have issues (whether fixed
or reported only). Useful as a pre-commit hook or CI step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

REQUIRED_NODE_FIELDS = ("flags", "order", "mode")


def audit_file(p: Path, fix: bool = False) -> tuple[list[str], bool]:
    """Returns (issues_list, was_fixed)."""
    try:
        text = p.read_text(encoding="utf-8")
        j = json.loads(text)
    except Exception as e:
        return [f"PARSE FAIL: {e}"], False

    issues: list[str] = []
    fixed = False

    # 1. Top-level id is UUID4
    wf_id = j.get("id", "")
    if not UUID_RE.match(str(wf_id)):
        if fix:
            new_uuid = str(uuid.uuid4())
            issues.append(
                f"id={wf_id!r} not UUID4 -> regenerated: {new_uuid}"
            )
            j["id"] = new_uuid
            fixed = True
        else:
            issues.append(f"id={wf_id!r} not UUID4")

    # 2. Every node has flags/order/mode
    if not isinstance(j.get("nodes"), list):
        issues.append("'nodes' missing or not a list")
        return issues, fixed

    max_order = max(
        (n.get("order") for n in j["nodes"] if isinstance(n.get("order"), int)),
        default=-1,
    )
    next_order = max_order + 1
    for n in j["nodes"]:
        for f in REQUIRED_NODE_FIELDS:
            if f not in n:
                if fix:
                    if f == "flags":
                        n[f] = {}
                    elif f == "order":
                        n[f] = next_order
                        next_order += 1
                    elif f == "mode":
                        n[f] = 0
                    issues.append(
                        f"node id={n.get('id')} type={n.get('type')} "
                        f"missing {f} -> filled"
                    )
                    fixed = True
                else:
                    issues.append(
                        f"node id={n.get('id')} type={n.get('type')} "
                        f"missing {f}"
                    )

    # 3. last_node_id / last_link_id sanity
    max_node_id = max((n.get("id", 0) for n in j["nodes"]), default=0)
    if not isinstance(j.get("last_node_id"), int) or j["last_node_id"] < max_node_id:
        if fix:
            j["last_node_id"] = max_node_id
            issues.append(f"last_node_id bumped to {max_node_id}")
            fixed = True
        else:
            issues.append(f"last_node_id missing or < max id {max_node_id}")

    links = j.get("links", []) or []
    max_link_id = max(
        (L[0] for L in links if isinstance(L, list) and L),
        default=0,
    )
    if not isinstance(j.get("last_link_id"), int) or j["last_link_id"] < max_link_id:
        if fix:
            j["last_link_id"] = max_link_id
            issues.append(f"last_link_id bumped to {max_link_id}")
            fixed = True
        else:
            issues.append(f"last_link_id missing or < max link id {max_link_id}")

    # 4. 'links' present
    if "links" not in j:
        if fix:
            j["links"] = []
            issues.append("'links' missing -> set to []")
            fixed = True
        else:
            issues.append("'links' missing")

    # 5. 'version' present
    if "version" not in j:
        if fix:
            j["version"] = 0.4
            issues.append("'version' missing -> set to 0.4")
            fixed = True
        else:
            issues.append("'version' missing")

    if fixed:
        p.write_text(json.dumps(j, indent=2), encoding="utf-8")

    return issues, fixed


# ---------------------------------------------------------------------------
# Sprint D D3 -- default-workflow creative-binding license-equivalence gate
# ---------------------------------------------------------------------------


class WorkflowSchemaError(RuntimeError):
    """Raised when a default-shipped workflow JSON violates a
    Sprint D D3 license-equivalence or prompt-profile gate."""


def _strip_badge(value: str) -> str:
    """Drop the picker's VRAM badge from a saved widget value.

    Deliberately NOT taken from ``catalog_module``: that parameter exists so
    tests can inject a FAKE catalog, and a fake has rows but no label helpers.
    Reaching through it for the stripper broke exactly that injection. The
    badge is a label convention, so it is stripped here, and the real
    catalog's implementation is preferred when it can be imported.
    """
    s = str(value or "")
    try:
        import sys as _sys
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from nodes import _otr_model_catalog as _cat  # noqa: PLC0415
        return _cat._strip_label_suffix(s)
    except Exception:  # noqa: BLE001 -- an audit tool must not die on import
        return s.rsplit(" (", 1)[0].rstrip() if s.endswith(")") and " (" in s else s


def check_default_workflow_creative_binding(
    workflow_json: dict, catalog_module=None,
) -> list[str]:
    """Sprint D D3 hard-fail validator.

    Walks every `OTR_LedgerScriptWriter` node in `workflow_json` and
    checks the `creative_writing_model` widget value against the
    catalog. Returns a list of violation messages (empty on pass).
    Calling code (or the audit_file flow) decides whether to raise
    `WorkflowSchemaError` or warn.

    Two contracts enforced per row bound to the creative slot:

      1. `license_audit_status == "mit_equivalent"`. Research-lane
         (talkie) or pending (Gemma) rows cannot ship default-bound.

      2. `prompt_profile == "modern"`. The default workflow should
         not silently route through the period system prompt; period
         routing is opt-in via the dropdown.

    `catalog_module` is a parameter so unit tests can inject a fake
    catalog; defaults to the real `nodes._otr_model_catalog`.
    """
    if catalog_module is None:
        import sys as _sys
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from nodes import _otr_model_catalog as catalog_module  # noqa: PLC0415

    rows_by_id = {m.repo_id: m for m in catalog_module.CURATED_LLM_MODELS}
    violations: list[str] = []
    for n in workflow_json.get("nodes") or []:
        if n.get("type") != "OTR_LedgerScriptWriter":
            continue
        widgets = n.get("widgets_values") or []
        # Inspect every widget value that matches a curated repo_id.
        # We don't try to be clever about widget-index -- if it's a
        # repo_id, it's bound to a slot (creative or technical).
        for v in widgets:
            if not isinstance(v, str) or not v:
                continue
            # NORMALIZE THE WAY THE RUNTIME DOES (2026-08-04). A saved widget
            # value carries the picker's VRAM badge --
            # "google/gemma-4-12b-it (11.9 GB)" -- and the loader strips it
            # before resolving the row. Matching the RAW string meant a
            # correctly-badged binding matched no curated row and this gate
            # inspected NOTHING: a non-mit_equivalent model could ship
            # default-bound and the licence check would pass in silence. The
            # same blindness was found in the test-side helper.
            bare = _strip_badge(v)
            if bare not in rows_by_id:
                continue
            v, row = bare, rows_by_id[bare]
            if row.license_audit_status != "mit_equivalent":
                violations.append(
                    f"writer node id={n.get('id')} binds {v!r} "
                    f"(license_audit_status="
                    f"{row.license_audit_status!r}); expected "
                    f"'mit_equivalent' for any default-shipped "
                    f"workflow binding"
                )
            if row.prompt_profile != "modern":
                violations.append(
                    f"writer node id={n.get('id')} binds {v!r} "
                    f"(prompt_profile={row.prompt_profile!r}); "
                    f"expected 'modern' for any default-shipped "
                    f"workflow binding -- period routing is "
                    f"opt-in not default-shipped"
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="Auto-rewrite invalid files in place")
    parser.add_argument("--workflows", type=Path,
                        default=Path(__file__).resolve().parent.parent / "workflows",
                        help="Workflows directory (default: workflows/)")
    args = parser.parse_args()

    print(f"=== Workflow JSON zod-schema audit ===")
    print(f"scanning: {args.workflows}")
    print(f"fix mode: {args.fix}")
    print()

    files = sorted(args.workflows.glob("*.json"))
    if not files:
        print("no *.json files found")
        return 0

    print(f"found {len(files)} workflow JSON files")
    print()

    any_issues = False
    summary: list[tuple[str, str, int]] = []
    for p in files:
        issues, fixed = audit_file(p, fix=args.fix)
        if not issues:
            print(f"  OK    {p.name}")
            summary.append((p.name, "OK", 0))
            continue
        any_issues = True
        tag = "FIXED" if fixed else "ISSUES"
        print(f"  {tag:<6} {p.name}")
        for i in issues:
            print(f"          - {i}")
        summary.append((p.name, tag, len(issues)))

    print()
    print("=== summary ===")
    for name, tag, n in summary:
        print(f"  {tag:<6} {name:<45} ({n} issue(s))")

    return 1 if any_issues else 0


if __name__ == "__main__":
    sys.exit(main())

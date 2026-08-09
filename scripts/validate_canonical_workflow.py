"""Structural audit of the canonical workflow JSON.

Runs OTR_WorkflowValidator's contract check + JSON round-trip + widget-count
audit + link referential integrity, as a real script (not inline `python -c`,
which CLAUDE.md forbids). Callable from the suite gate + from CI.

Usage:
    python scripts/validate_canonical_workflow.py                       # canonical
    python scripts/validate_canonical_workflow.py workflows/otr_canonical.json
    python scripts/validate_canonical_workflow.py workflows/variants/otr_foo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parents[1]))   # so `folder_paths` resolves from ComfyUI

DEFAULT_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)
    # JSON round-trip check: re-serialize + re-parse; the parsed dict must
    # match. A silent codepoint / trailing-comma anomaly surfaces here.
    round_trip = json.loads(json.dumps(data, ensure_ascii=False))
    if round_trip != data:
        raise SystemExit(f"[validate_canonical_workflow] JSON round-trip diverged for {path!r}")
    return data


def _audit_links(wf: dict, source: str) -> "list[str]":
    problems: "list[str]" = []
    nodes_by_id = {int(n["id"]): n for n in wf.get("nodes", [])}
    for link in wf.get("links", []):
        # litegraph link shape: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        if not isinstance(link, list) or len(link) < 6:
            problems.append(f"{source}: link {link!r} is not a 6-tuple")
            continue
        link_id, src_node, src_slot, dst_node, dst_slot, link_type = link[:6]
        if int(src_node) not in nodes_by_id:
            problems.append(f"{source}: link {link_id} src_node {src_node} not in nodes[]")
        if int(dst_node) not in nodes_by_id:
            problems.append(f"{source}: link {link_id} dst_node {dst_node} not in nodes[]")
    return problems


def _audit_widget_lengths(wf: dict, source: str) -> "list[str]":
    """Every widget-backed input on a node must have a slot in widgets_values.
    Non-widget inputs (real links) do not consume a widget slot."""
    problems: "list[str]" = []
    for node in wf.get("nodes", []):
        widget_inputs = [i for i in (node.get("inputs") or [])
                          if isinstance(i, dict) and i.get("widget")]
        wv = node.get("widgets_values") or []
        if len(widget_inputs) > len(wv):
            problems.append(
                f"{source}: node {node.get('id')} ({node.get('type')}) has "
                f"{len(widget_inputs)} widget-backed input(s) but only "
                f"{len(wv)} widgets_values entries")
    return problems


def _audit_deleted_types(wf: dict, source: str) -> "list[str]":
    try:
        from nodes._workflow_validation import DELETED_NODE_TYPES
    except Exception as e:  # noqa: BLE001
        return [f"{source}: could not import DELETED_NODE_TYPES: {e}"]
    problems: "list[str]" = []
    for node in wf.get("nodes", []):
        t = node.get("type")
        if t in DELETED_NODE_TYPES:
            problems.append(
                f"{source}: node {node.get('id')} type {t!r} is in "
                f"DELETED_NODE_TYPES; re-save the workflow without it")
    return problems


def _load_otr_package():
    """Import the OTR package the way ComfyUI does -- BY PATH, not by name.

    THIS IS THE WHOLE BUG. The package directory is ``ComfyUI-OldTimeRadio``
    with a HYPHEN, so ``importlib.import_module("ComfyUI_OldTimeRadio")`` --
    which is what this script used to do, after a dash->underscore rename --
    can never resolve. ComfyUI itself loads the pack by path. The lookup was
    therefore not merely fragile on a "bare venv sys.path issue" as the old
    docstring claimed: it was PERMANENTLY UNSATISFIABLE, so the contract check
    never ran ONCE, on any box, while the script still printed OK and exited 0.

    Loading from ``__init__.py`` via spec_from_file_location is the technique
    ``scripts/otr_macbeth_probe.py:load_otr_package`` already uses for exactly
    this reason -- that probe had to route around this script's fail-open to
    get a real contract check.
    """
    import importlib.util

    pkg_name = REPO_ROOT.name.replace("-", "_")
    existing = sys.modules.get(pkg_name)
    if existing is not None and getattr(existing, "NODE_CLASS_MAPPINGS", None):
        return existing

    init_py = REPO_ROOT / "__init__.py"
    if not init_py.is_file():
        raise RuntimeError(f"OTR package __init__.py missing: {init_py}")
    custom_nodes_parent = str(REPO_ROOT.parent)
    if custom_nodes_parent not in sys.path:
        sys.path.insert(0, custom_nodes_parent)
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_py, submodule_search_locations=[str(REPO_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(pkg_name, None)
        raise
    return module


def _run_validator_contract(wf: dict, source: str) -> "list[str]":
    """Call the shipped structural contract validator (validate_workflow_contract).
    Signature is `(workflow, node_class_mappings, ...)` -- feed the live
    NODE_CLASS_MAPPINGS from the OTR package.

    FAILS CLOSED. If the contract cannot be run, that is a PROBLEM and the
    script exits non-zero. It used to print a SKIPPED line to stderr and
    ``return []`` -- an empty problem list means "nothing wrong", so ``main()``
    reported OK and exited 0 having validated nothing. This script's own
    docstring advertises the contract check and calls itself callable from the
    suite gate and CI, so a green that skipped the headline check is a lie to
    every caller. Reproduced live and recorded in docs/HANDOFF_LOG.md."""
    try:
        from nodes._workflow_validation import validate_workflow_contract
    except Exception as e:  # noqa: BLE001
        return [f"{source}: could not import validate_workflow_contract: {e}"]
    try:
        pkg = _load_otr_package()
        ncm = getattr(pkg, "NODE_CLASS_MAPPINGS", None)
        if not ncm:
            raise RuntimeError("NODE_CLASS_MAPPINGS empty on the loaded package")
    except Exception as e:  # noqa: BLE001
        return [
            f"{source}: CONTRACT NOT RUN -- could not resolve "
            f"NODE_CLASS_MAPPINGS ({type(e).__name__}: {e}). This is a "
            f"failure, not a skip: the contract check is this script's "
            f"headline guarantee."
        ]
    problems: "list[str]" = []
    try:
        validate_workflow_contract(wf, ncm)
    except Exception as e:  # noqa: BLE001
        problems.append(f"{source}: validate_workflow_contract raised {type(e).__name__}: {e}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    args = parser.parse_args(argv)
    path = Path(args.workflow).resolve()
    if not path.is_file():
        print(f"[validate_canonical_workflow] not a file: {path}", file=sys.stderr)
        return 2
    wf = _load(path)
    problems: "list[str]" = []
    problems.extend(_run_validator_contract(wf, str(path)))
    problems.extend(_audit_deleted_types(wf, str(path)))
    problems.extend(_audit_widget_lengths(wf, str(path)))
    problems.extend(_audit_links(wf, str(path)))
    if problems:
        print(f"[validate_canonical_workflow] {len(problems)} PROBLEM(S) in {path}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[validate_canonical_workflow] OK {path} "
          f"({len(wf.get('nodes', []))} nodes, "
          f"{len(wf.get('links', []))} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""OTR_WorkflowValidator -- opt-in execution-time contract validator.

Per ADR docs/2026-05-13-S14_2-active-validation-ADR.md (Option B,
locked in S24/C12; implementation in S26 Sprint 3).

Placed as the first node in a workflow JSON, this validator reads the
workflow JSON file from disk (the live one ComfyUI is executing) and
runs the same `validate_workflow_contract` check that
`tests/test_workflow_live_passes_validator.py` runs in CI. Violations
raise the typed exception from `_workflow_validation`, which
ComfyUI surfaces as a red-bordered node error in the canvas -- the
same channel as every other OTR node failure.

INPUT_TYPES:
  - workflow_json_path (STRING, optional default): absolute path to the
    workflow JSON. If empty, falls back to the canonical fixture path
    under workflows/otr_scifi_16gb_full.json relative to this file.
  - validate_anyway (BOOLEAN, default True): set False to skip the
    check for diagnostic loads (e.g. running a deliberately-broken
    workflow to inspect intermediate state).
  - strict_unknown_types (BOOLEAN, default True): when True, an
    OTR_-prefixed type missing from NODE_CLASS_MAPPINGS raises
    `WorkflowUnknownNodeTypeError`. False matches the CI test default.

OUTPUT:
  - validation_report (STRING): on pass, a brief one-line OK report
    that downstream nodes can route to a Note or ignore. On fail, the
    node raises before producing a return value.

OUTPUT_NODE = True so ComfyUI executes this node even without a
downstream consumer.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("OTR.workflow_validator")

# Default workflow path -- the canonical fixture. The node is opt-in by
# workflow placement, so the user has already chosen to validate; pre-
# filling the canonical path keeps the widget usable for the common
# case (running the canonical workflow on this checkout).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WORKFLOW_PATH = _REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _load_workflow(path: str) -> dict[str, Any]:
    # Sprint E E5 / H5: explicit empty-string fallback. The shipped
    # workflow JSON ships the validator widget as "" (per the S29
    # Phase 1 cleanbreak that removed hardcoded `C:/Users/jeffr/...`
    # operator paths from the JSON surface). Pre-E5 this fell through
    # to `_DEFAULT_WORKFLOW_PATH` silently with no log line, leaving
    # soak diagnostics unable to tell whether the empty widget was
    # intentional or a wiring error. Post-E5 the fallback is explicit
    # and the resolved path is logged at INFO so the operator sees
    # which file actually got validated.
    if not path:
        log.info(
            "OTR_WorkflowValidator: workflow_json_path widget empty; "
            "resolved to canonical _DEFAULT_WORKFLOW_PATH=%s",
            _DEFAULT_WORKFLOW_PATH,
        )
        p = _DEFAULT_WORKFLOW_PATH
    else:
        p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"OTR_WorkflowValidator: workflow JSON not found at {p!r}"
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"OTR_WorkflowValidator: workflow JSON at {p!r} failed to parse: {e}"
        ) from e


class WorkflowValidator:
    """OTR workflow contract validator node.

    Side-effecting. Returns a one-line OK report on pass; raises on fail.
    """

    CATEGORY = "OldTimeRadio/diagnostics"
    DESCRIPTION = (
        "Opt-in execution-time workflow contract validator. Place as "
        "the first node in a workflow to catch contract drift at queue "
        "time. Reads the workflow JSON from disk and runs the same "
        "validate_workflow_contract check that runs in CI."
    )

    # Validator runs for its side effect even with no downstream consumer.
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("validation_report",)
    FUNCTION = "validate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json_path": ("STRING", {
                    "multiline": False,
                    "default": str(_DEFAULT_WORKFLOW_PATH),
                }),
                "validate_anyway": ("BOOLEAN", {"default": True}),
                "strict_unknown_types": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, workflow_json_path: str, validate_anyway: bool,
                   strict_unknown_types: bool) -> str:
        """Re-run on any change to the inputs OR to the workflow JSON
        on disk. mtime + path is the canonical change signal."""
        try:
            p = Path(workflow_json_path) if workflow_json_path else _DEFAULT_WORKFLOW_PATH
            mtime = p.stat().st_mtime_ns if p.is_file() else 0
        except OSError:
            mtime = 0
        return f"{workflow_json_path}|{mtime}|{validate_anyway}|{strict_unknown_types}"

    def validate(self, workflow_json_path: str,
                 validate_anyway: bool,
                 strict_unknown_types: bool):
        if not validate_anyway:
            msg = "OTR_WorkflowValidator: validate_anyway=False -- skipped."
            log.info(msg)
            return (msg,)

        from ._workflow_validation import validate_workflow_contract
        try:
            from .. import NODE_CLASS_MAPPINGS as _NCM  # type: ignore
        except (ImportError, ValueError):
            # Test environment: import the package root directly.
            try:
                import importlib
                _pkg = importlib.import_module(
                    "custom_nodes.ComfyUI-OldTimeRadio"
                )
                _NCM = getattr(_pkg, "NODE_CLASS_MAPPINGS", {})
            except Exception:
                _NCM = {}

        workflow = _load_workflow(workflow_json_path)
        # Raises a WorkflowValidationError subclass on first failure.
        validate_workflow_contract(
            workflow,
            _NCM,
            strict_unknown_types=strict_unknown_types,
        )
        n_nodes = len(workflow.get("nodes") or [])
        n_links = len(workflow.get("links") or [])
        msg = (
            f"OTR_WorkflowValidator: OK -- {n_nodes} nodes, {n_links} links, "
            f"strict_unknown_types={strict_unknown_types}, "
            f"path={workflow_json_path or str(_DEFAULT_WORKFLOW_PATH)!r}"
        )
        log.info(msg)
        return (msg,)


NODE_CLASS_MAPPINGS = {"OTR_WorkflowValidator": WorkflowValidator}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_WorkflowValidator": "OTR Workflow Validator (opt-in, S14.2)",
}

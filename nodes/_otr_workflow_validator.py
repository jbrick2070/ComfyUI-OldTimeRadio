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


def _resolve_workflow_path(path: str) -> Path:
    """GATE B S2 code-defect fix (2026-06-11, spec section 2 'verified ground
    truth'): non-empty RELATIVE paths used to resolve against the process CWD
    -- correct only by accident under ComfyUI Desktop and silently wrong for
    headless runs launched from any other directory (`IS_CHANGED` then hashed
    a phantom mtime=0, so on-disk edits never re-triggered validation).

    Resolution contract (shared by `_load_workflow` AND `IS_CHANGED`):
      * empty       -> the canonical `_DEFAULT_WORKFLOW_PATH` (explicit, logged
                       by _load_workflow -- the E5 behavior, kept);
      * relative    -> anchored at the REPO ROOT (never the process CWD);
      * absolute    -> taken as-is.
    """
    if not path:
        return _DEFAULT_WORKFLOW_PATH
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


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
    p = _resolve_workflow_path(path)
    if path and str(p) != path:
        log.info(
            "OTR_WorkflowValidator: relative workflow_json_path %r resolved "
            "against the repo root -> %s", path, p,
        )
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


# ---------------------------------------------------------------------------
# S14.3 widget-vector contract check (BUG-LOCAL-293)
# ---------------------------------------------------------------------------
# Mirrors scripts/otr_api.py::_serialized_slot_names so the validator agrees
# with the UI->API converter on how many widgets_values slots a node SHOULD
# carry. Two ComfyUI client-side serialization rules /object_info doesn't
# encode visibly: (1) forceInput=True widget inputs occupy NO slot;
# (2) an INT seed/noise_seed widget gains a hidden control_after_generate
# companion slot. Drift between this count and the saved widgets_values length
# is the BUG-210 (stale-slot shift) / BUG-253 (short-by-one) class.
_WIDGET_TYPE_NAMES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})
_COMPANION_INT_WIDGETS = ("seed", "noise_seed")


def _wv_type_of(spec: Any):
    if isinstance(spec, (list, tuple)) and len(spec) > 0:
        return spec[0]
    return spec


def _wv_is_widget_backed(spec: Any) -> bool:
    t = _wv_type_of(spec)
    if isinstance(t, (list, tuple)):  # COMBO choices (list OR tuple form) is a widget
        return True
    return isinstance(t, str) and t.upper() in _WIDGET_TYPE_NAMES


def _wv_has_force_input(spec: Any) -> bool:
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return False
    opts = spec[1]
    return isinstance(opts, dict) and bool(opts.get("forceInput"))


def _expected_slot_count(input_types: dict) -> int:
    """How many widgets_values slots a clean save should carry, per the
    otr_api serialized-slot rules (forceInput dropped, seed companion added)."""
    required = (input_types.get("required") or {})
    optional = (input_types.get("optional") or {})
    n = 0
    for name, spec in list(required.items()) + list(optional.items()):
        if not _wv_is_widget_backed(spec):
            continue
        if _wv_has_force_input(spec):
            continue
        n += 1
        t = _wv_type_of(spec)
        if isinstance(t, str) and t.upper() == "INT" and name in _COMPANION_INT_WIDGETS:
            n += 1  # hidden control_after_generate companion slot
    return n


def widget_vector_drift(workflow: dict, ncm: dict) -> list[str]:
    """Return human-readable drift findings for OTR nodes whose saved
    widgets_values length != expected serialized-slot count. OTR nodes only
    (others lack a loadable INPUT_TYPES here). Pure; never raises."""
    findings: list[str] = []
    for node in (workflow.get("nodes") or []):
        ntype = node.get("type")
        cls = ncm.get(ntype) if isinstance(ncm, dict) else None
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            continue
        wv = node.get("widgets_values")
        if not isinstance(wv, list):
            continue
        try:
            expected = _expected_slot_count(cls.INPUT_TYPES() or {})
        except Exception:
            continue
        if len(wv) != expected:
            findings.append(
                f"node {node.get('id')} {ntype}: widgets_values="
                f"{len(wv)} != expected {expected}"
            )
    return findings


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
            # GATE B S2 STAMP (the switchable-workflow decision doc, section
            # 4): the THREE stamp widgets are the ONLY new optional fields
            # allowed on this node. The MASTER ships them empty (unstamped);
            # emit_snapshot (S3) writes them on generated .gen.json tiers.
            # Node `properties` were rejected (not executable in API prompts).
            "optional": {
                "profile_id": ("STRING", {"multiline": False, "default": ""}),
                "master_hash": ("STRING", {"multiline": False, "default": ""}),
                "generated_by": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, workflow_json_path: str, validate_anyway: bool,
                   strict_unknown_types: bool, profile_id: str = "",
                   master_hash: str = "", generated_by: str = "") -> str:
        """Re-run on any change to the inputs OR to the workflow JSON
        on disk. mtime + path is the canonical change signal. Uses the
        SAME repo-root resolution as `_load_workflow` (GATE B S2 defect
        fix) so a relative path hashes the real file's mtime instead of
        a CWD-dependent phantom."""
        try:
            p = _resolve_workflow_path(workflow_json_path)
            mtime = p.stat().st_mtime_ns if p.is_file() else 0
        except OSError:
            mtime = 0
        return (f"{workflow_json_path}|{mtime}|{validate_anyway}|"
                f"{strict_unknown_types}|{profile_id}|{master_hash}|"
                f"{generated_by}")

    # ------------------------------------------------------------------ #
    # GATE B S2: stamp assertion + runtime env export (decision doc sec. 4)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_host() -> dict:
        """Cheap host reality: platform + CUDA presence + total VRAM (MB).
        Separated for testability (tests monkeypatch this)."""
        import platform as _platform
        sysname = _platform.system()
        info = {
            "platform": ("mac" if sysname == "Darwin"
                         else "win" if sysname == "Windows" else "any"),
            "has_cuda": False,
            "vram_mb": 0,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["has_cuda"] = True
                info["vram_mb"] = int(
                    torch.cuda.get_device_properties(0).total_memory
                    // (1024 * 1024))
        except Exception:  # noqa: BLE001 -- detection must never crash
            pass
        return info

    def _assert_stamp(self, workflow_json_path: str, profile_id: str,
                      master_hash: str, generated_by: str) -> str:
        """The S2 startup assertion: a STAMPED workflow must match the
        committed profile AND the detected host reality, else the prompt
        ABORTS with a reason -> suggestion table (no cuda -> cpu_floor;
        VRAM < 10 GB -> 8gb_lite; mac -> cpu_floor). ACTIVE whenever
        profile_id is non-empty; ``validate_anyway`` can NEVER skip it.
        On pass, exports the runtime env (every execution -- stale values
        from a previous prompt are overwritten; an operator-set conflicting
        ceiling warns LOUD and the SMALLER value wins)."""
        from ._otr_shared.capability_profiles import ProfileError, load_profile
        try:
            profile = load_profile(profile_id)
        except ProfileError as e:
            raise ValueError(
                f"OTR_WorkflowValidator: stamped profile_id {profile_id!r} "
                f"failed to load: {e}") from e

        host = self._detect_host()
        problems = []
        if profile["device_backend"] == "cuda" and not host["has_cuda"]:
            suggestion = "cpu_floor"
            problems.append(
                f"profile {profile_id!r} requires CUDA but this host has none"
                + (" (mac)" if host["platform"] == "mac" else "")
                + f" -> suggested tier: {suggestion}")
        elif (profile["device_backend"] == "cuda"
              and host["vram_mb"]
              and host["vram_mb"] < int(profile["vram_budget_mb"])):
            suggestion = ("8gb_lite" if host["vram_mb"] >= 7_300
                          else "cpu_floor")
            problems.append(
                f"profile {profile_id!r} budgets "
                f"{profile['vram_budget_mb']} MB VRAM but this host reports "
                f"{host['vram_mb']} MB -> suggested tier: {suggestion}")
        if profile["platform"] not in ("any", host["platform"]):
            problems.append(
                f"profile {profile_id!r} targets platform "
                f"{profile['platform']!r}; this host is "
                f"{host['platform']!r} -> suggested tier: cpu_floor")
        if problems:
            raise ValueError(
                "OTR_WorkflowValidator: STAMP ASSERTION FAILED (the stamped "
                "snapshot does not fit this machine; validate_anyway never "
                "skips this):\n  " + "\n  ".join(problems))

        # Runtime export -- EVERY execution, not "if unset" (a long-running
        # server persists env across prompts; stale values are overwritten).
        budget = int(profile["vram_budget_mb"])
        ceiling = budget
        cur = (os.environ.get("OTR_VRAM_CEILING_MB") or "").strip()
        if cur:
            try:
                cur_i = int(cur)
            except ValueError:
                cur_i = None
            if cur_i is not None and cur_i != budget:
                ceiling = min(cur_i, budget)
                log.warning(
                    "OTR_WorkflowValidator: OTR_VRAM_CEILING_MB=%s conflicts "
                    "with profile %r budget %d -- the SMALLER value (%d) "
                    "wins (LOUD)", cur, profile_id, budget, ceiling)
        os.environ["OTR_VRAM_CEILING_MB"] = str(ceiling)
        os.environ["OTR_ACTIVE_PROFILE"] = profile_id
        snapshot_hash = ""
        try:
            p = _resolve_workflow_path(workflow_json_path)
            if p.is_file():
                import hashlib
                snapshot_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            pass
        os.environ["OTR_SNAPSHOT_HASH"] = snapshot_hash
        msg = (f"stamp OK: profile={profile_id} ceiling={ceiling}MB "
               f"snapshot_sha={snapshot_hash[:12] or 'n/a'}"
               + (f" generated_by={generated_by}" if generated_by else ""))
        log.info("OTR_WorkflowValidator: %s (master_hash=%s)",
                 msg, master_hash[:12] or "n/a")
        return msg

    def validate(self, workflow_json_path: str,
                 validate_anyway: bool,
                 strict_unknown_types: bool,
                 profile_id: str = "",
                 master_hash: str = "",
                 generated_by: str = ""):
        # The stamp assertion + env export run FIRST whenever profile_id is
        # non-empty -- validate_anyway only skips the CONTRACT check below,
        # never this (decision doc section 4; CI rejects snapshots shipping
        # validate_anyway=false at S3).
        stamp_msg = ""
        if (profile_id or "").strip():
            stamp_msg = self._assert_stamp(
                workflow_json_path, profile_id.strip(),
                str(master_hash or ""), str(generated_by or ""))
        if not validate_anyway:
            msg = ("OTR_WorkflowValidator: validate_anyway=False -- contract "
                   "check skipped." + (f" {stamp_msg}" if stamp_msg else ""))
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
        # S14.3 (BUG-LOCAL-293): widget-vector contract check -- HARD GATE.
        # Each OTR node's saved widgets_values length must equal its INPUT_TYPES
        # serialized-slot count (forceInput slots dropped; INT seed/noise_seed
        # control_after_generate companion added; sockets + COMBO list/tuple
        # handled, per scripts/otr_api._serialized_slot_names). A mismatch is the
        # BUG-210 (stale-slot shift) / BUG-253 (short-by-one) / BUG-281 (stale
        # forceInput slot) realignment class -- ComfyUI maps later widgets to the
        # WRONG slots, silently shipping a mis-configured render. We RAISE here so
        # the run halts at queue time, before any model loads, naming the node.
        # validate_anyway=False bypasses the whole node if an operator needs to
        # run a deliberately-drifted workflow for diagnostics.
        drift = widget_vector_drift(workflow, _NCM)
        if drift:
            for d in drift:
                log.error("OTR_WorkflowValidator: WIDGET-VECTOR DRIFT -- %s", d)
            raise ValueError(
                "OTR_WorkflowValidator: widget-vector contract drift on "
                f"{len(drift)} node(s) -- " + "; ".join(drift) + ". A node's "
                "saved widgets_values length no longer matches its INPUT_TYPES "
                "serialized-slot count; ComfyUI would positionally map later "
                "widgets to the wrong slots. Fix the workflow JSON's node "
                "widgets_values (or set validate_anyway=False to bypass)."
            )
        n_nodes = len(workflow.get("nodes") or [])
        n_links = len(workflow.get("links") or [])
        msg = (
            f"OTR_WorkflowValidator: OK -- {n_nodes} nodes, {n_links} links, "
            f"strict_unknown_types={strict_unknown_types}, "
            f"path={workflow_json_path or str(_DEFAULT_WORKFLOW_PATH)!r}"
            f", widget_vector_drift=0"
            + (f" | {stamp_msg}" if stamp_msg else "")
        )
        log.info(msg)
        return (msg,)


NODE_CLASS_MAPPINGS = {"OTR_WorkflowValidator": WorkflowValidator}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_WorkflowValidator": "OTR Workflow Validator (opt-in, S14.2)",
}

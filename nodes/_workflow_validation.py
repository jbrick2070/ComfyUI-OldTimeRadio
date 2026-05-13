"""S14.1 -- Workflow contract validator (commit A: validator + test-only).

Asserts the workflow JSON matches every OTR node class's declared
``INPUT_TYPES()`` and contains no stale / forbidden surfaces. Six
independent checks, each surfacing as its own typed exception so
callers can handle granularly or catch the root for "any contract
violation".

Per the S10-S15 plan's S14.1 spec (with Q-D9 vote: WorkflowValidationError
root + 5 typed children, all ValueError subclasses) and Q-D10 vote
(half-measure: test-only this commit; auto-invoke on workflow load
lands one week later as commit B if false-positive count stays zero).
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Exception hierarchy (Q-D9: ValueError-rooted, 5 typed children)
# ---------------------------------------------------------------------------


class WorkflowValidationError(ValueError):
    """Root for all workflow contract violations.

    Catch this for "any contract violation"; catch a specific
    subclass for granular handling. Sublcasses ValueError so existing
    handlers that catch ValueError still see these errors -- no
    new-root surprise behavior."""


class WorkflowReservedLinkIDError(WorkflowValidationError):
    """A workflow link uses an ID from G5_RESERVED_LINK_IDS."""


class WorkflowInputSocketError(WorkflowValidationError):
    """A node has a rogue / forbidden / unwired-required input socket."""


class WorkflowWidgetDriftError(WorkflowValidationError):
    """A required input is unwired and has no widget value."""


class WorkflowDeletedNodeError(WorkflowValidationError):
    """The workflow uses a node type that has been deleted."""


class WorkflowUnknownNodeTypeError(WorkflowValidationError):
    """A node type starts with ``OTR_`` but is not in NODE_CLASS_MAPPINGS."""


# ---------------------------------------------------------------------------
# Reserved sets (S14 lockdown -- centralizes the constants the prior
# narrow S8.3 plan kept inline in test files)
# ---------------------------------------------------------------------------


# Link IDs reserved by FreezeCascade fanout. Adding a workflow link
# with one of these IDs would collide with the cascade's assignments.
G5_RESERVED_LINK_IDS = frozenset({111, 112})


# Node types that have been deleted and must not appear in any
# workflow JSON. Listing them here means a stale workflow surfaces
# at validation time rather than at runtime.
DELETED_NODE_TYPES = frozenset({
    "OTR_LLMDirector",         # deleted in S2 (commit 249bc06)
    "OTR_BarkTTS",             # legacy single-line node
    "OTR_SFXGenerator",        # legacy single-line node
    "OTR_VoiceRender",         # legacy aggregator
    "OTR_BatchKokoroGenerator",# replaced by OTR_KokoroAnnouncer
})


# Input socket names that have been retired from the wire-input
# vocabulary. Any node carrying one of these in its inputs[] is a
# stale wiring -- regardless of which class it belongs to.
FORBIDDEN_INPUT_SOCKETS = frozenset({
    "production_plan_json",   # the deleted Director's output socket
    "director_json",          # alternate Director-shape name
    "voice_map_json",         # Director's voice_assignments split-out
    "sfx_plan_json",          # Director's sfx_plan split-out
    "music_plan_json",        # Director's music_plan split-out
})


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _is_otr_type(type_name: str) -> bool:
    """Heuristic: only OTR_-prefixed node types are eligible for the
    deep INPUT_TYPES introspection. ComfyUI built-ins (CLIPLoader,
    UNETLoader, etc.) and other third-party nodes are passed through
    for the type-existence + deleted-node + forbidden-socket checks
    but skip the rogue-socket and unwired-required checks.

    This boundary is what keeps the validator focused on OTR's
    contract surface without dragging in the entire ComfyUI
    NODE_CLASS_MAPPINGS at test time.
    """
    return type_name.startswith("OTR_")


def validate_workflow_contract(
    workflow: dict,
    node_class_mappings: dict[str, Any],
    *,
    strict_unknown_types: bool = False,
) -> None:
    """Run all six checks against the workflow JSON.

    Raises the first violation it finds. Specific subclass per check
    type so callers can branch on the failure mode.

    ``node_class_mappings`` is the OTR registry (the dict exported
    from ``nodes/__init__.py``); only OTR_-prefixed types are
    introspected against it. Non-OTR types pass the unknown-type
    check unconditionally.

    ``strict_unknown_types`` (default False): when True, an
    OTR_-prefixed type missing from the mapping raises
    WorkflowUnknownNodeTypeError. When False (the test-runner
    default), missing OTR types are skipped for INPUT_TYPES
    introspection but no exception fires. This lets the validator
    run usefully in environments where not every OTR class can be
    imported (heavy optional deps, partial registry, etc.) while
    still catching the deliberate ungranted-type case via the
    explicit-opt-in adversarial test.
    """
    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []

    # --- Check 1: unknown OTR node types ----------------------------------
    if strict_unknown_types:
        for node in nodes:
            t = node.get("type") or ""
            if _is_otr_type(t) and t not in node_class_mappings:
                raise WorkflowUnknownNodeTypeError(
                    f"Node {node.get('id')} has OTR-prefixed type {t!r} "
                    f"that is not in NODE_CLASS_MAPPINGS. Either register "
                    f"the class or fix the workflow."
                )

    # --- Check 2 + 3: socket contract vs INPUT_TYPES (OTR types only) ----
    for node in nodes:
        t = node.get("type") or ""
        if not _is_otr_type(t):
            continue
        cls = node_class_mappings.get(t)
        if cls is None:
            continue  # already raised above; defensive
        try:
            declared = cls.INPUT_TYPES()
        except Exception as exc:
            raise WorkflowValidationError(
                f"{t}(id={node.get('id')}): INPUT_TYPES() raised "
                f"{type(exc).__name__}: {exc}"
            )
        decl_required = set((declared.get("required") or {}).keys())
        decl_optional = set((declared.get("optional") or {}).keys())
        decl_all = decl_required | decl_optional

        actual_inputs = node.get("inputs") or []
        actual = {i.get("name") for i in actual_inputs if isinstance(i, dict)}

        rogue = actual - decl_all
        if rogue:
            raise WorkflowInputSocketError(
                f"{t}(id={node.get('id')}): rogue sockets {sorted(rogue)} "
                f"not declared by INPUT_TYPES()."
            )

        # Required-input wiring check: every required-by-class input
        # must either have a link OR be widget-fulfilled. Widgets in
        # ComfyUI workflow JSON live in widgets_values (positional
        # list); this check uses a presence heuristic -- if the input
        # name appears in declared.required AND there's no link AND
        # no widget value present, it's drift.
        widget_values = node.get("widgets_values") or []
        # Heuristic: socket-only required inputs in declared.required
        # should appear in actual_inputs with a non-null link. If they
        # don't appear OR the link is null AND there's no widget for
        # them, raise WidgetDrift.
        for input_name in decl_required:
            wired = next(
                (i for i in actual_inputs
                 if isinstance(i, dict) and i.get("name") == input_name),
                None,
            )
            if wired is not None and wired.get("link") is not None:
                continue
            # Not wired -- needs a widget. ComfyUI's widget storage
            # is positional, so a strict positional-index check would
            # be brittle. We accept "widgets_values is non-empty" as
            # sufficient evidence the input has a widget value;
            # tighter positional pinning is a future enhancement.
            if widget_values:
                continue
            raise WorkflowWidgetDriftError(
                f"{t}(id={node.get('id')}): required input "
                f"{input_name!r} is unwired AND no widget value present."
            )

    # --- Check 4: deleted node types ---------------------------------
    for node in nodes:
        t = node.get("type") or ""
        if t in DELETED_NODE_TYPES:
            raise WorkflowDeletedNodeError(
                f"Deleted node type {t!r} present at id={node.get('id')}. "
                f"This type was retired; the workflow needs migration."
            )

    # --- Check 5: forbidden input sockets ----------------------------
    for node in nodes:
        for inp in (node.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            if inp.get("name") in FORBIDDEN_INPUT_SOCKETS:
                raise WorkflowInputSocketError(
                    f"{node.get('type')}(id={node.get('id')}): "
                    f"forbidden socket {inp.get('name')!r}. Names in "
                    f"FORBIDDEN_INPUT_SOCKETS are retired from the "
                    f"wire-input vocabulary."
                )

    # --- Check 6: link-table battery ---------------------------------
    if not links:
        return  # empty workflow / nothing to validate
    link_ids = []
    for L in links:
        if not isinstance(L, list) or len(L) < 5:
            raise WorkflowValidationError(
                f"Malformed link entry: {L!r}; expected list of >=5 elements."
            )
        link_ids.append(L[0])
    if len(link_ids) != len(set(link_ids)):
        seen: set = set()
        dups = []
        for lid in link_ids:
            if lid in seen:
                dups.append(lid)
            seen.add(lid)
        raise WorkflowValidationError(
            f"Duplicate link IDs present: {dups}"
        )
    last_link_id = workflow.get("last_link_id")
    if last_link_id is not None and last_link_id != max(link_ids):
        raise WorkflowValidationError(
            f"last_link_id={last_link_id} != max(links)={max(link_ids)}"
        )
    collision = set(link_ids) & G5_RESERVED_LINK_IDS
    if collision:
        raise WorkflowReservedLinkIDError(
            f"G5-reserved link IDs in workflow: {sorted(collision)}. "
            f"IDs in {sorted(G5_RESERVED_LINK_IDS)} are reserved by the "
            f"FreezeCascade fanout."
        )
    node_ids = {n.get("id") for n in nodes}
    for L in links:
        # Link tuple shape: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        if L[1] not in node_ids or L[3] not in node_ids:
            raise WorkflowValidationError(
                f"Orphan link {L[0]}: src/dst node missing "
                f"(src={L[1]}, dst={L[3]})"
            )


__all__ = [
    "WorkflowValidationError",
    "WorkflowReservedLinkIDError",
    "WorkflowInputSocketError",
    "WorkflowWidgetDriftError",
    "WorkflowDeletedNodeError",
    "WorkflowUnknownNodeTypeError",
    "G5_RESERVED_LINK_IDS",
    "DELETED_NODE_TYPES",
    "FORBIDDEN_INPUT_SOCKETS",
    "validate_workflow_contract",
]

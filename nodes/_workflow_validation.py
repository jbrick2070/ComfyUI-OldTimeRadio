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
    "OTR_LLMDirector",            # deleted in S2 (commit 249bc06)
    "OTR_BarkTTS",                # legacy single-line node
    "OTR_SFXGenerator",           # legacy single-line node
    "OTR_VoiceRender",            # legacy aggregator
    "OTR_BatchKokoroGenerator",   # replaced by OTR_KokoroAnnouncer
    "OTR_PostAudioVideoPipeline", # S27: subprocess HuMo trigger, superseded in-graph by OTR_BatchHumoRender + the (now also removed) legacy compositor
    # CW-4 legacy render-path teardown (2026-06-07): the legacy in-graph
    # compositor mixed audio with ffmpeg -shortest (forbidden by the
    # frozen-audio spine). Replaced by SignalLostVideo -> OTR_SilentComposite
    # -> OTR_MasterAudioMux (terminal mux, -c:a copy, no -shortest).
    "OTR_VideoComposite",
    # CW cleanbreak (2026-06-08): the legacy in-graph BATCH render path is
    # retired -- the FLUX/HuMo/LTX batch renderers + the VRAM-unload node +
    # the LTX topology gate. Replaced by the model-agnostic video platform
    # (OTR_VideoRenderBatch + the registry adapters). Tombstoned so a stale
    # workflow JSON naming one of these fails loudly at validation.
    "OTR_BatchHumoRender",
    "OTR_BatchLTXRender",
    "OTR_BatchFluxRender",
    "OTR_BatchFluxPortraitRender",
    "OTR_UnloadAll",
    "OTR_LtxBranchGate",
    # CW cleanbreak follow-up (2026-06-08): the HuMo tier loader only fed the
    # deleted batch HuMo node; the in-process eng_humo adapter loads its own.
    "OTR_HuMoTierLoader",
    # Lean-down 2026-05-29 (step 6): the dormant Story Room writers'-room
    # cluster (Director / Editor / Story Room / Extract / Commit) was
    # deleted. Tombstoned so a stale workflow JSON referencing one of
    # them fails loudly at validation instead of at runtime.
    "OTR_StoryRoom",
    "OTR_StoryRoomExtract",
    "OTR_StoryRoomCommit",
    "OTR_DirectorBrief",
    "OTR_EditorPass",
    # Lean-down 2026-05-29 (step 12): temporary BUG-LOCAL-231 bisect
    # STRING-emit node; deleted with its _bisect_flux_*.json graphs.
    "OTR_BisectStringSource",
    # Lean-down 2026-05-29 (step 7): the shadow-pass Stage 1 fan-out +
    # best-of-N beat-selector diagnostic nodes were deleted with the
    # shadow/fan-out cluster.
    "OTR_Stage1FanOut",
    "OTR_BeatSelector",
    # Chunk E cleanbreak (2026-06-08): legacy video-plan nodes superseded
    # by the model-agnostic platform (OTR_ShotLock owns all budget/plan
    # logic; OTR_VideoRenderBatch replaces the per-model batch renderers).
    # Tombstoned so a stale workflow JSON naming these fails at validation.
    "OTR_VideoPlan",           # superseded by OTR_ShotLock
    "OTR_RenderPlan",          # superseded by OTR_ShotLock
    "OTR_ShotDurationCalculator",  # old name, renamed to OTR_FixedShotDurationStub
                                   # then fully superseded by ShotLock budget
    "OTR_FixedShotDurationStub",   # stub replacement, now superseded
    # Chunk E cleanbreak completion (2026-06-09): the legacy gate-bound
    # loader shells + the FLUX topology gate are retired. V-5 (execution
    # plan): ALL model loading is adapter-internal via comfy
    # model_management -- no deferred-loader shell nodes survive. Their
    # only consumers were the legacy FLUX/LTX batch chains deleted in the
    # CW cleanbreak; the platform render path (OTR_VideoRenderBatch +
    # registry adapters) loads everything inside the adapters.
    "OTR_FluxBranchGate",              # legacy FLUX topology gate (Sprint H)
    "OTR_DeferredCheckpointLoader",    # gate-bound FLUX loader shell (V-5)
    "OTR_DeferredLtxTextEncoderLoader",  # gate-bound LTX loader shell (V-5)
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
    # rip-sfx-broll (2026-07-01): the SceneSequencer sfx overlay inputs
    # were removed with the sfx subsystem (they were never wired to a
    # producer). A stale workflow carrying them fails at validation.
    "sfx_audio_clips",
    "sfx_offset_ms",
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

        # --- Check 3: widget-drift (S16.3: positional + None-aware) ------
        # ComfyUI's widget storage is positional: widgets_values[i]
        # corresponds to the i-th entry in INPUT_TYPES().required that
        # is NOT also a wired socket. We walk decl_required in declared
        # order and consume widget_values in the same order.
        #
        # S16.3 plan deviation: the original spec said empty string
        # "" also counts as unfilled. ComfyUI's INPUT_TYPES.required
        # routinely declares ``("STRING", {"default": ""})`` for
        # fields like ``episode_title`` where blank means "auto-
        # derive at runtime". Failing on bare "" here breaks every
        # such node. The validator is for CONTRACT violations, not
        # operational nits; per-field "must be non-empty" checks
        # belong in the node's runtime ``generate()``.
        #
        # The remaining contract guarantees:
        #   - the widgets_values list has at least as many slots as
        #     decl_required has unwired entries (else, widget
        #     storage drift -- node will crash at runtime)
        #   - slot is not explicit None (Comfy treats None as the
        #     "this socket got dropped" marker)
        #
        # Python 3.7+ dict insertion order is guaranteed, so iterating
        # decl_required gives the same order ComfyUI uses when
        # materializing widget slots.
        widget_values = node.get("widgets_values") or []
        widget_iter = iter(widget_values)
        for input_name in decl_required:
            wired = next(
                (i for i in actual_inputs
                 if isinstance(i, dict) and i.get("name") == input_name),
                None,
            )
            if wired is not None and wired.get("link") is not None:
                continue  # wired -- no widget consumed
            # Not wired -- must have a widget slot at this position.
            try:
                slot = next(widget_iter)
            except StopIteration:
                raise WorkflowWidgetDriftError(
                    f"{t}(id={node.get('id')}): required input "
                    f"{input_name!r} unwired AND widgets_values "
                    f"exhausted before this slot."
                )
            if slot is None:
                raise WorkflowWidgetDriftError(
                    f"{t}(id={node.get('id')}): required input "
                    f"{input_name!r} unwired AND widget slot is "
                    f"explicit None (Comfy uses None for dropped "
                    f"sockets; widget storage drift)."
                )

    # --- Check 4: deleted node types ---------------------------------
    for node in nodes:
        t = node.get("type") or ""
        if t in DELETED_NODE_TYPES:
            raise WorkflowDeletedNodeError(
                f"Deleted node type {t!r} present at id={node.get('id')}. "
                f"This type was retired; the workflow needs migration."
            )

    # --- Check 5: forbidden input sockets (extended in S16.2) ------------
    # Scans the four user-facing surfaces where Director-era names can
    # survive a rename: socket name, nested widget name, node title, and
    # the Save/Restore identifier in properties. The original check only
    # read inp.get("name") -- the workflow JSON discovered three nested
    # widget.name and title hits in production (S16.1) that the narrow
    # version missed.
    for node in nodes:
        surfaces: list[tuple[str, Any]] = []
        for inp in (node.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            surfaces.append(("socket", inp.get("name")))
            widget = inp.get("widget") or {}
            if isinstance(widget, dict):
                surfaces.append(("widget", widget.get("name")))
        surfaces.append(("title", node.get("title")))
        sr_name = (node.get("properties") or {}).get("Node name for S&R")
        surfaces.append(("s_and_r", sr_name))
        for kind, val in surfaces:
            if val and val in FORBIDDEN_INPUT_SOCKETS:
                raise WorkflowInputSocketError(
                    f"{node.get('type')}(id={node.get('id')}): "
                    f"forbidden name {val!r} on {kind!r} surface. "
                    f"Names in FORBIDDEN_INPUT_SOCKETS are retired from "
                    f"every user-facing surface, not just the socket."
                )

    # --- Check 6: link-table battery ---------------------------------
    if not links:
        return  # empty workflow / nothing to validate
    link_ids = []
    for L in links:
        # S16.5 (IMP-22): ComfyUI's link tuple is 6 elements:
        # [link_id, src_node, src_slot, dst_node, dst_slot, type]
        # The original threshold (>=5) passed degenerate 5-tuples
        # missing the trailing type tag.
        if not isinstance(L, list) or len(L) < 6:
            raise WorkflowValidationError(
                f"Malformed link entry: {L!r}; expected list of >=6 "
                f"elements [link_id, src_node, src_slot, dst_node, "
                f"dst_slot, type]."
            )
        link_ids.append(L[0])
    # S16.5 (IMP-23): dedup duplicate-ID accumulator. An ID present
    # 3 times reported as [42, 42] under the old manual loop.
    from collections import Counter
    counts = Counter(link_ids)
    dups = sorted(lid for lid, n in counts.items() if n > 1)
    if dups:
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

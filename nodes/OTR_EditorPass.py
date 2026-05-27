"""nodes/OTR_EditorPass.py -- Editor agent ComfyUI node (DORMANT).

Sprint 10B Wave 1 Agent E. Per
docs/2026-05-26-good-story-writer-architecture__02_design.md Section
4 Wave 1 Agent E:

    Owns. New files: _otr_editor_pass.py, nodes/OTR_EditorPass.py,
    tests/test_otr_editor_pass.py. Owns the EditorVerdict dataclass
    (section 3.2).
    Frozen against. Agent D's DirectorBrief shape (section 3.1).

This node is registered (NODE_CLASS_MAPPINGS) but DORMANT in Wave 1
-- the workflow JSON gains an entry with the director_brief +
writer_draft sockets DISCONNECTED. Wave 2 Agent F (Story Room loop)
wires the Editor into the writer's revision cycles. Today's pipeline
is unchanged.

PD6 satisfied: no model_id widget on this node. The technical-slot
model id arrives over the wire from the writer's broadcast
`technical_model` output socket. An unwired socket raises
MissingModelInputError loud at run-time -- the recovery is
graph-level (connect the writer), not a code fallback.

PD3 satisfied: workflow JSON gains a new node entry.
LLM slot: technical (constrained-decode verdict).
"""

from __future__ import annotations

import json
import logging
import time


log = logging.getLogger("OTR")


__all__ = ["OTR_EditorPass"]


class OTR_EditorPass:
    """OTR Editor Pass -- in-loop rubric verdict for the Writer's draft.

    Inputs (all optional/forceInput so Wave 1 can ship dormant):
        director_brief    Serialized DirectorBrief JSON (Section 3.1)
                          from Agent D's OTR_DirectorBrief node. Wired
                          by Wave 2 Agent F; disconnected today.
        writer_draft      The current Writer draft as one prose
                          string. Wired by Wave 2 Agent F.
        cycle             0-indexed Editor pass number for this
                          episode (0..3 expected; widget pinned 0..3).
        technical_model   Resolved technical_model id from the
                          writer's broadcast output (PD6). No local
                          widget; an unwired socket raises
                          MissingModelInputError loud.

    Outputs (1 slot):
        editor_verdict    Serialized EditorVerdict JSON
                          (`EditorVerdict.to_dict()` then
                          `json.dumps`). Consumed by Wave 2 Agent F.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("editor_verdict",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_brief": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Serialized DirectorBrief JSON (Section 3.1) "
                        "from the Director node. Wire from Agent D's "
                        "OTR_DirectorBrief output socket. Wave 1 leaves "
                        "this disconnected; Wave 2 Agent F wires the "
                        "Story Room loop into the Editor."
                    ),
                }),
                "writer_draft": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The current Writer draft as one prose string. "
                        "Wave 1 leaves this disconnected; Wave 2 Agent F "
                        "wires the Story Room loop's Writer turn output "
                        "into this socket."
                    ),
                }),
                "cycle": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "tooltip": (
                        "0-indexed Editor pass number for this episode. "
                        "Section 4 caps the loop at 3 Editor cycles."
                    ),
                }),
            },
            "optional": {
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the writer's "
                        "broadcast output (PD6). No local widget; the "
                        "Editor must be wired into the writer's "
                        "technical_model output socket to receive its "
                        "verdict-LLM id. Validated at run-time via "
                        "_otr_model_inputs.require_model -- an unwired "
                        "socket raises MissingModelInputError loud."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Dormant in Wave 1; mark always-changed so Wave 2 wiring sees
        # a fresh run every queue (the Editor verdict depends on the
        # live Writer draft, which the loop re-emits each cycle).
        return time.time()

    def run(
        self,
        director_brief: str = "",
        writer_draft: str = "",
        cycle: int = 0,
        technical_model: str = "",
    ):
        # Lazy imports keep node-load cheap and let pytest collect
        # this module without dragging in torch / lm-format-enforcer.
        from . import _otr_constrained_generate as _OTRCG
        from . import _otr_model_inputs as _OTRMI
        from . import _otr_model_loader as _OTRML
        from ._otr_editor_pass import (
            EditorCallFailedError,
            EditorVerdict,
            EditorVerdictSchema,
            run_editor,
        )

        # Wave 1 dormant guard: if either upstream socket is empty,
        # return an empty pass-through verdict so workflow validation
        # passes even with the sockets disconnected. Wave 2 connects
        # them.
        brief_str = (director_brief or "").strip()
        draft_str = (writer_draft or "").strip()
        if not brief_str or not draft_str:
            log.info(
                "[OTR_EditorPass] dormant pass-through "
                "(brief_empty=%s draft_empty=%s); Wave 2 Agent F "
                "wires the live loop.",
                not brief_str, not draft_str,
            )
            placeholder = EditorVerdict(
                pass_decision=True,
                failing_axes=[],
                per_axis_notes={},
                overall_note=(
                    "OTR_EditorPass dormant pass-through: upstream "
                    "director_brief or writer_draft socket is empty. "
                    "Wave 2 Agent F wires the Story Room loop into "
                    "this node."
                ),
                cycle=int(cycle or 0),
            )
            return (json.dumps(placeholder.to_dict(), ensure_ascii=False),)

        # Live path: resolve the technical_model id and run the call.
        # LLM slot: technical -- structured constrained-decode verdict
        # per project rule 6. The Editor consumes the writer's
        # broadcast `technical_model` socket per the Section 2
        # writers'-room routing rule.
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        generate_fn = _OTRCG.make_constrained_generate_fn(
            cache_entry, EditorVerdictSchema,
        )

        try:
            verdict = run_editor(
                director_brief=brief_str,
                writer_draft=draft_str,
                generate_fn=generate_fn,
                cycle=int(cycle or 0),
            )
        except EditorCallFailedError as exc:
            log.warning(
                "[OTR_EditorPass] run_editor exhausted retry budget: "
                "%s", exc,
            )
            # Wave 2 Agent F decides loop-level recovery; here we
            # surface a typed verdict that names the failure rather
            # than silently passing.
            failure_verdict = EditorVerdict(
                pass_decision=False,
                failing_axes=[],
                per_axis_notes={},
                overall_note=(
                    f"EditorCallFailedError after {exc.attempts} "
                    f"attempt(s): {type(exc.last_error).__name__}: "
                    f"{exc.last_error}"
                ),
                cycle=int(cycle or 0),
            )
            return (
                json.dumps(failure_verdict.to_dict(), ensure_ascii=False),
            )

        return (
            json.dumps(verdict.to_dict(), ensure_ascii=False),
        )

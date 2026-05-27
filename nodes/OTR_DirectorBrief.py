"""nodes/OTR_DirectorBrief.py -- Sprint 10B Wave 1 Agent D (2026-05-27).

Comfy node wrapper around the Director module
(`nodes/_otr_director_brief.py`). Takes a news seed + comma-separated
cast names + the writer's technical_model id (over a STRING input
socket; no widget), and emits a serialized DirectorBrief JSON string
on its single output.

DORMANT in Wave 1: the workflow JSON entry lands with the output
socket disconnected. Wave 2 Agent F (Story Room loop) wires the
Director's output into the Writer's loop. Today's pipeline reads
nothing from this node.

Design references:
- `docs/2026-05-26-good-story-writer-architecture__02_design.md`
  Section 4 Wave 1 Agent D + Section 3.1 DirectorBrief dataclass.

PD3 (workflow JSON): this node is appended to
`workflows/otr_scifi_16gb_full.json` with the output socket
disconnected; downstream consumers land in Wave 2.
PD6 (LLM-slot tagging): no `model_id` widget on this node; the
`technical_model` STRING input socket carries the model id from the
writer's broadcast output. The slot acquisition call site is tagged
`# LLM slot: technical` per project rule 6.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

import json
import logging
from typing import Any


log = logging.getLogger("OTR")


__all__ = ["OTR_DirectorBrief"]


class OTR_DirectorBrief:
    """Director agent -- emits a structured DirectorBrief for one episode.

    Inputs (all required-or-socket; no `model_id` widget per PD6):
      news_seed         STRING (multiline) -- the news premise the
                        episode is about. Plain prose.
      cast_names        STRING (single line) -- comma-separated cast
                        names. The Director uses these names verbatim
                        in the brief's `opposed_desires` field.
      technical_model   STRING (force-input socket) -- the resolved
                        technical_model id from
                        OTR_LedgerScriptWriter.technical_model. An
                        unwired socket raises MissingModelInputError
                        loud (PD6 contract).

    Output (1 slot):
      director_brief    STRING -- a JSON object serialized from the
                        `DirectorBrief` dataclass via `.to_dict()`.
                        Downstream agents (Wave 2 Agent F's Story Room)
                        rebuild a `DirectorBrief` via
                        `DirectorBrief(**json.loads(payload))`.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("director_brief",)

    @classmethod
    def INPUT_TYPES(cls):
        # PD6: no `model_id` widget. The technical_model STRING input is
        # force-input only and arrives from the writer's broadcast
        # output. The two free-form text widgets carry the news seed and
        # the comma-separated cast list -- both small surfaces that test
        # cleanly without the full writer pipeline.
        return {
            "required": {
                "news_seed": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The news premise the episode is about. Plain "
                        "prose, one or two sentences. The Director will "
                        "name this premise concretely in the brief so "
                        "the Writer cannot drift into a generic frame."
                    ),
                }),
                "cast_names": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Comma-separated cast names (e.g. "
                        "'REN BLACK, DR. MAEVE COLE'). The Director "
                        "uses these names verbatim in the brief's "
                        "opposed_desires field."
                    ),
                }),
            },
            "optional": {
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the writer's "
                        "broadcast output (PD6). No local widget; the "
                        "Director node must be wired into the writer's "
                        "technical_model output socket. Validated at "
                        "run-time via _otr_model_inputs.require_model -- "
                        "an unwired socket raises "
                        "MissingModelInputError loud."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Director output is non-deterministic by design (the LLM picks
        # the frame). Re-run every queue so the operator sees the new
        # brief; the brief is the canonical reproducibility anchor for
        # the Story Room (PD7 relaxation, design Section 10 Q6).
        import time as _t
        return _t.time()

    def run(
        self,
        news_seed: str = "",
        cast_names: str = "",
        technical_model: str = "",
    ):
        # Lazy imports keep node-load cheap. Mirror the pattern used by
        # OTR_LedgerFreezeCascade: _otr_model_loader for slot
        # acquisition, _otr_model_inputs for socket validation,
        # _otr_constrained_generate for the JSON-schema-bound closure,
        # and the local Director module for the call entry point.
        from . import _otr_model_loader as _OTRML
        from . import _otr_model_inputs as _OTRMI
        from ._otr_constrained_generate import make_constrained_generate_fn
        from ._otr_director_brief import (
            DirectorBrief,
            DirectorBriefSchema,
            DirectorCallFailedError,
            run_director,
        )

        # Normalize the cast list: comma-separated, trimmed, drop empties.
        cast_list = [
            piece.strip()
            for piece in (cast_names or "").split(",")
            if piece.strip()
        ]

        log.info(
            "[OTR_DirectorBrief] news_seed=%d chars, %d cast name(s); "
            "requesting technical slot.",
            len(news_seed or ""),
            len(cast_list),
        )

        # PD6: fail loud if the technical_model socket is unwired. The
        # recovery is graph-level (connect the writer's
        # technical_model broadcast output).
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )

        # LLM slot: technical
        # Reason: the Director's brief is a structured constrained-
        # decode pass (JSON object validated against
        # DirectorBriefSchema). Per project rule 6 all structured-JSON
        # passes route to the writer's technical slot; the model id
        # arrives via the STRING input socket above.
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )

        generate_fn = make_constrained_generate_fn(
            cache_entry, DirectorBriefSchema,
        )

        # Run the Director call. On a clean run we serialize the
        # DirectorBrief to JSON and emit it on the output socket. On a
        # retry-budget exhaustion (DirectorCallFailedError) Wave 1 ships
        # a typed sentinel payload so the workflow does not crash the
        # queue: the node is DORMANT, no downstream consumer reads it
        # yet, and a hard raise here would block unrelated diagnostic
        # graphs from rendering. Wave 2 Agent F's Story Room loop will
        # decide whether the Director failure short-circuits the loop
        # or falls through to a manual director brief; for now the
        # sentinel keeps the queue alive and stamps the failure reason
        # on the JSON payload for forensic inspection.
        try:
            brief = run_director(
                news_seed=news_seed or "",
                cast_names=cast_list,
                generate_fn=generate_fn,
            )
            payload = brief.to_dict()
            payload["status"] = "ok"
        except DirectorCallFailedError as exc:
            log.warning(
                "[OTR_DirectorBrief] run_director exhausted retry budget "
                "after %d attempt(s); emitting failure sentinel "
                "(last error: %s).",
                exc.attempts, exc.last_error,
            )
            empty = DirectorBrief(
                news_premise="",
                dramatic_question="",
                opposed_desires="",
                arc_shape="",
                audience_feel="",
                forbidden_drift=[],
                raw_brief="",
            )
            payload = empty.to_dict()
            payload["status"] = "failed"
            payload["error"] = (
                f"{type(exc.last_error).__name__}: {exc.last_error}"
                if exc.last_error is not None
                else "no error captured"
            )

        return (json.dumps(payload, ensure_ascii=False, indent=2),)

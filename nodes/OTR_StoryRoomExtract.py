"""nodes/OTR_StoryRoomExtract.py -- Sprint 10B Wave 2 Agent G (2026-05-27).

ComfyUI node wrapper around
`_otr_story_room_extract.extract_from_transcript`. Consumes a
serialized StoryRoomTranscript (Wave 2 Agent F output) and emits a
serialized StoryRoomExtraction whose dict-shaped fields match the
keys downstream pipeline nodes already consume (per design Section
3.4 + the Section 4 Wave 2 cross-check).

DORMANT in Wave 2: when the upstream payload is empty / `dormant` /
malformed, the node returns a sentinel without calling the LLM. When
the upstream payload is a real transcript, the node runs the
extraction.

PD6: no `model_id` widget. The technical-slot model id arrives over a
STRING forceInput socket from the writer's broadcast output. An
unwired socket raises MissingModelInputError at run-time only when
the extraction path is actually taken (the dormant pass-through does
not need it).

PD3: workflow JSON wires Story Room (Wave 2 Agent F's
OTR_StoryRoom.story_room_transcript) into this node's
story_room_transcript input. The output socket ships disconnected --
Wave 3's bridge wires the downstream consumers when the operator
A/B confirms the room ships on by default.

LLM slot: technical (constrained-decode extraction).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any


log = logging.getLogger("OTR")


__all__ = ["OTR_StoryRoomExtract"]


class OTR_StoryRoomExtract:
    """Story Room Extract -- transcript-to-structured pass.

    Inputs (required):
        story_room_transcript  STRING (multiline) -- serialized
                               StoryRoomTranscript JSON from
                               OTR_StoryRoom. Wave 2 wires the socket;
                               an empty / dormant payload triggers the
                               dormant pass-through.

    Inputs (optional / forceInput):
        cast_names             STRING -- writers'-room cast names,
                               comma-separated. When present, the
                               extractor preserves the canonical names
                               verbatim.
        news_seed              STRING (multiline) -- the original news
                               premise. Helps the extractor fill the
                               `premise` field.
        technical_model        STRING (forceInput) -- the resolved
                               technical_model id from the writer's
                               broadcast output. PD6: no local widget.
                               Validated at run-time via
                               _otr_model_inputs.require_model when
                               the extraction path is taken.

    Output (1 slot):
        story_room_extraction  STRING -- serialized StoryRoomExtraction
                               JSON (`StoryRoomExtraction.to_dict()`
                               then `json.dumps`). Downstream
                               consumers (announcer / continuity
                               ledger / music+SFX / bark) read the
                               canonical ledger today; Wave 3's
                               bridge will map this extraction into
                               those consumers when use_story_room
                               flips on by default.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story_room_extraction",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_room_transcript": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Serialized StoryRoomTranscript JSON from "
                        "OTR_StoryRoom. Wave 2 wires this socket; "
                        "an empty / 'dormant' payload triggers the "
                        "dormant pass-through (no LLM call)."
                    ),
                }),
            },
            "optional": {
                "cast_names": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Comma-separated Director-locked cast names "
                        "(e.g. 'REN BLACK, DR. MAEVE COLE'). When "
                        "present the extractor preserves the names "
                        "verbatim."
                    ),
                }),
                "news_seed": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The original news premise plain prose. Used "
                        "to fill the `premise` field when the Writer's "
                        "draft does not name it concretely."
                    ),
                }),
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the writer's "
                        "broadcast output (PD6). No local widget; "
                        "validated at run-time via "
                        "_otr_model_inputs.require_model. The dormant "
                        "pass-through does not need this socket."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Extraction is deterministic given the transcript + cast +
        # seed inputs, but the upstream Story Room is non-deterministic
        # by design (PD7 relaxed per Section 10 Q6). Always-changed so
        # the operator sees a fresh extraction every queue.
        return time.time()

    def _build_dormant_payload(self, reason: str) -> str:
        """Return the JSON payload for the no-op path.

        Downstream consumers (when Wave 3 wires them) recognise
        `status == 'dormant'` as the pass-through marker and fall
        back to the legacy Stage 1 plan + ledger outputs. PD1 holds.
        """
        return json.dumps(
            {
                "status": "dormant",
                "reason": reason,
                "cast": [],
                "beats": [],
                "dialogue": [],
                "audio_cues": [],
                "running_facts": [],
                "arc": None,
                "premise": "",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _parse_transcript(self, transcript_json: str):
        """Deserialize a StoryRoomTranscript JSON payload.

        Returns the parsed dict OR None when the payload is empty /
        malformed / a `dormant` sentinel / a `failed` sentinel.
        The Extract node treats any None return as a dormant pass-
        through (no LLM call).
        """
        text = (transcript_json or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "[OTR_StoryRoomExtract] transcript is not valid JSON: %s",
                str(exc)[:200],
            )
            return None
        if not isinstance(payload, dict):
            log.warning(
                "[OTR_StoryRoomExtract] transcript JSON root must be an "
                "object, got %s.", type(payload).__name__,
            )
            return None
        status = payload.get("status")
        if status in ("dormant", "failed"):
            log.info(
                "[OTR_StoryRoomExtract] upstream transcript stamped "
                "status='%s'; pass-through.",
                status,
            )
            return None
        # A real transcript must have a non-empty final_draft for the
        # extractor to have anything to transcribe.
        draft = str(payload.get("final_draft") or "").strip()
        if not draft:
            log.info(
                "[OTR_StoryRoomExtract] upstream transcript has no "
                "final_draft; pass-through.",
            )
            return None
        return payload

    def _parse_cast(self, cast_names: str):
        return [
            piece.strip()
            for piece in (cast_names or "").split(",")
            if piece.strip()
        ]

    def run(
        self,
        story_room_transcript: str = "",
        cast_names: str = "",
        news_seed: str = "",
        technical_model: str = "",
    ):
        transcript_payload = self._parse_transcript(story_room_transcript)
        if transcript_payload is None:
            return (self._build_dormant_payload(
                "no usable StoryRoomTranscript on the input socket"
            ),)

        # Lazy imports keep node-load cheap.
        from . import _otr_constrained_generate as _OTRCG
        from . import _otr_model_inputs as _OTRMI
        from . import _otr_model_loader as _OTRML
        from ._otr_story_room_extract import (
            ExtractionCallFailedError,
            StoryRoomExtractionSchema,
            extract_from_transcript,
        )

        # PD6: fail loud if the technical_model socket is unwired.
        # Recovery is graph-level (connect the writer's broadcast).
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )

        # LLM slot: technical
        # Reason: structured constrained-decode extraction per
        # project rule 6.
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        generate_fn = _OTRCG.make_constrained_generate_fn(
            cache_entry, StoryRoomExtractionSchema,
        )

        try:
            extraction = extract_from_transcript(
                transcript_payload,
                generate_fn=generate_fn,
                cast_names=self._parse_cast(cast_names),
                news_seed=news_seed or "",
            )
        except ExtractionCallFailedError as exc:
            log.warning(
                "[OTR_StoryRoomExtract] extract_from_transcript "
                "exhausted retry budget after %d attempt(s); emitting "
                "failure sentinel (last error: %s).",
                exc.attempts, exc.last_error,
            )
            payload: dict[str, Any] = {
                "status": "failed",
                "reason": (
                    f"ExtractionCallFailedError after {exc.attempts} "
                    f"attempt(s); last error: "
                    f"{type(exc.last_error).__name__}: {exc.last_error}"
                    if exc.last_error is not None else
                    f"ExtractionCallFailedError after {exc.attempts} "
                    f"attempt(s); no error captured."
                ),
                "cast": [],
                "beats": [],
                "dialogue": [],
                "audio_cues": [],
                "running_facts": [],
                "arc": None,
                "premise": "",
            }
            return (json.dumps(payload, ensure_ascii=False, indent=2),)

        payload = extraction.to_dict()
        payload["status"] = "ok"
        return (json.dumps(payload, ensure_ascii=False, indent=2),)

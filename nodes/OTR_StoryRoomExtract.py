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
        from . import _otr_ledger as _OTRL
        from . import _otr_model_inputs as _OTRMI
        from . import _otr_model_loader as _OTRML
        from ._otr_story_room_extract import (
            DialogueOnlySchema,
            ExtractionCallFailedError,
            extract_dialogue_only,
        )

        # PD6: fail loud if the technical_model socket is unwired.
        # Recovery is graph-level (connect the writer's broadcast).
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )

        # Sprint 1 keystone (2026-05-28). Read the voice slot ids
        # from the in-flight ledger lines -- the ledger is the
        # source of truth post-init_lines_from_outline. Extract no
        # longer regenerates cast / beats / arc / premise /
        # running_facts; the consumer (OTR_StoryRoomCommit) joins
        # rows by dialogue_slot_id and raises StoryRoomCommitError
        # on any integrity violation.
        led_path = _OTRL.in_flight_ledger_path()
        ledger = (
            _OTRL.load_ledger_safe(led_path)
            if led_path is not None else None
        )
        if not isinstance(ledger, dict):
            log.warning(
                "[OTR_StoryRoomExtract] no in-flight ledger; "
                "pass-through dormant sentinel."
            )
            return (self._build_dormant_payload(
                "no in-flight ledger to read voice slot ids from"
            ),)
        ledger_lines = ledger.get("lines") or []
        voice_slot_ids: list[str] = [
            str(ln.get("dialogue_slot_id") or "").strip()
            for ln in ledger_lines
            if str(ln.get("dialogue_slot_id") or "").strip()
        ]
        if not voice_slot_ids:
            log.warning(
                "[OTR_StoryRoomExtract] in-flight ledger lacks "
                "dialogue_slot_id on its lines (pre-Sprint-1 "
                "shape). Pass-through dormant sentinel so the "
                "operator regenerates Stage 1 before committing."
            )
            return (self._build_dormant_payload(
                "ledger lines pre-date Sprint 1; missing "
                "dialogue_slot_id"
            ),)

        # LLM slot: technical
        # Reason: structured constrained-decode extraction per
        # project rule 6.
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        generate_fn = _OTRCG.make_constrained_generate_fn(
            cache_entry, DialogueOnlySchema,
        )

        # Auto-resolve cast + news_seed from in-flight ledger when
        # the widgets are empty (2026-05-27).
        from ._otr_writers_room_resolver import (
            resolve_cast_names as _resolve_cast,
            resolve_news_seed as _resolve_seed,
        )
        resolved_cast = _resolve_cast(cast_names) or self._parse_cast(cast_names)
        resolved_seed = _resolve_seed(news_seed)
        try:
            dialogue_rows = extract_dialogue_only(
                transcript_payload,
                generate_fn=generate_fn,
                voice_slot_ids=voice_slot_ids,
                cast_names=resolved_cast,
                news_seed=resolved_seed,
            )
        except ExtractionCallFailedError as exc:
            log.warning(
                "[OTR_StoryRoomExtract] extract_dialogue_only "
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

        # ---- Build 2 (2026-05-28): deterministic Tier-A craft floor ----
        # Run the format/integrity gate on the SAME rows Commit will join,
        # BEFORE the ledger is touched. On any Tier-A failure emit the
        # node's existing status="failed" sentinel so OTR_StoryRoomCommit
        # hard-fails under commit=True (integrity over recovery -- no
        # silent fallback to legacy lines). Deterministic; no LLM; no new
        # node/widget -> no workflow JSON change (build2/WIRING_SPEC.md
        # Option A + Placement 1).
        from ._otr_craft_floor import evaluate_tier_a, normalize_slot_line

        craft_raw = "\n".join(
            normalize_slot_line(
                r.get("dialogue_slot_id"), r.get("speaker"), r.get("text")
            )
            for r in dialogue_rows
        )
        # Manifest slot-id sequence is the ledger's voiced order -- the
        # exact sequence OTR_StoryRoomCommit validates position-by-position
        # (_commit_dialogue raises on draft_slot_ids != voice_slot_ids), so
        # the count / order / empty checks mirror Commit and yield zero
        # false-reds on a known-good draft. Speaker is paired from the
        # draft row per slot, making SPEAKER_MISMATCH inert: the in-flight
        # ledger LINE persists char_id + speaker_role, NOT the speaker
        # NAME, and the live Commit join does not validate speaker either.
        # Promoting speaker to a live check needs a slot->name resolver
        # confirmed against a live extract (operator N=3 follow-up). The
        # one genuinely new hard-fail here is BELOW_WORD_FLOOR (floor 4);
        # beats target 16-24 words so it only catches degenerate output.
        _draft_speaker_by_sid = {
            str(r.get("dialogue_slot_id") or "").strip(): str(
                r.get("speaker") or ""
            ).strip()
            for r in dialogue_rows
            if str(r.get("dialogue_slot_id") or "").strip()
        }
        craft_manifest = [
            {"slot_id": sid, "speaker": _draft_speaker_by_sid.get(sid, "")}
            for sid in voice_slot_ids
        ]
        craft_verdict = evaluate_tier_a(craft_raw, craft_manifest)
        if not craft_verdict.passed:
            log.warning(
                "[OTR_StoryRoomExtract] Tier-A craft floor REJECTED the "
                "extract (%d failure(s): %s); emitting failure sentinel "
                "so Commit fails loud rather than writing degraded lines.",
                len(craft_verdict.failures),
                ", ".join(craft_verdict.failure_codes),
            )
            payload = {
                "status": "failed",
                "reason": (
                    "Tier-A craft floor: "
                    + ", ".join(craft_verdict.failure_codes)
                ),
                "tier_a": craft_verdict.to_dict(),
                "cast": [],
                "beats": [],
                "dialogue": [],
                "audio_cues": [],
                "running_facts": [],
                "arc": None,
                "premise": "",
            }
            return (json.dumps(payload, ensure_ascii=False, indent=2),)
        # ---- end Build 2 craft floor ----

        # Reassemble the StoryRoomExtraction-shaped payload from the
        # in-flight ledger (cast / beats / running_facts / premise)
        # plus the LLM-extracted dialogue rows. Downstream consumers
        # see the same shape they always have; only the LLM's job
        # got narrower.
        meta = ledger.get("meta") or {}
        cast_rows = ledger.get("cast") or []
        beat_rows = ledger.get("beats") or []
        running_facts = (
            meta.get("running_facts")
            or (ledger.get("running_facts") or [])
        )
        premise_str = (
            (meta.get("news") or {}).get("script_brief")
            if isinstance(meta.get("news"), dict) else None
        ) or meta.get("episode_title") or ""
        payload = {
            "status": "ok",
            "cast": list(cast_rows),
            "beats": list(beat_rows),
            "dialogue": dialogue_rows,
            "audio_cues": [],
            "running_facts": list(running_facts),
            "arc": meta.get("arc"),
            "premise": str(premise_str or ""),
        }
        return (json.dumps(payload, ensure_ascii=False, indent=2),)

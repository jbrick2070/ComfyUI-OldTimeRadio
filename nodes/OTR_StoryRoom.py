"""nodes/OTR_StoryRoom.py -- Sprint 10B Wave 2 Agent F (2026-05-27).

ComfyUI node wrapper around `_otr_story_room.run_story_room`. Wires
the Director (Wave 1 Agent D, upstream), the Writer (Mistral-Nemo via
the writer's creative_writing_model slot), and the Editor (Wave 1
Agent E, embedded) into a bounded conversation that converges on a
publishable Writer draft.

Design references:
- docs/2026-05-26-good-story-writer-architecture__02_design.md
  Section 1 architecture, Section 3.3 StoryRoomTranscript contract,
  Section 4 Wave 2 Agent F, Section 7 PD7 reproducibility relaxation
  under use_story_room=True.

Wave 2 lands this node behind `use_story_room: bool = False`. The
node ships dormant: when use_story_room is False the node returns an
empty StoryRoomTranscript and the downstream Extract node (Wave 2
Agent G) recognises the empty payload as a pass-through. When True
the node runs the loop and emits the populated transcript.

PD6 (LLM-slot tagging): no `model_id` widget on this node. Both slot
ids arrive over `forceInput` STRING sockets from the writer's broadcast
outputs:
  creative_writing_model  -> Writer turns (free-form prose).
  technical_model         -> Editor turns (constrained verdict).

PD3 (workflow JSON): Wave 2 wires this node downstream of
OTR_DirectorBrief and upstream of OTR_StoryRoomExtract (Wave 2 Agent G,
landing next commit).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, List


log = logging.getLogger("OTR")


__all__ = ["OTR_StoryRoom"]


class OTR_StoryRoom:
    """Story Room loop -- emits a StoryRoomTranscript for one episode.

    Inputs:
      director_brief        STRING (multiline) -- serialized DirectorBrief
                            JSON from OTR_DirectorBrief. Empty/missing
                            triggers the dormant pass-through.
      news_seed             STRING (multiline) -- the real news premise.
                            Pass through from the news interpreter.
      cast_names            STRING (single line) -- comma-separated cast
                            names. Used verbatim in dialogue
                            attributions.
      use_story_room        BOOLEAN (widget, default False) -- master
                            switch. False -> dormant pass-through;
                            True -> run the writers' room loop.
      max_writer_turns      INT (widget, default 4) -- per-cycle Writer
                            turn cap (cycle 0 only -- revision cycles
                            get one turn each).
      max_editor_cycles     INT (widget, default 3) -- hard cap on
                            Editor cycles.
      max_total_turns       INT (widget, default 18) -- hard cap on
                            total turns across all roles.
      creative_writing_model  STRING (forceInput) -- resolved
                            creative_writing_model id from the writer.
                            No local widget; an unwired socket raises
                            MissingModelInputError when the loop runs.
      technical_model       STRING (forceInput) -- resolved
                            technical_model id from the writer.
                            Same PD6 contract as the Director / Editor
                            nodes.

    Output (1 slot):
      story_room_transcript STRING -- serialized StoryRoomTranscript
                            JSON (StoryRoomTranscript.to_dict() then
                            json.dumps). Wave 2 Agent G's
                            OTR_StoryRoomExtract consumes this payload.
                            When use_story_room is False the payload is
                            a `{"status": "dormant", ...}` sentinel that
                            the Extract node passes through.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("story_room_transcript",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_brief": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Serialized DirectorBrief JSON from "
                        "OTR_DirectorBrief. Wave 2 wires this socket; "
                        "Wave 1 left the Director node dormant."
                    ),
                }),
                "news_seed": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "The real news premise the episode is about. "
                        "Pass-through from the news interpreter."
                    ),
                }),
                "cast_names": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Comma-separated cast names. Used verbatim in "
                        "dialogue attributions inside the Writer's "
                        "drafts."
                    ),
                }),
                "use_story_room": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Master switch. False -> dormant pass-through "
                        "(PD1 byte-identity for the legacy audio "
                        "path). True -> run the writers' room loop "
                        "(Director -> Writer -> Editor with bounded "
                        "revision)."
                    ),
                }),
                "max_writer_turns": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "Per-cycle Writer turn cap. Cycle 0 gets up "
                        "to this many drafts before the first Editor "
                        "pass; later cycles get one revision turn "
                        "each."
                    ),
                }),
                "max_editor_cycles": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "tooltip": (
                        "Hard cap on Editor cycles. Per Section 1 the "
                        "design ships with 3 cycles; higher values "
                        "let an operator soak the loop without "
                        "rebuilding the node."
                    ),
                }),
                "max_total_turns": ("INT", {
                    "default": 18,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "tooltip": (
                        "Hard cap on total turns across all roles "
                        "(Director + Writer + Editor). Per Section 1 "
                        "the design ships at 18."
                    ),
                }),
            },
            "optional": {
                "creative_writing_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved creative_writing_model id from the "
                        "writer's broadcast output (PD6). The Writer "
                        "turns use this slot. An unwired socket "
                        "raises MissingModelInputError when "
                        "use_story_room is True; the dormant pass-"
                        "through does not need it."
                    ),
                }),
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the "
                        "writer's broadcast output (PD6). The Editor "
                        "turns use this slot. Same dormant-pass-"
                        "through exemption as creative_writing_model."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The Story Room is intentionally non-deterministic when active
        # (PD7 relaxed per design Section 10 Q6 -- the transcript is the
        # new reproducibility anchor). Always-changed so the operator
        # gets a fresh room every queue when use_story_room is True; the
        # dormant pass-through is byte-identical regardless.
        return time.time()

    def _build_dormant_payload(self, reason: str) -> str:
        """Return the JSON payload for the use_story_room=False path.

        The Extract node (Wave 2 Agent G) recognises a `status` key of
        `'dormant'` as a pass-through marker and emits the legacy
        Stage-1-plan outputs unchanged. PD1 byte-identity holds.
        """
        return json.dumps(
            {
                "status": "dormant",
                "reason": reason,
                "director_brief": None,
                "turns": [],
                "editor_verdicts": [],
                "final_draft": "",
                "terminated_clean": False,
                "total_turns": 0,
                "elapsed_seconds": 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _parse_director_brief(self, brief_json: str):
        """Deserialize a DirectorBrief JSON payload into a dataclass.

        Returns the populated DirectorBrief OR None when the payload is
        empty / malformed / a failure sentinel. The Story Room treats
        any None return as "no usable brief -> dormant pass-through".
        """
        from ._otr_director_brief import DirectorBrief

        text = (brief_json or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "[OTR_StoryRoom] director_brief is not valid JSON: %s",
                str(exc)[:200],
            )
            return None
        if not isinstance(payload, dict):
            log.warning(
                "[OTR_StoryRoom] director_brief JSON root must be an "
                "object, got %s.", type(payload).__name__,
            )
            return None
        if payload.get("status") == "failed":
            log.warning(
                "[OTR_StoryRoom] upstream director brief stamped "
                "status='failed'; treating as dormant pass-through.",
            )
            return None
        # Strip the 'status'/'error' wrapping keys that the Director
        # node adds before serialization; DirectorBrief itself only
        # carries the Section 3.1 fields.
        clean = {k: v for k, v in payload.items() if k not in ("status", "error")}
        try:
            return DirectorBrief(**clean)
        except TypeError as exc:
            log.warning(
                "[OTR_StoryRoom] director_brief payload does not match "
                "DirectorBrief shape: %s", str(exc)[:200],
            )
            return None

    def _parse_cast(self, cast_names: str) -> List[str]:
        """Comma-separated cast names -> list[str]."""
        return [
            piece.strip()
            for piece in (cast_names or "").split(",")
            if piece.strip()
        ]

    def run(
        self,
        director_brief: str = "",
        news_seed: str = "",
        cast_names: str = "",
        use_story_room: bool = False,
        max_writer_turns: int = 4,
        max_editor_cycles: int = 3,
        max_total_turns: int = 18,
        creative_writing_model: str = "",
        technical_model: str = "",
    ):
        # Dormant pass-through. Wave 2 lands the node with this widget
        # OFF by default so PD1 byte-identity holds in the legacy
        # pipeline. Wave 3 operator A/B flips it on.
        if not use_story_room:
            log.info(
                "[OTR_StoryRoom] use_story_room=False -> dormant "
                "pass-through. PD1 holds; downstream Extract emits "
                "legacy Stage-1 outputs unchanged.",
            )
            return (self._build_dormant_payload(
                "use_story_room widget is False"
            ),)

        # Parse the Director brief. A missing / malformed / failed
        # brief drops us back to the dormant path so a single upstream
        # hiccup does not crash the queue.
        brief = self._parse_director_brief(director_brief)
        if brief is None:
            log.info(
                "[OTR_StoryRoom] no usable DirectorBrief upstream; "
                "dormant pass-through. (use_story_room was True but "
                "the brief did not parse.)",
            )
            return (self._build_dormant_payload(
                "no usable DirectorBrief on the director_brief input"
            ),)

        cast = self._parse_cast(cast_names)

        # Lazy imports so node-load is cheap and pytest collect does
        # not drag in torch / lm-format-enforcer for unrelated suites.
        from . import _otr_constrained_generate as _OTRCG
        from . import _otr_model_inputs as _OTRMI
        from . import _otr_model_loader as _OTRML
        from ._otr_editor_pass import EditorVerdictSchema
        from ._otr_story_room import (
            StoryRoomCallFailedError,
            run_story_room,
        )

        # PD6: both slot sockets must be wired. The recovery is graph-
        # level (connect the writer's broadcast outputs).
        resolved_creative_id = _OTRMI.require_model(
            creative_writing_model, slot="creative",
        )
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )

        # LLM slot: creative
        # Reason: Writer turns are free-form prose drafts of the
        # episode body. Per project rule 6 creative narrative passes
        # route to the writer's `creative_writing_model` slot.
        writer_cache = _OTRML.request_slot(
            "creative", resolved_creative_id,
        )
        writer_generate_fn = _OTRML.make_generate_fn(writer_cache)

        # LLM slot: technical
        # Reason: Editor verdicts are JSON objects validated against
        # EditorVerdictSchema. Per project rule 6 structured-JSON
        # passes route to the writer's `technical_model` slot.
        editor_cache = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        editor_generate_fn = _OTRCG.make_constrained_generate_fn(
            editor_cache, EditorVerdictSchema,
        )

        log.info(
            "[OTR_StoryRoom] running room: cast=%d brief.news_premise="
            "%d chars; budgets writer=%d editor=%d total=%d.",
            len(cast),
            len(brief.news_premise or ""),
            max_writer_turns,
            max_editor_cycles,
            max_total_turns,
        )

        try:
            transcript = run_story_room(
                director_brief=brief,
                news_seed=news_seed or "",
                cast=cast,
                writer_generate_fn=writer_generate_fn,
                editor_generate_fn=editor_generate_fn,
                rubric=None,
                max_writer_turns=int(max_writer_turns),
                max_editor_cycles=int(max_editor_cycles),
                max_total_turns=int(max_total_turns),
            )
        except StoryRoomCallFailedError as exc:
            log.warning(
                "[OTR_StoryRoom] room produced no usable draft after "
                "%d attempt(s); emitting failure sentinel (last error: "
                "%s).",
                exc.attempts, exc.last_error,
            )
            payload: dict[str, Any] = {
                "status": "failed",
                "reason": (
                    f"StoryRoomCallFailedError after {exc.attempts} "
                    f"attempt(s); last error: "
                    f"{type(exc.last_error).__name__}: {exc.last_error}"
                    if exc.last_error is not None else
                    f"StoryRoomCallFailedError after {exc.attempts} "
                    f"attempt(s); no error captured."
                ),
                "director_brief": brief.to_dict(),
                "turns": [],
                "editor_verdicts": [],
                "final_draft": "",
                "terminated_clean": False,
                "total_turns": 0,
                "elapsed_seconds": 0.0,
            }
            return (json.dumps(payload, ensure_ascii=False, indent=2),)

        payload = transcript.to_dict()
        payload["status"] = (
            "ok" if transcript.terminated_clean else "ok_dirty"
        )
        return (json.dumps(payload, ensure_ascii=False, indent=2),)

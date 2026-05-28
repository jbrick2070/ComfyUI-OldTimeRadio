"""nodes/OTR_StoryRoomCommit.py -- Wave 3 bridge (2026-05-27).

Bridges the Story Room writers'-room loop's output into the
in-flight ledger so the freeze cascade + Bark see the new
dialogue instead of the legacy compose_line output.

Reads `story_room_extraction` (Wave 2 Agent G output -- a
serialized StoryRoomExtraction JSON with the `dialogue` list per
Section 3.4: `[{"beat_id", "speaker", "text"}, ...]`) and walks
the rows, calling `Ledger.update_line_text(beat_id, text)` on the
in-flight ledger singleton for each.

DORMANT by default (`commit: bool = False`). Operator opts in when
the Wave 3 listen-test A/B confirms Story Room ships better than
legacy. Until then this node passes the writer's `script_json`
through unchanged and the cascade reads the legacy lines.

When `commit=True` AND extraction.status == 'ok':
    1. Resolve the in-flight ledger path
       (`_otr_ledger.in_flight_ledger_path`).
    2. Load the ledger from disk.
    3. Walk `extraction.dialogue`; for each row with a non-empty
       `text`, find the matching `lines[*]` by `beat_id` and
       overwrite `text` + `char_count` + `word_count`. Skip rows
       whose beat_id doesn't resolve (defensive -- the Story Room
       transcript may not align beat-for-beat with the legacy
       outline).
    4. Save the ledger back to disk.
    5. Stamp `meta.story_room_commit` with audit data
       (rows_attempted / rows_committed / rows_skipped /
       skipped_beat_ids).

Output (1 slot): `script_json` passthrough so downstream wiring
(the existing writer -> freeze cascade STRING link) doesn't need
to change.

PD1: never raises. Any failure logs a warning, stamps a marker on
meta, and the writer's `script_json` passes through unchanged --
the legacy lines stay in the ledger and the cascade renders them.

PD3: workflow JSON wires this node BETWEEN the writer's
script_json output and the freeze cascade's script_json input,
plus the Extract output socket into the commit's
story_room_extraction input. Two new links.

PD6: no LLM call site. Pure deterministic ledger surgery.

UTF-8 no BOM. No em-dashes (Windows cp1252 decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional


log = logging.getLogger("OTR")


__all__ = ["OTR_StoryRoomCommit", "StoryRoomCommitError"]


class StoryRoomCommitError(RuntimeError):
    """Sprint 1 keystone (2026-05-28).

    Raised when the slot-id join cannot land cleanly -- count
    mismatch, order mismatch, empty text on a voiced slot, or
    pre-Sprint-1 ledger that lacks `dialogue_slot_id` on its lines.
    The node turns the ComfyUI graph red rather than silently falling
    back to legacy compose output. Plan principle: integrity over
    recovery; every prompt improvement is fake progress until the
    Story Room draft demonstrably lands on every voiced row.
    """


class OTR_StoryRoomCommit:
    """Wave 3 bridge -- commit Story Room dialogue into the in-flight
    ledger so the freeze cascade sees Story Room output.

    Inputs (required):
        script_json    STRING (passthrough from OTR_LedgerScriptWriter).
                       Returned verbatim on the output socket so
                       downstream wiring is unchanged.
        story_room_extraction
                       STRING -- serialized StoryRoomExtraction JSON
                       from OTR_StoryRoomExtract. Empty / 'dormant'
                       / 'failed' status triggers the no-op
                       pass-through.
        commit         BOOLEAN (widget, default False) -- master
                       switch. False keeps the legacy lines in the
                       ledger. True overwrites them from the
                       Story Room extraction.

    Output (1 slot):
        script_json    STRING -- the input script_json verbatim.
                       The actual ledger write happens as a
                       side-effect on the in-flight singleton; the
                       socket is just topology glue.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script_json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Passthrough from OTR_LedgerScriptWriter. "
                        "Returned verbatim on the output socket."
                    ),
                }),
                "story_room_extraction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Serialized StoryRoomExtraction JSON from "
                        "OTR_StoryRoomExtract. Empty / 'dormant' / "
                        "'failed' status triggers no-op pass-through."
                    ),
                }),
                "commit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Wave 3 commit switch. False -> ledger is "
                        "untouched (PD1 byte-identity for the legacy "
                        "compose path). True -> walk "
                        "extraction.dialogue and overwrite "
                        "ledger.lines[*].text per beat_id."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always-changed so the operator sees a fresh commit pass
        # every queue (the upstream Story Room is non-deterministic
        # by design when active; PD7 relaxed when commit=True).
        return time.time()

    def _parse_extraction(
        self, payload_json: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the extraction dict OR None when it's a no-op marker.

        Empty input, malformed JSON, status='dormant', status='failed',
        and no 'dialogue' rows all return None -> commit becomes a
        no-op pass-through.
        """
        text = (payload_json or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "[OTR_StoryRoomCommit] extraction is not valid JSON: "
                "%s -- pass-through.", str(exc)[:200],
            )
            return None
        if not isinstance(data, dict):
            log.warning(
                "[OTR_StoryRoomCommit] extraction root must be an "
                "object, got %s -- pass-through.",
                type(data).__name__,
            )
            return None
        status = data.get("status")
        if status in ("dormant", "failed"):
            log.info(
                "[OTR_StoryRoomCommit] upstream extraction stamped "
                "status='%s'; pass-through.", status,
            )
            return None
        dialogue = data.get("dialogue") or []
        if not isinstance(dialogue, list) or not dialogue:
            log.info(
                "[OTR_StoryRoomCommit] extraction has no dialogue "
                "rows; pass-through.",
            )
            return None
        return data

    def _commit_dialogue(
        self,
        ledger_dict: Dict[str, Any],
        dialogue_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Sprint 1 keystone (2026-05-28). Walk dialogue rows keyed by
        `dialogue_slot_id` and overwrite the matching ledger line's
        text. Fail loud on any integrity violation.

        Failure modes that raise StoryRoomCommitError (no silent
        fallback to legacy lines, no silent skip):
          * The ledger lines lack `dialogue_slot_id` entirely
            (pre-Sprint-1 ledger; operator must regenerate Stage 1).
          * Draft row count != voice slot count.
          * Draft slot id sequence != voice slot id sequence.
          * Any draft row carries empty text on a voiced slot.

        Returns the proof block stamped on
        `meta.story_room_commit`:
          {commit_mode, draft_rows, voice_slots, rows_committed,
           rows_skipped (always 0 on success), fallback_to_legacy
           (always False), committed_slot_ids}.

        PURE on the ledger_dict (mutates in place); the caller is
        responsible for saving.
        """
        lines = ledger_dict.get("lines") or []
        voice_lines = [
            ln for ln in lines
            if str(ln.get("dialogue_slot_id") or "").strip()
        ]
        voice_slot_ids = [
            str(ln.get("dialogue_slot_id") or "").strip()
            for ln in voice_lines
        ]
        if not voice_slot_ids:
            raise StoryRoomCommitError(
                "no ledger lines carry `dialogue_slot_id` -- ledger "
                "pre-dates Sprint 1 keystone. Regenerate the outline "
                "so init_lines_from_outline stamps slot ids before "
                "committing Story Room dialogue."
            )

        draft_slot_ids: List[str] = []
        for row in dialogue_rows:
            if not isinstance(row, dict):
                raise StoryRoomCommitError(
                    f"dialogue row is not a dict: {type(row).__name__}"
                )
            sid = str(row.get("dialogue_slot_id") or "").strip()
            if not sid:
                raise StoryRoomCommitError(
                    "draft row missing dialogue_slot_id "
                    f"(row keys: {sorted(row.keys())[:8]}). The narrow "
                    "extract_dialogue_only path emits one per row; "
                    "verify OTR_StoryRoomExtract is on the narrow path."
                )
            draft_slot_ids.append(sid)

        if len(draft_slot_ids) != len(voice_slot_ids):
            raise StoryRoomCommitError(
                f"slot count mismatch: draft={len(draft_slot_ids)} "
                f"voice={len(voice_slot_ids)}. The narrow Extract "
                "path is supposed to emit exactly one row per voiced "
                "slot. Operator: inspect the Extract attempt logs."
            )
        if draft_slot_ids != voice_slot_ids:
            head_draft = ", ".join(draft_slot_ids[:8])
            head_voice = ", ".join(voice_slot_ids[:8])
            raise StoryRoomCommitError(
                f"slot order mismatch: draft=[{head_draft}...] "
                f"voice=[{head_voice}...]. Slot ids must align "
                "position-by-position; the Extract LLM reordered "
                "rows or hallucinated a slot id."
            )

        idx = {sid: ln for sid, ln in zip(voice_slot_ids, voice_lines)}
        committed: List[str] = []
        for row in dialogue_rows:
            sid = str(row["dialogue_slot_id"]).strip()
            text = str(row.get("text") or "").strip()
            if not text:
                raise StoryRoomCommitError(
                    f"empty text for slot {sid}; fail loud rather "
                    "than write an empty ledger line."
                )
            target = idx[sid]
            target["text"] = text
            target["char_count"] = len(text)
            # Match the writer's word count convention: tokenize
            # on letters + apostrophes (mirrors
            # `production_ledger._word_count`).
            import re as _re
            target["word_count"] = len(_re.findall(
                r"[A-Za-z][A-Za-z0-9'\-]*", text,
            ))
            committed.append(sid)

        return {
            "commit_mode": "dialogue_slot_order",
            "draft_rows": len(draft_slot_ids),
            "voice_slots": len(voice_slot_ids),
            "rows_committed": len(committed),
            "rows_skipped": 0,
            "fallback_to_legacy": False,
            "committed_slot_ids": committed,
        }

    def run(
        self,
        script_json: str = "",
        story_room_extraction: str = "",
        commit: bool = False,
    ):
        # Dormant pass-through: commit OFF or extraction missing.
        if not commit:
            log.info(
                "[OTR_StoryRoomCommit] commit=False -- pass-through "
                "(legacy lines stay in ledger). PD1 holds.",
            )
            return (script_json,)

        extraction = self._parse_extraction(story_room_extraction)
        if extraction is None:
            return (script_json,)

        # Lazy imports.
        try:
            from . import _otr_ledger as _OTRL
        except ImportError as exc:
            log.warning(
                "[OTR_StoryRoomCommit] could not import _otr_ledger: "
                "%s -- pass-through.", exc,
            )
            return (script_json,)

        led_path = _OTRL.in_flight_ledger_path()
        if led_path is None:
            log.warning(
                "[OTR_StoryRoomCommit] no in-flight ledger path "
                "resolved; pass-through.",
            )
            return (script_json,)

        ledger = _OTRL.load_ledger_safe(led_path)
        if ledger is None:
            log.warning(
                "[OTR_StoryRoomCommit] could not load ledger from "
                "%s; pass-through.", led_path,
            )
            return (script_json,)

        # Sprint 1 keystone (2026-05-28): StoryRoomCommitError is the
        # red-graph signal. The plan's commit gate ("5/5 episodes prove
        # full commit") relies on operators seeing the failure rather
        # than silently shipping legacy compose output. Any non-commit
        # exception (e.g. unrelated bug) still falls through the broad
        # except and pass-throughs, preserving the pre-Sprint-1 PD1
        # contract for legacy code paths.
        dialogue_rows = extraction.get("dialogue") or []
        try:
            audit = self._commit_dialogue(ledger, dialogue_rows)
        except StoryRoomCommitError as exc:
            meta = ledger.setdefault("meta", {})
            meta["story_room_commit"] = {
                "enabled": True,
                "committed": False,
                "commit_mode": "dialogue_slot_order",
                "rows_committed": 0,
                "rows_skipped": len(dialogue_rows),
                "fallback_to_legacy": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                _OTRL.save_ledger_safe(led_path, ledger)
            except Exception:  # noqa: BLE001
                pass
            log.error(
                "[OTR_StoryRoomCommit] StoryRoomCommitError -- HALTING "
                "(no silent fallback to legacy lines): %s",
                exc,
            )
            raise
        except Exception as exc:  # noqa: BLE001 -- defensive
            log.warning(
                "[OTR_StoryRoomCommit] commit raised %s: %s -- "
                "ledger left unchanged.",
                type(exc).__name__, str(exc)[:200],
            )
            return (script_json,)

        # Stamp meta with the audit trail.
        meta = ledger.setdefault("meta", {})
        meta["story_room_commit"] = {
            "enabled": True,
            "committed": True,
            "premise": (extraction.get("premise") or "")[:300],
            **audit,
        }
        # Recompute totals so downstream consumers see consistent
        # numbers. word_count + char_count totals are the only
        # roll-ups the freeze cascade reads.
        try:
            total_words = sum(
                int(ln.get("word_count") or 0)
                for ln in (ledger.get("lines") or [])
                if not ln.get("skip")
            )
            total_chars = sum(
                int(ln.get("char_count") or 0)
                for ln in (ledger.get("lines") or [])
                if not ln.get("skip")
            )
            ledger["total_word_count"] = total_words
            ledger["total_char_count"] = total_chars
        except Exception:  # noqa: BLE001
            # Totals are nice-to-have; don't fail the commit on
            # arithmetic edge.
            pass

        saved = _OTRL.save_ledger_safe(led_path, ledger)
        if not saved:
            log.warning(
                "[OTR_StoryRoomCommit] save_ledger_safe returned "
                "False; pass-through.",
            )
            return (script_json,)

        log.info(
            "[OTR_StoryRoomCommit] Sprint 1 commit: rewrote %d/%d "
            "voiced line(s) by dialogue_slot_id (skipped=%d, "
            "fallback_to_legacy=%s). Ledger saved to %s.",
            audit["rows_committed"],
            audit["voice_slots"],
            audit["rows_skipped"],
            audit["fallback_to_legacy"],
            led_path,
        )
        return (script_json,)

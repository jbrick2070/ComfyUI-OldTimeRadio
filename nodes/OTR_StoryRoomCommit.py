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


__all__ = ["OTR_StoryRoomCommit"]


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
        """Walk dialogue rows and overwrite ledger.lines[*].text by
        beat_id match. Returns a per-row audit dict.

        PURE on the ledger_dict (mutates in place); the caller is
        responsible for saving.
        """
        lines = ledger_dict.get("lines") or []
        # Index lines by beat_id for O(1) lookup.
        idx: Dict[str, Dict[str, Any]] = {}
        for ln in lines:
            bid = str(ln.get("beat_id") or "").strip()
            if bid:
                idx[bid] = ln

        attempted = 0
        committed_ids: List[str] = []
        skipped: List[Dict[str, str]] = []

        for row in dialogue_rows:
            if not isinstance(row, dict):
                continue
            attempted += 1
            bid = str(row.get("beat_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not bid:
                skipped.append({
                    "row": json.dumps(row)[:120],
                    "reason": "missing beat_id",
                })
                continue
            if not text:
                skipped.append({
                    "beat_id": bid,
                    "reason": "empty text",
                })
                continue
            target_line = idx.get(bid)
            if target_line is None:
                skipped.append({
                    "beat_id": bid,
                    "reason": "no matching line in legacy ledger",
                })
                continue
            target_line["text"] = text
            target_line["char_count"] = len(text)
            # Match the writer's word count convention: tokenize
            # on letters + apostrophes (mirrors
            # `production_ledger._word_count`).
            import re as _re
            target_line["word_count"] = len(_re.findall(
                r"[A-Za-z][A-Za-z0-9'\-]*", text,
            ))
            committed_ids.append(bid)

        return {
            "rows_attempted": attempted,
            "rows_committed": len(committed_ids),
            "rows_skipped": len(skipped),
            "committed_beat_ids": committed_ids,
            "skipped": skipped,
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

        try:
            dialogue_rows = extraction.get("dialogue") or []
            audit = self._commit_dialogue(ledger, dialogue_rows)
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
            "[OTR_StoryRoomCommit] BUG-WAVE3 commit: rewrote %d/%d "
            "line(s) from Story Room extraction (skipped=%d). "
            "Ledger saved to %s.",
            audit["rows_committed"],
            audit["rows_attempted"],
            audit["rows_skipped"],
            led_path,
        )
        return (script_json,)

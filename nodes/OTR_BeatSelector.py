"""nodes/OTR_BeatSelector.py -- Sprint 4.1 wire-up (2026-05-28).

ComfyUI wrapper around `_otr_beat_selector.select_winning_beat_sheet`.
Accepts up to 3 candidate Stage1Plan JSON payloads, runs the pure-
Python mechanical selector, and emits the winning plan JSON plus a
serialized audit dict the operator can inspect.

PD6: no `model_id` widget. The selector is pure Python -- no LLM
call -- so no slot socket either. This is one of the few OTR nodes
that legitimately has zero model dependency; the selector is
deterministic mechanical scoring.

Wiring (operator decides):
  upstream Stage 1 fan-out -> 3 STRING outputs (candidate_a JSON,
  candidate_b JSON, candidate_c JSON) -> this node's inputs ->
  winning_plan_json output goes wherever the legacy single-plan
  output went.

PD7 reproducibility: the selector is fully deterministic given
candidate inputs (ties broken by lowest index); same inputs ->
same winner -> same audit.

UTF-8 no BOM. Lazy imports keep node-load cheap.
"""
from __future__ import annotations

import json
import logging
from typing import Any, List

log = logging.getLogger("OTR")


__all__ = ["OTR_BeatSelector"]


class OTR_BeatSelector:
    """Sprint 4 best-of-N selector ComfyUI node wrapper.

    Inputs (required):
        candidate_a_json   STRING -- serialized Stage1Plan JSON.

    Inputs (optional):
        candidate_b_json   STRING -- serialized Stage1Plan JSON.
                                     Empty triggers single-candidate
                                     mode (selector still runs;
                                     validates the single candidate
                                     and either ships it or raises
                                     NoValidBeatSheetError).
        candidate_c_json   STRING -- serialized Stage1Plan JSON.

    Outputs:
        winning_plan_json  STRING -- serialized winning Stage1Plan
                                     JSON. Downstream consumers see
                                     a single plan as they always
                                     have.
        selector_audit     STRING -- serialized BeatSelectorAudit
                                     JSON (winner_index, reason,
                                     per-candidate scores +
                                     defects, eligible_indices)
                                     for operator inspection / for
                                     stamping on meta.beat_selector
                                     in a follow-up wire-up.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("winning_plan_json", "selector_audit")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidate_a_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Serialized Stage1Plan JSON (candidate A). "
                        "Required; the selector runs on at least "
                        "one candidate."
                    ),
                }),
            },
            "optional": {
                "candidate_b_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional second candidate Stage1Plan JSON. "
                        "Empty string treats the call as single-"
                        "candidate mode."
                    ),
                }),
                "candidate_c_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional third candidate Stage1Plan JSON. "
                        "Empty string treats the call as 2-candidate "
                        "mode."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Selector is deterministic; rely on ComfyUI's default
        # change-detection over the JSON payloads.
        return ""

    def _parse_candidate(self, payload: str):
        """Parse a candidate JSON payload into a Stage1Plan object
        (which the selector duck-types). Returns None on empty or
        malformed payload; caller filters."""
        text = (payload or "").strip()
        if not text:
            return None
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "[OTR_BeatSelector] candidate JSON parse failed: %s",
                str(exc)[:200],
            )
            return None
        if not isinstance(d, dict):
            return None
        # Lazy-import Stage1Plan via the canonical parse +
        # stamp_dialogue_slot_ids entry point so candidates carry
        # slot ids before the selector scores them.
        try:
            from ._otr_stage1_plan import parse_and_validate_plan
            return parse_and_validate_plan(d)
        except Exception as exc:  # noqa: BLE001 -- log + skip
            log.warning(
                "[OTR_BeatSelector] candidate Stage1Plan parse/"
                "validate failed: %s",
                str(exc)[:200],
            )
            # Return the raw dict so the selector's duck-typing can
            # at least try to score (validate_beat_sheet treats
            # untyped beats charitably). The exception is logged so
            # the operator can fix the upstream.
            return d

    def run(
        self,
        candidate_a_json: str = "",
        candidate_b_json: str = "",
        candidate_c_json: str = "",
    ):
        from ._otr_beat_selector import (
            NoValidBeatSheetError,
            select_winning_beat_sheet,
        )

        candidates: List[Any] = []
        for payload in (candidate_a_json, candidate_b_json,
                        candidate_c_json):
            parsed = self._parse_candidate(payload)
            if parsed is not None:
                candidates.append(parsed)

        if not candidates:
            log.error(
                "[OTR_BeatSelector] zero usable candidates; emitting "
                "empty winning plan + failure audit."
            )
            audit = {
                "winner_index": -1,
                "reason": "no usable candidate payloads",
                "scores_per_candidate": [],
                "defects_per_candidate": [],
                "eligible_indices": [],
            }
            return ("", json.dumps(audit, ensure_ascii=False))

        try:
            winner, audit = select_winning_beat_sheet(candidates)
        except NoValidBeatSheetError as exc:
            log.error(
                "[OTR_BeatSelector] NoValidBeatSheetError: %s -- "
                "emitting empty winning plan; the caller's '5/5 "
                "episodes clean' contract should fail visibly.",
                exc,
            )
            audit = getattr(exc, "audit", None)
            audit_dict = audit.to_dict() if audit is not None else {
                "winner_index": -1,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            return ("", json.dumps(audit_dict, ensure_ascii=False))

        # Serialize the winner back to JSON. The downstream consumer
        # re-parses with parse_and_validate_plan if it needs typed
        # access.
        if hasattr(winner, "model_dump"):
            winner_payload = winner.model_dump()
        else:
            winner_payload = winner
        winning_plan_json = json.dumps(
            winner_payload, ensure_ascii=False, indent=2,
        )
        audit_json = json.dumps(audit.to_dict(), ensure_ascii=False)
        log.info(
            "[OTR_BeatSelector] winner=candidate[%d] eligible=%s "
            "scores=%s",
            audit.winner_index, audit.eligible_indices,
            [s.total for s in audit.scores_per_candidate],
        )
        return (winning_plan_json, audit_json)

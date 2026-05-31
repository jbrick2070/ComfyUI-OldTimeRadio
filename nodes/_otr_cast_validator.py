"""nodes/_otr_cast_validator.py -- cast LLM Pass-1 referee (S7).

The LLM Pass-1 call (S6) may return ONLY a name + texture per slot. This
validator rejects anything outside that contract; on ANY rejection the caller
falls back to the deterministic pool names (which are already gender-coherent
via the S2 repair) -- no LLM retry, terminates, stays reproducible.

Pure helper module, no widget / INPUT_TYPES / socket -> no workflow-JSON wiring
(PD3 N/A). Inert unless OTR_NAME_MODE=llm_slot_fill.

Frozen S0 contract -- LLM Pass-1 return (LLM -> validator), NAME + TEXTURE ONLY:

    { "char_id": "c02", "name": "Dr. Mara Venn",
      "one_line_presence": "Precise, tired, morally alert.",
      "dialogue_style": "short clinical sentences with buried warmth" }

Rejects: any key outside the Pass-1 set (so a gender/voice/role carry-back is
refused); any unknown/missing char_id; wrong count; duplicate names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

ALLOWED_PASS1_KEYS = frozenset(
    {"char_id", "name", "one_line_presence", "dialogue_style"})


@dataclass
class Pass1ValidationResult:
    ok: bool
    reason: str = ""
    names_by_char_id: Dict[str, str] = field(default_factory=dict)
    texture_by_char_id: Dict[str, dict] = field(default_factory=dict)


def _fail(reason: str) -> Pass1ValidationResult:
    return Pass1ValidationResult(ok=False, reason=reason)


def validate_pass1(llm_items, plan) -> Pass1ValidationResult:
    """Validate the LLM Pass-1 output against the immutable CastPlanner plan.

    `llm_items`: the parsed LLM response -- expected list of dicts.
    `plan`: list of CastPlanSlot (only char_ids are consulted; gender/voice/role
        are NOT accepted from the LLM, so any attempt to send them back is a
        rejected out-of-contract key).

    On ok: names_by_char_id maps char_id -> UPPER name; texture_by_char_id maps
    char_id -> {one_line_presence, dialogue_style}. On any violation: ok=False
    with a human-readable reason; the caller keeps the deterministic names.
    """
    plan_ids = [s.char_id for s in plan]
    plan_id_set = set(plan_ids)

    if not isinstance(llm_items, list):
        return _fail(f"Pass-1 output is not a list (got {type(llm_items).__name__})")
    if len(llm_items) != len(plan):
        return _fail(f"wrong count: got {len(llm_items)}, expected {len(plan)}")

    names_by: Dict[str, str] = {}
    texture_by: Dict[str, dict] = {}
    seen_ids: set = set()
    seen_names: set = set()

    for idx, item in enumerate(llm_items):
        if not isinstance(item, dict):
            return _fail(f"item {idx} is not an object")
        extra = set(item.keys()) - ALLOWED_PASS1_KEYS
        if extra:
            # A gender / voice_preset / dramatic_role key lands here -- the LLM
            # is forbidden from carrying structure back.
            return _fail(f"item {idx} has out-of-contract keys {sorted(extra)}")
        cid = item.get("char_id")
        if cid not in plan_id_set:
            return _fail(f"unknown char_id {cid!r}")
        if cid in seen_ids:
            return _fail(f"duplicate char_id {cid!r}")
        name = (item.get("name") or "").strip()
        if not name:
            return _fail(f"empty name for {cid!r}")
        name_u = name.upper()
        if name_u in seen_names:
            return _fail(f"duplicate name {name_u!r}")
        seen_ids.add(cid)
        seen_names.add(name_u)
        names_by[cid] = name_u
        texture_by[cid] = {
            "one_line_presence": (item.get("one_line_presence") or "").strip(),
            "dialogue_style": (item.get("dialogue_style") or "").strip(),
        }

    if seen_ids != plan_id_set:
        missing = sorted(plan_id_set - seen_ids)
        return _fail(f"missing char_ids {missing}")

    return Pass1ValidationResult(
        ok=True, names_by_char_id=names_by, texture_by_char_id=texture_by)

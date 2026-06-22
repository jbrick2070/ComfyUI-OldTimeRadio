"""nodes/_otr_slot_drama_contract.py -- Build 3 (2026-05-28).

GO_FORWARD_PLAN_v10 Build 3: `slot_drama_contract` + deterministic
contract validation.

PROMOTED from docs/sprint_drafts/build3/ and wired into
OTR_LedgerScriptWriter.run (after build_continuity_ledger + the
dramatic_state stamp, before the per-beat render loop) per
build3/WIRING_SPEC.md. Helper module only -- not a node, not in
_NODE_MODULES, so ComfyUI never auto-imports it; the writer imports it
lazily. Build 4 (compose_exchange) consumes meta["slot_drama_contracts"].

----------------------------------------------------------------------
What a slot drama contract is
----------------------------------------------------------------------
For each voiced dialogue slot, the contract is the per-line OBLIGATION
the exchange writer (Build 4) must honor: who speaks, what the line is
trying to do (`line_job`), the unspoken force under it
(`hidden_pressure`), at least one concrete detail it must ground in,
and the emotional/positional state before vs after the line. The
`must_turn` flag marks the slot that pivots the episode -- the
costly-choice beat -- where state_before MUST differ from state_after.

----------------------------------------------------------------------
LLM surface is deliberately tiny (plan critique 4 & 5)
----------------------------------------------------------------------
DERIVED deterministically (NO LLM):
  * speaker                  -- from the slot/line row.
  * concrete_detail_required -- candidates from (continuity
                                active_props  union  news key_terms).
  * state_before / state_after / must_turn -- from DramaticState +
                                beat position. The costly_choice_beat
                                slot gets must_turn=True and the
                                ending_change-derived after-state.

LLM-WRITTEN (one technical-slot pass per episode, two free-text
fields only):
  * line_job        -- one clause: what this line is trying to do.
  * hidden_pressure -- one clause: the unspoken force under it.

Everything else is derived, so a bad LLM pass can only corrupt two
short strings; the deterministic validator + minimal-contract fallback
cover the rest.

PURE module: pydantic + stdlib only. No torch, no I/O, no node
surface. The LLM pass takes a `generate_fn` by dependency injection
so tests pass a fake. UTF-8, no BOM. SFW.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError


log = logging.getLogger("OTR")

__all__ = [
    "SlotDramaContract",
    "SlotJobFields",
    "derive_contract_skeleton",
    "generate_slot_job_fields",
    "build_slot_drama_contract",
    "build_minimal_contract",
    "validate_contract",
    "validate_episode_contracts",
    "build_line_dramatic_fields",
    "compute_beat_tension_ramp",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SlotDramaContract(BaseModel):
    """The per-slot dramatic obligation (plan Build 3 schema).

    Field-by-field:
      * dialogue_slot_id -- d###; the ledger row this contract binds to.
      * speaker          -- speaker NAME (derived from the slot row).
      * line_job         -- LLM-written: what this line is trying to do.
      * hidden_pressure  -- LLM-written: the unspoken force under it.
      * concrete_detail_required -- one or more grounding details, each
        drawn from (continuity active_props  union  news key_terms).
        The exchange writer must land at least one across the exchange
        (Build 4 weights this per-exchange, not per-line).
      * state_before / state_after -- the emotional/positional state
        of the scene entering vs leaving this slot. Derived from
        DramaticState + beat position.
      * must_turn -- True ONLY for the costly-choice slot. When True
        the validator demands state_before != state_after.
    """

    dialogue_slot_id: str = Field(..., pattern=r"^d\d{3}$")
    speaker: str = Field(..., min_length=1, max_length=120)
    line_job: str = Field(..., min_length=1, max_length=240)
    hidden_pressure: str = Field(..., min_length=1, max_length=240)
    concrete_detail_required: List[str] = Field(default_factory=list)
    state_before: str = Field(..., min_length=1, max_length=200)
    state_after: str = Field(..., min_length=1, max_length=200)
    must_turn: bool = Field(default=False)


class SlotJobFields(BaseModel):
    """The TWO free-text fields the LLM is allowed to write.

    Bound as the constrained-decode schema for the technical-slot
    pass so the model can only emit a JSON object with exactly these
    two short strings. Keeps the LLM surface minimal (plan critique
    4 & 5): a bad pass can corrupt only `line_job` + `hidden_pressure`,
    both then re-checked by the deterministic validator.
    """

    line_job: str = Field(..., min_length=1, max_length=240)
    hidden_pressure: str = Field(..., min_length=1, max_length=240)


# ---------------------------------------------------------------------------
# Small shape adapters -- accept dicts OR pydantic-ish objects
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict OR an attribute off an object.

    DramaticState / continuity / line+beat rows reach this module in
    mixed shapes depending on the call site (model_dump dicts vs live
    pydantic objects). This keeps the derivation source-agnostic.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _norm_terms(*sources: Optional[List[str]]) -> List[str]:
    """Union the given term lists, trim blanks, de-dupe, preserve order.

    Used to build the concrete-detail candidate pool from
    (active_props  union  key_terms). Order-stable so derivation is
    deterministic: same inputs -> same candidate order -> same picks.
    """
    seen: set[str] = set()
    out: List[str] = []
    for src in sources:
        for raw in (src or []):
            t = " ".join(str(raw or "").split())
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Deterministic derivation -- everything except the two LLM fields
# ---------------------------------------------------------------------------


def _state_phrases(dramatic_state: Any) -> Tuple[str, str]:
    """Derive (opening_state, ending_state) phrases from DramaticState.

    opening_state -- the tension named by the dramatic_question plus
                     the two opposed wants (the world at the open).
    ending_state  -- the ending_change (the world at the close).

    Both clamped to the SlotDramaContract state field cap (200).
    """
    dq = " ".join(str(_get(dramatic_state, "dramatic_question", "") or "").split())
    a = " ".join(str(_get(dramatic_state, "character_a_wants", "") or "").split())
    b = " ".join(str(_get(dramatic_state, "character_b_wants", "") or "").split())
    ec = " ".join(str(_get(dramatic_state, "ending_change", "") or "").split())

    if a and b:
        opening = f"unresolved: {a} vs {b}"
    elif dq:
        opening = f"unresolved: {dq}"
    else:
        opening = "the dramatic question is still open"

    ending = ec or "the question is answered and the world has changed"

    return (_clamp(opening, 200), _clamp(ending, 200))


def _clamp(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    sp = cut.rfind(" ")
    if sp > n // 2:
        return cut[:sp].rstrip()
    return cut.rstrip()


def _pick_concrete_details(
    slot_index: int,
    candidates: List[str],
    *,
    must_turn: bool,
) -> List[str]:
    """Deterministically assign concrete-detail requirements to a slot.

    Rotates through the candidate pool by slot_index so different
    slots ground in different props/terms (avoids object spam --
    plan critique 6). The turning slot gets two candidates when the
    pool allows (it carries the most weight); other slots get one.
    Empty pool -> empty list (validator will flag if a turning slot
    needs grounding; minimal-contract fallback handles that case).
    """
    if not candidates:
        return []
    n = len(candidates)
    first = candidates[slot_index % n]
    if must_turn and n >= 2:
        second = candidates[(slot_index + 1) % n]
        if second != first:
            return [first, second]
    return [first]


def derive_contract_skeleton(
    *,
    slot_row: Any,
    slot_index: int,
    dramatic_state: Any,
    active_props: Optional[List[str]] = None,
    key_terms: Optional[List[str]] = None,
) -> dict:
    """Derive every contract field EXCEPT the two LLM-written ones.

    Args:
        slot_row: the ledger line / beat row for this slot. Must carry
            `dialogue_slot_id` and a speaker (read in priority order:
            `speaker`, then `speaker_name`, then `name`).
        slot_index: 0-based position of this slot among the episode's
            voiced slots (drives detail rotation + state phrasing).
        dramatic_state: DramaticState (dict or object). Supplies the
            costly_choice_beat slot id, the opposed wants, and the
            ending_change.
        active_props: continuity ledger active_props (ContinuityState).
        key_terms: news key_terms (meta["news"]["key_terms"]).

    Returns:
        A dict with all SlotDramaContract fields EXCEPT line_job /
        hidden_pressure -- ready to merge with the LLM (or fallback)
        fields and validate.
    """
    slot_id = str(_get(slot_row, "dialogue_slot_id", "") or "").strip()
    speaker = (
        str(_get(slot_row, "speaker", "") or "").strip()
        or str(_get(slot_row, "speaker_name", "") or "").strip()
        or str(_get(slot_row, "name", "") or "").strip()
    )

    costly_slot = str(_get(dramatic_state, "costly_choice_beat", "") or "").strip()
    must_turn = bool(slot_id) and slot_id == costly_slot

    opening, ending = _state_phrases(dramatic_state)
    if must_turn:
        # The turning slot is exactly where the world flips from the
        # open-state to the ending-change. Hard, derived, non-equal.
        state_before, state_after = opening, ending
    else:
        # Non-turning slots hold the line: same state on each side.
        # The validator only forbids equality when must_turn is True.
        held = opening if slot_index == 0 else _clamp(
            "tension carried forward, unresolved", 200
        )
        state_before, state_after = held, held

    candidates = _norm_terms(active_props, key_terms)
    details = _pick_concrete_details(slot_index, candidates, must_turn=must_turn)

    return {
        "dialogue_slot_id": slot_id,
        "speaker": speaker,
        "concrete_detail_required": details,
        "state_before": state_before,
        "state_after": state_after,
        "must_turn": must_turn,
    }


# ---------------------------------------------------------------------------
# The LLM pass -- writes ONLY line_job + hidden_pressure
# ---------------------------------------------------------------------------
# LLM slot: technical
# Reason: structured JSON pass bound to SlotJobFields (constrained
# decode) -- per project rule 6, schema-constrained calls route to the
# technical model slot. Wired from the writer's `technical_model`
# output socket; this module exposes NO model_id widget.


_SLOT_JOB_SYSTEM_PROMPT = """\
You write the dramatic OBLIGATION for a single line of an audio drama,
given the scene's dramatic frame and the slot's derived state. Return
EXACTLY one JSON object, no prose, no Markdown fences:
{
  "line_job":        "one clause: what this line is trying to DO (a verb-driven aim, not the words spoken)",
  "hidden_pressure": "one clause: the unspoken force under the line (what the speaker will not say outright)"
}
Keep each field under 240 characters. Be specific to THIS slot. Do not
restate the state phrases verbatim. Safe for work."""


def _build_slot_job_user_prompt(
    *,
    speaker: str,
    dramatic_state: Any,
    state_before: str,
    state_after: str,
    must_turn: bool,
    beat_intent: str,
    concrete_detail_required: List[str],
) -> str:
    dq = str(_get(dramatic_state, "dramatic_question", "") or "").strip()
    parts = [
        f"SPEAKER: {speaker}",
        f"DRAMATIC QUESTION: {dq}" if dq else "",
        f"BEAT INTENT: {beat_intent}" if beat_intent else "",
        f"STATE BEFORE: {state_before}",
        f"STATE AFTER: {state_after}",
        "THIS IS THE COSTLY-CHOICE TURN (the line that pivots the episode)."
        if must_turn
        else "",
        (
            "MUST GROUND IN (use at least one): "
            + ", ".join(concrete_detail_required)
        )
        if concrete_detail_required
        else "",
        "Write line_job and hidden_pressure for this slot only.",
    ]
    return "\n".join(p for p in parts if p)


def generate_slot_job_fields(
    generate_fn: Callable[..., str],
    *,
    speaker: str,
    dramatic_state: Any,
    state_before: str,
    state_after: str,
    must_turn: bool,
    beat_intent: str,
    concrete_detail_required: List[str],
    temperature: float = 0.6,
    max_new_tokens: int = 192,
) -> SlotJobFields:
    """Run the technical-slot LLM pass for the two free-text fields.

    `generate_fn` is dependency-injected (a constrained-decode closure
    from `make_constrained_generate_fn(cache_entry, SlotJobFields,
    heartbeat_label=...)` in production; a fake in tests). It is
    expected to return a JSON string conforming to SlotJobFields.

    Raises ValidationError / ValueError on unparseable or
    schema-invalid output so the caller's regenerate-then-fallback
    path can react.
    """
    messages = [
        {"role": "system", "content": _SLOT_JOB_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_slot_job_user_prompt(
                speaker=speaker,
                dramatic_state=dramatic_state,
                state_before=state_before,
                state_after=state_after,
                must_turn=must_turn,
                beat_intent=beat_intent,
                concrete_detail_required=concrete_detail_required,
            ),
        },
    ]
    # LLM slot: technical -- structured SlotJobFields JSON pass; the
    # injected generate_fn is the constrained-decode closure built from
    # the technical slot (see make_constrained_generate_fn call site).
    raw = generate_fn(
        messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    data = json.loads(raw)
    return SlotJobFields.model_validate(data)


# ---------------------------------------------------------------------------
# Deterministic minimal-contract fallback (NO LLM)
# ---------------------------------------------------------------------------


def build_minimal_contract(
    *,
    slot_row: Any,
    slot_index: int,
    dramatic_state: Any,
    beat_intent: str = "",
    active_props: Optional[List[str]] = None,
    key_terms: Optional[List[str]] = None,
) -> SlotDramaContract:
    """Build a valid contract with NO LLM -- the post-fail fallback.

    Used when the LLM contract fails validation after one regenerate.
    `line_job` + `hidden_pressure` are synthesized purely from
    DramaticState + beat_intent so the result is deterministic and
    always schema-valid for a well-formed slot row.
    """
    skeleton = derive_contract_skeleton(
        slot_row=slot_row,
        slot_index=slot_index,
        dramatic_state=dramatic_state,
        active_props=active_props,
        key_terms=key_terms,
    )

    intent = " ".join((beat_intent or "").split())
    if skeleton["must_turn"]:
        ec = str(_get(dramatic_state, "ending_change", "") or "").strip()
        line_job = _clamp(
            f"force the irreversible choice that brings about: {ec}"
            if ec
            else "force the irreversible choice that pivots the episode",
            240,
        )
        hidden_pressure = _clamp(
            "the cost of choosing -- what the speaker loses either way",
            240,
        )
    else:
        line_job = _clamp(
            f"advance the beat: {intent}" if intent else "advance the scene's tension",
            240,
        )
        a = str(_get(dramatic_state, "character_a_wants", "") or "").strip()
        b = str(_get(dramatic_state, "character_b_wants", "") or "").strip()
        if a and b:
            hidden_pressure = _clamp(f"the unspoken pull between {a} and {b}", 240)
        else:
            hidden_pressure = _clamp(
                "the unspoken stake neither party will name", 240
            )

    return SlotDramaContract(
        line_job=line_job,
        hidden_pressure=hidden_pressure,
        **skeleton,
    )


# ---------------------------------------------------------------------------
# Top-level builder -- derive skeleton, run LLM, regenerate once, else fallback
# ---------------------------------------------------------------------------


def build_slot_drama_contract(
    generate_fn: Optional[Callable[..., str]],
    *,
    slot_row: Any,
    slot_index: int,
    dramatic_state: Any,
    beat_intent: str = "",
    active_props: Optional[List[str]] = None,
    key_terms: Optional[List[str]] = None,
    temperature: float = 0.6,
) -> Tuple[SlotDramaContract, str]:
    """Build one slot's contract: derive -> LLM -> validate, with the
    regenerate-once-then-deterministic-fallback path.

    Returns:
        (contract, source) where source is one of:
          * "llm"             -- first LLM pass validated.
          * "llm_regenerate"  -- second LLM pass validated.
          * "minimal"         -- both LLM passes failed (or generate_fn
                                 is None); deterministic fallback used.

    `generate_fn` may be None to force the deterministic minimal
    contract (e.g. a dry run, or when the technical model is offline).
    The contract is always returned valid: the minimal fallback is
    constructed to pass `validate_contract` for any well-formed slot
    row, so no garbage contract reaches the writer (plan Build 3 gate).
    """
    active_props = active_props or []
    key_terms = key_terms or []

    skeleton = derive_contract_skeleton(
        slot_row=slot_row,
        slot_index=slot_index,
        dramatic_state=dramatic_state,
        active_props=active_props,
        key_terms=key_terms,
    )

    def _try_llm() -> Optional[SlotDramaContract]:
        if generate_fn is None:
            return None
        try:
            jobs = generate_slot_job_fields(
                generate_fn,
                speaker=skeleton["speaker"],
                dramatic_state=dramatic_state,
                state_before=skeleton["state_before"],
                state_after=skeleton["state_after"],
                must_turn=skeleton["must_turn"],
                beat_intent=beat_intent,
                concrete_detail_required=skeleton["concrete_detail_required"],
                temperature=temperature,
            )
            candidate = SlotDramaContract(
                line_job=jobs.line_job,
                hidden_pressure=jobs.hidden_pressure,
                **skeleton,
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            log.info(
                "[SlotDramaContract] LLM job-field pass failed for %s: %s",
                skeleton.get("dialogue_slot_id"),
                exc,
            )
            return None
        ok, reasons = validate_contract(candidate, active_props, key_terms)
        if not ok:
            log.info(
                "[SlotDramaContract] LLM contract for %s failed validation: %s",
                skeleton.get("dialogue_slot_id"),
                "; ".join(reasons),
            )
            return None
        return candidate

    # First attempt.
    contract = _try_llm()
    if contract is not None:
        return contract, "llm"

    # Regenerate once.
    contract = _try_llm()
    if contract is not None:
        return contract, "llm_regenerate"

    # Deterministic minimal fallback (no LLM).
    minimal = build_minimal_contract(
        slot_row=slot_row,
        slot_index=slot_index,
        dramatic_state=dramatic_state,
        beat_intent=beat_intent,
        active_props=active_props,
        key_terms=key_terms,
    )
    return minimal, "minimal"


# ---------------------------------------------------------------------------
# Deterministic validators
# ---------------------------------------------------------------------------


def validate_contract(
    contract: Any,
    active_props: Optional[List[str]],
    key_terms: Optional[List[str]],
) -> Tuple[bool, List[str]]:
    """Per-slot deterministic contract check (plan Build 3).

    Rules (each failure appends a reason; same input -> same verdict):
      1. schema-valid -- coerces to SlotDramaContract (catches bad
         slot id pattern, missing/empty required fields, wrong types).
      2. every text field non-empty after strip (line_job,
         hidden_pressure, speaker, state_before, state_after).
      3. concrete_detail_required subset of (active_props union
         key_terms), case-insensitive. Each entry must be non-empty.
      4. when must_turn is True: state_before != state_after (the turn
         must actually change the state) AND at least one concrete
         detail is required (the pivot must ground in something real).
      5. when must_turn is True: at least one concrete detail present
         (covered in rule 4; called out for the episode-level check).

    Returns (ok, reasons). ok == (reasons == []).
    """
    reasons: List[str] = []

    # Rule 1 -- schema validity.
    if isinstance(contract, SlotDramaContract):
        c = contract
    else:
        try:
            c = SlotDramaContract.model_validate(
                contract if isinstance(contract, dict) else _as_dict(contract)
            )
        except (ValidationError, ValueError) as exc:
            return False, [f"schema_invalid: {exc}"]

    # Rule 2 -- non-empty text fields.
    for field in ("speaker", "line_job", "hidden_pressure", "state_before", "state_after"):
        if not str(getattr(c, field, "") or "").strip():
            reasons.append(f"empty_field: {field}")

    # Rule 3 -- concrete details subset of the allowed pool.
    pool = {t.lower() for t in _norm_terms(active_props, key_terms)}
    for det in c.concrete_detail_required:
        d = " ".join(str(det or "").split())
        if not d:
            reasons.append("empty_concrete_detail")
            continue
        if d.lower() not in pool:
            reasons.append(f"concrete_detail_not_in_pool: {d!r}")

    # Rule 4 -- the turning slot must actually turn AND must ground.
    if c.must_turn:
        if _eq_norm(c.state_before, c.state_after):
            reasons.append("must_turn_but_state_unchanged")
        if not [d for d in c.concrete_detail_required if str(d or "").strip()]:
            reasons.append("must_turn_but_no_concrete_detail")

    return (not reasons), reasons


def validate_episode_contracts(
    contracts: List[Any],
    active_props: Optional[List[str]],
    key_terms: Optional[List[str]],
) -> Tuple[bool, List[str]]:
    """Episode-level deterministic check across ALL slot contracts.

    Per-slot rules (delegated to `validate_contract`) PLUS the
    cross-slot invariant from the plan:
      * EXACTLY ONE slot carries the costly-choice obligation
        (must_turn == True). Zero -> the episode never turns; more
        than one -> ambiguous pivot.

    Returns (ok, reasons). Reasons are prefixed with the slot id for
    per-slot failures and "episode:" for the cross-slot rule.
    """
    reasons: List[str] = []

    turn_slots: List[str] = []
    for c in contracts:
        ok, sub = validate_contract(c, active_props, key_terms)
        sid = str(_get(c, "dialogue_slot_id", "") or "").strip() or "?"
        if not ok:
            reasons.extend(f"{sid}: {r}" for r in sub)
        if bool(_get(c, "must_turn", False)):
            turn_slots.append(sid)

    if len(turn_slots) == 0:
        reasons.append("episode: no slot carries the costly-choice turn")
    elif len(turn_slots) > 1:
        reasons.append(
            "episode: more than one costly-choice turn: " + ", ".join(turn_slots)
        )

    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _eq_norm(a: str, b: str) -> bool:
    return " ".join((a or "").lower().split()) == " ".join((b or "").lower().split())


def _as_dict(obj: Any) -> dict:
    """Best-effort coerce an object to a dict for schema validation."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise ValueError(f"cannot coerce {type(obj).__name__} to dict for validation")


# ---------------------------------------------------------------------------
# A5 (story-quality Phase 1): deliver the contract drama to the line writer
# ---------------------------------------------------------------------------


def build_line_dramatic_fields(
    contract: Any,
    dramatic_state: Any,
    *,
    speaker: str = "",
    a_name: str = "",
    b_name: str = "",
) -> dict:
    """Map a SlotDramaContract (+ the episode DramaticState) onto the line
    composer's per-line dramatic fields, so the line PLAYS its objective /
    obstacle / turn / subtext instead of restating the theme (A5 delivery of
    the news-driven drama from B1).

    Returns ``{"beat_objective", "beat_obstacle", "beat_turn",
    "beat_subtext"}`` (all strings; empty when not applicable -- the composer
    drops empty blocks, keeping the legacy prompt byte-identical):
      * beat_objective <- contract.line_job, with the concrete_detail_required
        grounding folded in (there is no dedicated LineRequest detail slot);
      * beat_subtext  <- contract.hidden_pressure;
      * beat_turn     <- the state_before -> state_after shift, ONLY on the
        must_turn (costly-choice) slot;
      * beat_obstacle <- the OPPOSING lead's want (the speaker's counterpart),
        so distinct leads pull against each other.

    `contract` / `dramatic_state` may be dicts or model instances. Pure; never
    raises (a bad input degrades to empty fields). The caller must NOT mutate
    locked cast rows -- distinctness rides entirely on these per-line fields.
    """
    try:
        line_job = str(_get(contract, "line_job", "") or "").strip()
        hidden = str(_get(contract, "hidden_pressure", "") or "").strip()
        must_turn = bool(_get(contract, "must_turn", False))
        state_before = str(_get(contract, "state_before", "") or "").strip()
        state_after = str(_get(contract, "state_after", "") or "").strip()
        details = _get(contract, "concrete_detail_required", []) or []
        a_wants = str(_get(dramatic_state, "character_a_wants", "") or "").strip()
        b_wants = str(_get(dramatic_state, "character_b_wants", "") or "").strip()

        det_list = [str(d).strip() for d in details if str(d).strip()]
        objective = line_job
        if objective and det_list:
            objective = f"{objective} (ground it in: {', '.join(det_list)})"
        elif not objective and det_list:
            objective = f"ground the line in: {', '.join(det_list)}"

        sp = (speaker or "").strip().lower()
        if a_name and sp == a_name.strip().lower():
            obstacle = b_wants
        elif b_name and sp == b_name.strip().lower():
            obstacle = a_wants
        else:
            # non-lead speaker: the antagonist pressure (B want) by default.
            obstacle = b_wants or a_wants

        turn = ""
        if (must_turn and state_before and state_after
                and not _eq_norm(state_before, state_after)):
            turn = f"{state_before} -> {state_after}"

        return {
            "beat_objective": objective,
            "beat_obstacle": obstacle,
            "beat_turn": turn,
            "beat_subtext": hidden,
        }
    except Exception:  # noqa: BLE001 -- adapter must never break audio
        return {
            "beat_objective": "",
            "beat_obstacle": "",
            "beat_turn": "",
            "beat_subtext": "",
        }


def compute_beat_tension_ramp(character_beat_ids: list) -> dict:
    """STEP 6 (2026-06-22 story+cast fix, roundtable-converged): a deterministic
    escalating intensity signal (1..5) over the CHARACTER beats of an episode.

    The outline's arc_phase already escalates monotonically (setup ->
    complication -> resolution, validated), but `beat_tension` -- the composer's
    "Tension: N/5" cue -- was never assigned a value, so the line writer saw no
    per-beat intensity gradient. This wires one.

    Curve (roundtable R1, GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude
    judge): an ordinal ramp over the character-beat ordinal `i` (0-based), peak
    at the FINAL character beat, no easing (a synthetic falling action would
    contradict "escalating" and the budget has no falling phase):
        n <= 1  -> 3   (a lone beat sits mid-scale)
        else    -> clamp(round(1 + 4*i/(n-1)), 1, 5)
    Announcer / music / sfx are excluded by the caller (character beats only).

    Pure + deterministic (no RNG, no clock): same beat list -> same map. Returns
    ``{beat_id: tension}``; an empty / malformed input returns ``{}``.
    """
    ids = [str(b).strip() for b in (character_beat_ids or []) if str(b).strip()]
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 3}
    out: dict = {}
    for i, bid in enumerate(ids):
        t = int(round(1 + 4 * i / (n - 1)))
        out[bid] = max(1, min(5, t))
    return out

r"""Build 4 -- compose_exchange + repair-by-exchange-group.

PROMOTED (2026-05-28) from docs/sprint_drafts/build4/ to nodes/. Pure,
fully-injected engine (generate_fn + tier_a_check + the two wiring
helpers' Build 2 deps are all injected) -- not a node, not in
_NODE_MODULES, so ComfyUI never auto-imports it. The live render-loop
wiring (a use_exchange feature flag on OTR_LedgerScriptWriter mirroring
use_multiturn_dialogue, + the workflow JSON rewire) is STAGED for an
operator session: Build 4's gate is entirely live N=3 metrics (slot
drift, exposition count, scene movement, VRAM <= 14.5 GB) which cannot
be measured headless. See build4/WIRING_SPEC.md + the session handoff
for the exact wiring recipe. group_voiced_beats + make_tier_a_adapter
(bottom of this file) are the ready-made seams for that wiring.


Per workflows/GO_FORWARD_PLAN_v10_four_builds_2026-05-28.md, Build 4:

    * Replace isolated compose_line with compose_exchange over a beat
      group (2-3 voiced slots), with prior committed lines in context.
    * HARD RULE: the exchange returns exactly one text block per slot id
      in the group. Internal pauses are allowed inside a block, but the
      slot COUNT is fixed -- one slot in, one block out, same ids, same
      speakers, no added/skipped slots (critique 8).
    * De-exposition lives in the writer prompt, weighted to avoid object
      spam: require ONE concrete grounding per EXCHANGE, not per line
      (critique 6). Authored vocabulary is not filtered or scored by Python.
    * Repair by exchange GROUP, not single slot (critique 11): if any
      slot fails the injected Tier-A check, re-run the WHOLE exchange
      ONCE with the failure reasons appended; preserve slot ids +
      speakers; one repair attempt only; still failing -> fail loud /
      legacy fallback. No multi-cycle churn.

This module is a DRAFT under docs/sprint_drafts/build4/. It is not
imported by ComfyUI. It depends conceptually on:

    * Build 2's slot format (`d###|SPEAKER: text`, one slot = one
      committed text block) and Build 2's Tier-A integrity validator.
      Both are STUBBED here as injected interfaces -- see
      TierACheckFn and the slot-format constants below.
    * Build 3's slot_drama_contract (one contract per slot id). STUBBED
      here as a duck-typed mapping -- see SlotContract / the `contracts`
      argument shape in build_exchange_prompt.

Everything that touches a live model or a Build 2/3 surface is injected
so tests can pass fakes (no torch, no live model).

# LLM slot: creative
# Reason: compose_exchange renders multi-line dialogue (subtext, refusal,
# reversal) -- this is creative-axis narrative work per project rule 6.
# The model id is wired from the writer's creative_writing_model output
# socket; this module exposes no model_id widget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


log = logging.getLogger("OTR")


# ===========================================================================
# Stubbed interfaces the integration must satisfy
# ===========================================================================
#
# These are the seams between Build 4 and Builds 2/3. They are described
# here as plain protocols / duck-typed shapes so this draft stays free of
# any import from the live node package. The WIRING_SPEC.md restates them
# as the integration contract.


# ---- Build 2: slot format ------------------------------------------------
#
# A voiced slot is identified by a dialogue_slot_id matching d### and a
# speaker (a cast member name, ALL CAPS by convention). Build 2's Story
# Room emits one line per slot as:
#
#     d001|MARLOW: I'm not signing that.
#
# One slot = exactly one committed text block. The block may contain
# internal pauses (e.g. "...", em-dashes), but it is ONE ledger row.

SLOT_ID_PATTERN: str = r"^d\d{3}$"
"""dialogue_slot_id regex (Build 2 slot format). d001..d999."""

_SLOT_ID_RE = re.compile(SLOT_ID_PATTERN)

# Line prefix the exchange writer is asked to emit and parse_exchange
# expects back: `d###|SPEAKER: text`. Mirrors Build 2's committed-row
# surface so a parsed exchange drops straight into the ledger.
SLOT_LINE_RE = re.compile(
    r"^\s*(?P<slot_id>d\d{3})\s*\|\s*(?P<speaker>[^:]+?)\s*:\s*(?P<text>.*\S.*)$"
)


@dataclass
class VoicedSlot:
    """One voiced slot in a beat group (Build 2 slot-format shape).

    The integration may pass any object exposing these attributes.
    compose_exchange reads only the structural slot identity and prompt
    context carried here.
    """

    dialogue_slot_id: str       # d### (Build 2)
    speaker: str                # cast member name, ALL CAPS
    intent: str = ""            # beat intent (prompt context)


# ---- Build 3: slot_drama_contract ----------------------------------------
#
# One contract per voiced slot, keyed by dialogue_slot_id. Build 3's
# schema (pydantic): dialogue_slot_id, speaker, line_job, hidden_pressure,
# concrete_detail_required[], state_before, state_after, must_turn:bool.
# Build 4 only READS the contract to shape the prompt; it does not
# validate it (Build 3 owns contract validation).


@dataclass
class SlotContract:
    """Duck-typed mirror of Build 3's slot_drama_contract.

    The integration passes the real Build 3 pydantic objects; this draft
    only reads the fields below, so a plain dataclass / mapping with the
    same surface is interchangeable in tests.
    """

    dialogue_slot_id: str
    speaker: str
    line_job: str = ""
    hidden_pressure: str = ""
    concrete_detail_required: List[str] = field(default_factory=list)
    state_before: str = ""
    state_after: str = ""
    must_turn: bool = False


# ---- Build 2: Tier-A integrity check (injected) --------------------------
#
# Tier-A is format/integrity only, deterministic, hard-fail: slot count,
# slot order, speaker match, and empty lines. Prose length and vocabulary are
# never gated. Build 4 calls it through this injected callable so tests can
# pass a fake. The real Build 2 validator is adapted to this signature at
# wire-up (see WIRING_SPEC.md).


@dataclass
class TierAResult:
    """Result of the injected Tier-A check over a parsed exchange.

    `ok` is the overall verdict. `reasons` is a list of human-readable
    failure strings -- one or more per failing slot. `failing_slot_ids`
    lists which slots tripped a check (used only for diagnostics; repair
    re-runs the WHOLE exchange regardless, per critique 11).
    """

    ok: bool
    reasons: List[str] = field(default_factory=list)
    failing_slot_ids: List[str] = field(default_factory=list)


# Injected check: (parsed_exchange, expected_slots) -> TierAResult.
#   parsed_exchange: dict[slot_id] -> text (the parse_exchange output)
#   expected_slots:  the VoicedSlot sequence for the group (order + speaker)
TierACheckFn = Callable[[Mapping[str, str], Sequence[VoicedSlot]], TierAResult]


# Free-form creative generate fn (from _otr_model_loader.make_generate_fn
# or the constrained factory). Signature mirrors the real closure:
#   (messages, *, temperature, max_new_tokens) -> str
GenerateFn = Callable[..., str]


# ===========================================================================
# Defaults
# ===========================================================================

DEFAULT_EXCHANGE_TEMPERATURE: float = 0.85
"""Creative-axis temperature. Matches the Stage 2 single-line composer
default so an A/B comparison stays apples-to-apples."""

DEFAULT_EXCHANGE_MAX_NEW_TOKENS: int = 320
"""Token budget for a 2-3 slot exchange. Larger than the single-line
budget (200) because the model emits several lines at once. See
WIRING_SPEC.md for the context-cap / VRAM caution."""

DEFAULT_N: int = 1
"""Candidates per exchange. Build 4 ships N=1 + one repair (best-of-N
exchange tuning is explicitly deferred in the plan). Kept as a parameter
so the integration can experiment without an API change."""

# Lane-enablement chunk 2 (2026-07-05): the STATIC portion of the exchange
# system prompt, extracted so the seam can be pack-routed. This constant is
# the SCIENCE EXTRACTION FIXTURE (byte-identity pinned against the science
# pack's `exchange_system` seam by tests/test_exchange_seam_lane2.py) and
# the default when no pack-routed prompt is injected. The dynamic grounding
# clause is appended by build_exchange_prompt at assembly time.
EXCHANGE_SYSTEM_PROMPT: str = (
    "You write old-time-radio drama dialogue. Write an EXCHANGE "
    "between the speakers below -- naturalistic, with subtext.\n\n"
    "Craft rules:\n"
    "  - Characters should not answer each other too directly.\n"
    "  - At least one line avoids the real question.\n"
    "  - At least one line reveals pressure through a concrete "
    "object or action, not by naming the feeling.\n"
    "  - Do NOT summarize the situation. Do NOT explain the theme.\n"
)

# ===========================================================================
# Result envelope
# ===========================================================================


@dataclass
class ExchangeResult:
    """Typed result the caller acts on.

    status is one of:
      * "ok"             -- exchange parsed + passed Tier-A on the first
                            attempt. lines is the committed-ready map.
      * "ok_repaired"    -- first attempt failed Tier-A; the single
                            repair pass passed. lines is the repaired map.
      * "fail"           -- repair attempt also failed Tier-A, OR the
                            output could not be parsed into one block per
                            slot. The caller falls back to legacy
                            per-line compose (PD1: never break audio).

    lines maps dialogue_slot_id -> text for EVERY slot in the group when
    status is ok / ok_repaired. On "fail" it carries whatever parsed (may
    be partial / empty) for diagnostics only -- the caller must NOT commit
    a "fail" result.
    """

    status: str                                  # ok | ok_repaired | fail
    lines: Dict[str, str] = field(default_factory=dict)
    slot_ids: List[str] = field(default_factory=list)
    repaired: bool = False
    tier_a: Optional[TierAResult] = None         # final Tier-A verdict
    fail_reason: Optional[str] = None
    attempts: int = 0                            # generate calls made (1 or 2)


# ===========================================================================
# Prompt builder
# ===========================================================================


def _slot_ids_of(beat_group: Sequence[VoicedSlot]) -> List[str]:
    return [s.dialogue_slot_id for s in beat_group]


def _contract_for(
    contracts: Mapping[str, Any],
    slot_id: str,
) -> Optional[Any]:
    """Look up the Build 3 contract for a slot id. Tolerates either a
    mapping keyed by slot id or a sequence of contract objects.
    """
    if isinstance(contracts, Mapping):
        return contracts.get(slot_id)
    for c in contracts or ():
        if getattr(c, "dialogue_slot_id", None) == slot_id:
            return c
    return None


def _grounding_pool(
    beat_group: Sequence[VoicedSlot],
    contracts: Mapping[str, Any],
) -> List[str]:
    """Collect concrete-detail candidates from the group's contracts.

    Build 3 derives concrete_detail_required from active_props U
    news.key_terms. Build 4 asks the model to ground the EXCHANGE in ONE
    of these (not one per line) to avoid object spam.
    """
    pool: List[str] = []
    for slot in beat_group:
        c = _contract_for(contracts, slot.dialogue_slot_id)
        if c is None:
            continue
        for d in (getattr(c, "concrete_detail_required", None) or ()):
            d = str(d).strip()
            if d and d not in pool:
                pool.append(d)
    return pool


def build_exchange_prompt(
    beat_group: Sequence[VoicedSlot],
    contracts: Mapping[str, Any],
    prior_committed_lines: Sequence[str],
    cast: Sequence[Any],
    *,
    failure_reasons: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    source_block: str = "",
) -> List[dict]:
    """Build the chat messages for one exchange over a 2-3 slot group.

    Args:
        beat_group: the VoicedSlot sequence (2-3 voiced slots). Slot ids
            and speakers are authoritative -- the model must preserve them.
        contracts: Build 3 slot_drama_contracts, keyed by slot id (or a
            sequence of contract objects). Read-only here.
        prior_committed_lines: already-committed lines (as displayed,
            e.g. "d004|MARLOW: ...") providing scene context. Truncated
            by the caller against context_cap (see WIRING_SPEC.md).
        cast: cast members (objects exposing .name / .persona). Used to
            give the model each speaker's voice.
        failure_reasons: on the REPAIR pass, the Tier-A failure strings
            from the first attempt are appended so the rewrite is
            targeted. None on the first attempt.
        system_prompt: the STATIC system-prompt portion. None (default)
            uses EXCHANGE_SYSTEM_PROMPT (the science extraction fixture).
            The writer injects the bank's pack-routed `exchange_system`
            seam here (lane-enablement chunk 2) -- this module stays
            import-free of the routing layer by design. The dynamic grounding
            guidance is always appended after it.

    Returns:
        chat messages (list of {"role", "content"}) for generate_fn.

    The craft instructions (from the plan): characters shouldn't answer
    too directly; at least one line avoids the real question; at least
    one line reveals pressure through a concrete object or action; do not
    summarize the situation; do not explain the theme. De-exposition:
    require ONE concrete grounding per EXCHANGE (not per line). The
    authored vocabulary is never filtered or scored by this module.
    """
    slot_ids = _slot_ids_of(beat_group)
    persona_by_name = {
        getattr(c, "name", ""): getattr(c, "persona", "")
        for c in (cast or ())
    }

    # Speaker roster for this exchange.
    roster_lines = []
    for slot in beat_group:
        persona = persona_by_name.get(slot.speaker, "")
        roster_lines.append(
            f"  - {slot.speaker}"
            + (f": {persona}" if persona else "")
        )
    roster_block = "\n".join(roster_lines) or "  (none)"

    # Per-slot obligations from the Build 3 contract.
    slot_blocks = []
    for slot in beat_group:
        c = _contract_for(contracts, slot.dialogue_slot_id)
        line_job = getattr(c, "line_job", "") if c else ""
        hidden = getattr(c, "hidden_pressure", "") if c else ""
        must_turn = bool(getattr(c, "must_turn", False)) if c else False
        intent = (slot.intent or "").strip()
        parts = [f"  {slot.dialogue_slot_id} ({slot.speaker}):"]
        if intent:
            parts.append(f"    beat intent: {intent}")
        if line_job:
            parts.append(f"    line job: {line_job}")
        if hidden:
            parts.append(f"    hidden pressure (do NOT state outright): {hidden}")
        if must_turn:
            parts.append(
                "    this line must TURN the scene "
                "(a refusal, reversal, or shift -- not a restatement)"
            )
        slot_blocks.append("\n".join(parts))
    slot_block = "\n".join(slot_blocks)

    grounding = _grounding_pool(beat_group, contracts)
    grounding_clause = (
        "Ground the exchange in ONE concrete detail -- a single object or "
        "physical action somewhere in these lines, not one per line. "
        + (
            f"Draw it from: {', '.join(grounding[:8])}."
            if grounding
            else "Pick one concrete object or action that fits the scene."
        )
    )

    prior_block = "\n".join(prior_committed_lines) if prior_committed_lines else "  (scene opens here)"

    # Lane chunk 2: static portion is injectable (pack-routed by the writer);
    # None keeps the module constant -- byte-identical assembly either way
    # when the injected value equals the constant (science lane, test-pinned).
    base = EXCHANGE_SYSTEM_PROMPT if system_prompt is None else system_prompt
    system = base + f"  - {grounding_clause}\n"
    from ._otr_dialogue_policy import append_dialogue_policy
    # Only the speakers this exchange actually voices. `cast` stays where it
    # belongs -- rendering voice guidance for those speakers in the user
    # message -- and no longer votes on the accent policy.
    system = append_dialogue_policy(
        system,
        active_speakers=tuple(slot.speaker for slot in beat_group),
    )

    fmt = (
        "Output EXACTLY one line per slot id, in this order, nothing else:\n"
        + "\n".join(f"  {sid}|SPEAKER: <line>" for sid in slot_ids)
        + "\n\n"
        "Keep each slot id and its speaker EXACTLY as given -- do not add, "
        "drop, reorder, merge, or rename slots. Internal pauses (...) are "
        "fine inside a line, but emit one line per slot id."
    )

    user_parts = [
        "Speakers:",
        roster_block,
        "",
        "Scene so far:",
        prior_block,
        "",
    ]

    # The source passage rides the USER message as delimited DATA, never the
    # pack-routed system seam. Two reasons, both load-bearing: the seam is a
    # static, test-pinned constant that must stay byte-identical, and a source
    # is quoted material whose imperative sentences ("Come thy ways, Signior")
    # must never read as direction to the model. It sits BEFORE the slot
    # instructions so the last thing the model reads is what to write, not
    # what to read.
    if source_block and not isinstance(source_block, str):
        # Fail loud rather than coerce. Passing a SourceSpan or SourceGrounding
        # here would embed a repr in a live prompt -- and because the repr is
        # deliberately body-free, the model would receive a description of the
        # passage instead of the passage, which reads as plausible grounding
        # while grounding nothing.
        raise TypeError(
            f"source_block must be a pre-rendered str from "
            f"render_source_block, got {type(source_block).__name__}"
        )
    if source_block:
        user_parts.extend([
            "The passage below is the SOURCE this scene adapts. Carry its "
            "people, place, period and events; where it gives these "
            "characters words, carry those words.",
            source_block,
            "",
        ])

    user_parts.extend([
        "Write these slots as one exchange:",
        slot_block,
        "",
        fmt,
    ])

    if failure_reasons:
        user_parts.extend([
            "",
            "Your previous attempt failed these integrity checks -- fix "
            "ALL of them while keeping every slot id and speaker exact:",
            "\n".join(f"  - {r}" for r in failure_reasons),
        ])

    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ===========================================================================
# Parser -- enforce one block per slot
# ===========================================================================


class ExchangeParseError(ValueError):
    """Raised when raw output cannot be parsed into exactly one block per
    expected slot id (wrong count, missing slot, extra/unknown slot,
    duplicate slot)."""


def parse_exchange(
    raw: str,
    expected_slot_ids: Sequence[str],
) -> Dict[str, str]:
    """Parse raw exchange output into {slot_id: text}, enforcing one
    block per expected slot id.

    HARD RULE (critique 8): the result must contain EXACTLY the expected
    slot ids -- same set, no duplicates, no extras, none missing. A block
    may contain internal pauses; the count is what is fixed.

    Args:
        raw: model output. Expected line shape `d###|SPEAKER: text`.
        expected_slot_ids: the slot ids the group requires.

    Returns:
        dict mapping each expected slot id to its (stripped, non-empty)
        text.

    Raises:
        ExchangeParseError on any slot-count / id mismatch or empty block.
    """
    expected = list(expected_slot_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise ExchangeParseError(
            f"expected_slot_ids contains duplicates: {expected}"
        )

    parsed: Dict[str, str] = {}
    seen_order: List[str] = []
    for line in (raw or "").splitlines():
        m = SLOT_LINE_RE.match(line)
        if not m:
            continue
        sid = m.group("slot_id")
        text = m.group("text").strip()
        if not text:
            # Empty block -- treat as a parse failure for this slot so the
            # one-block-per-slot invariant cannot be silently satisfied by
            # a blank line.
            raise ExchangeParseError(f"slot {sid} parsed an empty text block")
        if sid in parsed:
            raise ExchangeParseError(
                f"slot {sid} appears more than once (one block per slot)"
            )
        parsed[sid] = text
        seen_order.append(sid)

    got = set(parsed)
    if got != expected_set:
        missing = sorted(expected_set - got)
        extra = sorted(got - expected_set)
        raise ExchangeParseError(
            "slot id mismatch -- one block per expected slot required. "
            f"missing={missing} extra={extra} "
            f"(expected={expected}, got={seen_order})"
        )

    # Return in the expected order for deterministic downstream commit.
    return {sid: parsed[sid] for sid in expected}


# ===========================================================================
# compose_exchange -- generate, parse, Tier-A, repair-by-group
# ===========================================================================


def _run_once(
    beat_group: Sequence[VoicedSlot],
    contracts: Mapping[str, Any],
    prior_lines: Sequence[str],
    cast: Sequence[Any],
    *,
    generate_fn: GenerateFn,
    temperature: float,
    max_new_tokens: int,
    failure_reasons: Optional[Sequence[str]],
    system_prompt: Optional[str],
    source_block: str = "",
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """One generate + parse cycle. Returns (parsed_or_None, parse_error).

    Does NOT run Tier-A -- the caller composes that so the repair branch
    can append the failure reasons.

    # LLM slot: creative
    # Reason: the single creative-axis dialogue render for the exchange.
    """
    expected = _slot_ids_of(beat_group)
    messages = build_exchange_prompt(
        beat_group,
        contracts,
        prior_lines,
        cast,
        failure_reasons=failure_reasons,
        system_prompt=system_prompt,
        source_block=source_block,
    )
    # LLM slot: creative
    # Reason: exchange dialogue rendering is creative-axis work (rule 6).
    raw = generate_fn(
        messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    try:
        parsed = parse_exchange(raw or "", expected)
    except ExchangeParseError as exc:
        return None, str(exc)
    return parsed, None


def compose_exchange(
    beat_group: Sequence[VoicedSlot],
    contracts: Mapping[str, Any],
    prior_lines: Sequence[str],
    cast: Sequence[Any],
    *,
    generate_fn: GenerateFn,
    tier_a_check: TierACheckFn,
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_EXCHANGE_TEMPERATURE,
    max_new_tokens: int = DEFAULT_EXCHANGE_MAX_NEW_TOKENS,
    system_prompt: Optional[str] = None,
    source_block: str = "",
) -> ExchangeResult:
    """Compose one exchange over a 2-3 voiced beat group.

    Flow:
      1. Generate + parse (one block per slot, ids enforced).
      2. Run the injected Tier-A integrity check.
      3. If Tier-A passes -> status "ok".
      4. If Tier-A fails (or parse failed) -> re-run the WHOLE exchange
         ONCE with the failure reasons appended (repair by group, not by
         slot -- critique 11). If the repair passes -> "ok_repaired".
      5. Still failing after one repair -> status "fail" (caller falls
         back to legacy per-line compose; PD1: never break audio).

    ONE repair attempt only -- no multi-cycle churn.

    Args:
        beat_group: 2-3 VoicedSlots. Slot ids + speakers are authoritative.
        contracts: Build 3 slot_drama_contracts (read-only).
        prior_lines: already-committed scene lines (caller truncates to
            context_cap).
        cast: cast members (.name / .persona).
        generate_fn: creative-slot generate closure
            (messages, *, temperature, max_new_tokens) -> str. Injected.
        tier_a_check: Build 2 integrity check adapted to TierACheckFn.
            Injected so tests pass a fake.
        n: candidates per attempt (default 1; kept for future best-of-N).
            With n>1 the first parsed candidate that passes Tier-A wins;
            if none pass, the last parsed candidate's reasons drive repair.
        system_prompt: static system-prompt portion (lane chunk 2); None
            uses EXCHANGE_SYSTEM_PROMPT. Threaded to build_exchange_prompt
            on both the first attempt and the repair pass.

    Returns:
        ExchangeResult. Only "ok" / "ok_repaired" results are safe to
        commit; "fail" signals legacy fallback.

    # LLM slot: creative
    # Reason: orchestrates the creative-axis exchange render + one
    # targeted repair render (rule 6).
    """
    expected = _slot_ids_of(beat_group)
    result = ExchangeResult(status="fail", slot_ids=list(expected))

    if not beat_group:
        result.fail_reason = "empty beat_group"
        return result

    n = max(1, int(n))

    # ---- First attempt (best-of-n parse+TierA on the first pass) --------
    last_reasons: List[str] = []
    last_parsed: Optional[Dict[str, str]] = None
    first_attempt_failed = False

    for _ in range(n):
        parsed, parse_err = _run_once(
            beat_group, contracts, prior_lines, cast,
            generate_fn=generate_fn,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            failure_reasons=None,
            system_prompt=system_prompt,
            source_block=source_block,
        )
        result.attempts += 1
        if parsed is None:
            first_attempt_failed = True
            last_reasons = [f"parse failure: {parse_err}"]
            log.info(
                "[ComposeExchange] slots=%s parse failed: %s",
                expected, parse_err,
            )
            continue
        last_parsed = parsed
        verdict = tier_a_check(parsed, beat_group)
        if verdict.ok:
            result.status = "ok"
            result.lines = dict(parsed)
            result.tier_a = verdict
            log.info("[ComposeExchange] slots=%s ok (attempts=%d)",
                     expected, result.attempts)
            return result
        first_attempt_failed = True
        last_reasons = list(verdict.reasons) or ["Tier-A failed (no reasons given)"]
        result.tier_a = verdict
        log.info(
            "[ComposeExchange] slots=%s Tier-A failed: %s",
            expected, last_reasons,
        )

    if not first_attempt_failed:
        # Defensive: loop should have returned on success.
        result.fail_reason = "internal: no first-attempt verdict produced"
        return result

    # ---- Single repair pass (whole exchange, reasons appended) ----------
    repair_parsed, repair_parse_err = _run_once(
        beat_group, contracts, prior_lines, cast,
        generate_fn=generate_fn,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        failure_reasons=last_reasons,
        system_prompt=system_prompt,
        # THE SAME window as the first attempt. Reselecting here would mean
        # the repair quoted different material than the attempt whose
        # failure reasons it is answering, and the receipt could no longer
        # say which passage the accepted line came from.
        source_block=source_block,
    )
    result.attempts += 1
    result.repaired = True

    if repair_parsed is None:
        result.status = "fail"
        result.lines = dict(last_parsed) if last_parsed else {}
        result.fail_reason = (
            f"repair parse failure: {repair_parse_err} "
            f"(first-attempt reasons: {last_reasons})"
        )
        log.info(
            "[ComposeExchange] slots=%s repair parse failed: %s -- fail-loud",
            expected, repair_parse_err,
        )
        return result

    repair_verdict = tier_a_check(repair_parsed, beat_group)
    result.tier_a = repair_verdict
    if repair_verdict.ok:
        result.status = "ok_repaired"
        result.lines = dict(repair_parsed)
        log.info(
            "[ComposeExchange] slots=%s ok after one repair (attempts=%d)",
            expected, result.attempts,
        )
        return result

    # Still failing after one repair -> fail loud / legacy fallback.
    result.status = "fail"
    result.lines = dict(repair_parsed)
    result.fail_reason = (
        "Tier-A still failing after one repair: "
        + "; ".join(repair_verdict.reasons or ["(no reasons)"])
    )
    log.info(
        "[ComposeExchange] slots=%s still failing after repair -- fail-loud "
        "(caller falls back to legacy)",
        expected,
    )
    return result


# ===========================================================================
# Integration seams (Build 4 wiring helpers)
# ===========================================================================
#
# The two helpers below are the seams the writer integration uses to drive
# compose_exchange. Both are pure -- the Build 2 functions are injected --
# so they unit-test without torch or a live model. See
# build4/WIRING_SPEC.md sections 1 (grouping) and 3 (Tier-A adapter).


def group_voiced_beats(
    slots: Sequence[Any],
    *,
    min_size: int = 2,
    max_size: int = 3,
    reserved_speakers: Sequence[str] = ("ANNOUNCER",),
) -> List[List[Any]]:
    """Partition an ordered slot sequence into exchange groups.

    Grouping rules (build4/WIRING_SPEC.md section 1):
      * Only consecutive VOICED, non-reserved slots are grouped. A
        reserved-speaker slot (ANNOUNCER / MUSIC, matched case-
        insensitively against `reserved_speakers`) or any slot whose
        speaker is empty is emitted as its own singleton group and is
        NOT merged into a neighbour -- it keeps its dedicated pass.
      * A run of voiced slots is chunked into groups of size
        `min_size`..`max_size`, choosing sizes so a groupable run never
        leaves a trailing singleton (a run of 4 -> [2, 2], not [3, 1]).
        A run of length 1 stays a singleton.
      * Order is preserved end to end.

    Returns a list of groups (each a list of the original slot objects).
    The caller routes by size: a group of >= `min_size` goes to
    compose_exchange; a singleton goes to the existing per-slot path
    (legacy single-line compose, or the dedicated ANNOUNCER / MUSIC pass).

    `slots` items only need a `.speaker` attribute (read for the
    reserved / voiced test); compose_exchange reads the rest later.
    """
    reserved_lower = {
        str(s or "").strip().lower() for s in reserved_speakers
    }

    def _is_groupable(slot: Any) -> bool:
        spk = str(getattr(slot, "speaker", "") or "").strip()
        return bool(spk) and spk.lower() not in reserved_lower

    groups: List[List[Any]] = []
    run: List[Any] = []

    def _flush_run() -> None:
        # Chunk a run of groupable slots into sizes in [min_size, max_size]
        # avoiding a trailing singleton wherever the run length allows.
        i = 0
        n = len(run)
        while i < n:
            remaining = n - i
            if remaining <= max_size:
                size = remaining
            elif remaining - max_size < min_size:
                # taking max_size would strand < min_size -> take min_size
                size = min_size
            else:
                size = max_size
            groups.append(run[i:i + size])
            i += size
        run.clear()

    for slot in slots or ():
        if _is_groupable(slot):
            run.append(slot)
        else:
            _flush_run()
            groups.append([slot])
    _flush_run()
    return groups


def make_tier_a_adapter(
    evaluate_tier_a: Callable[..., Any],
    normalize_slot_line: Callable[[str, str, str], str],
) -> TierACheckFn:
    """Adapt Build 2's deterministic Tier-A validator to a TierACheckFn.

    Build 2 exposes `evaluate_tier_a(raw, manifest)` over
    the `d###|SPEAKER: text` slot format plus `normalize_slot_line` to
    serialize one block. This wraps them into the
    `(parsed, expected_slots) -> TierAResult` signature compose_exchange
    calls, so the integration injects the real Build 2 functions and this
    module stays import-free / unit-testable with fakes.

    The serialized rows use each expected slot's authoritative speaker, so
    the adapter exercises Build 2's count / order / empty checks; prose
    length and vocabulary never participate. The speaker is
    bound to the slot id by construction
    (parse_exchange already fixed the ids), so SPEAKER_MISMATCH cannot
    false-fire here.
    """

    def _adapter(
        parsed: Mapping[str, str],
        expected_slots: Sequence[VoicedSlot],
    ) -> TierAResult:
        raw = "\n".join(
            normalize_slot_line(
                s.dialogue_slot_id,
                s.speaker,
                parsed.get(s.dialogue_slot_id, ""),
            )
            for s in expected_slots
        )
        manifest = [
            {"slot_id": s.dialogue_slot_id, "speaker": s.speaker}
            for s in expected_slots
        ]
        verdict = evaluate_tier_a(raw, manifest)
        failures = list(getattr(verdict, "failures", []) or [])
        reasons = [
            str(getattr(f, "detail", "") or getattr(f, "code", ""))
            for f in failures
        ]
        failing = [
            str(getattr(f, "slot_id", "") or "")
            for f in failures
            if str(getattr(f, "slot_id", "") or "")
        ]
        return TierAResult(
            ok=bool(getattr(verdict, "passed", False)),
            reasons=reasons,
            failing_slot_ids=failing,
        )

    return _adapter


# ===========================================================================
# run_exchange_prepass -- the writer-loop orchestration (testable)
# ===========================================================================
#
# The writer integration calls this ONCE before its per-beat render loop.
# It groups consecutive voiced beats, composes each group as an exchange,
# and returns {beat_id: text} for the beats that composed cleanly. The
# per-beat loop then short-circuits those beats to the returned text and
# leaves every other beat (singletons, ANNOUNCER / MUSIC, failed groups)
# on its existing path -- so audio is never blocked (Prime Directive 1).
# Pure given the injected generate_fn + tier_a_check (no torch, no I/O),
# so the whole orchestration unit-tests with fakes.


class _CastShim:
    """Adapter so dict cast rows expose .name / .persona to the prompt."""

    __slots__ = ("name", "persona")

    def __init__(self, name: str, persona: str) -> None:
        self.name = name
        self.persona = persona


def _normalize_cast(cast: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for c in (cast or ()):
        if isinstance(c, dict):
            out.append(
                _CastShim(
                    str(c.get("name") or ""),
                    str(c.get("persona") or c.get("voice") or ""),
                )
            )
        else:
            out.append(c)
    return out


def run_exchange_prepass(
    beats: Sequence[Any],
    contracts: Mapping[str, Any],
    cast: Sequence[Any],
    *,
    generate_fn: GenerateFn,
    tier_a_check: TierACheckFn,
    prior_window: int = 4,
    min_size: int = 2,
    max_size: int = 3,
    reserved_speakers: Sequence[str] = ("ANNOUNCER",),
    temperature: float = DEFAULT_EXCHANGE_TEMPERATURE,
    max_new_tokens: int = DEFAULT_EXCHANGE_MAX_NEW_TOKENS,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Compose voiced beat groups as exchanges; return {beat_id: text}.

    Walks `beats` in order, accumulating runs of consecutive voiced
    (non-reserved, non-empty-speaker, valid d### slot id) beats. Each run
    is chunked by group_voiced_beats; every group of >= min_size is
    rendered with compose_exchange. Only "ok" / "ok_repaired" results
    contribute lines; a "fail" group or a singleton contributes nothing,
    so the caller renders those beats via its existing path.

    `beats` items are duck-typed: .dialogue_slot_id, .speaker, .beat_id,
    and .intent.

    prior_window: how many already-composed display lines to feed the
    next group as scene context. Composed lines accumulate as the prepass
    advances so later groups see earlier dialogue.

    system_prompt: static system-prompt portion (lane chunk 2). The writer
    resolves the bank's pack-routed `exchange_system` seam and injects it
    here; None uses EXCHANGE_SYSTEM_PROMPT (science fixture). This module
    never imports the routing layer -- injection preserves its purity.

    Returns {beat_id: text} for beats whose group composed cleanly. Pure
    given the injected generate_fn + tier_a_check.
    """
    reserved_lower = {
        str(s or "").strip().lower() for s in reserved_speakers
    }
    cast_norm = _normalize_cast(cast)

    def _slot_id(b: Any) -> str:
        return str(getattr(b, "dialogue_slot_id", "") or "").strip()

    def _speaker(b: Any) -> str:
        return str(getattr(b, "speaker", "") or "").strip()

    def _groupable(b: Any) -> bool:
        spk = _speaker(b)
        return (
            bool(_SLOT_ID_RE.match(_slot_id(b)))
            and bool(spk)
            and spk.lower() not in reserved_lower
        )

    out: Dict[str, str] = {}
    prior: List[str] = []

    def _compose_run(run_beats: List[Any]) -> None:
        if not run_beats:
            return
        slot_to_beat = {_slot_id(b): b for b in run_beats}
        voiced_slots = [
            VoicedSlot(
                dialogue_slot_id=_slot_id(b),
                speaker=_speaker(b),
                intent=str(getattr(b, "intent", "") or ""),
            )
            for b in run_beats
        ]
        for grp in group_voiced_beats(
            voiced_slots,
            min_size=min_size,
            max_size=max_size,
            reserved_speakers=reserved_speakers,
        ):
            if len(grp) < min_size:
                continue
            res = compose_exchange(
                grp,
                contracts,
                prior[-prior_window:] if prior_window > 0 else [],
                cast_norm,
                generate_fn=generate_fn,
                tier_a_check=tier_a_check,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
            if res.status not in ("ok", "ok_repaired"):
                continue
            for vs in grp:
                sid = vs.dialogue_slot_id
                text = res.lines.get(sid, "")
                b = slot_to_beat.get(sid)
                if not text or b is None:
                    continue
                bid = str(getattr(b, "beat_id", "") or "").strip()
                if bid:
                    out[bid] = text
                prior.append(f"{sid}|{vs.speaker}: {text}")

    run: List[Any] = []
    for b in beats or ():
        if _groupable(b):
            run.append(b)
        else:
            _compose_run(run)
            run = []
    _compose_run(run)
    return out

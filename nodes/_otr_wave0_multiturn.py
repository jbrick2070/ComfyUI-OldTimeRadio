"""Sprint 10B Wave 0 -- in-loop wire-up of the dormant Stage 2 multiturn composer.

The dormant Stage 2 multi-turn machinery (nodes/_otr_stage2_call.py +
nodes/_otr_stage2_prompt.py, 30 green tests in
tests/test_stage2_multiturn.py) has the right shape but nothing wired
to it. The production writer at
OTR_LedgerScriptWriter.py:2872 and :2938 calls the LEGACY one-shot
_OTRLC.compose_line. This module is the wire-up: it builds the
Stage 2 inputs (Stage1Plan + Stage1Beat) from the writer's in-loop
state and dispatches to either the legacy composer or the multiturn
composer based on the new `use_multiturn_dialogue` widget.

Per design doc `docs/2026-05-26-good-story-writer-architecture__02_
design.md` Section 4 Wave 0:

    * One commit, one widget, one adapter.
    * Default `use_multiturn_dialogue=False` so the legacy byte-identity
      contract holds out-of-the-box.
    * Reserved speakers (ANNOUNCER / MUSIC) always route to legacy --
      Stage 2 compose_line short-circuits on reserved speakers anyway,
      so routing them to legacy avoids an unnecessary fallback path.
    * On any multiturn failure (empty record, exception), fall back to
      the legacy composer rather than ship a blank line. PD1 ("audio
      is king") -- no path that breaks audio output.

The plan adapter reuses the existing `_otr_legacy_to_stage1_adapter`
logic by feeding it a synthetic ledger-shaped dict built from the
writer's in-loop state. Wave 1 Agent B (Stage 3 validators in legacy
compose_line) will reuse the per-beat adapter here.

PURE module. No I/O, no model loads. The dispatch wrapper accepts a
generate_fn (a closure from the writer's slot scheduler) and a
legacy-compose callable so this module imports nothing from the writer.

# LLM slot: creative
# Reason: the dispatch wrapper forwards a creative-slot generate_fn
# to the Stage 2 multiturn composer; Stage 2 itself is tagged creative
# in _otr_stage2_call.compose_line. No new LLM call introduced here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pydantic import ValidationError

from ._otr_legacy_to_stage1_adapter import legacy_ledger_to_stage1_plan
from ._otr_stage1_plan import Stage1Beat, Stage1Plan
from ._otr_stage2_call import (
    DEFAULT_BEST_OF_N,
    Stage2LineRecord,
    compose_line as stage2_compose_line,
)


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reserved speakers Stage 2 does not generate for. The Stage 2
# compose_line returns a fallback record on these speakers; we route
# them to legacy unconditionally so the widget toggle is a no-op on
# announcer / music beats.
RESERVED_SPEAKERS: frozenset[str] = frozenset({"ANNOUNCER", "MUSIC"})

# Default mode per the design doc + the Stage 2 module's own default.
DEFAULT_MULTITURN_MODE: str = "roleplay_multiturn"


# ---------------------------------------------------------------------------
# Per-beat adapter -- legacy Beat -> Stage1Beat
# ---------------------------------------------------------------------------


# Stage1Beat schema requires beat_id to match b\d{3}; the legacy
# ledger's beat_ids are mostly already in that shape (b001, b002, ...).
# This coerces any stray shape (e.g. raw int, "beat_1") into bNNN.
def _coerce_beat_id(raw: Any, fallback_index: int = 0) -> str:
    """Coerce a legacy beat_id to the Stage1Beat bNNN pattern.

    Args:
        raw: whatever the legacy beat carries as beat_id.
        fallback_index: 0-based position used to synthesize an id if
            the raw value cannot be coerced.

    Returns:
        A string matching b\\d{3} (e.g. 'b001').
    """
    if isinstance(raw, str) and raw.startswith("b"):
        digits = "".join(c for c in raw[1:] if c.isdigit())
        if digits:
            return f"b{int(digits):03d}"
    if isinstance(raw, int) and raw >= 0:
        return f"b{raw:03d}"
    return f"b{max(0, int(fallback_index)):03d}"


def line_request_to_stage1_beat(
    legacy_beat: Any,
    *,
    fallback_index: int = 0,
) -> Stage1Beat:
    """Build a Stage1Beat from a legacy outline Beat.

    Stage1Beat field requirements (per Stage1Plan schema):
        beat_id            b\\d{3}
        speaker            cast member name OR 'ANNOUNCER' / 'MUSIC'
        intent             5..200 chars
        length_target_words 5..200
        emotional_register 3..80 chars
        callback_to        None or b\\d{3}

    Legacy Beat fields the writer constructs upstream:
        beat_id, speaker, speaker_role, intent, mood, target_words,
        arc_phase, sfx_cue

    Args:
        legacy_beat: an outline Beat (pydantic) or a duck-typed object
            with the same attribute surface. Tests pass dataclass
            mocks here.
        fallback_index: 0-based loop position used if beat_id is
            unrecoverable; the writer's render loop passes the
            enumeration index.

    Returns:
        Stage1Beat ready for Stage 2 compose_line.

    Never raises. Short / missing fields are padded with placeholders
    that satisfy the schema; the rendered line goes through the
    multiturn composer regardless. The caller is responsible for
    routing reserved speakers (ANNOUNCER / MUSIC) to legacy -- this
    adapter happily produces a Stage1Beat for them, but Stage 2
    compose_line will return a fallback record on reserved speakers.

    # LLM slot: creative
    # Reason: pure adapter -- no LLM call -- but lives in the
    # creative-axis dispatch path so the slot sweep stays happy.
    """
    beat_id = _coerce_beat_id(
        getattr(legacy_beat, "beat_id", None),
        fallback_index=fallback_index,
    )
    raw_speaker = getattr(legacy_beat, "speaker", "") or ""
    speaker = str(raw_speaker).strip()
    if not speaker:
        speaker = "ANNOUNCER"  # safe placeholder; reserved speaker = caller routes to legacy

    raw_intent = (getattr(legacy_beat, "intent", "") or "").strip()
    if len(raw_intent) < 5:
        intent = "Beat intent not preserved by the legacy outline."
    else:
        intent = raw_intent[:200]

    raw_target = getattr(legacy_beat, "target_words", None)
    try:
        target = int(raw_target) if raw_target is not None else 20
    except (TypeError, ValueError):
        target = 20
    target = max(5, min(200, target))

    raw_register = (getattr(legacy_beat, "mood", "") or "").strip()
    if len(raw_register) < 3:
        register = "neutral"
    else:
        register = raw_register[:80]

    try:
        return Stage1Beat(
            beat_id=beat_id,
            speaker=speaker[:40],
            intent=intent,
            length_target_words=target,
            emotional_register=register,
            callback_to=None,
        )
    except ValidationError as exc:
        # Final guard: if anything still trips pydantic, produce a
        # safe placeholder beat. Stage 2 will produce SOMETHING; the
        # dispatch wrapper falls back to legacy on empty output.
        log.warning(
            "[Wave0Adapter] Stage1Beat validation failed for "
            "beat_id=%r speaker=%r: %s -- using placeholder beat",
            beat_id, speaker, str(exc)[:200],
        )
        return Stage1Beat(
            beat_id=beat_id,
            speaker="ANNOUNCER",
            intent="Adapter placeholder (legacy beat failed validation).",
            length_target_words=20,
            emotional_register="neutral",
            callback_to=None,
        )


# ---------------------------------------------------------------------------
# In-loop Stage1Plan builder
# ---------------------------------------------------------------------------


def _build_synthetic_led_data(
    outline: Any,
    cast_rows: List[dict],
    meta: dict,
) -> dict:
    """Shape the writer's in-loop state into the dict the existing
    legacy_ledger_to_stage1_plan adapter expects.

    The legacy adapter was written for the POST-render ledger (where
    meta and lines exist). In-loop we have outline (with .beats) but no
    rendered lines yet. We synthesize empty 'lines' from outline.beats
    so the adapter populates plan.beats; Stage 2 compose_line reads
    plan.cast / plan.arc / plan.premise / plan.running_facts, not
    plan.beats, so empty texts are fine.
    """
    cast_rows = list(cast_rows or [])

    # Build char_id lookup from cast_rows.
    char_id_by_name: dict[str, Any] = {}
    for row in cast_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or ""
        char_id = row.get("char_id")
        if name and char_id is not None:
            char_id_by_name[name] = char_id

    synth_lines: List[dict] = []
    beats = list(getattr(outline, "beats", None) or [])
    for i, b in enumerate(beats):
        speaker = (getattr(b, "speaker", "") or "").strip()
        role_attr = (getattr(b, "speaker_role", "") or "").strip().lower()
        char_id = char_id_by_name.get(speaker)
        synth_lines.append({
            "beat_id": getattr(b, "beat_id", f"b{i+1:03d}"),
            "speaker_role": role_attr or "character",
            "char_id": char_id,
            "text": "",
            "intent": getattr(b, "intent", "") or "",
            "mood": getattr(b, "mood", "") or "",
            "emotional_register": getattr(b, "mood", "") or "",
        })

    premise = (getattr(outline, "premise", "") or "").strip()

    return {
        "cast": cast_rows,
        "lines": synth_lines,
        "meta": {
            "episode_canon": {
                "premise": premise,
                "arc": {
                    "setup": "",
                    "complication": "",
                    "resolution": "",
                },
            },
            "continuity_ledger": (meta or {}).get("continuity_ledger", {}),
        },
    }


def build_inloop_stage1_plan(
    outline: Any,
    cast_rows: List[dict],
    meta: dict,
) -> Optional[Stage1Plan]:
    """Build a best-effort Stage1Plan from the writer's in-loop state.

    Returns None if the plan cannot be built (e.g., no usable cast
    rows). The caller should route to legacy compose_line in that case.

    The plan's arc is placeholder text -- the legacy outline does not
    carry structured setup/complication/resolution fields. Stage 2's
    Turn 2 prompt embeds the arc; Wave 1 Agent D's Director will
    produce real arc text from a news brief, replacing the placeholder.
    For Wave 0 the placeholder is acceptable -- the multiturn chain
    still gets premise + cast + running_facts grounding.

    Reuses legacy_ledger_to_stage1_plan so all the pydantic edge cases
    (gender coercion, voice_id uniqueness, intent padding) are
    handled in one place.
    """
    led_data = _build_synthetic_led_data(outline, cast_rows, meta or {})
    return legacy_ledger_to_stage1_plan(led_data)


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------


@dataclass
class MultiturnDispatchResult:
    """Result envelope from a dispatch call.

    Carries the cleaned line text and the path taken so the writer can
    stamp meta.dialogue_path per episode (or per beat in soak
    diagnostics).
    """

    text: str
    path: str               # one of: 'multiturn', 'legacy_fallback', 'legacy'
    record: Optional[Stage2LineRecord] = None
    fallback_reason: Optional[str] = None


def dispatch_compose_line(
    *,
    use_multiturn_dialogue: bool,
    # multiturn inputs (only used when widget is True AND speaker is
    # not reserved):
    plan: Optional[Stage1Plan],
    beat: Stage1Beat,
    creative_generate_fn: Callable[..., str],
    multiturn_mode: str = DEFAULT_MULTITURN_MODE,
    multiturn_n: int = DEFAULT_BEST_OF_N,
    previous_speaker: Optional[str] = None,
    previous_line: Optional[str] = None,
    banned_phrases: Optional[List[str]] = None,
) -> MultiturnDispatchResult:
    """Dispatch one compose_line call to either multiturn or legacy.

    Returns a MultiturnDispatchResult. If the multiturn path produces
    a non-empty line, the result carries path='multiturn'. If the
    multiturn path returns an empty record (reserved speaker,
    all-empty candidates, exception), the result carries
    path='legacy_fallback' AND text=''. The CALLER is responsible for
    invoking the legacy composer in that case -- this module owns
    only the multiturn dispatch.

    When use_multiturn_dialogue=False, the result is always
    path='legacy' with text=''; the caller invokes the legacy composer
    unchanged.

    PD1: this function NEVER raises. Any exception in the multiturn
    path is caught and turned into a path='legacy_fallback' result.

    Args:
        use_multiturn_dialogue: the widget value.
        plan: Stage1Plan built from in-loop state. None forces
            legacy_fallback.
        beat: Stage1Beat for this line. Speakers in RESERVED_SPEAKERS
            force legacy_fallback (Stage 2 doesn't generate for them).
        creative_generate_fn: the writer's creative_generate_fn.
        multiturn_mode: 'roleplay_multiturn' (default) or
            'roleplay_single_turn'.
        multiturn_n: best-of-N candidate count. Default 4.
        previous_speaker / previous_line / banned_phrases: forwarded
            to Stage 2 compose_line.

    Returns:
        MultiturnDispatchResult carrying the path taken and the line.

    # LLM slot: creative
    # Reason: forwards a creative-slot generate_fn into Stage 2
    # compose_line (creative slot per Stage 2 module's tag).
    """
    if not use_multiturn_dialogue:
        return MultiturnDispatchResult(text="", path="legacy")

    if plan is None:
        return MultiturnDispatchResult(
            text="",
            path="legacy_fallback",
            fallback_reason="build_inloop_stage1_plan returned None",
        )

    if (beat.speaker or "").strip().upper() in RESERVED_SPEAKERS:
        return MultiturnDispatchResult(
            text="",
            path="legacy_fallback",
            fallback_reason=(
                f"speaker {beat.speaker!r} is reserved; Stage 2 does "
                f"not generate for reserved speakers"
            ),
        )

    if (beat.speaker or "").strip() not in {c.name for c in plan.cast}:
        return MultiturnDispatchResult(
            text="",
            path="legacy_fallback",
            fallback_reason=(
                f"speaker {beat.speaker!r} is not in plan.cast "
                f"(in-loop adapter could not resolve)"
            ),
        )

    try:
        # LLM slot: creative
        # Reason: Stage 2 Turn 4 is creative-axis dialogue rendering
        # (tagged inside _otr_stage2_call.compose_line as well).
        record = stage2_compose_line(
            plan=plan,
            beat=beat,
            generate_fn=creative_generate_fn,
            mode=multiturn_mode,
            n=multiturn_n,
            previous_speaker=previous_speaker,
            previous_line=previous_line,
            banned_phrases=banned_phrases,
        )
    except Exception as exc:  # noqa: BLE001 -- PD1: never break audio
        log.warning(
            "[Wave0Dispatch] beat=%s multiturn compose raised %s: %s "
            "-- falling back to legacy composer",
            beat.beat_id, type(exc).__name__, str(exc)[:200],
        )
        return MultiturnDispatchResult(
            text="",
            path="legacy_fallback",
            fallback_reason=(
                f"multiturn raised {type(exc).__name__}: "
                f"{str(exc)[:200]}"
            ),
        )

    if not (record.chosen_text or "").strip():
        log.info(
            "[Wave0Dispatch] beat=%s multiturn produced empty text "
            "(fallback_reason=%r) -- caller will use legacy",
            beat.beat_id, record.fallback_reason,
        )
        return MultiturnDispatchResult(
            text="",
            path="legacy_fallback",
            record=record,
            fallback_reason=(
                record.fallback_reason or "multiturn returned empty text"
            ),
        )

    return MultiturnDispatchResult(
        text=record.chosen_text,
        path="multiturn",
        record=record,
        fallback_reason=None,
    )

"""Legacy ledger -> Stage1Plan adapter.

Builds a best-effort Stage1Plan + rendered_lines list from a legacy
writer ledger: cast rows in led.data['cast'], beat/outline rows in
led.data['lines'], premise/arc text in meta.episode_canon, etc.

Retained because the writer's Stage 3 path reuses this adapter
(OTR_LedgerScriptWriter imports legacy_ledger_to_stage1_plan). The
Sprint 10A whole-episode shadow critic that originally consumed it
was removed in the 2026-05-29 lean-down.

Adapter philosophy: loss-tolerant + non-fatal. If the legacy ledger
is missing a field the Stage 1 schema requires, we fill with a
sensible placeholder so the critic prompt can still build. None is
returned only when the adapter CANNOT produce a plan that would
parse through Stage1Plan -- in which case the caller skips the
shadow critic for this episode.

Module is PURE: no LLM, no I/O.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import ValidationError

from ._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants -- gender/voice defaults for fields the legacy ledger lacks
# ---------------------------------------------------------------------------

# When the adapter can't resolve a field from legacy data, fall back to
# placeholders that satisfy the schema. The critic still scores the
# RENDERED transcript fine; these are background details.

_PLACEHOLDER_PERSONA: str = (
    "Legacy cast member -- persona not preserved by the v1 writer flow."
)
_PLACEHOLDER_ARC_ROLE: str = "supporting role"
_PLACEHOLDER_VOICE_ID: str = "v2/en_speaker_5"
_PLACEHOLDER_PREMISE: str = (
    "Legacy episode -- premise not preserved by the v1 writer flow."
)
_PLACEHOLDER_ARC_PART: str = (
    "(legacy ledger did not preserve this arc segment)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_news_seed_text(raw: Any) -> str:
    """Coerce meta.news_seed (which is a dict in current ledgers and was a
    plain string in older ones) to a text excerpt the adapter can use as
    a premise placeholder.

    Current ledger shape (post-Sprint-10A NewsFetcher refactor):
        meta.news_seed = {
            "headline": str, "source": str, "url": str,
            "date": str,     "body_chars": int, "style": str,
            "selected_at": str
        }
    Older / handwritten ledgers may carry meta.news_seed as a single
    string (the full body excerpt). We accept either shape.

    BUG-LOCAL-277 (2026-05-26): the live soak run
    signal_lost_..._154105 surfaced an AttributeError on the dict shape
    because the adapter did `(meta.get('news_seed') or '').strip()`,
    assuming the field was always a string. Fix: coerce centrally with
    type-aware logic and let `.strip()` see real strings only.

    Args:
        raw: whatever lives at meta.news_seed -- str, dict, None,
             or something unexpected.

    Returns:
        Best-effort text excerpt, possibly empty. Never raises.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        # Prefer the headline; fall back to source as a last resort so
        # the critic at least sees a topic anchor.
        headline = raw.get("headline")
        if isinstance(headline, str) and headline.strip():
            return headline.strip()
        source = raw.get("source")
        if isinstance(source, str) and source.strip():
            return source.strip()
    return ""


def _gender_to_pronouns(gender: str) -> str:
    """Map cast_pools gender literal -> Stage 1 pronouns literal."""
    g = (gender or "").strip().lower()
    if g == "female":
        return "she/her"
    if g == "nonbinary":
        return "they/them"
    return "he/him"   # default for 'male' AND for unknown / empty


def _normalize_voice_preset(preset: Optional[str]) -> str:
    """Map legacy voice_preset to a Stage 1 voice_id that matches the
    schema regex. Kokoro presets (bm_*, bf_*) and missing values fall
    back to a v2/en_speaker_* placeholder.
    """
    if isinstance(preset, str) and preset.startswith("v2/en_speaker_"):
        # Match Stage 1 schema: only single-digit speaker indices.
        # Strip anything past the first digit so 'v2/en_speaker_10' -> 'v2/en_speaker_1'.
        rest = preset[len("v2/en_speaker_"):]
        if rest and rest[0].isdigit():
            return f"v2/en_speaker_{rest[0]}"
    return _PLACEHOLDER_VOICE_ID


# Reserved speakers used by both the legacy and Stage 1 schemas; these
# are NOT cast rows.
_RESERVED_SPEAKERS = frozenset({"ANNOUNCER", "MUSIC"})


def _stage1_beat_role_from_legacy(role: str) -> str:
    """Map legacy speaker_role to a Stage 1 beat speaker literal."""
    r = (role or "").strip().lower()
    if r == "announcer":
        return "ANNOUNCER"
    if r.startswith("music"):
        return "MUSIC"
    return ""  # 'character' -> resolved from cast lookup below


def _safe_int(value: Any, default: int) -> int:
    try:
        n = int(value)
        if n < 1:
            return default
        return n
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Adapter entry point
# ---------------------------------------------------------------------------


def legacy_ledger_to_stage1_plan(led_data: dict) -> Optional[Stage1Plan]:
    """Build a best-effort Stage1Plan from a legacy ledger dict.

    Returns None if the legacy ledger lacks essentials (no cast or
    no lines) -- the caller should skip the shadow critic in that
    case.

    Args:
        led_data: the in-memory ledger dict (production_ledger
            .Ledger.data shape). Expected keys: 'cast' (list of cast
            rows), 'lines' (list of line rows), 'meta' (dict).

    Returns:
        Stage1Plan if conversion succeeded, None otherwise.
    """
    if not isinstance(led_data, dict):
        return None

    cast_rows: List[dict] = led_data.get("cast") or []
    line_rows: List[dict] = led_data.get("lines") or []
    meta: dict = led_data.get("meta") or {}

    if not cast_rows:
        log.info(
            "[Stage7Shadow] adapter: legacy ledger has no cast rows; "
            "skipping shadow critic"
        )
        return None
    if not line_rows:
        log.info(
            "[Stage7Shadow] adapter: legacy ledger has no line rows; "
            "skipping shadow critic"
        )
        return None

    # ---- Premise + arc ------------------------------------------------
    # Legacy episode_canon JSON (when present) carries premise + arc;
    # otherwise fall back to meta.news_seed prefix or placeholder.
    premise = ""
    episode_canon = meta.get("episode_canon") or {}
    if isinstance(episode_canon, dict):
        premise = (episode_canon.get("premise") or "").strip()
    if not premise:
        # BUG-LOCAL-277: news_seed is a dict in current ledgers; route
        # through the type-aware coercion so the adapter handles both
        # dict and legacy-string shapes without raising.
        news_seed = _coerce_news_seed_text(meta.get("news_seed"))
        # Trim to fit the schema's 10..300 char band.
        if news_seed:
            premise = news_seed[:280]
    if len(premise) < 10:
        premise = _PLACEHOLDER_PREMISE

    arc_data = (episode_canon.get("arc") or {}) if isinstance(episode_canon, dict) else {}
    setup = (arc_data.get("setup") or "").strip()
    complication = (arc_data.get("complication") or "").strip()
    resolution = (arc_data.get("resolution") or "").strip()
    if len(setup) < 10:
        setup = _PLACEHOLDER_ARC_PART
    if len(complication) < 10:
        complication = _PLACEHOLDER_ARC_PART
    if len(resolution) < 10:
        resolution = _PLACEHOLDER_ARC_PART

    # ---- Cast ---------------------------------------------------------
    stage1_cast: List[Stage1CastMember] = []
    seen_names: set[str] = set()
    seen_voices: set[str] = set()
    for row in cast_rows:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        if not name or name in _RESERVED_SPEAKERS:
            continue   # skip ANNOUNCER row + invalid rows
        if name in seen_names:
            continue   # dedupe
        seen_names.add(name)
        gender_raw = row.get("gender") or "male"
        gender = "male" if gender_raw not in ("male", "female", "nonbinary") else gender_raw
        pronouns = _gender_to_pronouns(gender)
        voice_id = _normalize_voice_preset(row.get("voice_preset"))
        # Make voice_id unique by appending the cast-row index if collision.
        if voice_id in seen_voices:
            # Try the next available digit.
            for d in range(10):
                candidate = f"v2/en_speaker_{d}"
                if candidate not in seen_voices:
                    voice_id = candidate
                    break
        seen_voices.add(voice_id)
        # Persona must satisfy Stage1CastMember.persona min_length=20.
        # Legacy ledgers may carry short descriptions; pad with a
        # neutral suffix rather than drop the cast row. Same for
        # arc_role (min_length=3).
        raw_persona = (row.get("character_description") or "").strip()
        if len(raw_persona) < 20:
            persona = (
                f"{raw_persona} {_PLACEHOLDER_PERSONA}".strip()
                if raw_persona
                else _PLACEHOLDER_PERSONA
            )
        else:
            persona = raw_persona
        persona = persona[:400]
        raw_arc_role = (row.get("role") or "").strip()
        if len(raw_arc_role) < 3:
            arc_role = _PLACEHOLDER_ARC_ROLE
        else:
            arc_role = raw_arc_role[:80]
        try:
            stage1_cast.append(Stage1CastMember(
                name=name[:40],
                gender=gender,
                pronouns=pronouns,
                voice_id=voice_id,
                persona=persona,
                arc_role=arc_role,
            ))
        except ValidationError as exc:
            log.warning(
                "[Stage7Shadow] adapter: dropping cast row %r due to "
                "pydantic mismatch: %s", name, str(exc)[:200],
            )
            continue

    if not stage1_cast:
        log.info(
            "[Stage7Shadow] adapter: no usable cast members after "
            "filtering; skipping shadow critic"
        )
        return None

    # Cap at Stage 1's max cast size (6 per schema).
    stage1_cast = stage1_cast[:6]
    valid_cast_names = {c.name for c in stage1_cast}

    # ---- Beats --------------------------------------------------------
    stage1_beats: List[Stage1Beat] = []
    seen_beat_ids: set[str] = set()
    for i, row in enumerate(line_rows):
        if not isinstance(row, dict):
            continue
        beat_id_raw = row.get("beat_id") or row.get("line_id") or f"b{i+1:03d}"
        # Coerce to bNNN format.
        if isinstance(beat_id_raw, str) and beat_id_raw.startswith("b") and len(beat_id_raw) >= 2:
            try:
                # Match the schema's b\d{3} regex.
                num_part = "".join(c for c in beat_id_raw[1:] if c.isdigit())
                if num_part:
                    beat_id = f"b{int(num_part):03d}"
                else:
                    beat_id = f"b{i+1:03d}"
            except ValueError:
                beat_id = f"b{i+1:03d}"
        else:
            beat_id = f"b{i+1:03d}"
        if beat_id in seen_beat_ids:
            beat_id = f"b{i+1:03d}"
            if beat_id in seen_beat_ids:
                continue
        seen_beat_ids.add(beat_id)

        # Resolve speaker: legacy uses speaker_role + char_id.
        legacy_role = (row.get("speaker_role") or "").strip().lower()
        reserved = _stage1_beat_role_from_legacy(legacy_role)
        if reserved:
            speaker = reserved
        else:
            # Lookup by char_id -> cast.name
            char_id = row.get("char_id")
            cast_row = next(
                (c for c in cast_rows if c.get("char_id") == char_id),
                None,
            )
            speaker = (cast_row.get("name") if cast_row else "") or ""
            if not speaker or speaker not in valid_cast_names:
                # Drop beats whose speaker can't be resolved to a valid
                # Stage 1 cast member (already filtered reserved above).
                continue

        intent = (
            row.get("beat_intent")
            or row.get("intent")
            or (row.get("text") or "")[:200]
            or "Legacy beat (no intent preserved)."
        )
        if len(intent) < 5:
            intent = "Legacy beat (no intent preserved)."
        text = row.get("text") or ""
        # Use actual rendered word count as the target -- the critic
        # can still evaluate length-target-appropriateness.
        actual_words = max(5, min(200, len([t for t in text.split() if t])))
        register = (
            row.get("emotional_register")
            or row.get("register")
            or "neutral"
        )[:80] or "neutral"
        try:
            stage1_beats.append(Stage1Beat(
                beat_id=beat_id,
                speaker=speaker,
                intent=intent[:200],
                length_target_words=actual_words,
                emotional_register=register,
                callback_to=None,
            ))
        except ValidationError as exc:
            log.warning(
                "[Stage7Shadow] adapter: dropping beat %r: %s",
                beat_id, str(exc)[:200],
            )
            continue

    # Schema requires min 3 beats. If we don't have 3, pad with
    # placeholder beats so the schema parses -- the critic will see
    # the actual rendered transcript regardless.
    while len(stage1_beats) < 3:
        idx = len(stage1_beats) + 1
        bid = f"b{900 + idx:03d}"   # 901, 902, 903 -- avoid collision
        if bid in seen_beat_ids:
            bid = f"b{990 + idx:03d}"
        seen_beat_ids.add(bid)
        stage1_beats.append(Stage1Beat(
            beat_id=bid,
            speaker=stage1_cast[0].name,
            intent="Placeholder beat to satisfy Stage1 schema (legacy ledger had <3 valid beats).",
            length_target_words=20,
            emotional_register="neutral",
            callback_to=None,
        ))

    # Cap at schema max 40.
    stage1_beats = stage1_beats[:40]

    # ---- Running facts ------------------------------------------------
    running_facts: List[str] = []
    continuity_ledger = meta.get("continuity_ledger") or {}
    if isinstance(continuity_ledger, dict):
        facts = continuity_ledger.get("facts") or []
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, dict):
                    txt = (f.get("text") or f.get("fact") or "").strip()
                elif isinstance(f, str):
                    txt = f.strip()
                else:
                    txt = ""
                if txt:
                    running_facts.append(txt[:400])
                if len(running_facts) >= 20:
                    break

    # ---- Build the Stage1Plan ----------------------------------------
    try:
        plan = Stage1Plan(
            premise=premise,
            arc=Stage1Arc(
                setup=setup,
                complication=complication,
                resolution=resolution,
            ),
            cast=stage1_cast,
            beats=stage1_beats,
            running_facts=running_facts,
        )
        return plan
    except ValidationError as exc:
        log.warning(
            "[Stage7Shadow] adapter: Stage1Plan construction failed "
            "after best-effort conversion: %s",
            str(exc)[:300],
        )
        return None


def extract_rendered_lines(led_data: dict) -> List[dict]:
    """Extract the (beat_id, speaker, text) rows the critic needs from
    the legacy ledger. Used by the FreezeCascade shadow critic call.

    The critic's transcript formatter accepts dicts with 'beat_id',
    'speaker' (or 'char_name'), and 'text' keys. Legacy line rows
    have 'speaker_role' + 'char_id'; we resolve char_id to the cast
    row name for display.
    """
    if not isinstance(led_data, dict):
        return []
    line_rows = led_data.get("lines") or []
    cast_rows = led_data.get("cast") or []
    char_id_to_name = {
        c.get("char_id"): c.get("name", "")
        for c in cast_rows
        if isinstance(c, dict)
    }
    out: List[dict] = []
    for row in line_rows:
        if not isinstance(row, dict):
            continue
        bid = row.get("beat_id") or row.get("line_id") or ""
        text = row.get("text") or ""
        if not text.strip():
            continue
        role = (row.get("speaker_role") or "").strip().lower()
        if role == "announcer":
            speaker = "ANNOUNCER"
        elif role.startswith("music"):
            speaker = "MUSIC"
        else:
            speaker = char_id_to_name.get(row.get("char_id"), "") or "UNKNOWN"
        out.append({
            "beat_id": str(bid),
            "speaker": speaker,
            "text": text.strip(),
        })
    return out


# ---------------------------------------------------------------------------
# In-loop Stage1Plan adapters
# ---------------------------------------------------------------------------
# Migrated here 2026-05-29 (lean-down step 5) from _otr_wave0_multiturn.py
# when the dormant Stage 2 multiturn dispatch was deleted. These pure
# adapters are NOT multiturn-specific -- the kept Stage 3 production
# validators path (enable_production_stage3_validators) builds an in-loop
# Stage1Plan + per-beat Stage1Beat from live writer state via these, so
# they outlive the multiturn cluster. build_inloop_stage1_plan reuses the
# module-local legacy_ledger_to_stage1_plan for all pydantic edge cases.
# ---------------------------------------------------------------------------


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
        arc_phase

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

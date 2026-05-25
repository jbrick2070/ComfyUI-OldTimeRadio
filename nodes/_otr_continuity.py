"""nodes/_otr_continuity.py -- continuity ledger (Sprint 5A).

A continuity contradiction is a script defect where a character
references something they should not yet know: a name not yet
introduced, a discovery not yet made, a prop not yet revealed. The
single LLM macro-plan (the outline) names beats but does not track
WHO KNOWS WHAT WHEN, so the line composer can ask the model to write
beat 4 with no signal that beat 4's speaker was off-stage when the
secret in beat 2 was uncovered.

Sprint 5A is the direct fix. It adds ONE structured LLM call made
right after the episode outline is generated. That call reads the
outline beats + the locked cast and extracts a `ContinuityState` --
the narrative facts of the episode, each tagged with who is aware of
it and who must not reference it, plus the beat index where the fact
becomes true. Later, when the line composer renders each dialogue
line, `render_continuity_slice` projects a per-speaker, per-beat
"continuity slice" from this state; the slice is injected into the
line-composition prompt as hard constraints, so a character cannot
reference a fact they should not know yet.

This module is PURE: no ComfyUI node class, no `INPUT_TYPES`, no
`model_id` widget (Prime Directive 6 -- the LLM model id is the
`generate_fn` slot callable threaded in by the writer; this module
exposes no widget and no model_id parameter). The single LLM call
site is a structured-extraction pass, so it is tagged
`# LLM slot: technical` and routed through the shared
`structured_call` retry ladder.

Prime Directive 1 -- audio output must never break. This pass NEVER
raises into the pipeline: every failure path (an exhausted retry
ladder, a raising slot fn, any other exception) is caught, logged at
warning level, and collapsed to `ContinuityState.neutral()` -- an
empty state that `render_continuity_slice` renders as the empty
string, i.e. "no continuity block". A failed continuity pass degrades
the script to pre-5A behaviour; it never aborts the run.

Conventions mirrored from `_otr_story_brief.run_story_brief_reflection`
(another never-raises structured pass): the `try/except
StructuredCallFailedError` + broad `except Exception` arms, the
neutral-fallback return, the `helper_name` log attribution, and the
shared-import guard (package import in production, flat import under
standalone test).

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


log = logging.getLogger("OTR")


__all__ = [
    "ContinuityFact",
    "ContinuityState",
    "build_continuity_ledger",
    "render_continuity_slice",
]


# Shared structured-JSON retry ladder. Package import in production;
# flat import when this module is loaded standalone / under test.
# Mirrors the import guard used by every other structured-pass module.
try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
    )

# Sprint 2C typed repair-prompt factories. The continuity pass passes a
# dispatching factory so structured_call's Attempt 3 routes the repair
# turn by failure class (bad JSON syntax / bad schema field).
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


# ---------------------------------------------------------------------------
# Constants -- LLM ladder settings for the continuity extraction pass
# ---------------------------------------------------------------------------

# The continuity pass extracts structured facts from an outline; it is
# NOT creative prose. A low base temperature keeps the model honest
# about the JSON shape. The structural-retry temperature is STRICTLY
# LOWER per the 2B principle (a JSON-schema re-roll lowers entropy, it
# never raises it); structured_call asserts this invariant at entry.
_CONTINUITY_BASE_TEMPERATURE: float = 0.30
_CONTINUITY_STRUCTURAL_RETRY_TEMPERATURE: float = 0.15

# Token budget for the continuity object. A handful of short facts plus
# their known_by / hidden_from arrays fits well under this; a generous
# headroom keeps a model from being truncated mid-object on a large
# cast. Structured passes set their own budget (see structured_call).
_CONTINUITY_MAX_NEW_TOKENS: int = 1200

# Caps on what the prompt feeds the model, so the input stays bounded
# regardless of episode length.
_BEAT_INTENT_CHAR_CAP: int = 200      # truncate each beat intent
_CAST_DESC_CHAR_CAP: int = 220        # truncate each cast description
_PROMPT_BEAT_CAP: int = 48            # never feed more than this many beats


# ---------------------------------------------------------------------------
# Pydantic schema -- ContinuityFact + ContinuityState
# ---------------------------------------------------------------------------


class ContinuityFact(BaseModel):
    """One concrete narrative fact of the episode plus its knowledge map.

    A fact is a single thing that is true in the story world (a
    revealed identity, a discovered object, a decision made, a place
    reached). The two name lists turn that fact into a hard per-speaker
    constraint for the line composer:

      * `known_by`   -- characters AWARE of the fact. The line composer
                        may let one of these speakers reference it.
      * `hidden_from` -- characters who must NOT reference the fact (it
                        is a secret they have not learned, a twist not
                        yet revealed to them). The composer renders
                        these into an explicit "must NOT reference"
                        list in the speaker's continuity slice.

    `established_beat` is the beat INDEX at which the fact becomes true
    -- the position of the beat in `outline.beats` (0-based). A fact
    with `established_beat == 3` does not yet exist at beats 0..2, so
    `render_continuity_slice` filters it out for any earlier beat. The
    default 0 means "true from the start of the episode".
    """

    fact: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One concrete narrative fact, plain prose, one clause.",
    )
    known_by: list[str] = Field(
        default_factory=list,
        description="Character names / ids aware of this fact.",
    )
    hidden_from: list[str] = Field(
        default_factory=list,
        description="Characters who must NOT reference this fact.",
    )
    established_beat: int = Field(
        default=0,
        ge=0,
        description="Beat index (0-based) where the fact becomes true.",
    )


class ContinuityState(BaseModel):
    """The full continuity ledger for one episode.

    Produced once, right after the outline, by `build_continuity_ledger`.
    Consumed per-line by `render_continuity_slice`. An empty / neutral
    state (see `ContinuityState.neutral`) is the safe fallback the
    builder returns on any failure -- it renders to the empty string,
    so a failed continuity pass simply means "no continuity block",
    never a broken run.
    """

    location: str = Field(
        default="",
        max_length=200,
        description="Primary location of the episode, plain prose.",
    )
    active_props: list[str] = Field(
        default_factory=list,
        description="Notable props / objects in play across the episode.",
    )
    facts: list[ContinuityFact] = Field(
        default_factory=list,
        description="The per-character continuity facts of the episode.",
    )

    @classmethod
    def neutral(cls) -> "ContinuityState":
        """Return an empty ContinuityState -- the safe fallback.

        Every field is at its default (empty location, empty
        `active_props`, empty `facts`). `render_continuity_slice`
        treats this as "nothing to constrain" and returns the empty
        string, so a failed continuity pass degrades the script to
        pre-5A behaviour rather than aborting the run.
        """
        return cls()

    def is_neutral(self) -> bool:
        """True when this state carries no continuity signal at all.

        A neutral state has no location, no active props, and no facts.
        `render_continuity_slice` short-circuits to "" on a neutral
        state.
        """
        return (
            not (self.location or "").strip()
            and not self.active_props
            and not self.facts
        )


# ---------------------------------------------------------------------------
# Prompt assembly -- outline + cast -> continuity-extraction prompt
# ---------------------------------------------------------------------------


_CONTINUITY_SYSTEM_PROMPT = """\
You are a continuity supervisor for a short audio drama. You are given
the episode outline (an ordered list of beats) and the locked cast.
Your job is to extract the CONTINUITY STATE: the concrete narrative
facts of the episode and, for each fact, which characters know it and
which characters must not reference it yet.

Return EXACTLY one JSON object, no prose, no Markdown fences:
{
  "location":      "primary place of the episode, one short phrase",
  "active_props":  ["notable objects in play, short nouns"],
  "facts": [
    {
      "fact":             "one concrete narrative fact, one clause",
      "known_by":         ["character names aware of this fact"],
      "hidden_from":      ["character names who must NOT reference it"],
      "established_beat": 0
    }
  ]
}

Rules:
- A fact is something TRUE in the story world: a revealed identity, a
  discovered object, a decision made, a place reached, a secret kept.
- `known_by` lists the characters aware of the fact. `hidden_from`
  lists characters who must not mention or rely on it -- a secret they
  have not learned, a twist not yet revealed to them.
- Use ONLY the exact character names from the Cast block. Never invent
  a name.
- `established_beat` is the 0-based index of the beat where the fact
  first becomes true. Use 0 for facts true from the episode's start.
- Extract the handful of facts that actually matter for continuity.
  Do not pad. An episode with no secrets may return an empty `facts`
  list.
"""


def _safe_text(value: Any, cap: int) -> str:
    """Coerce an arbitrary value to a trimmed, length-capped string."""
    text = str(value or "").strip()
    if len(text) > cap:
        text = text[: cap - 1].rstrip() + "..."
    return text


def _beat_field(beat: Any, name: str, default: str = "") -> str:
    """Read a named field off a Beat object OR a plain dict.

    The outline's `Beat` is a pydantic model (attribute access), but
    tests and early-stage callers may pass plain dicts. Duck-typed so
    both shapes work without importing `_otr_outline` at module load.
    """
    if isinstance(beat, dict):
        return _safe_text(beat.get(name, default), 10_000)
    return _safe_text(getattr(beat, name, default), 10_000)


def _iter_beats(outline: Any) -> list[Any]:
    """Return the list of beats from an Outline object or a plain dict.

    `outline.beats` is the real field name on `_otr_outline.Outline`
    (a pydantic model). A plain dict with a `"beats"` key is also
    accepted for tests / early-stage callers.
    """
    if outline is None:
        return []
    if isinstance(outline, dict):
        beats = outline.get("beats")
    else:
        beats = getattr(outline, "beats", None)
    if not beats:
        return []
    return list(beats)


def _render_beats_block(outline: Any) -> str:
    """Render the outline beats as a numbered, index-tagged block.

    Each line carries the 0-based beat index (so the model can fill
    `established_beat` correctly), the beat's speaker, its speaker_role,
    its intent, and its mood -- the real `Beat` fields read off
    `_otr_outline.Beat`. The index is the line composer's `beat_index`
    coordinate, so the prompt and `render_continuity_slice` agree on
    what an index means.
    """
    beats = _iter_beats(outline)
    if not beats:
        return "(no beats in outline)"
    lines: list[str] = []
    for index, beat in enumerate(beats[:_PROMPT_BEAT_CAP]):
        speaker = _beat_field(beat, "speaker") or "?"
        role = _beat_field(beat, "speaker_role") or "?"
        intent = _safe_text(_beat_field(beat, "intent"), _BEAT_INTENT_CHAR_CAP)
        mood = _beat_field(beat, "mood")
        bits = [f"beat {index}: speaker={speaker} role={role}"]
        if intent:
            bits.append(f"intent={intent}")
        if mood:
            bits.append(f"mood={mood}")
        lines.append("  " + " | ".join(bits))
    if len(beats) > _PROMPT_BEAT_CAP:
        lines.append(
            f"  (...{len(beats) - _PROMPT_BEAT_CAP} further beats omitted)"
        )
    return "\n".join(lines)


def _cast_name(row: Any) -> str:
    """Extract a character name from a cast row (dict or object)."""
    if isinstance(row, dict):
        return _safe_text(row.get("name") or row.get("char_id"), 80)
    return _safe_text(
        getattr(row, "name", None) or getattr(row, "char_id", None), 80
    )


def _cast_description(row: Any) -> str:
    """Extract a character description from a cast row (dict or object)."""
    if isinstance(row, dict):
        value = row.get("character_description") or row.get("description")
    else:
        value = (
            getattr(row, "character_description", None)
            or getattr(row, "description", None)
        )
    return _safe_text(value, _CAST_DESC_CHAR_CAP)


def _render_cast_block(cast_rows: Any) -> str:
    """Render the locked cast as a name + description block.

    `cast_rows` is the writer's locked cast -- a list of dicts (the
    production ledger shape) or objects. Each emitted line gives the
    model the exact name it must use in `known_by` / `hidden_from` and
    a short description so it can reason about who would plausibly
    know what.
    """
    rows = list(cast_rows or [])
    if not rows:
        return "(no cast supplied)"
    lines: list[str] = []
    for row in rows:
        name = _cast_name(row)
        if not name:
            continue
        desc = _cast_description(row)
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    if not lines:
        return "(no cast supplied)"
    return "\n".join(lines)


def _build_continuity_user_prompt(outline: Any, cast_rows: Any) -> str:
    """Assemble the user-message body for the continuity-extraction call."""
    title = ""
    premise = ""
    if isinstance(outline, dict):
        title = _safe_text(outline.get("title"), 120)
        premise = _safe_text(outline.get("premise"), 400)
    elif outline is not None:
        title = _safe_text(getattr(outline, "title", ""), 120)
        premise = _safe_text(getattr(outline, "premise", ""), 400)
    parts: list[str] = ["Extract the continuity state for this episode.", ""]
    if title:
        parts.append(f"Title: {title}")
    if premise:
        parts.append(f"Premise: {premise}")
    parts.extend([
        "",
        "Cast (use ONLY these exact names):",
        _render_cast_block(cast_rows),
        "",
        "Outline beats (the number after 'beat' is the 0-based index "
        "you use for established_beat):",
        _render_beats_block(outline),
        "",
        "Return only the JSON continuity-state object.",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# build_continuity_ledger -- the single LLM call (never raises)
# ---------------------------------------------------------------------------


def build_continuity_ledger(
    generate_fn: Callable[..., str],
    outline: Any,
    cast_rows: Any,
    *,
    technical_repo_id: Optional[str] = None,
) -> ContinuityState:
    """Run the continuity-extraction pass and return a ContinuityState.

    ONE LLM call, made right after the episode outline is generated.
    It reads the outline beats + the locked cast and extracts the
    narrative facts of the episode, each tagged with who is aware of it
    and who must not reference it.

    Parameters:
      generate_fn
        The LLM generate callable -- the writer's TECHNICAL slot. Its
        contract is the project-standard slot-fn signature:
        `generate_fn(messages, *, temperature, max_new_tokens) -> str`.
        Per Prime Directive 6 the model id is NOT a widget on this
        module: the writer threads the technical-slot callable in here,
        and `technical_repo_id` (below) is the optional STRING model id
        wired from the writer's `technical_model` broadcast output.
      outline
        The episode outline -- an `_otr_outline.Outline` pydantic object
        (real field used: `outline.beats`, a list of `Beat` objects
        with `speaker`, `speaker_role`, `intent`, `mood`, `beat_id`,
        `arc_phase`). A plain dict with a `"beats"` key is also accepted
        for tests / early-stage callers.
      cast_rows
        The locked cast -- a list of dicts (production-ledger shape:
        `name`, `char_id`, `character_description`) or objects.
      technical_repo_id
        OPTIONAL. The technical-slot model id, for log attribution
        only. Accepted as a STRING per Prime Directive 6; this module
        never exposes a `model_id` widget.

    NEVER RAISES (Prime Directive 1 -- audio output must never break).
    The single `structured_call` is wrapped in a `try/except
    StructuredCallFailedError` arm AND a broad `except Exception` arm.
    On any failure -- an exhausted retry ladder, a slot fn that raised,
    any other exception -- the failure is logged at warning level and
    `ContinuityState.neutral()` is returned. A neutral state renders to
    the empty string, so a failed continuity pass simply degrades the
    script to pre-5A behaviour.
    """
    beats = _iter_beats(outline)
    if not beats:
        # Nothing to extract continuity from -- not a failure, just an
        # empty outline. Return neutral without spending an LLM call.
        log.info(
            "[OTR_Continuity] build_continuity_ledger: outline carries "
            "no beats; returning neutral ContinuityState (no LLM call)."
        )
        return ContinuityState.neutral()

    user_prompt = _build_continuity_user_prompt(outline, cast_rows)
    messages = [
        {"role": "system", "content": _CONTINUITY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    model_note = technical_repo_id or "(unspecified)"

    # LLM slot: technical -- this is structured fact extraction from an
    # outline (JSON object output, validated against a pydantic schema),
    # not creative narrative prose, so it belongs on the writer's
    # technical model. The model id arrives via the `generate_fn` slot
    # callable threaded in by OTR_LedgerScriptWriter, which reads it
    # from the writer's `technical_model` broadcast socket -- no new
    # widget, no new model_id parameter (Prime Directive 6).
    try:
        state = structured_call(
            prompt=messages,
            schema=ContinuityState,
            slot_fn=generate_fn,
            base_temperature=_CONTINUITY_BASE_TEMPERATURE,
            structural_retry_temperature=_CONTINUITY_STRUCTURAL_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=_CONTINUITY_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="build_continuity_ledger",
        )
    except StructuredCallFailedError as exc:
        # The retry ladder ran out of attempts with no schema-valid
        # result. Per Prime Directive 1 this must not abort the run --
        # degrade to a neutral state so the line composer simply gets
        # no continuity block.
        log.warning(
            "[OTR_Continuity] build_continuity_ledger: structured_call "
            "exhausted the retry ladder after %d attempt(s) (last error: "
            "%s; technical_repo_id=%s); returning neutral ContinuityState.",
            exc.attempts, exc.last_error, model_note,
        )
        return ContinuityState.neutral()
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        # structured_call does not catch slot-fn exceptions: a VRAM /
        # loader / framework failure inside the generate_fn closure
        # lands here. A bare reraise would crash the whole script
        # generation over a continuity-planning pass -- the never-raises
        # contract (Prime Directive 1) catches it broadly.
        log.warning(
            "[OTR_Continuity] build_continuity_ledger: generate_fn raised "
            "%s: %s (technical_repo_id=%s); returning neutral "
            "ContinuityState.",
            type(exc).__name__, exc, model_note,
        )
        return ContinuityState.neutral()

    log.info(
        "[OTR_Continuity] build_continuity_ledger: extracted %d fact(s), "
        "location=%r, %d active prop(s) (technical_repo_id=%s).",
        len(state.facts), state.location, len(state.active_props),
        model_note,
    )
    return state


# ---------------------------------------------------------------------------
# render_continuity_slice -- per-speaker, per-beat hard-constraint block
# ---------------------------------------------------------------------------


def _speaker_matches(speaker: str, name: str) -> bool:
    """Loose, case-insensitive speaker / name comparison.

    The line composer's `speaker` may be an ALL-CAPS name, a mixed-case
    name, or a char_id; the continuity facts carry whatever the model
    emitted from the cast block. Match if the trimmed, case-folded
    forms are equal, or if one is a word-level prefix of the other
    (so "DR. LEMMY" matches "LEMMY" and vice versa). Empty strings
    never match.
    """
    a = (speaker or "").strip().casefold()
    b = (name or "").strip().casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    # Word-level containment: handle "DR. LEMMY" vs "LEMMY".
    a_words = set(a.replace(".", " ").replace(",", " ").split())
    b_words = set(b.replace(".", " ").replace(",", " ").split())
    if not a_words or not b_words:
        return False
    return a_words <= b_words or b_words <= a_words


def _name_in_list(speaker: str, names: list[str]) -> bool:
    """True when `speaker` loosely matches any name in `names`."""
    return any(_speaker_matches(speaker, n) for n in (names or []))


def render_continuity_slice(
    state: ContinuityState,
    speaker: str,
    beat_index: int,
) -> str:
    """Render the continuity constraints for one speaker at one beat.

    PURE function -- no LLM call. Given the full `ContinuityState`, the
    current `speaker`, and the current `beat_index` (0-based, the index
    of the beat being composed in `outline.beats`), returns a compact
    hard-constraint text block to inject into the line-composition
    prompt:

      * the episode location;
      * the active props ESTABLISHED on or before this beat (a prop is
        treated as established from the episode start -- props carry no
        per-prop beat index, so they are shown once `state` is in use);
      * the facts this `speaker` is `known_by` AND whose
        `established_beat` is <= `beat_index` -- the things this speaker
        legitimately knows at this point;
      * the facts this `speaker` is `hidden_from`, rendered as an
        explicit "must NOT reference" list -- regardless of
        `established_beat`, because a fact a speaker is hidden from is
        off-limits whether or not it has been established for others.

    Returns "" (the empty string) when:
      * `state` is None or neutral / empty (`ContinuityState.neutral()`);
      * nothing applies to this speaker at this beat (no location, no
        props, no known facts, no hidden facts).

    The caller treats an empty string as "no continuity block" -- it
    injects nothing into the prompt. Speakers are matched
    case-insensitively and loosely (a name or a char_id, see
    `_speaker_matches`).
    """
    # Defensive: a None or non-state argument degrades to "" rather
    # than raising -- consistent with the never-break contract.
    if state is None or not isinstance(state, ContinuityState):
        return ""
    if state.is_neutral():
        return ""

    try:
        beat_index = int(beat_index)
    except (TypeError, ValueError):
        beat_index = 0
    if beat_index < 0:
        beat_index = 0

    facts = list(state.facts or [])

    # Facts this speaker knows: known_by match AND already established
    # at (or before) the current beat.
    known_now = [
        f for f in facts
        if _name_in_list(speaker, f.known_by)
        and int(f.established_beat or 0) <= beat_index
    ]
    # Facts this speaker must not reference: hidden_from match. A
    # hidden fact is off-limits irrespective of established_beat -- the
    # speaker is barred from it whether or not others know it yet.
    hidden = [f for f in facts if _name_in_list(speaker, f.hidden_from)]

    location = (state.location or "").strip()
    props = [p for p in (state.active_props or []) if str(p).strip()]

    # Nothing to say for this speaker at this beat -> no block.
    if not (location or props or known_now or hidden):
        return ""

    lines: list[str] = ["CONTINUITY CONSTRAINTS (hard -- do not violate):"]
    if location:
        lines.append(f"- Location: {location}")
    if props:
        lines.append(f"- Active props: {', '.join(str(p) for p in props)}")
    if known_now:
        lines.append(
            f"- {speaker} KNOWS the following and may reference them:"
        )
        for f in known_now:
            lines.append(f"    * {f.fact}")
    if hidden:
        lines.append(
            f"- {speaker} must NOT reference, mention, or rely on the "
            "following (not yet known to this character):"
        )
        for f in hidden:
            lines.append(f"    * {f.fact}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional smoke block -- exercised only when run directly, never on import.
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual smoke only
    import json

    _outline_stub = {
        "title": "The Quiet Frequency",
        "premise": "A radio operator decodes a signal that should not exist.",
        "beats": [
            {"speaker": "MARA", "speaker_role": "character",
             "intent": "Mara tunes the dial and hears the anomaly.",
             "mood": "curious"},
            {"speaker": "DELL", "speaker_role": "character",
             "intent": "Dell dismisses it as interference.",
             "mood": "skeptical"},
            {"speaker": "MARA", "speaker_role": "character",
             "intent": "Mara realizes the signal names her.",
             "mood": "alarmed"},
        ],
    }
    _cast_stub = [
        {"name": "MARA", "char_id": "c01",
         "character_description": "A night-shift radio operator."},
        {"name": "DELL", "char_id": "c02",
         "character_description": "A weary station supervisor."},
    ]

    def _canned_generate_fn(messages, *, temperature, max_new_tokens):
        return json.dumps({
            "location": "A coastal radio relay station, after midnight",
            "active_props": ["the receiver dial", "a logbook"],
            "facts": [
                {
                    "fact": "The anomalous signal contains Mara's name.",
                    "known_by": ["MARA"],
                    "hidden_from": ["DELL"],
                    "established_beat": 2,
                },
            ],
        })

    _state = build_continuity_ledger(_canned_generate_fn, _outline_stub, _cast_stub)
    print("state:", _state.model_dump())
    print("--- MARA @ beat 2 ---")
    print(render_continuity_slice(_state, "MARA", 2))
    print("--- DELL @ beat 2 ---")
    print(render_continuity_slice(_state, "DELL", 2))
    print("--- MARA @ beat 0 (fact not yet established) ---")
    print(repr(render_continuity_slice(_state, "MARA", 0)))
    print("--- neutral ---")
    print(repr(render_continuity_slice(ContinuityState.neutral(), "MARA", 5)))

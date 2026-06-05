"""nodes/_otr_editor_constraints.py -- Sprint 5 (2026-05-28).

Editor downgrade to constraint checker. Strips taste verbs from the
editor prompt, narrows the editor's job to five concrete constraints
that small local models can check reliably, and provides a pure-
Python rule-based check_constraints() that the wire-up sprint plugs
into the Story Room loop in place of the taste-based EditorVerdict.

Why downgrade: the round-robin agreed that revision cannot
ORIGINATE quality. The pre-Sprint-5 editor ran up to three cycles
of free-form "make it better / more drama / improve pacing"
revision; under live observation those cycles either rubber-stamped
or chased their tails. Sprint 4's best-of-N selection produces the
ceiling; Sprint 5's editor only enforces the floor.

Five constraints (all checkable by a small model OR by Python):
  WRONG_SPEAKER         -- a line is attributed to a name not in cast.
  PHANTOM_CHARACTER     -- a non-cast proper noun speaks or is named.
  MISSING_COSTLY_CHOICE -- dramatic_state.costly_choice_beat does
                           not resolve (the named slot is missing,
                           or its state does not change).
  NO_FINAL_THIRD_TURN   -- the final third of the voiced beats has
                           no state change (state_before ==
                           state_after across the run).
  FORMAT_FAILURE        -- the draft is malformed / off-schema.

This module ships PURE -- no LLM call, no I/O. The wire-up sprint
threads it into `run_story_room` as the single repair loop after
the Sprint 4 selector picks a winner; revision cap drops from 3 to
1 (one targeted repair, then ship-or-fail).

UTF-8 no BOM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from ._otr_beat_validators import is_dead_beat


__all__ = [
    "EditorConstraint",
    "EditorConstraintVerdict",
    "EDITOR_CONSTRAINTS_SYSTEM_PROMPT",
    "DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES",
    "check_constraints",
]


# Single-source-of-truth enum of constraint codes. Strings (not
# IntEnum) so the constrained-decode schema can pin them as a
# Literal[...] union when the LLM-based variant lands.
class EditorConstraint:
    WRONG_SPEAKER = "WRONG_SPEAKER"
    PHANTOM_CHARACTER = "PHANTOM_CHARACTER"
    MISSING_COSTLY_CHOICE = "MISSING_COSTLY_CHOICE"
    NO_FINAL_THIRD_TURN = "NO_FINAL_THIRD_TURN"
    FORMAT_FAILURE = "FORMAT_FAILURE"

    ALL = (
        WRONG_SPEAKER,
        PHANTOM_CHARACTER,
        MISSING_COSTLY_CHOICE,
        NO_FINAL_THIRD_TURN,
        FORMAT_FAILURE,
    )


# Sprint 5 (2026-05-28) -- hard cap on serial revision cycles. The
# pre-Sprint-5 default was 3; under the new constraint-checker
# regime quality comes from the Sprint 4 selector, not from editor
# cycles. One targeted repair turn is the maximum; if the repair
# does not clear the failing constraints the loop ships or fails
# loud (caller's choice).
DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES: int = 1


# Sprint 5 system prompt -- strictly no taste verbs. The wire-up
# sprint feeds this into a constrained-decode call against an enum
# Literal schema so the LLM cannot wander.
EDITOR_CONSTRAINTS_SYSTEM_PROMPT = """\
You are the EDITOR. You are NOT a taste editor. You check five
constraints only and return a verdict listing the constraints
that FAIL on the current draft.

Do NOT touch pacing. Do NOT ask for additional dramatic intensity.
Do NOT redraft the line. Your job is to list failing constraints
and stop.

The five constraints are:
  WRONG_SPEAKER         A line is attributed to a name not in the
                        locked cast.
  PHANTOM_CHARACTER     A proper noun that is not in the cast (and
                        not on the canonical journalistic-terms
                        roster) is named or speaks.
  MISSING_COSTLY_CHOICE The dramatic_state.costly_choice_beat does
                        not resolve in the draft -- either the
                        named slot is missing, or its state does
                        not change between before and after.
  NO_FINAL_THIRD_TURN   The final third of the voiced beats has no
                        state change (state_before ==
                        state_after across the run).
  FORMAT_FAILURE        The draft is malformed -- bracket stage
                        directions inside dialogue, JSON inside
                        dialogue, markdown headers, etc.

Output a single JSON object: {"pass_decision": <bool>,
"failing_constraints": [<one or more of the five codes above>],
"repair_note": "<one-sentence concrete repair instruction, only
when pass_decision is False>"}. JSON only. No prose preamble.

pass_decision MUST be True iff failing_constraints is empty.
"""


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass
class EditorConstraintVerdict:
    """Verdict for the constraint-only editor.

    Shape intentionally narrow vs the pre-Sprint-5 EditorVerdict
    (no per_axis_notes, no overall_note, no failing_axes -- those
    were taste-rubric artifacts). `repair_note` is a single
    actionable sentence the Writer receives for the one targeted
    repair turn; populated only when pass_decision is False.
    """

    pass_decision: bool
    failing_constraints: List[str] = field(default_factory=list)
    repair_note: str = ""
    cycle: int = 0

    def to_dict(self) -> dict:
        return {
            "pass_decision": bool(self.pass_decision),
            "failing_constraints": list(self.failing_constraints),
            "repair_note": str(self.repair_note),
            "cycle": int(self.cycle),
        }


# ---------------------------------------------------------------------------
# Pure-Python rule-based check_constraints
# ---------------------------------------------------------------------------


def _norm(s: Any) -> str:
    return " ".join((str(s or "").strip()).lower().split())


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Dict-or-attribute read (Sprint 4.1 wire-up)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _voiced_beats(plan: Any) -> List[Any]:
    return [
        b for b in (getattr(plan, "beats", None) or [])
        if str(getattr(b, "speaker", "") or "").strip() != "MUSIC"
    ]


def _cast_names(plan: Any) -> set[str]:
    names: set[str] = set()
    for c in (getattr(plan, "cast", None) or []):
        nm = str(getattr(c, "name", "") or "").strip().upper()
        if nm:
            names.add(nm)
    return names


_FORMAT_FAILURE_NEEDLES = (
    "[sfx",        # bracket stage direction
    "[music",
    "[sound",
    "```",          # markdown code fence
    "**",           # markdown bold/italic
)


def _cast_first_last_aliases(cast: set) -> set:
    """Unambiguous first/last-name aliases of the locked cast.

    A weaker writer model often abbreviates a locked character -- e.g. it
    writes 'NIA' for the locked 'NIA HIBBERT' -- which the exact-membership
    cast check then flags as WRONG_SPEAKER / PHANTOM_CHARACTER and halts the
    freeze gate. This returns the first- and last-name tokens of multi-word
    cast names that map to EXACTLY ONE cast member; a token shared by two
    characters (ambiguous) is excluded, so a genuine off-cast name still
    fails. Full-name writers are unaffected -- their exact names already
    match -- so this only ADDS valid aliases (no previously-valid name
    becomes invalid). In line with the phantom-ship policy, resolving an
    unambiguous abbreviation is preferred over halting the render.
    """
    owner: dict = {}
    ambiguous: set = set()
    for full in cast or ():
        toks = str(full).strip().upper().split()
        if len(toks) < 2:
            continue  # single-token names are already their own full form
        for tok in (toks[0], toks[-1]):
            if len(tok) < 3:
                continue
            if tok in owner and owner[tok] != full:
                ambiguous.add(tok)
            else:
                owner[tok] = full
    return {t for t in owner if t not in ambiguous}


def check_constraints(
    plan: Any,
    *,
    cast_names_extra: Optional[Iterable[str]] = None,
    allowed_proper_nouns: Optional[Iterable[str]] = None,
) -> EditorConstraintVerdict:
    """Run all five constraints against a Stage1Plan-shaped object
    and return a single EditorConstraintVerdict.

    Pure function. The wire-up sprint calls this from
    `run_story_room` after the writer drafts; on a failed verdict
    the repair_note is threaded into a single Writer revision turn.

    cast_names_extra        Optional iterable of additional cast
                            names to allow (e.g. ANNOUNCER, MUSIC).
                            Defaulted to add ANNOUNCER + MUSIC.
    allowed_proper_nouns    Optional iterable of journalistic terms
                            (places, agencies, things) that PHANTOM_
                            CHARACTER must not flag. Defaulted to
                            empty; caller threads from
                            news_interpreter.key_terms.
    """
    failing: list[str] = []
    notes: list[str] = []

    voiced = _voiced_beats(plan)
    cast = _cast_names(plan)
    extra = {str(n).strip().upper() for n in (cast_names_extra or [])}
    extra.update({"ANNOUNCER", "MUSIC"})
    # Accept unambiguous first/last-name abbreviations of locked cast members
    # (e.g. 'NIA' for 'NIA HIBBERT') so a weak model's shorthand does not trip
    # WRONG_SPEAKER / PHANTOM_CHARACTER and halt the freeze gate. Additive only.
    valid_speakers = cast | extra | _cast_first_last_aliases(cast)
    allowed_pns = {
        str(n).strip().upper()
        for n in (allowed_proper_nouns or [])
    }

    # ---- WRONG_SPEAKER -------------------------------------------------
    for b in voiced:
        sp = str(getattr(b, "speaker", "") or "").strip().upper()
        if sp and sp not in valid_speakers:
            failing.append(EditorConstraint.WRONG_SPEAKER)
            notes.append(
                f"beat {getattr(b, 'beat_id', '?')} attributed to "
                f"{sp!r}, which is not in the locked cast "
                f"({sorted(cast)})."
            )
            break  # one report is enough for this constraint

    # ---- PHANTOM_CHARACTER --------------------------------------------
    # Scan voiced beat intent + objective + state_* for ALL-CAPS
    # words that are not in cast/valid/allowed sets. This is a
    # conservative phantom check -- the line-composer Phase 0 gate
    # already does the rigorous version; this constraint is a
    # backup at the BEAT-SHEET level (the writer may have planned
    # an off-cast character into the structure).
    #
    # Key signal: the SOURCE token must already be ALL CAPS (proper
    # name convention). Lowercase / titlecase tokens are common
    # English words and never count as phantoms.
    phantom_candidates: set[str] = set()
    for b in voiced:
        for attr in ("intent", "objective", "obstacle", "turn",
                     "state_before", "state_after"):
            v = str(getattr(b, attr, "") or "")
            for tok in v.split():
                bare_src = tok.strip(".,;:!?\"'()[]{}")
                if (
                    len(bare_src) >= 3
                    and bare_src.isalpha()
                    and bare_src == bare_src.upper()  # was ALL CAPS in source
                ):
                    bare = bare_src.upper()
                    if (
                        bare not in valid_speakers
                        and bare not in allowed_pns
                    ):
                        phantom_candidates.add(bare)
    # Filter out obviously generic ALL-CAPS words that crop up in
    # English (SFX, OK, TV, AM, PM, USA, NASA, FBI, CIA, NIST). The
    # caller's allowed_proper_nouns is the better signal; this is
    # just a noise floor.
    generic_caps = {"OK", "TV", "AM", "PM", "USA", "NASA", "FBI",
                    "CIA", "NIST", "MIT", "SFX", "DC", "LA", "NYC",
                    "GMT", "UTC", "AI", "ML", "CPU", "GPU"}
    phantom_candidates -= generic_caps
    if phantom_candidates:
        failing.append(EditorConstraint.PHANTOM_CHARACTER)
        notes.append(
            "non-cast proper noun(s) in the beat sheet: "
            f"{sorted(phantom_candidates)}. Either add them to "
            "the cast or remove them from the beats."
        )

    # ---- MISSING_COSTLY_CHOICE ----------------------------------------
    ds = getattr(plan, "dramatic_state", None)
    if ds is not None:
        slot = str(_attr(ds,"costly_choice_beat", "") or "").strip()
        if not slot:
            failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
            notes.append(
                "dramatic_state.costly_choice_beat is empty -- the "
                "third act needs a typed pivot."
            )
        else:
            by_slot = {
                str(getattr(b, "dialogue_slot_id", "") or "").strip(): b
                for b in voiced
            }
            by_slot.pop("", None)
            pivot = by_slot.get(slot)
            if pivot is None:
                failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
                notes.append(
                    f"costly_choice_beat={slot!r} does not name any "
                    "voiced beat's dialogue_slot_id."
                )
            else:
                sb = _norm(getattr(pivot, "state_before", ""))
                sa = _norm(getattr(pivot, "state_after", ""))
                if sb and sa and sb == sa:
                    failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
                    notes.append(
                        f"beat {getattr(pivot, 'beat_id', '?')} "
                        f"(slot {slot}) is the costly_choice_beat "
                        "but its state does not change. The choice "
                        "must move the situation."
                    )

    # ---- NO_FINAL_THIRD_TURN ------------------------------------------
    # Take the last 1/3 of voiced beats (at least 1). Every one of
    # them must NOT be a dead beat. Untyped beats are ignored.
    if voiced:
        n = len(voiced)
        cut = max(1, n // 3)
        final_third = voiced[-cut:]
        typed_dead = [
            b for b in final_third
            if (
                getattr(b, "state_before", None)
                and getattr(b, "state_after", None)
                and is_dead_beat(b)
            )
        ]
        if typed_dead and len(typed_dead) == len(final_third):
            # Every typed beat in the final third is dead -> no turn.
            failing.append(EditorConstraint.NO_FINAL_THIRD_TURN)
            notes.append(
                "the final third of the voiced beats carries no "
                "state change. The episode must turn before close."
            )

    # ---- FORMAT_FAILURE (rule-based; scans typed beat strings) -------
    fmt_offenders: list[str] = []
    for b in voiced:
        for attr in ("intent", "objective", "obstacle", "turn"):
            v = (str(getattr(b, attr, "") or "")).lower()
            if any(n in v for n in _FORMAT_FAILURE_NEEDLES):
                fmt_offenders.append(
                    f"beat {getattr(b, 'beat_id', '?')}.{attr}"
                )
                break
    if fmt_offenders:
        failing.append(EditorConstraint.FORMAT_FAILURE)
        notes.append(
            "bracket stage directions / markdown fragments in the "
            f"beat sheet: {fmt_offenders[:4]}."
        )

    # ---- Build the verdict --------------------------------------------
    if failing:
        repair_note = " | ".join(notes)
        return EditorConstraintVerdict(
            pass_decision=False,
            failing_constraints=failing,
            repair_note=repair_note,
        )
    return EditorConstraintVerdict(
        pass_decision=True,
        failing_constraints=[],
        repair_note="",
    )



# ---------------------------------------------------------------------------
# Sprint 5.1 wire-up helper (2026-05-28)
# ---------------------------------------------------------------------------


__all__.append("check_constraints_from_ledger")


class _BeatLike:
    """Minimal Stage1Beat-shaped view for the constraint checker.

    Reconstructed from a ledger line row + the matching outline beat
    when possible. Carries only the attributes the checker reads
    (beat_id / speaker / dialogue_slot_id / intent + the Sprint 2
    state_* fields when present). Defaults to None for typed-state
    fields so untyped beats are charitably skipped by validators 3
    (each_beat_changes_situation) + 4 (NO_FINAL_THIRD_TURN).
    """

    def __init__(self, *, beat_id, speaker, dialogue_slot_id, intent="",
                 state_before=None, state_after=None, objective=None,
                 obstacle=None, turn=None, subtext=None,
                 tension=None, next_turn=None):
        self.beat_id = beat_id
        self.speaker = speaker
        self.dialogue_slot_id = dialogue_slot_id
        self.intent = intent
        self.state_before = state_before
        self.state_after = state_after
        self.objective = objective
        self.obstacle = obstacle
        self.turn = turn
        self.subtext = subtext
        self.tension = tension
        self.next_turn = next_turn


class _PlanLike:
    """Stage1Plan-shaped view -- cast + beats + dramatic_state.

    The constraint checker duck-types all three attributes, so this
    is enough to run all five constraints against an in-flight
    ledger without round-tripping through pydantic.
    """

    def __init__(self, *, cast, beats, dramatic_state):
        self.cast = cast
        self.beats = beats
        self.dramatic_state = dramatic_state


def _cast_view_from_ledger(ledger_data: dict):
    """Map ledger.cast rows to objects exposing .name."""
    rows = ledger_data.get("cast") or []
    out = []
    for row in rows:
        if isinstance(row, dict):
            nm = str(row.get("name", "") or "").strip()
        else:
            nm = str(getattr(row, "name", "") or "").strip()
        if nm:
            out.append(type("_CastRow", (), {"name": nm})())
    return out


def _beats_view_from_ledger(ledger_data: dict):
    """Build _BeatLike rows from ledger.lines (Sprint 1 stamped
    dialogue_slot_id; voiced rows carry char_id / text we don't
    need for constraints, but we extract speaker from beat_id ->
    cast lookup when possible)."""
    lines = ledger_data.get("lines") or []
    cast_id_by_name = {}
    for c in (ledger_data.get("cast") or []):
        if isinstance(c, dict):
            cast_id_by_name[c.get("char_id", "")] = c.get("name", "")
        else:
            cast_id_by_name[getattr(c, "char_id", "")] = (
                getattr(c, "name", "")
            )
    beats = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        beat_id = str(ln.get("beat_id", "") or "").strip()
        dsi = str(ln.get("dialogue_slot_id", "") or "").strip()
        char_id = str(ln.get("char_id", "") or "").strip()
        speaker_role = str(ln.get("speaker_role", "") or "").strip()
        # Speaker resolution: announcer -> ANNOUNCER; otherwise look up
        # cast_id_by_name; fall back to the raw char_id uppercased.
        if speaker_role == "announcer":
            speaker = "ANNOUNCER"
        elif speaker_role in ("music_open", "music_close", "music_inter"):
            speaker = "MUSIC"
        else:
            speaker = cast_id_by_name.get(char_id, char_id.upper())
        intent = str(ln.get("beat_intent", "") or "").strip()
        beats.append(_BeatLike(
            beat_id=beat_id,
            speaker=speaker,
            dialogue_slot_id=(dsi or None),
            intent=intent,
        ))
    return beats


def check_constraints_from_ledger(
    ledger_data: dict,
    *,
    cast_names_extra=None,
    allowed_proper_nouns=None,
) -> EditorConstraintVerdict:
    """Run check_constraints against the in-flight ledger.

    Reads ledger.cast / ledger.lines / ledger.meta.dramatic_state
    and reconstructs a minimal Stage1Plan-shaped view so the pure-
    Python check_constraints can run without round-tripping through
    pydantic. Returns the same EditorConstraintVerdict
    check_constraints would return.

    The Sprint 5.1 wire-up calls this in OTR_LedgerScriptWriter
    after DramaticState is stamped (Sprint 2.1 site) to produce a
    diagnostic `meta.editor_constraints` audit. The live editor
    cycle is unchanged this sprint -- the actual swap (replacing
    the taste editor with the constraint editor at
    max_editor_cycles=1) lives in a follow-up sprint that touches
    run_story_room.
    """
    cast = _cast_view_from_ledger(ledger_data)
    beats = _beats_view_from_ledger(ledger_data)
    meta = ledger_data.get("meta") or {}
    ds = meta.get("dramatic_state")  # dict OR None
    plan_view = _PlanLike(cast=cast, beats=beats, dramatic_state=ds)
    return check_constraints(
        plan_view,
        cast_names_extra=cast_names_extra,
        allowed_proper_nouns=allowed_proper_nouns,
    )



# ---------------------------------------------------------------------------
# Sprint 5.2 wire-up helper (2026-05-28)
# ---------------------------------------------------------------------------


__all__.extend([
    "derive_repair_prompt",
    "MAX_REPAIR_TURNS",
])


# Sprint 5.2: hard cap on the number of repair turns the live swap
# fires per failed constraint check. Sprint 5 default was 1 cycle;
# the live swap honors that contract (one targeted repair turn then
# ship-or-fail, no open-ended improvement loop).
MAX_REPAIR_TURNS: int = 1


_CONSTRAINT_REPAIR_HEADERS: dict[str, str] = {
    EditorConstraint.WRONG_SPEAKER: (
        "WRONG_SPEAKER: a line is attributed to a name not in the "
        "locked cast. Use only the cast names supplied above."
    ),
    EditorConstraint.PHANTOM_CHARACTER: (
        "PHANTOM_CHARACTER: a non-cast proper noun is named or "
        "speaks. Drop or rename to a cast member; generic roles "
        "like 'the tech' / 'the lab' are fine."
    ),
    EditorConstraint.MISSING_COSTLY_CHOICE: (
        "MISSING_COSTLY_CHOICE: the dramatic_state.costly_choice_beat "
        "does not resolve. The third act needs a beat whose "
        "state_after differs concretely from its state_before."
    ),
    EditorConstraint.NO_FINAL_THIRD_TURN: (
        "NO_FINAL_THIRD_TURN: the final third of the voiced beats "
        "has no state change. Add a concrete turn before the close."
    ),
    EditorConstraint.FORMAT_FAILURE: (
        "FORMAT_FAILURE: the draft carries bracket stage directions, "
        "markdown fragments, or other malformed text. Emit plain "
        "prose only."
    ),
}


def derive_repair_prompt(verdict: EditorConstraintVerdict) -> str:
    """Turn a failed EditorConstraintVerdict into a Writer-side
    repair prompt block.

    Sprint 5.2: the live constraint-editor swap in run_story_room
    runs check_constraints (or the LLM-based variant via
    EDITOR_CONSTRAINTS_SYSTEM_PROMPT) AFTER the Writer's first
    draft. On verdict.pass_decision == False, the loop threads the
    output of this helper into one targeted Writer revision turn
    (per MAX_REPAIR_TURNS), then re-checks and ships or fails loud.

    The returned string is a single, opinionated block: header +
    per-constraint hint + concrete repair_note. The Writer reads
    it as "fix these constraints; do not touch anything else."

    Returns an empty string when verdict.pass_decision is True
    (no repair needed) -- so callers can unconditionally splice
    the helper into prompt construction.
    """
    if verdict.pass_decision:
        return ""

    lines: list[str] = [
        "REVISE: the previous draft failed the editor's "
        "constraint check. Fix the listed constraints below in "
        "the revision; do not rewrite passing passages.",
    ]
    for constraint in verdict.failing_constraints:
        header = _CONSTRAINT_REPAIR_HEADERS.get(constraint, "")
        if header:
            lines.append(f"  - {header}")
        else:
            lines.append(f"  - {constraint}")
    note = (verdict.repair_note or "").strip()
    if note:
        lines.append("")
        lines.append(f"Editor's note: {note}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Sprint 5.3 (2026-05-28) -- LLM-side constraint editor surface
# ---------------------------------------------------------------------------
#
# Mirror of `_otr_editor_pass.run_editor` for the Sprint 5 constraint
# editor: same call shape (prompt builder + run() entry that consumes
# a constrained-decode generate_fn), narrower schema (no taste
# rubric, no per-axis-notes), single repair turn (Sprint 5 cap).
#
# The Sprint 5.4 follow-up swaps `_otr_story_room._call_editor` for
# `_call_constraint_editor` (a thin wrapper around run_constraint_
# editor) when the operator flips use_constraint_editor=True on the
# OTR_StoryRoom widget. That swap also forces max_editor_cycles=1.
# This sprint lands the pure LLM-side surface + tests; the loop swap
# is a separate sprint because it touches OTR_StoryRoom's
# constrained-decode binding layer.

import json as _json_for_constraint_editor

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field
from typing import Literal as _Literal


__all__.extend([
    "EditorConstraintVerdictSchema",
    "ConstraintEditorCallFailedError",
    "build_constraint_editor_prompt",
    "run_constraint_editor",
])


_CONSTRAINT_EDITOR_BASE_TEMPERATURE: float = 0.20
_CONSTRAINT_EDITOR_RETRY_TEMPERATURE: float = 0.10
# Sprint 5.3: the constraint verdict carries at most 5 enum codes +
# a one-sentence repair_note. ~600 chars of JSON + overhead fits
# comfortably in 1024 tokens at the base temperature; tighter than
# the taste editor's 4096 cap.
_CONSTRAINT_EDITOR_MAX_NEW_TOKENS: int = 1024
_CONSTRAINT_EDITOR_MAX_ATTEMPTS: int = 2


_ConstraintCodeLiteral = _Literal[
    "WRONG_SPEAKER",
    "PHANTOM_CHARACTER",
    "MISSING_COSTLY_CHOICE",
    "NO_FINAL_THIRD_TURN",
    "FORMAT_FAILURE",
]


class EditorConstraintVerdictSchema(_BaseModel):
    """Constrained-decode binding for the LLM-side constraint editor.

    Mirrors EditorConstraintVerdict one for one; the Literal union on
    `failing_constraints` keeps the sampler inside the 5 canonical
    codes so the constraint check can never invent a sixth axis. The
    consumer json.loads + Pydantic-validates the raw output (belt and
    braces vs the constrained sampler).
    """

    pass_decision: bool = _Field(
        ...,
        description=(
            "True iff failing_constraints is empty. The loop ships "
            "on True; on False it runs ONE targeted Writer revision "
            "turn (Sprint 5 cap)."
        ),
    )
    failing_constraints: List[_ConstraintCodeLiteral] = _Field(
        default_factory=list,
        description=(
            "Subset of the five EditorConstraint codes the draft "
            "currently fails. Empty when pass_decision is True."
        ),
    )
    repair_note: str = _Field(
        default="",
        max_length=400,
        description=(
            "One-sentence concrete repair instruction for the "
            "Writer's next revision turn. Required when "
            "pass_decision is False; ignored when True."
        ),
    )


class ConstraintEditorCallFailedError(RuntimeError):
    """Raised when the LLM-side constraint editor exhausts its retry
    budget. The run_story_room caller decides whether to soft-pass
    the Writer's last draft (per Section 4 Wave 2 risk row) or hard-
    fail the room; this module surfaces the failure loud."""

    def __init__(
        self,
        *,
        attempts: int,
        last_raw_output: str,
        last_error: Exception,
    ) -> None:
        self.attempts = attempts
        self.last_raw_output = last_raw_output
        self.last_error = last_error
        super().__init__(
            f"Constraint editor exhausted retry budget after "
            f"{attempts} attempt(s); last error: "
            f"{type(last_error).__name__}: {last_error}"
        )


def _coerce_director_brief_dict(brief: Any) -> dict:
    """Local copy of _otr_editor_pass._coerce_director_brief's logic
    so this module stays decoupled from the taste editor's prompt
    builder."""
    if isinstance(brief, str):
        try:
            brief = _json_for_constraint_editor.loads(brief)
        except _json_for_constraint_editor.JSONDecodeError as exc:
            raise ConstraintEditorCallFailedError(
                attempts=0,
                last_raw_output=brief[:200],
                last_error=exc,
            )
    if isinstance(brief, dict):
        d = brief
    else:
        d = {
            "news_premise": getattr(brief, "news_premise", ""),
            "dramatic_question": getattr(brief, "dramatic_question", ""),
            "opposed_desires": getattr(brief, "opposed_desires", ""),
            "arc_shape": getattr(brief, "arc_shape", ""),
            "audience_feel": getattr(brief, "audience_feel", ""),
            "forbidden_drift": list(getattr(brief, "forbidden_drift", []) or []),
            "raw_brief": getattr(brief, "raw_brief", ""),
        }
    return {
        "news_premise": str(d.get("news_premise", "") or ""),
        "dramatic_question": str(d.get("dramatic_question", "") or ""),
        "opposed_desires": str(d.get("opposed_desires", "") or ""),
        "arc_shape": str(d.get("arc_shape", "") or ""),
        "audience_feel": str(d.get("audience_feel", "") or ""),
        "forbidden_drift": [
            str(x) for x in (d.get("forbidden_drift", []) or [])
        ],
        "raw_brief": str(d.get("raw_brief", "") or ""),
    }


def build_constraint_editor_prompt(
    director_brief: Any,
    writer_draft: str,
    *,
    cast_names: Optional[List[str]] = None,
    cycle: int = 0,
) -> List[dict]:
    """Build [system, user] messages for the constraint editor call.

    Args:
        director_brief: DirectorBriefLike OR dict OR serialized JSON.
        writer_draft: the current Writer draft as one prose string.
        cast_names: optional locked cast roster; rendered into the
            prompt so the LLM has a concrete reference when scoring
            WRONG_SPEAKER / PHANTOM_CHARACTER.
        cycle: 0-indexed editor pass number. Stamped on the verdict.

    Returns a two-message list ready for generate_fn.
    """
    brief = _coerce_director_brief_dict(director_brief)
    draft = (writer_draft or "").strip() or "(empty draft)"
    cast_block = ""
    if cast_names:
        rows = [str(n).strip() for n in cast_names if str(n).strip()]
        if rows:
            cast_block = (
                "LOCKED CAST (use these names verbatim; any other "
                "proper noun fails PHANTOM_CHARACTER):\n  - "
                + "\n  - ".join(rows)
                + "\n\n"
            )

    user = (
        f"This is constraint-editor cycle {int(cycle)} for this "
        "episode.\n\n"
        f"{cast_block}"
        f"DIRECTOR'S BRIEF:\n"
        f"  News premise: {brief['news_premise']}\n"
        f"  Dramatic question: {brief['dramatic_question']}\n"
        f"  Opposed desires: {brief['opposed_desires']}\n"
        f"  Arc shape: {brief['arc_shape']}\n"
        f"  Audience feel: {brief['audience_feel']}\n"
        f"\n"
        f"WRITER'S CURRENT DRAFT:\n"
        f"--- begin draft ---\n"
        f"{draft}\n"
        f"--- end draft ---\n\n"
        "EDITOR INSTRUCTIONS:\n"
        "  1. Decide pass_decision (True iff failing_constraints is "
        "empty).\n"
        "  2. List EVERY constraint that fails. Use only these enum "
        "codes verbatim: WRONG_SPEAKER, PHANTOM_CHARACTER, "
        "MISSING_COSTLY_CHOICE, NO_FINAL_THIRD_TURN, FORMAT_FAILURE.\n"
        "  3. When pass_decision is False, write ONE concrete "
        "sentence in repair_note naming what the Writer should fix.\n"
        "  4. Do NOT rewrite the draft. Do NOT add taste notes.\n\n"
        "Output a single JSON object matching:\n"
        '{"pass_decision": <true|false>, "failing_constraints": '
        '[<codes>], "repair_note": "<one sentence>"}'
    )
    return [
        {"role": "system", "content": EDITOR_CONSTRAINTS_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def run_constraint_editor(
    director_brief: Any,
    writer_draft: str,
    *,
    generate_fn: Callable[..., str],
    cast_names: Optional[List[str]] = None,
    cycle: int = 0,
    max_attempts: int = _CONSTRAINT_EDITOR_MAX_ATTEMPTS,
) -> EditorConstraintVerdict:
    """Run one constraint-editor pass against the current Writer
    draft. Returns an EditorConstraintVerdict.

    Args:
        director_brief: DirectorBriefLike OR dict OR JSON string.
        writer_draft: the current draft prose.
        generate_fn: a constrained-decode closure bound to
            EditorConstraintVerdictSchema via
            _otr_constrained_generate.make_constrained_generate_fn.
            Signature: (messages, *, temperature, max_new_tokens) -> str.
        cast_names: optional locked cast roster.
        cycle: 0-indexed pass number; stamped on the returned verdict.
        max_attempts: retry budget (default 2 -- one base attempt +
            one structural retry at lower temperature).

    Raises:
        ConstraintEditorCallFailedError on retry exhaustion.

    # LLM slot: technical
    # Reason: structured constrained-decode verdict per project
    # rule 6. Constraint-editor turns route to the writer's
    # technical_model slot.
    """
    messages = build_constraint_editor_prompt(
        director_brief,
        writer_draft,
        cast_names=cast_names,
        cycle=cycle,
    )
    last_raw_output = ""
    last_error: Optional[Exception] = None
    for attempt_idx in range(1, max_attempts + 1):
        temperature = (
            _CONSTRAINT_EDITOR_BASE_TEMPERATURE
            if attempt_idx == 1
            else _CONSTRAINT_EDITOR_RETRY_TEMPERATURE
        )
        try:
            # LLM slot: technical
            # Reason: constrained-decode against
            # EditorConstraintVerdictSchema; rule 6.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_CONSTRAINT_EDITOR_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            last_raw_output = ""
            continue
        last_raw_output = raw or ""
        try:
            payload = _json_for_constraint_editor.loads(raw or "")
            parsed = EditorConstraintVerdictSchema.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 -- pydantic + JSONDecodeError
            last_error = exc
            continue
        return EditorConstraintVerdict(
            pass_decision=bool(parsed.pass_decision),
            failing_constraints=list(parsed.failing_constraints),
            repair_note=str(parsed.repair_note or ""),
            cycle=int(cycle),
        )

    raise ConstraintEditorCallFailedError(
        attempts=max_attempts,
        last_raw_output=last_raw_output,
        last_error=(
            last_error
            if last_error is not None
            else RuntimeError("no error captured")
        ),
    )

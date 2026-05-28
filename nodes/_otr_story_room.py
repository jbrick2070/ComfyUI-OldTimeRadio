"""nodes/_otr_story_room.py -- Sprint 10B Wave 2 Agent F (2026-05-27).

The writers' room loop. Wires the Director (Wave 1 Agent D), the Writer
(this module), and the Editor (Wave 1 Agent E) into a bounded
conversation that converges on a publishable Writer draft.

Design references:
- docs/2026-05-26-good-story-writer-architecture__02_design.md
  Section 1 architecture, Section 3.3 StoryRoomTranscript contract,
  Section 4 Wave 2 Agent F (loop body + termination rules),
  Section 7 PD7 reproducibility relaxation under use_story_room=True.

This module is PURE: no I/O, no GPU, no ComfyUI imports, no torch.
The two LLM call sites are the Writer pass and (re-uses) the Editor
pass; both are tagged at the call site per project rule 6:

    # LLM slot: creative   (Writer turns -- free-form prose)
    # LLM slot: technical  (Editor turns -- constrained JSON verdict)

The Director call happens BEFORE run_story_room is called; the brief
is consumed as data here, not generated. Wave 1 Agent D owns the
Director call site.

Wave 2 lands this module + node + tests; the node ships dormant
behind `use_story_room: bool = False`. Wave 3 (operator listen-test
gate) decides whether the room ships on by default for production.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional

from ._otr_director_brief import DirectorBrief
from ._otr_editor_pass import EditorCallFailedError, EditorVerdict, run_editor


log = logging.getLogger("OTR")


__all__ = [
    "StoryRoomTurn",
    "StoryRoomTranscript",
    "StoryRoomCallFailedError",
    "build_writer_system_prompt",
    "build_writer_user_prompt",
    "run_story_room",
    "WRITER_BASE_TEMPERATURE",
    "WRITER_REVISION_TEMPERATURE",
    "WRITER_MAX_NEW_TOKENS",
    "DEFAULT_MAX_WRITER_TURNS",
    "DEFAULT_MAX_EDITOR_CYCLES",
    "DEFAULT_MAX_TOTAL_TURNS",
]


# ---------------------------------------------------------------------------
# Loop budget defaults (design Section 1 + Section 4 Wave 2)
# ---------------------------------------------------------------------------

# Section 1: "Director once -> Writer 1-4 turns -> Editor -> if fail,
# Writer 1-3 more turns -> Editor again. Hard cap: 3 Editor cycles, 18
# total turns." These defaults are bound here as module constants so
# they appear in one place; the node and callers can still override.
DEFAULT_MAX_WRITER_TURNS: int = 4
DEFAULT_MAX_EDITOR_CYCLES: int = 3
DEFAULT_MAX_TOTAL_TURNS: int = 18


# ---------------------------------------------------------------------------
# Writer sampling defaults
# ---------------------------------------------------------------------------

# The Writer is the creative slot. The first draft draws from a wider
# distribution; revision drafts narrow the sampling so the Writer
# converges on a publishable text rather than wandering across rewrites.
# These are middle-of-the-road numbers; Wave 3's operator A/B may tune.
WRITER_BASE_TEMPERATURE: float = 0.85
WRITER_REVISION_TEMPERATURE: float = 0.65

# 2400 tokens is enough room for a ~600 word episode body (the Section
# 2.5 worked example is 502 words) with revision headroom. The Writer
# is free-form prose, NOT constrained-decode, so the model is allowed
# to stop earlier when it has a clean ending.
WRITER_MAX_NEW_TOKENS: int = 2400


# ---------------------------------------------------------------------------
# StoryRoomTurn / StoryRoomTranscript -- the frozen contract (Section 3.3)
# ---------------------------------------------------------------------------


@dataclass
class StoryRoomTurn:
    """One role turn inside the story room.

    role:    'director' | 'writer' | 'editor'.
    cycle:   which Editor cycle the turn belongs to. 0 means
             pre-first-Editor (the initial Writer drafts ride cycle 0
             before any Editor pass has run); 1 means the cycle that
             follows the first Editor pass; etc.
    content: free-form prose output for the writers'-room turn.
             For Director and Writer turns this is the raw model
             output (or a brief string for the Director's brief);
             for Editor turns it is
             the JSON verdict serialized as a string (or
             EditorVerdict.to_dict() rendered as JSON by the caller).
    """

    role: str
    cycle: int
    content: str


@dataclass
class StoryRoomTranscript:
    """Full transcript of one writers' room session.

    Per design Section 3.3, this is the wire format Wave 2 Agent G's
    extraction pass reads. Field shapes are frozen for the duration of
    Sprint 10B.
    """

    director_brief: "DirectorBrief"
    turns: List[StoryRoomTurn] = field(default_factory=list)
    editor_verdicts: List["EditorVerdict"] = field(default_factory=list)
    final_draft: str = ""
    terminated_clean: bool = False
    total_turns: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict for serialization.

        The Director brief and Editor verdicts each expose `.to_dict()`;
        StoryRoomTurn is a plain dataclass so asdict suffices.
        """
        return {
            "director_brief": self.director_brief.to_dict(),
            "turns": [asdict(t) for t in self.turns],
            "editor_verdicts": [v.to_dict() for v in self.editor_verdicts],
            "final_draft": self.final_draft,
            "terminated_clean": bool(self.terminated_clean),
            "total_turns": int(self.total_turns),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class StoryRoomCallFailedError(RuntimeError):
    """Raised when the Story Room cannot produce any Writer draft.

    Distinct from EditorCallFailedError: the Editor failing is
    recoverable inside the loop (we surface a soft-pass verdict and
    keep the last Writer draft as final_draft with
    terminated_clean=False). This exception fires when even the FIRST
    Writer turn fails to produce a non-empty draft -- there is nothing
    to ship and the consuming node decides whether to crash the queue
    or surface a sentinel.
    """

    def __init__(
        self,
        *,
        attempts: int,
        last_error: Optional[BaseException],
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        last_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "no error captured"
        )
        super().__init__(
            f"[OTR_StoryRoom] no Writer draft produced after "
            f"{attempts} attempt(s); last error -> {last_text}"
        )


# ---------------------------------------------------------------------------
# Writer prompt construction
# ---------------------------------------------------------------------------


_WRITER_SYSTEM_PROMPT = """\
You are the writer in a small writers' room for a short audio drama.
You write a single-take, two-hander radio episode in the spirit of
1940s-1970s late-night anthology radio: spare staging, character voice
above plot fireworks, a beginning + middle + end that lands cleanly
inside roughly five hundred words of dialogue.

You are working from the director's brief and (after the first pass)
from the editor's notes. Your job is to draft and revise the body of
one episode.

WRITING RULES:
- Name the news premise concretely. A listener who never read the
  source story must be able to identify it from the dialogue alone --
  not a generic frame, the specific detail.
- Two named characters carry the scene. Use the cast names verbatim
  in dialogue attributions, no nicknames, no renaming.
- One bookend each: a short ANNOUNCER line at the open and at the
  close. The rest is character dialogue with stage directions kept
  minimal (a beat of breath, a sound off in the hall) -- the audio
  bus will paint atmosphere; the dialogue carries the story.
- Arc shape: setup, turn, landing. The protagonist faces a choice
  that costs them something. Avoid the "alarm bell + countdown +
  rescue" beat shape that the editor flags as code-red theatrics.
- Safe for work. No profanity. Non-violent. Good narrative arc.

OUTPUT FORMAT:
- Prose only. No JSON, no Markdown, no commentary outside the script.
- One character speech per paragraph. Begin each paragraph with the
  SPEAKER NAME in all caps followed by a colon and a single space,
  then the speech.
- ANNOUNCER bookends use the same speaker-prefix format.
- Do NOT label the brief, the cycle, or the notes -- write the
  episode itself.
"""


def _format_cast_block(cast: List[Any]) -> str:
    """Render the cast list for the Writer prompt.

    Accepts either Stage1CastMember objects or plain strings; emits a
    bulleted list with name + (optional) persona excerpt so the Writer
    has voice texture to draw from. Empty / missing names are dropped.
    """
    rows: list[str] = []
    for entry in (cast or []):
        if entry is None:
            continue
        if isinstance(entry, str):
            name = entry.strip()
            persona = ""
        else:
            name = str(getattr(entry, "name", "") or "").strip()
            persona = str(getattr(entry, "persona", "") or "").strip()
        if not name:
            continue
        if persona:
            # Cap the persona to keep the prompt prefix stable across
            # turns (prefix caching favours a fixed-shape cast block).
            persona_short = persona if len(persona) <= 240 else (
                persona[:240].rstrip() + "..."
            )
            rows.append(f"- {name}: {persona_short}")
        else:
            rows.append(f"- {name}")
    if not rows:
        return "- (no cast supplied)"
    return "\n".join(rows)


def _format_brief_block(brief: DirectorBrief) -> str:
    """Render the Director's brief as a prose block for the Writer.

    Mirrors the field order the Director's prompt asks for so a
    Writer reading the block sees the brief in the same shape the
    Director wrote it.
    """
    drift = brief.forbidden_drift or []
    drift_block = (
        "\n".join(f"  - {phrase}" for phrase in drift)
        if drift
        else "  (none called out)"
    )
    return (
        "DIRECTOR'S BRIEF:\n"
        f"News premise:      {brief.news_premise}\n"
        f"Dramatic question: {brief.dramatic_question}\n"
        f"Opposed desires:   {brief.opposed_desires}\n"
        f"Arc shape:         {brief.arc_shape}\n"
        f"Audience feel:     {brief.audience_feel}\n"
        f"Forbidden drift:\n{drift_block}"
    )


def _format_editor_notes_block(verdict: EditorVerdict) -> str:
    """Render the last Editor verdict as a notes block for the Writer.

    Only failing axes appear -- the Writer doesn't need praise, just
    direction on what to revise. Pulls the per-axis notes verbatim
    from the Editor, in the order failing_axes lists them.
    """
    if not verdict.failing_axes:
        return (
            "EDITOR NOTES:\n"
            "  (no failing axes named -- revise for the overall_note)"
        )
    lines = ["EDITOR NOTES:"]
    for axis_key in verdict.failing_axes:
        note = verdict.per_axis_notes.get(axis_key, "").strip()
        if note:
            lines.append(f"  {axis_key}: {note}")
        else:
            lines.append(f"  {axis_key}: (no per-axis note)")
    overall = (verdict.overall_note or "").strip()
    if overall:
        lines.append("")
        lines.append(f"Overall: {overall}")
    return "\n".join(lines)


def _format_constraint_repair_block(verdict: EditorVerdict) -> str:
    """Sprint 5.5 (2026-05-28) -- render the Writer's revision
    directive from a constraint-editor verdict.

    Sprint 5.4 stores the constraint editor's output in an
    EditorVerdict-shape adapter where:
      verdict.failing_axes   == EditorConstraintVerdict.failing_constraints
      verdict.overall_note   == EditorConstraintVerdict.repair_note

    The Sprint 5.2 `derive_repair_prompt` helper produces a richer
    block than `_format_editor_notes_block` -- it labels each
    constraint code with a concrete per-code repair header so the
    Writer doesn't have to guess what "WRONG_SPEAKER" means.
    Sprint 5.5 splices that block in directly when the
    constraint-editor path is on.

    Pass-decision verdicts return an empty block (Writer has
    nothing to revise) -- callers can splice unconditionally.
    """
    from ._otr_editor_constraints import (
        EditorConstraintVerdict,
        derive_repair_prompt,
    )

    constraint_verdict = EditorConstraintVerdict(
        pass_decision=bool(verdict.pass_decision),
        failing_constraints=list(verdict.failing_axes or []),
        repair_note=str(verdict.overall_note or ""),
        cycle=int(getattr(verdict, "cycle", 0) or 0),
    )
    return derive_repair_prompt(constraint_verdict)


def build_writer_system_prompt() -> str:
    """Return the Writer's system prompt verbatim.

    Stable across all Writer turns inside one episode so prefix caching
    in the backend has a chance to amortize KV-cache warmup. Wave 3
    operator tuning may tweak this; the function indirection exists so
    a tweak lands in one place.
    """
    return _WRITER_SYSTEM_PROMPT


def build_writer_user_prompt(
    brief: DirectorBrief,
    cast: List[Any],
    news_seed: str,
    last_editor_verdict: Optional[EditorVerdict] = None,
    last_draft: Optional[str] = None,
    cycle: int = 0,
    *,
    use_constraint_editor: bool = False,
) -> str:
    """Build the user-message body for one Writer turn.

    On the first Writer turn (no `last_editor_verdict`), the prompt is
    director's brief + cast + news seed + a "draft the episode" cue.

    On revision turns the prompt is brief + cast + news seed + the last
    Writer draft + the Editor's notes + a "revise the draft addressing
    these notes" cue. The brief and cast are repeated verbatim each
    turn so the model isn't relying on chat history to keep them in
    scope.
    """
    seed = (news_seed or "").strip() or "(no news seed supplied)"
    brief_block = _format_brief_block(brief)
    cast_block = _format_cast_block(cast)

    parts: list[str] = [
        f"This is editor cycle {int(cycle)}.",
        "",
        brief_block,
        "",
        "CAST (use these names verbatim in dialogue attributions):",
        cast_block,
        "",
        "NEWS SEED (the real story this episode is about):",
        seed,
    ]

    if last_editor_verdict is not None and last_draft is not None:
        draft_clean = (last_draft or "").strip() or "(empty draft)"
        # Sprint 5.5 (2026-05-28): when the constraint editor is on,
        # render the per-code repair block via derive_repair_prompt
        # (labelled headers, no taste-rubric framing). Legacy path
        # keeps the unchanged _format_editor_notes_block.
        if use_constraint_editor:
            repair_block = _format_constraint_repair_block(
                last_editor_verdict,
            )
            # When derive_repair_prompt returns empty (the verdict
            # passed -- shouldn't happen in a revision cycle, but
            # defensive) fall through to a minimal block so the
            # Writer still has signal.
            if not repair_block:
                repair_block = (
                    "EDITOR NOTES:\n  (the constraint editor "
                    "passed; revise only for clarity)"
                )
            revision_directive = (
                "Revise the draft addressing each named constraint "
                "above. Do NOT rewrite passing passages. Output the "
                "FULL revised episode as prose now."
            )
        else:
            repair_block = _format_editor_notes_block(last_editor_verdict)
            revision_directive = (
                "Revise the draft addressing every failing axis named "
                "above. Output the FULL revised episode as prose now, "
                "not a diff or a change list."
            )
        parts.extend([
            "",
            "YOUR LAST DRAFT (the editor's notes are about THIS):",
            "--- begin last draft ---",
            draft_clean,
            "--- end last draft ---",
            "",
            repair_block,
            "",
            revision_directive,
        ])
    else:
        parts.extend([
            "",
            "Draft the episode now. Output the FULL episode as prose, "
            "speaker-prefixed paragraphs only, no preamble.",
        ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Writer turn
# ---------------------------------------------------------------------------


def _call_writer(
    brief: DirectorBrief,
    cast: List[Any],
    news_seed: str,
    *,
    writer_generate_fn: Callable[..., str],
    last_editor_verdict: Optional[EditorVerdict],
    last_draft: Optional[str],
    cycle: int,
    is_first_pass_in_cycle: bool,
    use_constraint_editor: bool = False,
) -> str:
    """Run one Writer turn and return the raw draft text.

    The Writer is a free-form prose pass -- creative slot, no
    constrained-decode schema. The caller decides whether the result
    is a usable draft (non-empty after stripping).

    Sprint 5.5 (2026-05-28): `use_constraint_editor` propagates into
    `build_writer_user_prompt` so revision turns under the
    constraint-editor path render `derive_repair_prompt`-style
    per-code repair blocks instead of the taste-rubric notes block.
    """
    user = build_writer_user_prompt(
        brief=brief,
        cast=cast,
        news_seed=news_seed,
        last_editor_verdict=last_editor_verdict,
        last_draft=last_draft,
        cycle=cycle,
        use_constraint_editor=use_constraint_editor,
    )
    messages = [
        {"role": "system", "content": build_writer_system_prompt()},
        {"role": "user", "content": user},
    ]
    # First Writer turn of an episode draws from a wider distribution;
    # revision turns narrow it so the Writer converges on the brief
    # rather than re-imagining the episode each cycle.
    temperature = (
        WRITER_BASE_TEMPERATURE
        if is_first_pass_in_cycle and cycle == 0
        else WRITER_REVISION_TEMPERATURE
    )
    # LLM slot: creative
    # Reason: Writer turns are free-form prose drafts of the episode
    # body. Per project rule 6 creative narrative passes route to the
    # writer's `creative_writing_model` slot.
    return writer_generate_fn(
        messages,
        temperature=temperature,
        max_new_tokens=WRITER_MAX_NEW_TOKENS,
    )


# ---------------------------------------------------------------------------
# Editor turn (delegates to Wave 1 Agent E)
# ---------------------------------------------------------------------------


def _call_editor(
    brief: DirectorBrief,
    draft: str,
    *,
    editor_generate_fn: Callable[..., str],
    rubric: Any,
    cycle: int,
) -> EditorVerdict:
    """Run one Editor pass against the current Writer draft.

    Delegates to `_otr_editor_pass.run_editor`; the constrained-decode
    generate_fn must already be bound to EditorVerdictSchema. On
    EditorCallFailedError we soft-pass at the loop level (Section 4
    Wave 2 risk row: "Editor call fails -> verdict stamped as
    soft-pass with overall_note describing the failure"). The Writer's
    last draft is preserved as final_draft so the episode can still
    ship; terminated_clean rides False to record the unhealthy exit.
    """
    # LLM slot: technical
    # Reason: Editor verdicts are JSON objects validated against
    # EditorVerdictSchema. Per project rule 6 structured-JSON passes
    # route to the writer's `technical_model` slot.
    try:
        return run_editor(
            director_brief=brief,
            writer_draft=draft,
            generate_fn=editor_generate_fn,
            rubric=rubric,
            cycle=cycle,
        )
    except EditorCallFailedError as exc:
        log.warning(
            "[OTR_StoryRoom] Editor call failed at cycle %d; soft-passing "
            "the last Writer draft so the episode can ship. Error: %s",
            cycle, exc,
        )
        return EditorVerdict(
            pass_decision=True,
            failing_axes=[],
            per_axis_notes={},
            overall_note=(
                f"EditorCallFailedError at cycle {cycle} after "
                f"{exc.attempts} attempt(s); soft-pass to keep the "
                f"episode shippable. Last error: "
                f"{type(exc.last_error).__name__}: {exc.last_error}"
            ),
            cycle=cycle,
        )


def _call_constraint_editor(
    brief: DirectorBrief,
    draft: str,
    *,
    editor_generate_fn: Callable[..., str],
    cast: List[Any],
    cycle: int,
) -> EditorVerdict:
    """Sprint 5.4 (2026-05-28) -- the constraint-editor sibling of
    _call_editor. Delegates to `_otr_editor_constraints
    .run_constraint_editor` and adapts the EditorConstraintVerdict
    into an EditorVerdict-shaped object so the run_story_room loop
    + transcript stay back-compat.

    Adapter mapping:
      EditorConstraintVerdict.pass_decision  -> EditorVerdict.pass_decision
      EditorConstraintVerdict.failing_constraints -> failing_axes
      EditorConstraintVerdict.repair_note    -> overall_note
                                                (the Writer revision
                                                turn reads this via
                                                _format_editor_notes_block)
      EditorConstraintVerdict.cycle          -> cycle
      per_axis_notes                         -> {}  (no taste rubric)

    On ConstraintEditorCallFailedError we soft-pass identically to the
    taste editor's failure path so Section 4 Wave 2's "Editor call
    fails -> soft-pass with note" contract still holds.

    The editor_generate_fn passed in MUST be constrained-decode-bound
    to EditorConstraintVerdictSchema (the OTR_StoryRoom node layer
    binds this when use_constraint_editor=True is on the widget).
    """
    # LLM slot: technical
    # Reason: structured constrained-decode verdict; rule 6.
    from ._otr_editor_constraints import (
        ConstraintEditorCallFailedError,
        run_constraint_editor,
    )

    cast_names = []
    for c in (cast or []):
        nm = ""
        if isinstance(c, dict):
            nm = str(c.get("name", "") or "").strip()
        else:
            nm = str(getattr(c, "name", "") or "").strip()
        if nm:
            cast_names.append(nm)

    try:
        verdict = run_constraint_editor(
            brief,
            draft,
            generate_fn=editor_generate_fn,
            cast_names=cast_names or None,
            cycle=cycle,
        )
    except ConstraintEditorCallFailedError as exc:
        log.warning(
            "[OTR_StoryRoom] constraint editor failed at cycle %d; "
            "soft-passing the Writer's last draft so the episode "
            "ships. Error: %s",
            cycle, exc,
        )
        return EditorVerdict(
            pass_decision=True,
            failing_axes=[],
            per_axis_notes={},
            overall_note=(
                f"ConstraintEditorCallFailedError at cycle {cycle} "
                f"after {exc.attempts} attempt(s); soft-pass to keep "
                f"the episode shippable. Last error: "
                f"{type(exc.last_error).__name__}: {exc.last_error}"
            ),
            cycle=cycle,
        )

    # Adapter -> EditorVerdict shape. The Writer revision turn reads
    # `overall_note` via _format_editor_notes_block; mapping
    # `repair_note` here means the Writer sees the constraint
    # editor's repair note as the revision directive without any
    # other code change.
    return EditorVerdict(
        pass_decision=bool(verdict.pass_decision),
        failing_axes=list(verdict.failing_constraints),
        per_axis_notes={},
        overall_note=str(verdict.repair_note or ""),
        cycle=int(verdict.cycle),
    )


# ---------------------------------------------------------------------------
# Public entry point -- run_story_room
# ---------------------------------------------------------------------------


def run_story_room(
    *,
    director_brief: DirectorBrief,
    news_seed: str,
    cast: List[Any],
    writer_generate_fn: Callable[..., str],
    editor_generate_fn: Callable[..., str],
    rubric: Any = None,
    max_writer_turns: int = DEFAULT_MAX_WRITER_TURNS,
    max_editor_cycles: int = DEFAULT_MAX_EDITOR_CYCLES,
    max_total_turns: int = DEFAULT_MAX_TOTAL_TURNS,
    use_constraint_editor: bool = False,
) -> StoryRoomTranscript:
    """Run the writers' room loop and return a populated transcript.

    Loop body (per design Section 1 + Section 4 Wave 2):
        1. The Director has already produced a brief (passed in).
        2. Writer takes up to `max_writer_turns` turns producing
           drafts in cycle 0. The first non-empty Writer draft is the
           current `final_draft` candidate; subsequent first-cycle
           turns OVERWRITE it (the model is free to throw away an
           earlier draft and try again until it has something to ship
           for the Editor to read).
        3. Editor reads the current draft.
           - pass_decision=True  -> terminate clean.
           - pass_decision=False -> push EditorVerdict.notes back into
             a Writer revision turn (one revision per cycle past
             cycle 0; the design says 1-3 turns per revision cycle,
             implemented as 1 revision turn per Editor cycle here so
             the budget stays predictable).
        4. Cap on Editor cycles (default 3) and total turns (default
           18). When either fires, terminate with terminated_clean
           depending on the last Editor verdict.

    Args:
        director_brief: the Director's brief for this episode.
            Wave 1 Agent D produces this; this loop CONSUMES it.
        news_seed: the news premise the episode is about. The Writer
            and Editor both see it so the room cannot drift off the
            news while staying technically schema-clean.
        cast: cast list. Either Stage1CastMember objects or plain
            strings. The Writer prints names verbatim in dialogue;
            the Editor reads cast names from the Director's brief.
        writer_generate_fn: free-form generate fn from
            `_otr_model_loader.make_generate_fn(cache_entry)`. Signature:
            (messages, *, temperature, max_new_tokens) -> str.
        editor_generate_fn: constrained-decode generate fn from
            `_otr_constrained_generate.make_constrained_generate_fn(
                cache_entry, EditorVerdictSchema)`.
            Signature identical to the writer fn.
        rubric: optional pre-loaded `Rubric` instance. When None the
            Editor loads it on first call.
        max_writer_turns: per-cycle Writer turn cap. Cycle 0 gets up
            to this many drafts before the first Editor pass; each
            later cycle gets one revision turn.
        max_editor_cycles: hard cap on Editor cycles (default 3).
        max_total_turns: hard cap on TOTAL turns regardless of role
            (default 18 per Section 1).

    Returns:
        A populated StoryRoomTranscript with:
            director_brief preserved verbatim,
            turns logging every Director / Writer / Editor turn,
            editor_verdicts in cycle order,
            final_draft = the canonical Writer draft (the one the
                Editor passed on, or the last produced draft if the
                cap fired),
            terminated_clean True iff the Editor said pass,
            total_turns counted across all roles,
            elapsed_seconds wall-clock.

    Raises:
        StoryRoomCallFailedError when no Writer turn produces a
        non-empty draft AND the Editor has nothing to read. This is
        the only hard failure path -- everything else (Editor failing,
        cap firing, Writer producing empty drafts but at least one
        usable one) is recoverable with terminated_clean=False.

    PD7 (reproducibility) is RELAXED for this loop per design Section
    10 Q6 -- the transcript is the new reproducibility anchor. Same
    transcript in => same downstream audio out.
    """
    if max_writer_turns < 1:
        raise ValueError("max_writer_turns must be >= 1")
    if max_editor_cycles < 1:
        raise ValueError("max_editor_cycles must be >= 1")
    if max_total_turns < 2:
        # Need room for at least one Writer turn + one Editor turn.
        raise ValueError("max_total_turns must be >= 2")

    # Sprint 5.4 (2026-05-28) -- when the constraint editor is on,
    # honor the Sprint 5 hard cap of ONE editor cycle. Quality
    # comes from the Sprint 4 selector (when wired) and the Sprint 4
    # / 5 structural validators; the editor only enforces a floor.
    # Pre-Sprint-5 max_editor_cycles widget defaults remain in
    # place for the taste-editor path -- this cap only fires when
    # use_constraint_editor is True.
    if use_constraint_editor:
        from ._otr_editor_constraints import (
            DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES,
        )
        if max_editor_cycles > DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES:
            log.info(
                "[OTR_StoryRoom] use_constraint_editor=True -- "
                "clamping max_editor_cycles %d -> %d (Sprint 5 cap; "
                "the constraint editor enforces a structural floor, "
                "not taste).",
                max_editor_cycles,
                DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES,
            )
            max_editor_cycles = DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES

    t_start = time.monotonic()

    transcript = StoryRoomTranscript(director_brief=director_brief)

    # Stamp the Director turn first. The Director's prose brief is
    # already a typed object; we surface the raw_brief in turn-log form
    # so Agent G's extraction can read the room as a single transcript.
    transcript.turns.append(StoryRoomTurn(
        role="director",
        cycle=0,
        content=(director_brief.raw_brief or "").strip(),
    ))

    current_draft: str = ""
    last_writer_error: Optional[BaseException] = None
    writer_attempts_made: int = 0

    def total_turns_used() -> int:
        return len(transcript.turns)

    def cap_hit() -> bool:
        # -1 because the Director turn was already counted; the rest
        # of the budget is for Writer + Editor turns.
        return total_turns_used() >= max_total_turns

    # ---- Cycle 0: initial Writer drafts ----
    for draft_idx in range(max_writer_turns):
        if cap_hit():
            log.info(
                "[OTR_StoryRoom] cycle 0 stopped early: max_total_turns "
                "reached at %d before draft #%d.",
                total_turns_used(), draft_idx + 1,
            )
            break
        writer_attempts_made += 1
        try:
            raw = _call_writer(
                brief=director_brief,
                cast=cast,
                news_seed=news_seed,
                writer_generate_fn=writer_generate_fn,
                last_editor_verdict=None,
                last_draft=None,
                cycle=0,
                is_first_pass_in_cycle=True,
                use_constraint_editor=use_constraint_editor,
            )
        except Exception as exc:  # noqa: BLE001 -- writer LLM call
            log.warning(
                "[OTR_StoryRoom] cycle 0 Writer draft #%d raised %s: %s",
                draft_idx + 1, type(exc).__name__, str(exc)[:200],
            )
            last_writer_error = exc
            continue

        raw_clean = (raw or "").strip()
        transcript.turns.append(StoryRoomTurn(
            role="writer",
            cycle=0,
            content=raw_clean,
        ))
        if raw_clean:
            current_draft = raw_clean
            # First non-empty draft is good enough to send to the
            # Editor. We do NOT consume the rest of the cycle 0
            # budget speculatively; the design says "up to" four
            # turns, not "always" four.
            break

    if not current_draft:
        # The Editor has nothing to read and the Writer never
        # produced a non-empty draft. This is the only hard failure
        # path.
        transcript.total_turns = total_turns_used()
        transcript.elapsed_seconds = time.monotonic() - t_start
        raise StoryRoomCallFailedError(
            attempts=writer_attempts_made,
            last_error=last_writer_error,
        )

    # ---- Editor / revision loop ----
    last_verdict: Optional[EditorVerdict] = None
    for cycle_idx in range(max_editor_cycles):
        if cap_hit():
            log.info(
                "[OTR_StoryRoom] editor cycle %d skipped: "
                "max_total_turns reached at %d.",
                cycle_idx, total_turns_used(),
            )
            break

        # Sprint 5.4 (2026-05-28) -- pick the constraint editor when
        # use_constraint_editor is on. The editor_generate_fn must be
        # bound to EditorConstraintVerdictSchema in that case; the
        # OTR_StoryRoom node layer handles the binding choice. The
        # adapter inside _call_constraint_editor converts the
        # EditorConstraintVerdict to an EditorVerdict-shape so the
        # transcript + Writer revision turn stay byte-compat.
        if use_constraint_editor:
            verdict = _call_constraint_editor(
                brief=director_brief,
                draft=current_draft,
                editor_generate_fn=editor_generate_fn,
                cast=cast,
                cycle=cycle_idx,
            )
        else:
            verdict = _call_editor(
                brief=director_brief,
                draft=current_draft,
                editor_generate_fn=editor_generate_fn,
                rubric=rubric,
                cycle=cycle_idx,
            )
        transcript.editor_verdicts.append(verdict)
        transcript.turns.append(StoryRoomTurn(
            role="editor",
            cycle=cycle_idx,
            content=str(verdict.overall_note or "").strip(),
        ))
        last_verdict = verdict

        if verdict.pass_decision:
            transcript.terminated_clean = True
            log.info(
                "[OTR_StoryRoom] terminated clean at editor cycle %d "
                "(turns used: %d).",
                cycle_idx, total_turns_used(),
            )
            break

        # Editor failed; queue a Writer revision turn for the NEXT
        # cycle (cycle_idx + 1) provided the budget allows.
        if cap_hit():
            log.info(
                "[OTR_StoryRoom] no revision turn after editor cycle %d: "
                "max_total_turns reached at %d.",
                cycle_idx, total_turns_used(),
            )
            break
        if cycle_idx + 1 >= max_editor_cycles:
            # No more Editor cycles, so no point queueing another
            # revision; the loop is about to terminate dirty.
            break

        try:
            raw = _call_writer(
                brief=director_brief,
                cast=cast,
                news_seed=news_seed,
                writer_generate_fn=writer_generate_fn,
                last_editor_verdict=verdict,
                last_draft=current_draft,
                cycle=cycle_idx + 1,
                is_first_pass_in_cycle=True,
                use_constraint_editor=use_constraint_editor,
            )
        except Exception as exc:  # noqa: BLE001 -- writer LLM call
            log.warning(
                "[OTR_StoryRoom] revision turn at cycle %d raised %s: %s",
                cycle_idx + 1, type(exc).__name__, str(exc)[:200],
            )
            # Keep the previous draft as the candidate; the next
            # Editor cycle re-scores it. If we exhaust cycles the
            # loop terminates with terminated_clean=False and the
            # last good draft as final_draft.
            transcript.turns.append(StoryRoomTurn(
                role="writer",
                cycle=cycle_idx + 1,
                content="",
            ))
            continue

        raw_clean = (raw or "").strip()
        transcript.turns.append(StoryRoomTurn(
            role="writer",
            cycle=cycle_idx + 1,
            content=raw_clean,
        ))
        if raw_clean:
            current_draft = raw_clean
        # If the revision turn produced an empty string we KEEP the
        # last good draft -- the Editor can still re-score the
        # previous version on the next cycle.

    if last_verdict is None:
        # We never got to run the Editor (cap hit immediately after
        # the first Writer draft). Fall through with terminated_clean
        # = False.
        transcript.terminated_clean = False
    elif not last_verdict.pass_decision:
        transcript.terminated_clean = False

    transcript.final_draft = current_draft
    transcript.total_turns = total_turns_used()
    transcript.elapsed_seconds = time.monotonic() - t_start

    log.info(
        "[OTR_StoryRoom] room closed: terminated_clean=%s "
        "total_turns=%d editor_cycles=%d elapsed=%.2fs.",
        transcript.terminated_clean,
        transcript.total_turns,
        len(transcript.editor_verdicts),
        transcript.elapsed_seconds,
    )
    return transcript

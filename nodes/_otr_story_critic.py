"""nodes/_otr_story_critic.py -- Sprint 5B whole-script story-quality critic.

The pipeline today has NO pass anywhere that judges story QUALITY. Every
existing gate is deterministic (word counts, regex strips) or structural
(the cast-contract audit in `_otr_ledger_reviewer.py`). That leaves a real
hole: the pipeline can emit a script that is structurally valid,
budget-correct, and cast-clean yet dramatically empty -- flat exposition,
characters who all sound alike, an arc that collapses in the middle -- and
nothing catches it.

This module is that missing pass. It runs ONE LLM call AFTER the Script
Doctor cleanup (Sprint 3C) has finished -- the script is already
cast-clean and structurally sound, so the critic spends its whole budget
on dramatic quality. It reads the assembled episode script, walks a
6-section rubric, and returns a Pydantic-validated `StoryCriticReport`.

The report feeds two downstream consumers:

  * Sprint 5C -- a targeted reroll loop re-composes the lines named in
    `reroll_targets`; each `RerollTarget.hint` becomes the concrete
    instruction handed to the line composer.
  * Sprint 6 -- the render stage reads `render_priority` (dramatic peaks
    first), `flat_lines`, and `arc_verdict` to budget render effort.

Design mirrors `_otr_ledger_reviewer.py`:

  * PURE module. No ComfyUI node class, no `INPUT_TYPES`, no `model_id`
    widget (Prime Directive 6). The LLM consumer receives a `generate_fn`
    callable; the model id is threaded in by the freeze cascade from the
    writer's `technical_model` broadcast socket.
  * The single LLM call routes through the shared `structured_call`
    3-attempt retry ladder (base -> structural retry -> typed repair).
  * NEVER raises into the pipeline (Prime Directive 1 -- audio output
    must never break). Any failure -- exhausted ladder, raising slot fn,
    anything else -- is logged and `StoryCriticReport.clean()` is
    returned: an all-empty report with `arc_verdict="strong"`, the safe
    fallback that asks no reroll and reorders no render.

The critic is a structured rubric evaluator -- it emits typed verdicts and
ordered id lists, no creative prose -- so its one LLM call is tagged
`# LLM slot: technical`.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("OTR")


__all__ = [
    "ContinuityIssue",
    "VoiceDriftNote",
    "FlatLine",
    "RerollTarget",
    "StanceIssue",
    "StoryCriticReport",
    "run_story_critic",
]


# ---------------------------------------------------------------------------
# Shared imports -- structured_call ladder + typed repair factory.
# Package import in production; flat import when loaded standalone / under
# test. Mirrors the import guard in `_otr_ledger_reviewer.py`.
# ---------------------------------------------------------------------------

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

try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore

# The Script Doctor's per-line row renderer and episode-context header.
# Sprint 5B reuses both rather than mirroring them: the critic judges the
# SAME 7-field per-line context the Doctor does (line_id, beat_id,
# arc_phase, mood, beat_intent, target_words, actual_words, text), and a
# second renderer would only risk drift from the canonical one. One
# renderer, one header, both passes.
try:
    from ._otr_ledger_reviewer import (
        _render_lines_for_doctor,
        _render_doctor_episode_context,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_ledger_reviewer import (  # type: ignore
        _render_lines_for_doctor,
        _render_doctor_episode_context,
    )


# ---------------------------------------------------------------------------
# Generation params for the single critic LLM call.
# ---------------------------------------------------------------------------

# A whole-script rubric walk needs more room than a small JSON object: the
# report can carry several reroll targets with prose hints plus a full
# render-priority ordering. ~3000 tokens is comfortable for an episode of
# 12-25 lines and stays well inside a structured pass's budget.
_CRITIC_MAX_NEW_TOKENS = 3000

# Base attempt runs warm enough that the model engages with dramatic
# nuance, not so warm it drifts off the schema. The structural retry runs
# STRICTLY COOLER (the 2B principle: lowering entropy during a JSON-schema
# repair, never raising it). structured_call asserts this invariant at
# entry.
_CRITIC_TEMPERATURE = 0.4
_CRITIC_RETRY_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Pydantic models -- the 6-section rubric.
# Mirrors the BaseModel style of `_otr_ledger_reviewer.py`: typed Literal
# verdicts, `Field(default_factory=list)` for every collection.
# ---------------------------------------------------------------------------


class ContinuityIssue(BaseModel):
    """One rubric §1 finding: a continuity break on a specific line.

    `issue` is a one-sentence description of the break -- a fact that
    contradicts an earlier line, a prop that vanishes, a timeline that
    does not add up.
    """
    line_id: str
    issue: str


class VoiceDriftNote(BaseModel):
    """One rubric §2 finding: a character whose voice drifts.

    `note` describes how this character's dialogue stops sounding like
    the character's established register / vocabulary / attitude. Keyed
    by `char_id` because voice drift is a per-character judgement, not a
    per-line one.
    """
    char_id: str
    note: str


class FlatLine(BaseModel):
    """One rubric §3 finding: a dramatically inert line.

    `reason` names why the line lands flat -- exposition with no dramatic
    life, a beat that states a fact instead of playing a moment.
    """
    line_id: str
    reason: str


# STEP 5 (2026-06-22 story+cast fix, roundtable-converged): the named flatness
# dimension a reroll target failed -- the operational vocabulary the critic and
# composer now share. The five craft dimensions a live line should do >=1 of, plus
# `tension` (failed to meet its per-beat target intensity) and `unspecified` (the
# safe default). OPTIONAL with a default so older stored
# `meta.story_critic_report` dicts + `StoryCriticReport.clean()` still validate
# (no new required field). It is telemetry + an optional hint prefix -- NOT a
# deterministic re-craft mapper (the critic's `hint` is already actionable and is
# threaded verbatim to the composer).
FailedDimension = Literal[
    "knowledge",     # change what a character/the audience knows
    "pressure",      # shift the pressure on a character
    "relationship",  # move a relationship
    "decision",      # force or avoid a decision
    "obstacle",      # raise or clear an obstacle
    "tension",       # failed to meet the beat's target intensity
    "unspecified",
]


class RerollTarget(BaseModel):
    """One rubric §5 finding: a line the critic asks Sprint 5C to reroll.

    `hint` is CONCRETE, ACTIONABLE feedback -- this string becomes the
    reroll instruction handed verbatim to the line composer in Sprint 5C.
    It must tell the composer what to change, not merely that something is
    wrong (e.g. "give Edna a sharper retort that lands the betrayal" --
    not "this line is weak").

    `failed_dimension` (STEP 5) names which shared craft dimension the line
    missed -- for telemetry and an optional hint prefix. Optional/defaulted for
    back-compat.
    """
    line_id: str
    hint: str
    failed_dimension: FailedDimension = "unspecified"


class StanceIssue(BaseModel):
    """D2 (2026-06-22, story-quality lift) -- TELEMETRY ONLY.

    A character whose stance toward the protagonist / central object REVERSES
    across the episode with no turn beat that earns it (the "Chandra's Echo"
    antagonist who flip-flops). This is a pure observation surfaced on
    `meta.story_critic_report`; it is DELIBERATELY NOT wired to a `RerollTarget`,
    a `FailedDimension`, a freeze gate, or `needs_full_rerun` -- R3 grounding
    proved no cross-run repair channel survives a JSON-frozen rerun, so any such
    wiring would be a silent dead-end repair path. Every field defaults so a
    partial response still validates.

      character_id     the character whose stance reversed.
      target           who/what the stance is toward (FREE-FORM string).
      prior_stance     the stance established earlier.
      new_stance       the contradicting stance.
      missing_turn_beat a beat_id OR a plain reason there is no earned turn.
      line_ids         the lines that evidence the reversal.
      severity         the critic's free-form severity label.
    """
    character_id: str = ""
    target: str = ""
    prior_stance: str = ""
    new_stance: str = ""
    missing_turn_beat: str = ""
    line_ids: list[str] = Field(default_factory=list)
    severity: str = ""


# The four arc verdicts. `strong` is the all-clear; `uneven` flags a
# script whose arc has soft patches; `flat` flags an arc with no real
# rise or fall; `mid_collapse` flags the specific failure where the
# middle of the episode loses dramatic tension. `strong` is the safe
# fallback default (and the value `StoryCriticReport.clean()` returns).
ArcVerdict = Literal["strong", "uneven", "flat", "mid_collapse"]


class StoryCriticReport(BaseModel):
    """The whole-script story-quality report -- the 6-section rubric.

    Every section defaults empty / `strong` so a partial response still
    validates and a downstream consumer reading any one section never
    hits a missing field.

      §1 continuity_issues -- factual / prop / timeline breaks.
      §2 voice_drift       -- per-character voice consistency notes.
      §3 flat_lines        -- dramatically inert lines.
      §4 arc_verdict       -- the whole-episode dramatic-arc verdict.
      §5 reroll_targets    -- lines Sprint 5C should re-compose, with a
                              concrete hint each.
      §6 render_priority   -- line_ids ordered with the dramatic peaks
                              FIRST, so Sprint 6 budgets render effort
                              toward the moments that carry the episode.
    """
    continuity_issues: list[ContinuityIssue] = Field(default_factory=list)
    voice_drift: list[VoiceDriftNote] = Field(default_factory=list)
    flat_lines: list[FlatLine] = Field(default_factory=list)
    arc_verdict: ArcVerdict = "strong"
    reroll_targets: list[RerollTarget] = Field(default_factory=list)
    render_priority: list[str] = Field(default_factory=list)
    # D2 (2026-06-22, story-quality lift) -- TELEMETRY ONLY. Defaulted empty so
    # older stored reports + clean() still validate. NEVER read by the reroll /
    # freeze path (see StanceIssue docstring).
    stance_issues: list[StanceIssue] = Field(default_factory=list)

    @classmethod
    def clean(cls) -> "StoryCriticReport":
        """Return the safe-fallback report: all sections empty,
        `arc_verdict="strong"`.

        This is what `run_story_critic` returns on ANY failure. It asks
        for no reroll (Sprint 5C does nothing) and supplies no render
        ordering (Sprint 6 falls back to ledger order) -- a critic that
        could not run leaves the pipeline exactly as it found it.
        """
        return cls()


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


_CRITIC_SYSTEM_PROMPT = """\
You are a story-quality critic for an audio drama. The script you are
shown has ALREADY passed every structural gate: word budgets are correct,
the cast contract is clean, and a script doctor has done a cleanup pass.
Your job is the one thing no earlier pass does -- judge the DRAMA.

A script can be structurally perfect and still be empty: flat exposition,
characters who all sound the same, an arc that sags in the middle. You are
the pass that catches that.

Work the rubric ONE SECTION AT A TIME, in order. Do not blur the sections
together; finish your thinking on each before the next.

SECTION 1 -- CONTINUITY. Walk the lines in order. Flag any line that
breaks continuity: a fact that contradicts an earlier line, a prop or
location that appears or vanishes with no setup, a timeline that does not
add up. Emit a continuity_issues entry {line_id, issue} for each. A
coherent script has none.

SECTION 2 -- VOICE DRIFT. Consider each character across all their lines.
Flag any character whose dialogue stops sounding like that character --
register, vocabulary, attitude all matter. Emit a voice_drift entry
{char_id, note} for each affected character. A well-cast script has none.

SECTION 3 -- FLAT LINES. A character line is ALIVE when it does at least
ONE of these five things AND advances its beat_intent:
  - knowledge:    changes what a character or the audience knows
  - pressure:     shifts the pressure on a character
  - relationship: moves a relationship between characters
  - decision:     forces or avoids a decision
  - obstacle:     raises a new obstacle or clears an existing one
A line is FLAT only when it does NONE of the five AND fails to advance its
beat_intent. Be sparing -- most lines in a competent script are alive; do
not invent flatness to look thorough.

TENSION FIT: each line may carry a `target_tension=N/5` (1 = orientation /
unease, 3 = active pressure / a choice, 5 = an irreversible decision, reveal,
or cost). Judge a line against its OWN target -- a calm setup line at target 1
is NOT flat for failing to be explosive; a line at target 5 that merely
restates is flat. When a line clearly fails to reach its target intensity,
that is the `tension` dimension.

Emit a flat_lines entry {line_id, reason} for each flat line, where `reason`
names the missing dimension in plain words.

SECTION 4 -- ARC VERDICT. Judge the whole episode's dramatic arc as ONE
of:
  strong       -- the arc rises, turns, and resolves with real tension.
  uneven       -- the arc works but has soft patches.
  flat         -- the arc has no real rise or fall; nothing builds.
  mid_collapse -- the arc starts and ends well but loses all tension in
                  the middle.

SECTION 5 -- REROLL TARGETS. From the lines you flagged above, pick the
ones most worth re-composing. Emit a reroll_targets entry ONLY for a line
that fails BOTH to advance its beat_intent AND at least one named dimension
-- that double test keeps the reroll list short. For each, emit
{line_id, hint, failed_dimension} where:
  - `hint` is CONCRETE, ACTIONABLE feedback the line composer can act on
    directly -- tell it WHAT to change, not merely that the line is weak.
  - `failed_dimension` is the single most important dimension the line
    missed, one of: knowledge, pressure, relationship, decision, obstacle,
    tension. (Use `tension` when the line failed to meet its target
    intensity.) If you cannot name one, use "unspecified".

SECTION 6 -- RENDER PRIORITY. Order EVERY line_id you were shown into a
single list, dramatic peaks FIRST -- the moments that carry the episode at
the top, the quiet connective lines at the bottom. The render stage uses
this order to budget effort.

SECTION 7 -- STANCE CONSISTENCY (observation only). Consider each character's
stance toward the protagonist and toward the central conflict across all their
lines. Flag a character whose stance REVERSES with no turn beat that earns it
-- an adversary who suddenly relents, an ally who suddenly turns, with nothing
shown to motivate the change. Emit a stance_issues entry
{character_id, target, prior_stance, new_stance, missing_turn_beat, line_ids,
severity} for each. This is an OBSERVATION for the record only -- it does NOT
ask for a reroll and does NOT change any verdict. A coherent script has none.

Return EXACTLY one JSON object matching this schema:
{
  "continuity_issues": [{"line_id": "...", "issue": "..."}, ...],
  "voice_drift": [{"char_id": "...", "note": "..."}, ...],
  "flat_lines": [{"line_id": "...", "reason": "..."}, ...],
  "arc_verdict": "strong" | "uneven" | "flat" | "mid_collapse",
  "reroll_targets": [{"line_id": "...", "hint": "...", "failed_dimension": "knowledge|pressure|relationship|decision|obstacle|tension|unspecified"}, ...],
  "render_priority": ["line_id", "line_id", ...],
  "stance_issues": [{"character_id": "...", "target": "...", "prior_stance": "...", "new_stance": "...", "missing_turn_beat": "...", "line_ids": ["..."], "severity": "..."}, ...]
}

Every list may be empty. No prose before or after the JSON. No markdown
fences.
"""


def _critic_character_lines(candidate_ledger: dict) -> list[dict]:
    """Character-role dialogue lines only -- the critic's judging scope.

    The critic judges DRAMA -- continuity, voice, flatness, arc -- which
    lives in character dialogue. Announcer / music / sfx beats are locked
    structural content; showing them would only tempt the critic to flag
    or reroll a line nothing downstream can act on. Mirrors the Script
    Doctor's `_doctor_character_lines` scope so both passes judge the same
    surface.
    """
    return [
        ln for ln in (candidate_ledger.get("lines", []) or [])
        if ln.get("speaker_role") == "character"
    ]


def _scope_character_lines(
    character_lines: list[dict],
    scope_line_ids,
) -> list[dict]:
    """STEP 4 (2026-06-22 story+cast fix): restrict the critic's judging
    surface to the targeted ``line_ids`` PLUS their immediate continuity
    neighbors (the prev/next CHARACTER line in canonical order).

    ``scope_line_ids is None`` -> the whole-episode pass (unchanged; the
    freeze-cascade initial critic + every legacy caller). When the reroll
    loop passes the patched target set, the critic re-scores ONLY that
    window -- the fix for the whack-a-mole re-scoring that never converged
    (each whole-episode re-run re-surfaced lines the loop never touched).
    Neighbors are included so a continuity break a reroll INTRODUCES at the
    seam is still caught and folded into the next cycle's scope.

    Missing/duplicate ids are ignored (de-duped via the index); the reroll
    loop only ever passes ids it just patched, so this is defensive.
    """
    if scope_line_ids is None:
        return character_lines
    wanted = {str(x) for x in scope_line_ids if x is not None}
    if not wanted:
        return []
    keep: set = set()
    n = len(character_lines)
    for i, ln in enumerate(character_lines):
        if str(ln.get("line_id")) in wanted:
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    keep.add(j)
    return [ln for k, ln in enumerate(character_lines) if k in keep]


def _render_critic_user_prompt(
    candidate_ledger: dict,
    cast_rows: list[dict],
    scope_line_ids=None,
) -> str:
    """Build the story critic's user prompt.

    Header is the shared Script Doctor episode context (budget, summary,
    locked cast) so the critic judges against the same episode frame the
    Doctor saw. Body is the enriched 7-field per-line rows from
    `_render_lines_for_doctor` -- the critic needs `arc_phase`,
    `beat_intent`, `mood`, and word counts to judge arc and flatness with
    real per-line context.

    ``scope_line_ids`` (STEP 4): when set, the body is restricted to the
    targeted lines + continuity neighbors (see `_scope_character_lines`);
    None judges the whole episode.
    """
    parts = _render_doctor_episode_context(candidate_ledger, cast_rows)
    character_lines = _scope_character_lines(
        _critic_character_lines(candidate_ledger), scope_line_ids,
    )
    # STEP 6 (2026-06-22 story+cast fix): surface each beat's target intensity to
    # the critic so SECTION 3 can judge a line AGAINST its target_tension (a calm
    # setup beat at target 1 is not flagged for failing to be explosive). The map
    # is stamped by the writer on meta.line_dramatic_frame[line_id].tension.
    _frames = (candidate_ledger.get("meta") or {}).get("line_dramatic_frame")
    tension_by_id = {}
    if isinstance(_frames, dict):
        for _lid, _fr in _frames.items():
            if isinstance(_fr, dict):
                _t = _fr.get("tension")
                if isinstance(_t, int) and 1 <= _t <= 5:
                    tension_by_id[str(_lid)] = _t
    parts.append(
        "EPISODE SCRIPT (CHARACTER DIALOGUE BEATS ONLY -- "
        "post-script-doctor, ready for the story-quality critic):"
    )
    parts.append(_render_lines_for_doctor(character_lines, tension_by_id))
    parts.append("")
    parts.append(
        "You are seeing ONLY character dialogue beats -- announcer, music, "
        "and SFX beats are locked structural content and are not your "
        "concern. Walk the 6-section rubric in order on the lines above, "
        "then emit the single JSON StoryCriticReport. Use only line_ids "
        "shown above; do not invent any."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# post_validator -- light hallucination guard.
# ---------------------------------------------------------------------------


def _make_critic_post_validator(candidate_ledger: dict):
    """Build the optional content-check passed to `structured_call`.

    The guard is deliberately LENIENT. It rejects a report ONLY when the
    report references a `line_id` that is not present in the ledger -- a
    small model occasionally invents an id, and a fabricated id would
    point Sprint 5C's reroll loop and Sprint 6's render ordering at
    nothing. It does NOT second-guess the critic's dramatic judgement, the
    arc verdict, or how many lines were flagged: those are exactly what
    the critic is for.

    `char_id`s in `voice_drift` are intentionally NOT checked -- the cast
    is small and stable, the consequence of a stray one is cosmetic, and
    keeping the guard to line_ids only keeps it light.

    Returns a `post_validator(report) -> str | None` closure: an error
    string rejects (and advances the ladder), `None` accepts.
    """
    valid_line_ids: set[str] = {
        str(ln.get("line_id"))
        for ln in (candidate_ledger.get("lines", []) or [])
        if ln.get("line_id")
    }

    def post_validator(report: "StoryCriticReport") -> Optional[str]:
        # An empty ledger means there is nothing to cross-check against;
        # accept whatever validated rather than reject every report.
        if not valid_line_ids:
            return None
        unknown: list[str] = []
        for issue in report.continuity_issues:
            if issue.line_id and issue.line_id not in valid_line_ids:
                unknown.append(issue.line_id)
        for flat in report.flat_lines:
            if flat.line_id and flat.line_id not in valid_line_ids:
                unknown.append(flat.line_id)
        for target in report.reroll_targets:
            if target.line_id and target.line_id not in valid_line_ids:
                unknown.append(target.line_id)
        for line_id in report.render_priority:
            if line_id and line_id not in valid_line_ids:
                unknown.append(line_id)
        if unknown:
            # A short, stable rejection message -- the dispatching repair
            # factory routes an unrecognised content failure to the
            # generic CRITICAL-prefix repair turn, which is the right
            # fallback for a hallucinated-id failure.
            preview = ", ".join(sorted(set(unknown))[:8])
            return (
                f"report references {len(set(unknown))} line_id(s) not "
                f"present in the ledger: {preview}"
            )
        return None

    return post_validator


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


def run_story_critic(
    generate_fn,
    candidate_ledger: dict,
    cast_rows: list[dict],
    scope_line_ids=None,
) -> StoryCriticReport:
    """Run the Sprint 5B whole-script story-quality critic.

    ONE LLM call, AFTER the Script Doctor cleanup. Walks the 6-section
    rubric and returns a Pydantic-validated `StoryCriticReport` whose
    output feeds Sprint 5C (the `reroll_targets` reroll loop) and Sprint 6
    (the `render_priority` / `flat_lines` / `arc_verdict` render stage).

    Parameters:
      generate_fn
        The LLM generate callable -- `generate_fn(messages, *,
        temperature, max_new_tokens) -> str`. Threaded in by the freeze
        cascade from the writer's `technical_model` broadcast socket; no
        `model_id` widget exists anywhere (Prime Directive 6).
      candidate_ledger
        `Ledger.data` -- the in-memory ledger dict, exactly the shape
        `run_script_doctor` receives. Read keys: `lines` (per-line
        records) and `meta` (episode budget / summary).
      cast_rows
        `candidate_ledger["cast"]` -- the locked cast rows.

    NEVER raises. Prime Directive 1: this pass runs in the freeze cascade
    after `review_ledger`, and a critic failure must NEVER break audio
    output. Any failure -- an exhausted `structured_call` retry ladder, a
    raising slot fn, or anything else -- is logged and
    `StoryCriticReport.clean()` is returned (all sections empty,
    `arc_verdict="strong"`): the pipeline is left exactly as it was found.
    """
    user_prompt = _render_critic_user_prompt(
        candidate_ledger, cast_rows, scope_line_ids,
    )
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # post_validator keeps the WHOLE-ledger valid id set even when scoped, so a
    # neighbor reference the critic emits is never falsely rejected; the loop
    # only acts on ids it can find anyway.
    post_validator = _make_critic_post_validator(candidate_ledger)

    # The single critic call routes through the shared structured_call
    # 3-attempt ladder (base -> structural retry -> typed repair). An
    # exhausted ladder raises StructuredCallFailedError; the slot fn (the
    # LLM call) may raise arbitrary loader exceptions structured_call does
    # not catch. BOTH are caught below -- every failure path returns
    # StoryCriticReport.clean(), keeping the never-raises contract
    # absolute (Prime Directive 1).
    #
    # LLM slot: technical -- the critic is a structured rubric evaluator.
    # It emits typed verdicts and ordered id lists against a fixed JSON
    # schema, not creative prose, so it routes to the technical model
    # exactly as the Script Doctor diagnosis / edits passes do. The
    # technical model id is the `generate_fn` slot threaded in by the
    # freeze cascade from the writer's technical_model broadcast socket --
    # no new widget, no new model_id parameter (Prime Directive 6).
    try:
        report = structured_call(
            prompt=messages,
            schema=StoryCriticReport,
            slot_fn=generate_fn,
            base_temperature=_CRITIC_TEMPERATURE,
            structural_retry_temperature=_CRITIC_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=post_validator,
            max_new_tokens=_CRITIC_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_story_critic",
        )
    except StructuredCallFailedError as exc:
        log.warning(
            "[OTR_StoryCritic] structured_call exhausted the retry ladder "
            "after %d attempt(s) (last error: %s); returning clean() "
            "fallback report",
            exc.attempts, exc.last_error,
        )
        return StoryCriticReport.clean()
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        log.warning(
            "[OTR_StoryCritic] structured_call raised %s: %s; returning "
            "clean() fallback report",
            type(exc).__name__, exc,
        )
        return StoryCriticReport.clean()

    log.info(
        "[OTR_StoryCritic] critic complete: arc_verdict=%s, %d continuity "
        "issue(s), %d voice-drift note(s), %d flat line(s), %d reroll "
        "target(s), %d line(s) in render priority",
        report.arc_verdict,
        len(report.continuity_issues),
        len(report.voice_drift),
        len(report.flat_lines),
        len(report.reroll_targets),
        len(report.render_priority),
    )
    return report


# ---------------------------------------------------------------------------
# Optional smoke block -- runs with a canned generate_fn, no GPU.
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    import json as _json

    logging.basicConfig(level=logging.INFO)

    _ledger = {
        "meta": {"target_words": 600, "style": "noir mystery"},
        "cast": [
            {"char_id": "c01", "name": "EDNA", "speaker_role": "character"},
            {"char_id": "c02", "name": "MARLOWE", "speaker_role": "character"},
        ],
        "lines": [
            {
                "line_id": "l001", "beat_id": "l001", "char_id": "c01",
                "speaker_role": "character", "arc_phase": "setup",
                "text": "The safe was open when I got here.",
            },
            {
                "line_id": "l002", "beat_id": "l002", "char_id": "c02",
                "speaker_role": "character", "arc_phase": "climax",
                "text": "Then you saw who took the file.",
            },
        ],
    }

    def _smoke_generate_fn(messages, *, temperature, max_new_tokens):
        return _json.dumps({
            "continuity_issues": [],
            "voice_drift": [],
            "flat_lines": [
                {"line_id": "l001", "reason": "states a fact, plays no moment"}
            ],
            "arc_verdict": "uneven",
            "reroll_targets": [
                {"line_id": "l001",
                 "hint": "open on Edna's fear, not the inventory of the safe"}
            ],
            "render_priority": ["l002", "l001"],
        })

    _report = run_story_critic(
        _smoke_generate_fn, _ledger, _ledger["cast"],
    )
    print(_report.model_dump_json(indent=2))

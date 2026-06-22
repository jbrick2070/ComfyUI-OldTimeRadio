"""nodes/_otr_reroll.py -- Sprint 5C targeted reroll loop.

Sprint 5B added the whole-script story critic (`_otr_story_critic.py`):
ONE LLM pass that judges DRAMA and emits a `StoryCriticReport`. For 5B
that report was purely ADVISORY -- it named lines worth re-composing in
`reroll_targets` but changed no text.

This module is Sprint 5C: it ACTS on `reroll_targets`. It re-composes
each flagged line with the critic's concrete `hint` threaded in as a hard
REVISE instruction, then re-SCORES the critic SCOPED to the patched lines
+ their continuity neighbors (STEP 4, 2026-06-22). The loop converges when
the targeted ids clear; a newly-failed neighbor joins the next cycle's
scope; it HALTS (to repair-then-ship) only on the MAX_REROLL_CYCLES cap OR
a rise in the outstanding flag count (divergence). On a halt the re-composed
lines are KEPT (repair-then-ship, never a pre-reroll restore) and the
episode is stamped `needs_full_rerun`, which the freeze cascade's A2 path
ships through the normal freeze (Phase 10's gap audit is the structural
backstop).

ARCHITECTURE -- Sprint 5C fork, Option A (technical slot, in-cascade).
The reroll runs INSIDE `OTR_LedgerFreezeCascade`, which keeps only the
TECHNICAL model resident. Re-composition therefore runs on that resident
technical model -- exactly as the Script Doctor's existing post-cleanup
edit pass does -- with NO creative-slot VRAM swap and NO node-surface /
workflow-JSON change. This deliberately deviates from the v4 plan's
literal "compose_line_draft / creative slot" wording; the alternative
(Option B) added a technical->creative->technical VRAM swap inside the
cascade for marginal gain. The deviation is recorded in the Build
Progress Log.

PRIME DIRECTIVES.
  * PD1 (audio is king): `run_targeted_reroll` NEVER raises. A failed
    re-composition leaves the original line untouched; an unexpected
    error degrades to "freeze what we have" -- the reroll is an
    improvement pass, never a gate that can abort a run.
  * PD2 (VRAM): no model swap -- the reroll reuses the resident
    technical model. No `force_vram_offload`.
  * PD6 (LLM slot tagging): the two LLM calls -- `compose_line` and
    `run_story_critic` -- both run on the technical slot. Tagged
    `# LLM slot: technical` at each call site. No node exposes a
    `model_id` widget; the model id is the `generate_fn` callable
    threaded in by the freeze cascade.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("OTR.reroll")


__all__ = [
    "MAX_REROLL_CYCLES",
    "RerollDisposition",
    "build_reroll_line_request",
    "run_targeted_reroll",
]


# ---------------------------------------------------------------------------
# Shared imports. Package import in production; flat import when loaded
# standalone / under test. Mirrors the import guard in
# `_otr_story_critic.py`.
# ---------------------------------------------------------------------------

try:
    from ._otr_line_composer import (
        LineRequest,
        LineCompositionFailedError,
        build_voice_card,
        compose_line,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_line_composer import (  # type: ignore
        LineRequest,
        LineCompositionFailedError,
        build_voice_card,
        compose_line,
    )

try:
    from ._otr_story_critic import StoryCriticReport, run_story_critic
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_story_critic import (  # type: ignore
        StoryCriticReport,
        run_story_critic,
    )

try:
    from ._otr_continuity import ContinuityState, render_continuity_slice
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_continuity import (  # type: ignore
        ContinuityState,
        render_continuity_slice,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v4 plan, Sprint 5C: "Cap at 2 critic->reroll cycles. Cycle 3 ->
# needs_full_rerun." One cycle = one whole reroll_targets batch plus one
# critic re-run on the mutated script.
MAX_REROLL_CYCLES = 2

# How many preceding voiced lines to rebuild into the LineRequest's
# rolling `last_lines` window for a reroll. The first-pass composer's
# window is scene-local; the reroll reconstruction walks back the same
# way (stopping at a non-voiced beat) and caps at this many.
_REROLL_LAST_LINES_WINDOW = 3

# Line roles that carry spoken audio. A non-voiced row (music / sfx
# marker) is a scene boundary for the `last_lines` reconstruction.
_VOICED_ROLES = frozenset({"character", "announcer"})

# Fallback target_words when a line row carries neither target_words nor
# a usable word_count -- the composer's size band needs a positive int.
_FALLBACK_TARGET_WORDS = 30


# ---------------------------------------------------------------------------
# Disposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerollDisposition:
    """End-of-reroll summary. Stamped (as a dict) to meta.reroll_disposition.

    verdict:
      "no_reroll"        the incoming critic report named no
                         reroll_targets -- the loop never ran.
      "reroll_resolved"  at least one cycle ran and the final critic
                         re-run named no further reroll_targets.
      "needs_full_rerun" the cap (MAX_REROLL_CYCLES) was reached or the
                         loop diverged and the critic STILL wants rerolls.
                         STEP 4 repair-then-ship: the re-composed lines are
                         KEPT (no pre-reroll restore) and the freeze
                         cascade's A2 path ships the candidate through the
                         normal freeze; Phase 10's gap audit is the
                         structural backstop.
      "reroll_error"     an unexpected error degraded the pass -- the
                         cascade proceeds to freeze with whatever lines
                         exist (PD1: never abort a run).
    """

    verdict: str
    cycles_run: int
    lines_rerolled: int
    final_report: StoryCriticReport

    def to_dict(self) -> dict:
        """JSON-friendly view for stamping on meta.reroll_disposition."""
        return {
            "verdict": self.verdict,
            "cycles_run": self.cycles_run,
            "lines_rerolled": self.lines_rerolled,
            "final_arc_verdict": self.final_report.arc_verdict,
            "final_reroll_targets": len(self.final_report.reroll_targets),
        }


# ---------------------------------------------------------------------------
# Ledger / cast helpers
# ---------------------------------------------------------------------------


def _cast_name_by_id(cast_rows: list) -> dict:
    """Map char_id -> canonical name from the locked cast rows."""
    out: dict = {}
    for row in cast_rows or []:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("char_id") or "").strip()
        name = str(row.get("name") or "").strip()
        if cid and name:
            out[cid] = name
    return out


def _cast_row_by_id(cast_rows: list) -> dict:
    """Map char_id -> the full cast row dict."""
    out: dict = {}
    for row in cast_rows or []:
        if isinstance(row, dict):
            cid = str(row.get("char_id") or "").strip()
            if cid:
                out[cid] = row
    return out


def _line_index(lines: list, line_id) -> int:
    """Return the 0-based index of the row matching `line_id` (by line_id
    or beat_id), or -1 when no row matches.
    """
    target = str(line_id)
    for i, row in enumerate(lines or []):
        if not isinstance(row, dict):
            continue
        if (str(row.get("line_id")) == target
                or str(row.get("beat_id")) == target):
            return i
    return -1


def _find_line_row(ledger_data: dict, line_id):
    """Return the ledger line row matching `line_id`, or None."""
    lines = ledger_data.get("lines", []) or []
    idx = _line_index(lines, line_id)
    return lines[idx] if idx >= 0 else None


def _reconstruct_last_lines(
    ledger_data: dict,
    line_index: int,
    name_by_id: dict,
) -> list:
    """Rebuild the composer's scene-local `last_lines` window for a reroll.

    Walks backward from the rerolled line. Each preceding VOICED row with
    non-empty text is collected as a (speaker_name, text) pair; the walk
    stops at the first non-voiced row (music / sfx marker -- a scene
    boundary) or once _REROLL_LAST_LINES_WINDOW pairs are gathered. The
    result is returned chronological (oldest first), the order
    `_format_last_lines` expects.
    """
    lines = ledger_data.get("lines", []) or []
    window: list = []
    i = line_index - 1
    while i >= 0 and len(window) < _REROLL_LAST_LINES_WINDOW:
        row = lines[i]
        if not isinstance(row, dict):
            break
        role = str(row.get("speaker_role") or "").strip()
        if role not in _VOICED_ROLES:
            break  # scene boundary -- stop the scene-local window here
        text = str(row.get("text") or "").strip()
        if text:
            cid = str(row.get("char_id") or "").strip()
            if role == "announcer":
                speaker = name_by_id.get(cid, "ANNOUNCER")
            else:
                speaker = name_by_id.get(cid, cid or "")
            window.append((speaker, text))
        i -= 1
    window.reverse()
    return window


# ---------------------------------------------------------------------------
# LineRequest reconstruction
# ---------------------------------------------------------------------------


def build_reroll_line_request(
    ledger_data: dict,
    line_row: dict,
    cast_rows: list,
) -> LineRequest:
    """Reconstruct a `LineRequest` for re-composing one critic-flagged line.

    Pulls the composer's static-prefix context from `meta` (the writer
    stamps `canon_header` / `outline_spine` / `theme` / `style_descriptor`
    / `allowed_roster` there in Sprint 5C) and the per-beat context from
    the ledger line row itself (`beat_intent` / `traits` (mood) /
    `target_words` / `arc_phase` -- the Sprint 3C row enrichment). The
    rolling `last_lines` window is rebuilt scene-locally from the
    preceding ledger rows.

    `continuity_slice` is rebuilt from ``meta["continuity"]`` via
    ``_rebuild_continuity_slice`` so the rerolled line gets the same
    per-speaker hard constraints the first-pass compose received (Sprint
    5C v2 follow-up, 2026-05-25). Beat coordinate: the 0-based position
    of the line row in ``ledger_data["lines"]`` -- ``init_lines_from_outline``
    stamps one ledger row per outline beat in beat order and rows are
    only mutated in place after that, so the row position matches the
    coordinate ``build_continuity_ledger`` used for ``established_beat``.
    A missing or malformed ``meta["continuity"]`` renders empty (Prime
    Directive 1).

    The per-line DRAMATIC FRAME (STEP 6, 2026-06-22) is reconstructed from
    ``meta["line_dramatic_frame"][line_id]`` (beat_objective / beat_obstacle /
    beat_turn / beat_subtext / beat_tension / dramatic_question / next_turn) so
    the rerolled line composes against the SAME frame + tension target the
    first-pass line had -- before this it rebuilt only arc_phase and lost the
    frame. A missing frame -> empty defaults (PD1).

    `position`, `sfx_cue`, `current_beat_block` remain at their empty
    defaults: those blocks would only be approximate reconstructions and
    the critic's `hint` (threaded separately as `reroll_hint`) already
    carries the actionable signal for a reroll. The phantom-name gate
    still runs on the reroll output inside `compose_line` via
    `allowed_roster`, so cast safety is unaffected.
    """
    meta = ledger_data.get("meta", {}) or {}
    cast_rows = cast_rows or []
    name_by_id = _cast_name_by_id(cast_rows)
    row_by_id = _cast_row_by_id(cast_rows)

    cid = str(line_row.get("char_id") or "").strip()
    speaker = name_by_id.get(cid, cid or "SPEAKER")
    cast_row = row_by_id.get(cid)
    voice_card = build_voice_card(cast_row) if cast_row else ""

    intent = str(line_row.get("beat_intent") or "").strip()
    mood = str(line_row.get("traits") or "").strip()
    arc_phase = str(line_row.get("arc_phase") or "").strip()

    # target_words: prefer the enriched per-beat allocation, fall back to
    # the line's current word_count, then a constant -- the composer's
    # size band needs a positive integer.
    target_words = _coerce_positive_int(line_row.get("target_words"))
    if target_words <= 0:
        target_words = _coerce_positive_int(line_row.get("word_count"))
    if target_words <= 0:
        target_words = _FALLBACK_TARGET_WORDS

    allowed_roster = frozenset(
        str(n) for n in (meta.get("allowed_roster") or []) if str(n).strip()
    )
    allowed_people = frozenset(
        name for name in name_by_id.values() if name
    )
    all_voice_cards = "\n".join(
        card for card in (build_voice_card(r) for r in cast_rows) if card
    )

    lines = ledger_data.get("lines", []) or []
    line_index = _line_index(
        lines, line_row.get("line_id") or line_row.get("beat_id"),
    )
    last_lines = (
        _reconstruct_last_lines(ledger_data, line_index, name_by_id)
        if line_index >= 0 else []
    )
    prev_speaker = last_lines[-1][0] if last_lines else ""

    continuity_slice = _rebuild_continuity_slice(
        meta, speaker, line_index if line_index >= 0 else 0,
    )

    # STEP 6 (2026-06-22 story+cast fix): reconstruct the per-line DRAMATIC FRAME
    # the first-pass compose received. The writer stamps it on
    # meta["line_dramatic_frame"][line_id] (objective/obstacle/turn/subtext/
    # tension/dramatic_question/next_turn); before this, build_reroll_line_request
    # rebuilt only arc_phase, so a rerolled line lost the entire dramatic frame
    # (and its tension target) the original line had -- the reroll composed blind
    # to the very signal it was meant to repair against. A missing frame -> empty
    # defaults (PD1: never raise).
    _line_key = str(line_row.get("line_id") or line_row.get("beat_id") or "")
    _frames = meta.get("line_dramatic_frame")
    _frame = _frames.get(_line_key) if isinstance(_frames, dict) else None
    _frame = _frame if isinstance(_frame, dict) else {}

    def _coerce_tension(value) -> int:
        try:
            t = int(value)
        except (TypeError, ValueError):
            return 0
        return t if 1 <= t <= 5 else 0

    return LineRequest(
        speaker=speaker,
        intent=intent,
        mood=mood,
        target_words=target_words,
        canon_header=str(meta.get("canon_header") or ""),
        last_lines=last_lines,
        allowed_roster=allowed_roster,
        style_descriptor=str(meta.get("style_descriptor") or ""),
        outline_spine=str(meta.get("outline_spine") or ""),
        character_voice_card=voice_card,
        arc_phase=arc_phase,
        allowed_people=allowed_people,
        prev_speaker=prev_speaker,
        theme=str(meta.get("theme") or ""),
        all_voice_cards=all_voice_cards,
        speaker_role="character",
        continuity_slice=continuity_slice,
        # STEP 6 -- the reconstructed dramatic frame.
        dramatic_question=str(_frame.get("dramatic_question") or ""),
        next_turn=str(_frame.get("next_turn") or ""),
        beat_objective=str(_frame.get("objective") or ""),
        beat_obstacle=str(_frame.get("obstacle") or ""),
        beat_turn=str(_frame.get("turn") or ""),
        beat_subtext=str(_frame.get("subtext") or ""),
        beat_tension=_coerce_tension(_frame.get("tension")),
    )


def _coerce_positive_int(value) -> int:
    """Return int(value) when it parses to a positive int, else 0."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return 0
    return out if out > 0 else 0


def _rebuild_continuity_slice(
    meta: dict,
    speaker: str,
    beat_index: int,
) -> str:
    """Reconstruct the per-speaker continuity slice for a reroll.

    Mirrors the writer's first-pass per-beat call (see
    ``OTR_LedgerScriptWriter._build_line_request_for_beat``): rebuilds the
    episode `ContinuityState` from ``meta["continuity"]`` (the writer
    stamps ``continuity_state.model_dump()`` at Sprint 5A) and projects it
    through ``render_continuity_slice`` for the given speaker at the given
    beat_index.

    PRIME DIRECTIVE 1 -- NEVER raises. Returns ``""`` on every failure
    path (no ``meta["continuity"]`` stamped, schema validation rejects the
    dump, render itself throws). An empty string means "no continuity
    block" -- ``_build_user_prompt`` drops the section entirely, so a
    rerolled line still composes; it just loses the per-speaker hard
    constraints the first pass enjoyed. The phantom-name gate inside
    ``compose_line`` still runs via ``allowed_roster`` regardless, so
    cast safety is unaffected.

    `beat_index` is the 0-based position of the line row in
    ``ledger_data["lines"]`` -- the same coordinate the first-pass
    continuity render used, because ``init_lines_from_outline`` stamps
    one ledger row per outline beat in beat order and the writer only
    mutates row text in place via ``update_line_text``; no row is
    inserted, deleted, or reordered after stamping. A negative index
    (line not found) maps to 0, matching the writer's fallback
    (``beat_index_by_id.get(beat.beat_id, 0)``).
    """
    if beat_index < 0:
        beat_index = 0
    raw = (meta or {}).get("continuity")
    if not isinstance(raw, dict) or not raw:
        return ""
    try:
        state = ContinuityState.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - never block the reroll
        log.warning(
            "[OTR_Reroll] meta.continuity did not validate (%s); rerolled "
            "line composes without a continuity slice",
            exc,
        )
        return ""
    try:
        return render_continuity_slice(state, speaker, beat_index) or ""
    except Exception as exc:  # noqa: BLE001 - render is pure but defensive
        log.warning(
            "[OTR_Reroll] render_continuity_slice raised for speaker=%r "
            "beat_index=%d (%s); slice degraded to empty",
            speaker, beat_index, exc,
        )
        return ""


def _coerce_report(raw) -> StoryCriticReport:
    """Rebuild a StoryCriticReport from the meta.story_critic_report dict.

    Returns `StoryCriticReport.clean()` on anything unparseable -- a
    reroll loop with no usable report simply does nothing.
    """
    if isinstance(raw, StoryCriticReport):
        return raw
    if not isinstance(raw, dict):
        return StoryCriticReport.clean()
    try:
        return StoryCriticReport.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Reroll] meta.story_critic_report did not validate "
            "(%s); treating as an empty report", exc,
        )
        return StoryCriticReport.clean()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_targeted_reroll(generate_fn, led) -> RerollDisposition:
    """Run the Sprint 5C targeted reroll loop on a post-critic ledger.

    Reads `meta.story_critic_report` (stamped by the Sprint 5B critic in
    the freeze cascade). For each `RerollTarget`, re-composes the flagged
    line with the critic's `hint` as a hard REVISE instruction, writes the
    new text back in place via `Ledger.update_line_text`, then re-SCORES the
    critic SCOPED to the patched lines + neighbors (STEP 4). Capped at
    MAX_REROLL_CYCLES cycles; the loop halts on the cap OR a rise in the
    outstanding flag count (divergence). On a halt the re-composed lines are
    KEPT (STEP 4 repair-then-ship -- NO pre-reroll restore) and the verdict is
    `needs_full_rerun`, which the freeze cascade's A2 path ships through the
    normal freeze (Phase 10's gap audit is the structural backstop).

    Parameters:
      generate_fn
        The technical-slot LLM callable resident in the freeze cascade.
        Used for BOTH the re-composition (`compose_line`) and the critic
        re-run (`run_story_critic`) -- Sprint 5C fork Option A.
      led
        A `production_ledger.Ledger`-like object exposing `.data` and
        `.update_line_text(beat_id, text)`.

    Stamps on meta:
      cycle_count          number of critic->reroll cycles actually run
      reroll_history       per-attempt audit list
      reroll_disposition   RerollDisposition.to_dict()
      reroll_verdict       the disposition verdict string
      story_critic_report  refreshed with the FINAL critic re-run

    NEVER raises (Prime Directive 1). Returns a RerollDisposition.
    """
    try:
        return _run_targeted_reroll_inner(generate_fn, led)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Reroll] unexpected error in the reroll loop (%s: %s); "
            "degrading to freeze-what-we-have (Prime Directive 1)",
            type(exc).__name__, exc,
        )
        report = StoryCriticReport.clean()
        disp = RerollDisposition("reroll_error", 0, 0, report)
        try:
            meta = led.data.setdefault("meta", {})
            report = _coerce_report(meta.get("story_critic_report"))
            disp = RerollDisposition("reroll_error", 0, 0, report)
            meta["reroll_verdict"] = disp.verdict
            meta["reroll_disposition"] = disp.to_dict()
        except Exception:  # noqa: BLE001 - never let cleanup raise
            pass
        return disp


def _run_targeted_reroll_inner(generate_fn, led) -> RerollDisposition:
    """The reroll loop body. `run_targeted_reroll` wraps this so any
    unexpected failure degrades instead of raising (Prime Directive 1).
    """
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    cast_rows = ledger_data.get("cast", []) or []

    report = _coerce_report(meta.get("story_critic_report"))

    if not report.reroll_targets:
        meta["cycle_count"] = 0
        disp = RerollDisposition("no_reroll", 0, 0, report)
        meta["reroll_verdict"] = disp.verdict
        meta["reroll_disposition"] = disp.to_dict()
        log.info(
            "[OTR_Reroll] critic named no reroll targets -- nothing to do"
        )
        return disp

    # STEP 4 (2026-06-22 story+cast fix): SCOPED convergence loop. The initial
    # `report` is the freeze-cascade WHOLE-EPISODE pass. Each cycle rerolls ONLY
    # the lines in `scope`, then re-scores the critic SCOPED to scope+neighbors
    # (run_story_critic(scope_line_ids=scope)). This replaces the whack-a-mole
    # whole-episode re-run that re-surfaced untouched lines and never converged.
    #
    # CONVERGENCE INVARIANT: the targeted ids must CLEAR; any newly-failed
    # continuity neighbor JOINS the next cycle's scope. HALT (to repair-then-
    # ship) only when cycle_count reaches MAX_REROLL_CYCLES OR the outstanding
    # flag count INCREASES vs the previous cycle (divergence) -- NOT on a mere
    # failure-to-strictly-decrease (fixing N that surfaces N is a plateau, not a
    # regression, and must keep iterating within the cap).
    #
    # Repair-then-ship: we KEEP every re-composed line (no pre-reroll restore).
    # The freeze cascade's A2 path ships a residual needs_full_rerun through the
    # normal freeze, and Phase 10's gap audit is the structural backstop.
    history: list = []
    cycles_run = 0
    total_rerolled = 0

    def _scope_and_hints(rep):
        ids: set = set()
        hints: dict = {}
        for t in rep.reroll_targets:
            lid = str(t.line_id or "").strip()
            if lid:
                ids.add(lid)
                hint = str(t.hint or "").strip()
                # STEP 5 (2026-06-22): fold the named flatness dimension in as a
                # lightweight PREFIX so the composer's REVISE carries it -- NOT a
                # deterministic re-craft (the critic's hint is already
                # actionable). "unspecified" / missing -> no prefix (back-compat).
                dim = str(getattr(t, "failed_dimension", "") or "").strip()
                if dim and dim != "unspecified":
                    hint = (
                        f"[{dim}] {hint}" if hint
                        else f"make this beat earn its {dim}"
                    )
                hints[lid] = hint
        return ids, hints

    scope, hint_by_id = _scope_and_hints(report)
    prev_outstanding = len(scope)
    diverged = False

    while scope and cycles_run < MAX_REROLL_CYCLES:
        cycles_run += 1
        cycle_rerolled = 0
        log.info(
            "[OTR_Reroll] cycle %d/%d: %d target(s) in scope",
            cycles_run, MAX_REROLL_CYCLES, len(scope),
        )
        for line_id in sorted(scope):
            hint = hint_by_id.get(line_id, "")
            row = _find_line_row(ledger_data, line_id)
            if row is None:
                history.append({
                    "cycle": cycles_run, "line_id": line_id,
                    "hint": hint, "status": "line_not_found",
                })
                log.warning(
                    "[OTR_Reroll] reroll target %r not in the ledger -- "
                    "skipped", line_id,
                )
                continue
            if str(row.get("speaker_role") or "") != "character":
                history.append({
                    "cycle": cycles_run, "line_id": line_id,
                    "hint": hint, "status": "not_a_character_line",
                })
                continue
            old_text = str(row.get("text") or "")
            try:
                req = build_reroll_line_request(
                    ledger_data, row, cast_rows,
                )
                # LLM slot: technical -- Sprint 5C fork Option A. The
                # reroll re-composes on the technical model resident in
                # the freeze cascade (the same model the Script Doctor
                # uses for its post-cleanup edit pass); no creative-slot
                # VRAM swap. polish is OFF -- a reroll is itself a
                # rewrite -- but the deterministic strip pipeline inside
                # compose_line (cast_strip / phantom gate / vocative
                # strip) still runs, so the rerolled line gets the same
                # cast-safety guards a first-pass line does.
                result = compose_line(
                    creative_fn=generate_fn,
                    req=req,
                    reroll_hint=hint,
                )
            except LineCompositionFailedError as exc:
                history.append({
                    "cycle": cycles_run, "line_id": line_id,
                    "hint": hint, "status": "compose_failed",
                    "detail": str(exc),
                })
                log.warning(
                    "[OTR_Reroll] re-composition exhausted its attempts "
                    "for %r -- keeping the original line", line_id,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - slot fn varies
                history.append({
                    "cycle": cycles_run, "line_id": line_id,
                    "hint": hint, "status": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                log.warning(
                    "[OTR_Reroll] unexpected error re-composing %r (%s) "
                    "-- keeping the original line", line_id, exc,
                )
                continue
            new_text = (result.text or "").strip()
            if not new_text:
                history.append({
                    "cycle": cycles_run, "line_id": line_id,
                    "hint": hint, "status": "empty_result",
                })
                continue
            led.update_line_text(line_id, new_text)
            total_rerolled += 1
            cycle_rerolled += 1
            history.append({
                "cycle": cycles_run, "line_id": line_id, "hint": hint,
                "old_text": old_text, "new_text": new_text,
                "status": "rerolled",
                "compose_flags": list(result.compose_flags),
            })
        # LLM slot: technical -- SCOPED critic re-run (STEP 4). Only the
        # patched lines + their continuity neighbors are re-scored, so the
        # loop converges on the lines it actually touched instead of chasing
        # untouched lines a whole-episode re-run keeps re-surfacing.
        # run_story_critic NEVER raises (clean() on any failure).
        log.info(
            "[OTR_Reroll] cycle %d re-composed %d line(s); re-scoring the "
            "critic on %d scoped line(s)+neighbors",
            cycles_run, cycle_rerolled, len(scope),
        )
        report = run_story_critic(
            generate_fn, ledger_data, cast_rows, scope_line_ids=scope,
        )
        flagged, hint_by_id = _scope_and_hints(report)
        next_scope = flagged  # targeted-still-failing + newly-flagged neighbors
        if not next_scope:
            scope = set()  # CONVERGED: every targeted id cleared, no new neighbor
            break
        if len(next_scope) > prev_outstanding:
            # Outstanding flag count INCREASED -- divergence. Stop and ship
            # what we have rather than loop toward a worse script.
            diverged = True
            scope = next_scope
            break
        prev_outstanding = len(next_scope)
        scope = next_scope

    meta["cycle_count"] = cycles_run
    meta["reroll_history"] = history

    # Refresh the FINAL critic report with a WHOLE-EPISODE pass so the Sprint 6
    # render-coupling stage (render_priority / flat_lines / arc_verdict) sees
    # the true end state -- the loop's last report is only the scoped window.
    final_report = run_story_critic(generate_fn, ledger_data, cast_rows)
    meta["story_critic_report"] = final_report.model_dump()
    meta["reroll_outstanding"] = sorted(scope)
    meta["reroll_diverged"] = bool(diverged)

    if final_report.reroll_targets:
        # Repair-then-ship: the bounded loop hit the cap or diverged and the
        # whole-episode critic still names targets. We KEEP the re-composed
        # lines (an improvement pass, never a regression to the pre-reroll
        # text) and stamp needs_full_rerun so the freeze cascade's A2 path
        # ships the candidate through the normal freeze (Phase 10's gap audit
        # remains the structural backstop). No pre-reroll restore.
        disp = RerollDisposition(
            "needs_full_rerun", cycles_run, total_rerolled, final_report,
        )
        log.warning(
            "[OTR_Reroll] critic still names %d target(s) after %d cycle(s) "
            "(diverged=%s) -- repair-then-ship: KEEPING the %d re-composed "
            "line(s), stamping needs_full_rerun for the cascade A2 "
            "ship-through", len(final_report.reroll_targets), cycles_run,
            diverged, total_rerolled,
        )
    else:
        disp = RerollDisposition(
            "reroll_resolved", cycles_run, total_rerolled, final_report,
        )
        log.info(
            "[OTR_Reroll] reroll resolved after %d cycle(s): %d line(s) "
            "re-composed, critic now clean", cycles_run, total_rerolled,
        )

    meta["reroll_verdict"] = disp.verdict
    meta["reroll_disposition"] = disp.to_dict()
    return disp

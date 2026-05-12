"""nodes/_otr_lfc_phase_6_episode_arc.py — LFC Phase 6 (ADR 6.13 / 6.14).

Episode-arc audit. Reads cached meta.scene_synopses from Phase 4
(commit 8) and runs ONE two-step LLM call that emits Editor notes
(ADR section 6.14). The formatter translates Editor notes into a
list of edits + an overall arc rationale.

Pipeline:
  1. Read meta.scene_synopses. When Phase 4 didn't fire / failed,
     fall back to `build_micro_recap(scope="episode")` -- ADR
     section 6.13 outline-intent fallback.
  2. Build the Macro-Bible (commit 7).
  3. two_step_generate (commit 6):
       reasoning  Macro-Bible + scene synopses + voiced beats
                  list. EDITOR role-induction. Free Markdown
                  Editor notes scoped per-scene or episode-wide.
       formatter  schema-extraction prompt -> EditorNotesOutput.
                  Each note has a scope + note text + nested
                  suggested_edits.
  4. Apply suggested_edits with the existing reviewer's edit cap
     `min(8, max(3, voiced_beats // 3))`. Track remaining budget
     on meta.doctor_edit_budget_remaining so cumulative cascade
     doctor edits don't thrash the original script.

Mutation policy + guards: same shape as Phase 4 / 5 -- length
drift >50% rejected, wrong scope (non-character / non-announcer)
rejected, unknown line_id rejected.

Failure cascade per ADR section 8: skip-and-continue. On
two_step None, Phase 6 records the failure on its report and
advances; cascade reaches Phase 10 normally.

Status: LFC sprint commit 10 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("OTR.lfc_phase_6")


__all__ = [
    "Phase6EpisodeArcReport",
    "phase_6_episode_arc",
    "EPISODE_ARC_REASONING_SYSTEM_PROMPT",
    "EPISODE_ARC_FORMATTER_TEMPLATE",
    "EDIT_LENGTH_DRIFT_REJECT_THRESHOLD",
    "compute_arc_edit_cap",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EDIT_LENGTH_DRIFT_REJECT_THRESHOLD: float = 0.50


# ADR section 6.14: Editor-note scaffold.
EPISODE_ARC_REASONING_SYSTEM_PROMPT = """\
You are the EDITOR auditing the OVERALL ARC of an audio drama
episode. The cast is locked. Cast contract is clean. Per-scene
coherence has run. Your job is the episode-level arc:

  * Does the episode have a beginning, middle, and end?
  * Does each scene SERVE the arc, or does some scene drift?
  * Is the tension curve coherent? Or does Scene 2 spike too high
    and Scene 4 fizzle?
  * Are the character motivations consistent end-to-end?

You will receive:
  MACRO-BIBLE     episode logline + cast voice cards + arc
                  skeleton.
  SCENE SYNOPSES  one paragraph per scene (cached from Phase 4).
                  When Phase 4 didn't fire, you'll see an OUTLINE
                  INTENTS list instead -- treat it as the
                  skeletal version of the same content.
  VOICED BEATS    every character / announcer line_id + speaker
                  + text. Reference these when proposing edits.

OUTPUT: free-form prose Editor notes. Each note has a SCOPE
("scene_<n>" or "speaker_<char_id>" or "episode") and a brief
explanation. Optionally each note carries 0-3 SUGGESTED EDITS
referencing specific line_ids.

Editor note format (write each note as a paragraph, this prose
will be JSON-extracted next stage):

  SCOPE: scene_2
  NOTE: The tension in Scene 2 collapses after line l007. The
  characters reach the discovery too early; the reveal lacks
  the slow-burn that scene_1 set up.
  SUGGESTED EDITS:
    line_id=l008 revise: "..."  reason: "..."

Be tight. Six notes maximum. The next stage translates this
prose into structured JSON. Do NOT output JSON here.
"""


EPISODE_ARC_FORMATTER_TEMPLATE = """\
You are a JSON formatter. Translate the editor's prose arc audit
into the following schema EXACTLY. Output ONLY the JSON object
(no markdown fences, no prose before or after).

Schema:
{
  "notes": [
    {
      "scope": "<scene_N | speaker_cXX | episode>",
      "note": "<editor's note text>",
      "suggested_edits": [
        {
          "line_id": "<line_id>",
          "original": "<original text or empty>",
          "revised": "<editor's revised text>",
          "reason": "<one-sentence rationale>"
        }
      ]
    }
  ]
}

Rules:
  * One JSON note per editor note in the prose.
  * suggested_edits may be empty for notes that only diagnose
    without proposing changes.
  * Preserve every line_id exactly.

Editor's review:
{{REASONING}}
"""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Phase6EpisodeArcReport:
    voiced_beats: int = 0
    edit_cap: int = 0
    cap_source: str = ""    # "min(8, max(3, voiced_beats // 3))"
    notes_proposed: int = 0
    edits_proposed: int = 0
    edits_applied: int = 0
    edits_rejected_unknown_line: int = 0
    edits_rejected_wrong_role: int = 0
    edits_rejected_length_drift: int = 0
    edits_rejected_cap_exceeded: int = 0
    rationale: str = ""
    skipped: bool = False
    skipped_reason: str = ""
    used_synopses_fallback: bool = False
    failed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "voiced_beats": int(self.voiced_beats),
            "edit_cap": int(self.edit_cap),
            "cap_source": self.cap_source,
            "notes_proposed": int(self.notes_proposed),
            "edits_proposed": int(self.edits_proposed),
            "edits_applied": int(self.edits_applied),
            "edits_rejected_unknown_line": int(
                self.edits_rejected_unknown_line
            ),
            "edits_rejected_wrong_role": int(
                self.edits_rejected_wrong_role
            ),
            "edits_rejected_length_drift": int(
                self.edits_rejected_length_drift
            ),
            "edits_rejected_cap_exceeded": int(
                self.edits_rejected_cap_exceeded
            ),
            "rationale": self.rationale,
            "skipped": bool(self.skipped),
            "skipped_reason": self.skipped_reason,
            "used_synopses_fallback": bool(self.used_synopses_fallback),
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Edit budget
# ---------------------------------------------------------------------------


def compute_arc_edit_cap(voiced_beats: int) -> int:
    """min(8, max(3, voiced_beats // 3)) -- matches reviewer cap shape.

    ADR section 4 + go-forward §6 plan: Phase 6 draws from the
    SAME pool as the reviewer, so cumulative doctor edits across
    the cascade don't thrash the script. The remaining budget
    stamps on meta.doctor_edit_budget_remaining after Phase 6.
    """
    return min(8, max(3, voiced_beats // 3))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _voiced_beats_block(ledger_data: dict) -> tuple[str, int]:
    """Render every voiced beat as a single-line entry. Returns the
    rendered string + the count."""
    lines = ledger_data.get("lines") or []
    out: list[str] = []
    count = 0
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        count += 1
        line_id = ln.get("line_id", "")
        cid = ln.get("char_id", "")
        text = (ln.get("text") or "").strip()
        out.append(f"  {line_id} | {cid}: {text}")
    return "\n".join(out), count


def _build_arc_reasoning_prompt(ledger_data: dict) -> tuple[str, bool]:
    """Build the Phase 6 reasoning prompt. Returns (prompt, used_fallback)."""
    from . import _otr_lfc_context as _LFC_CTX

    meta = ledger_data.get("meta") or {}
    synopses = meta.get("scene_synopses") or []
    used_fallback = False

    parts: list[str] = []
    parts.append(EPISODE_ARC_REASONING_SYSTEM_PROMPT.strip())
    parts.append("")
    parts.append("=" * 60)
    parts.append("MACRO-BIBLE:")
    parts.append(_LFC_CTX.build_macro_bible(ledger_data))
    parts.append("")
    parts.append("=" * 60)
    if synopses and any(s.strip() for s in synopses):
        parts.append("SCENE SYNOPSES (from Phase 4):")
        for idx, syn in enumerate(synopses):
            if syn and syn.strip():
                parts.append(f"  scene {idx}: {syn.strip()}")
            else:
                parts.append(f"  scene {idx}: (no synopsis -- Phase 4 failed for this scene)")
    else:
        # Fallback per ADR section 6.13 / go-forward plan.
        used_fallback = True
        parts.append(
            "SCENE SYNOPSES NOT AVAILABLE (Phase 4 disabled or failed). "
            "Fallback to outline-intent overview:"
        )
        parts.append(_LFC_CTX.build_micro_recap(
            ledger_data, scope="episode", target=None,
        ))
    parts.append("")
    parts.append("=" * 60)
    parts.append("VOICED BEATS:")
    voiced_block, voiced_count = _voiced_beats_block(ledger_data)
    parts.append(voiced_block if voiced_block else "  (none)")
    parts.append("")
    cap = compute_arc_edit_cap(voiced_count)
    parts.append(
        f"You may propose AT MOST {cap} suggested edits TOTAL "
        f"across all Editor notes. Six notes maximum. Free prose."
    )
    return "\n".join(parts), used_fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase_6_episode_arc(
    reasoning_generate_fn: Callable,
    led,
    *,
    enable: bool = False,
    formatter_generate_fn: Optional[Callable] = None,
) -> Phase6EpisodeArcReport:
    """LFC Phase 6: episode-arc audit with Editor-note scaffold.

    `enable=False` (the default) short-circuits. Returns a report
    and stamps `meta.phase_6_episode_arc_record` +
    `meta.doctor_edit_budget_remaining`.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = Phase6EpisodeArcReport()
    if not enable:
        rep.skipped = True
        rep.skipped_reason = "enable=False"
        ledger_data.setdefault("meta", {})[
            "phase_6_episode_arc_record"
        ] = rep.to_dict()
        log.debug("[LFC:phase_6] disabled (enable=False)")
        return rep

    # Voiced-beat count + cap.
    _, voiced_count = _voiced_beats_block(ledger_data)
    rep.voiced_beats = voiced_count
    rep.edit_cap = compute_arc_edit_cap(voiced_count)
    rep.cap_source = "min(8, max(3, voiced_beats // 3))"

    if voiced_count == 0:
        ledger_data.setdefault("meta", {})[
            "phase_6_episode_arc_record"
        ] = rep.to_dict()
        log.info("[LFC:phase_6] no voiced beats; no LLM call fired")
        return rep

    # Build prompt + run two-step.
    reasoning_prompt, used_fallback = _build_arc_reasoning_prompt(ledger_data)
    rep.used_synopses_fallback = used_fallback

    from . import _otr_lfc_llm_helpers as _LFC_LLM
    result = _LFC_LLM.two_step_generate(
        reasoning_generate_fn,
        formatter_generate_fn,
        reasoning_prompt=reasoning_prompt,
        formatter_prompt_template=EPISODE_ARC_FORMATTER_TEMPLATE,
        schema=_LFC_LLM.EditorNotesOutput,
        reasoning_temperature=0.7,
        formatter_temperature=0.1,
    )
    if result is None:
        rep.failed = True
        rep.failure_reason = "two_step_generate returned None"
        log.warning(
            "[LFC:phase_6] two_step_generate failed; dropping all edits"
        )
        ledger_data.setdefault("meta", {})[
            "phase_6_episode_arc_record"
        ] = rep.to_dict()
        return rep

    rep.notes_proposed = len(result.notes)
    proposed_edits: list = []
    for note in result.notes:
        rationale_parts = [note.note]
        # Track suggested_edits for application below.
        for edit in note.suggested_edits:
            proposed_edits.append(edit)
        if note.note:
            rationale_parts.append(note.note)
    rep.edits_proposed = len(proposed_edits)
    # Build a coarse rationale: scope summary across notes.
    rep.rationale = " | ".join(
        f"{note.scope}: {note.note[:80]}" for note in result.notes
    )

    # Apply edits with the global edit cap.
    lines = ledger_data.get("lines") or []
    lines_by_id = {
        ln.get("line_id", ""): ln for ln in lines
        if isinstance(ln, dict)
    }

    applied = 0
    for edit in proposed_edits:
        if applied >= rep.edit_cap:
            rep.edits_rejected_cap_exceeded += 1
            continue
        target = lines_by_id.get(edit.line_id)
        if target is None:
            rep.edits_rejected_unknown_line += 1
            continue
        role = (target.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            rep.edits_rejected_wrong_role += 1
            continue
        revised = edit.revised or ""
        if not revised:
            rep.edits_rejected_unknown_line += 1
            continue
        orig = target.get("text") or ""
        orig_wc = max(1, sum(1 for _ in orig.split()))
        new_wc = max(1, sum(1 for _ in revised.split()))
        deviation = abs(new_wc - orig_wc) / orig_wc
        if deviation > EDIT_LENGTH_DRIFT_REJECT_THRESHOLD:
            rep.edits_rejected_length_drift += 1
            continue
        target["text"] = revised
        target["char_count"] = len(revised)
        target["word_count"] = sum(1 for _ in revised.split())
        applied += 1

    rep.edits_applied = applied

    # Stamp remaining budget. ADR / go-forward plan: this is the
    # SHARED pool with the reviewer's Doctor; future phases that
    # also spend doctor-style edits should respect what's left.
    meta = ledger_data.setdefault("meta", {})
    budget_remaining = max(0, rep.edit_cap - applied)
    meta["doctor_edit_budget_remaining"] = budget_remaining
    meta["phase_6_episode_arc_record"] = rep.to_dict()
    log.info(
        "[LFC:phase_6] notes=%d proposed=%d applied=%d "
        "rej_unknown=%d rej_role=%d rej_length=%d rej_cap=%d "
        "fallback_synopses=%s budget_remaining=%d",
        rep.notes_proposed, rep.edits_proposed, rep.edits_applied,
        rep.edits_rejected_unknown_line,
        rep.edits_rejected_wrong_role,
        rep.edits_rejected_length_drift,
        rep.edits_rejected_cap_exceeded,
        used_fallback, budget_remaining,
    )
    return rep

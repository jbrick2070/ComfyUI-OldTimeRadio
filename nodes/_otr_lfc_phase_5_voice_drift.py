"""nodes/_otr_lfc_phase_5_voice_drift.py — LFC Phase 5 (ADR section 6.5).

Per-speaker voice-drift detection + targeted rewrites.

Pipeline:
  1. `compute_speaker_stats` walks ledger.lines once -- one
     SpeakerStats per char_id (mean_line_length + vocab_diversity).
  2. `flag_line_drift` per voiced line; collect ALL flagged lines
     into a single global batch (ADR section 6.5: "one batched LLM
     call" -- avoids per-speaker quadratic cost).
  3. `two_step_generate` over the batch with:
       reasoning prompt   Macro-Bible + per-speaker Micro-Recap +
                          the flagged lines spelled out with their
                          drift reason. Free Markdown / chain-of-
                          thought.
       formatter prompt   schema-extraction prompt that returns
                          ReviewerOutput (a list of ReviewerEdit
                          {line_id, original, revised, reason}).
  4. Apply each edit via line_id lookup. Reject edits that:
       * target a line not in the flagged subset,
       * target a non-character / non-announcer beat,
       * lengthen or shorten the line by >50%.
     Apply the rest in place; recompute word_count + char_count
     in lockstep.

Default OFF -- the cascade widget `enable_phase_5_voice_drift`
gates the call. The flagging heuristic is conservative
(40% word-count deviation OR vocab diversity below 60% of speaker
mean), so even at enable=True most episodes will see zero or
one flagged lines and zero LLM cost.

Status: LFC sprint commit 9 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("OTR.lfc_phase_5")


__all__ = [
    "Phase5VoiceDriftReport",
    "FlaggedLine",
    "phase_5_voice_drift",
    "VOICE_DRIFT_REASONING_SYSTEM_PROMPT",
    "VOICE_DRIFT_FORMATTER_TEMPLATE",
    "EDIT_LENGTH_DRIFT_REJECT_THRESHOLD",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Reject post-rewrite text whose word count differs from the original
# by more than 50%. A "voice drift" repair that doubles or halves the
# line length is changing the line's content, not its voice.
EDIT_LENGTH_DRIFT_REJECT_THRESHOLD: float = 0.50


# ADR section 6.14: Narrator-Character-Editor scaffold for Phase 4 /
# 6. Phase 5 is single-role (the Editor), but we still anchor on the
# same vocabulary so the prompt set stays coherent across phases.
VOICE_DRIFT_REASONING_SYSTEM_PROMPT = """\
You are the EDITOR reviewing one episode of an audio drama for
per-speaker voice drift. The cast is locked; the script is past
the cast-contract reviewer.

A "drifted" line is one that, after deterministic stats analysis,
diverges from a character's established voice -- either by a
sudden change in word-count (line too long or too short) or by a
sudden collapse in vocabulary diversity (the character starts
repeating a single word or generic phrase).

You will receive:
  * MACRO-BIBLE  episode logline + cast voice cards + arc skeleton
  * SPEAKER RECAPS  per-flagged-speaker: voice card + every line
                    that character spoke
  * FLAGGED LINES  one row per flagged line: line_id, char_id,
                   original text, drift reason

For each flagged line, decide if it should be revised. When it
should, write a revised version that:
  * preserves the line's narrative beat (what it accomplishes),
  * matches the character's established voice in the speaker recap,
  * stays within plus-or-minus 50 percent of the original word count,
  * contains only spoken dialogue -- no bracket stage direction,
    no asterisk action, no parenthesized cue verbs.

Some flagged lines are actually FINE for narrative reasons (a
character's emotional state shifts; a single-word panic response
is by design). When in doubt, leave the line unchanged.

Output: free-form prose. Walk through each flagged line. For each:
  - DECISION  REVISE or LEAVE
  - If REVISE, the revised text
  - One-sentence rationale

The next stage of this pipeline will translate your prose into
structured JSON. Do NOT output JSON here.
"""


VOICE_DRIFT_FORMATTER_TEMPLATE = """\
You are a JSON formatter. Translate the editor's prose review below
into the following JSON schema EXACTLY. Output ONLY the JSON object
(no markdown fences, no prose before or after).

Schema:
{
  "edits": [
    {
      "line_id": "<line_id from the editor's review>",
      "original": "<original line text>",
      "revised": "<editor's revised text>",
      "reason": "<one-sentence rationale>"
    }
  ],
  "rationale": "<one-sentence summary of the editor's overall approach>"
}

Rules:
  * Skip any line the editor marked LEAVE -- do NOT include in edits.
  * For every REVISE: the revised field is the editor's revised
    text verbatim. Do NOT paraphrase or trim.
  * Preserve line_id exactly as written by the editor.

Editor's review:
{{REASONING}}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlaggedLine:
    line_id: str
    char_id: str
    text: str
    reason: str  # drift reason from flag_line_drift

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "char_id": self.char_id,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass
class Phase5VoiceDriftReport:
    lines_scanned: int = 0
    speakers_examined: int = 0
    flagged_lines: list = field(default_factory=list)  # FlaggedLine.to_dict()
    edits_proposed: int = 0
    edits_applied: int = 0
    edits_rejected_unknown_line: int = 0
    edits_rejected_wrong_role: int = 0
    edits_rejected_length_drift: int = 0
    edits_rejected_not_flagged: int = 0
    skipped: bool = False
    skipped_reason: str = ""
    # G3 fix (commit 12.13): explicit failure flag. When
    # two_step_generate returns None (parse exhaustion / LLM
    # crash) the report previously left edits_applied=0 with
    # flagged_lines populated -- indistinguishable from a real
    # "editor declined to revise" outcome. failed=True
    # distinguishes those two paths.
    failed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "lines_scanned": int(self.lines_scanned),
            "speakers_examined": int(self.speakers_examined),
            "flagged_lines": list(self.flagged_lines),
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
            "edits_rejected_not_flagged": int(
                self.edits_rejected_not_flagged
            ),
            "skipped": bool(self.skipped),
            "skipped_reason": self.skipped_reason,
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_reasoning_prompt(
    led_or_data,
    flagged: list[FlaggedLine],
) -> str:
    from . import _otr_lfc_context as _LFC_CTX

    parts: list[str] = []
    parts.append(VOICE_DRIFT_REASONING_SYSTEM_PROMPT.strip())
    parts.append("")
    parts.append("=" * 60)
    parts.append("MACRO-BIBLE:")
    parts.append(_LFC_CTX.build_macro_bible(led_or_data))
    parts.append("")
    parts.append("=" * 60)
    parts.append("SPEAKER RECAPS:")
    seen_speakers: set[str] = set()
    for fl in flagged:
        if fl.char_id in seen_speakers:
            continue
        seen_speakers.add(fl.char_id)
        parts.append(
            _LFC_CTX.build_micro_recap(
                led_or_data, scope="speaker", target=fl.char_id,
            )
        )
        parts.append("")
    parts.append("=" * 60)
    parts.append("FLAGGED LINES (single batch -- review every one):")
    for fl in flagged:
        parts.append(
            f"  line_id={fl.line_id}  char_id={fl.char_id}\n"
            f"    drift reason: {fl.reason}\n"
            f"    original text: {fl.text!r}"
        )
    parts.append("")
    parts.append(
        "Walk through every flagged line above. DECISION + revised "
        "text (if REVISE) + rationale. Free prose. The next stage "
        "translates to JSON."
    )
    return "\n".join(parts)


def phase_5_voice_drift(
    reasoning_generate_fn: Callable,
    led,
    *,
    enable: bool = False,
    formatter_generate_fn: Optional[Callable] = None,
) -> Phase5VoiceDriftReport:
    """LFC Phase 5: per-speaker voice drift.

    `enable=False` (the default) short-circuits to a skipped report;
    callers can probe `report.skipped` to distinguish from a real
    no-flagged-lines outcome.

    Mutates line.text on accepted edits + recomputes word/char
    counts. Does NOT add or remove lines. Does NOT touch
    char_id / speaker_role / line_id.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = Phase5VoiceDriftReport()
    if not enable:
        rep.skipped = True
        rep.skipped_reason = "enable=False"
        ledger_data.setdefault("meta", {})[
            "phase_5_voice_drift_record"
        ] = rep.to_dict()
        log.debug("[LFC:phase_5] disabled (enable=False)")
        return rep

    lines = ledger_data.get("lines") or []
    rep.lines_scanned = len(lines)
    if not lines:
        ledger_data.setdefault("meta", {})[
            "phase_5_voice_drift_record"
        ] = rep.to_dict()
        return rep

    # Stats + flag.
    from . import _otr_lfc_watchdog as _WD
    stats = _WD.compute_speaker_stats(lines)
    rep.speakers_examined = len(stats)

    flagged: list[FlaggedLine] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        cid = ln.get("char_id") or ""
        if cid not in stats:
            continue
        text = ln.get("text") or ""
        if not text:
            continue
        drifted, reason = _WD.flag_line_drift(text, stats[cid])
        if drifted:
            flagged.append(FlaggedLine(
                line_id=ln.get("line_id", ""),
                char_id=cid, text=text, reason=reason,
            ))

    rep.flagged_lines = [fl.to_dict() for fl in flagged]
    if not flagged:
        ledger_data.setdefault("meta", {})[
            "phase_5_voice_drift_record"
        ] = rep.to_dict()
        log.info("[LFC:phase_5] no flagged lines; no LLM call fired")
        return rep

    # Two-step generate.
    from . import _otr_lfc_llm_helpers as _LFC_LLM
    reasoning_prompt = _build_reasoning_prompt(ledger_data, flagged)
    result = _LFC_LLM.two_step_generate(
        reasoning_generate_fn,
        formatter_generate_fn,
        reasoning_prompt=reasoning_prompt,
        formatter_prompt_template=VOICE_DRIFT_FORMATTER_TEMPLATE,
        schema=_LFC_LLM.ReviewerOutput,
        reasoning_temperature=0.7,
        formatter_temperature=0.1,
    )
    if result is None:
        # G3 fix (commit 12.13): flag the failure explicitly so the
        # standalone Phase 5 node can return verdict "failed"
        # instead of conflating it with "completed_no_edits_after_flag".
        rep.failed = True
        rep.failure_reason = "two_step_generate returned None"
        log.warning(
            "[LFC:phase_5] two_step_generate returned None on %d "
            "flagged lines; dropping all edits", len(flagged),
        )
        ledger_data.setdefault("meta", {})[
            "phase_5_voice_drift_record"
        ] = rep.to_dict()
        return rep

    rep.edits_proposed = len(result.edits)
    flagged_ids = {fl.line_id for fl in flagged}
    lines_by_id = {
        ln.get("line_id", ""): ln for ln in lines
        if isinstance(ln, dict)
    }

    for edit in result.edits:
        if edit.line_id not in flagged_ids:
            # Editor proposed an edit on a line we never flagged.
            rep.edits_rejected_not_flagged += 1
            continue
        target = lines_by_id.get(edit.line_id)
        if target is None:
            rep.edits_rejected_unknown_line += 1
            continue
        role = (target.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            rep.edits_rejected_wrong_role += 1
            continue
        # Length drift guard.
        orig = target.get("text") or ""
        orig_wc = max(1, sum(1 for _ in orig.split()))
        new_wc = max(1, sum(1 for _ in (edit.revised or "").split()))
        deviation = abs(new_wc - orig_wc) / orig_wc
        if deviation > EDIT_LENGTH_DRIFT_REJECT_THRESHOLD:
            rep.edits_rejected_length_drift += 1
            log.info(
                "[LFC:phase_5] rejecting edit on %s -- length "
                "deviation %.0f%% > threshold %.0f%%",
                edit.line_id, deviation * 100,
                EDIT_LENGTH_DRIFT_REJECT_THRESHOLD * 100,
            )
            continue
        # Apply.
        target["text"] = edit.revised
        target["char_count"] = len(edit.revised)
        target["word_count"] = sum(1 for _ in (edit.revised or "").split())
        rep.edits_applied += 1

    ledger_data.setdefault("meta", {})[
        "phase_5_voice_drift_record"
    ] = rep.to_dict()
    log.info(
        "[LFC:phase_5] flagged=%d proposed=%d applied=%d "
        "rej_unknown=%d rej_wrong_role=%d rej_length=%d rej_not_flagged=%d",
        len(flagged), rep.edits_proposed, rep.edits_applied,
        rep.edits_rejected_unknown_line,
        rep.edits_rejected_wrong_role,
        rep.edits_rejected_length_drift,
        rep.edits_rejected_not_flagged,
    )
    return rep

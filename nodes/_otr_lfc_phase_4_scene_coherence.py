"""nodes/_otr_lfc_phase_4_scene_coherence.py — LFC Phase 4 (ADR 6.13 / 6.14 / 6.15).

Per-scene coherence pass. Iterates scenes via `scenes_from_lines`
(music_inter dividers, single-scene fallback). For each scene:

  1. `build_micro_recap(scope="scene")` -- previous scene's last 3
     lines + this scene's beat intents + active speakers' voice cards.
  2. `two_step_generate` with the Narrator-Character-Editor scaffold
     (ADR 6.14) + audio-first directives (ADR 6.15).
  3. Apply at most `min(3, scene_lines // 2)` accepted edits.
  4. Cache the per-scene synopsis on `meta.scene_synopses` for
     Phase 6 to read (ADR 6.13 cross-phase memory).

Failure cascade per ADR section 8: skip-and-continue. When the
formatter pass returns None (parse exhaustion) for a SINGLE scene,
drop just that scene's edits, log a warning, advance to the next
scene. Do NOT fail Phase 4 as a whole.

Default OFF -- cascade widget `enable_phase_4_scene_coherence` gates
the call. Each scene that fires costs 1 reasoning + 1 formatter call;
a 3-scene episode at full opt-in is ~6 LLM calls.

Status: LFC sprint commit 8 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("OTR.lfc_phase_4")


__all__ = [
    "Phase4SceneCoherenceReport",
    "phase_4_scene_coherence",
    "SCENE_COHERENCE_REASONING_SYSTEM_PROMPT",
    "SCENE_COHERENCE_FORMATTER_TEMPLATE",
    "EDIT_LENGTH_DRIFT_REJECT_THRESHOLD",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Length-drift guard: reject any edit that >50% lengthens or shortens
# a line. Same threshold as Phase 5 -- a coherence repair that
# doubles or halves a line is changing the line's content, not its
# coherence.
EDIT_LENGTH_DRIFT_REJECT_THRESHOLD: float = 0.50


# ADR section 6.14 + 6.15. Three roles, audio-first directives, no
# visual stage direction.
SCENE_COHERENCE_REASONING_SYSTEM_PROMPT = """\
You are reviewing ONE scene of an audio drama. The cast is locked;
the cast-contract reviewer has already validated speakers and
character names. Your job is per-scene coherence: pacing, beat-to-
beat continuity, and the audio-first contract.

THREE ROLES at play in any scene -- you adjudicate, you do not
become any one of them:

  NARRATOR     handles time transitions, environmental cues, pacing
               shifts. When dialogue stalls, the Narrator forces
               forward motion.
  CHARACTERS   constrained to spoken dialogue + immediate phonetic /
               vocal cues only.
  EDITOR       the meta-role -- when you (the reviewer) decide the
               scene needs a structural nudge, write Editor notes
               that the next stage will translate into edits.

AUDIO-FIRST CONTRACT (this is a radio drama; no visual channel):

  BANNED         visual stage direction
                 ("she glares", "sunlight streams through the window")
                 visual emotion descriptions
                 ("his eyes widen", "she looks away")
                 mute action ("he stands silently")
  REQUIRED       acoustic translations
                 (SOUND: throat clears; SOUND: distant traffic;
                  VOICE: sharp intake of breath)
                 explicit Music cues in structured-tag form
                 (MUSIC: Cinematic, Deep Sadness, Cello, Swelling
                  Strings, 60 BPM, Moody)
                 -- NOT bare prose like "music plays sadly".

PACING (lock into the scene's tension):

  HIGH TENSION   short fragments, overlapping exchanges, 5-15 word
                 character lines.
  LOW TENSION    longer thoughts, breathing room, 15-30 word lines.
  MONOLOGUES > 30 words must be broken across SOUND interjections
                 or SKIPPED entirely.

You will receive the MACRO-BIBLE (episode identity), the
MICRO-RECAP for this scene (prior-scene tail, beat intents, active
speakers' voice cards), and the scene's lines verbatim.

For each line in the scene, decide REVISE / LEAVE / SKIP. Write a
SCENE SYNOPSIS at the end -- ONE paragraph summarizing what this
scene accomplishes; Phase 6 (episode arc) reads it.

Output: free-form prose. Walk through each line. End with:
    SCENE SYNOPSIS: <one paragraph>
The next stage of this pipeline translates your prose into
structured JSON. Do NOT output JSON here.
"""


SCENE_COHERENCE_FORMATTER_TEMPLATE = """\
You are a JSON formatter. Translate the editor's prose scene review
below into the following JSON schema EXACTLY. Output ONLY the JSON
object (no markdown fences, no prose before or after).

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
  "rationale": "<SCENE SYNOPSIS paragraph from the editor's review>"
}

Rules:
  * Include only REVISE decisions. SKIP gets emitted as revised=""
    with reason starting "SKIP:". LEAVE is omitted entirely.
  * `rationale` is the SCENE SYNOPSIS paragraph verbatim. Required.
  * Preserve line_id exactly as written by the editor.

Editor's review:
{{REASONING}}
"""


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class ScenePassResult:
    scene_idx: int
    edits_proposed: int = 0
    edits_applied: int = 0
    edits_skipped: int = 0
    edits_rejected_unknown_line: int = 0
    edits_rejected_wrong_scope: int = 0
    edits_rejected_length_drift: int = 0
    synopsis: str = ""
    failed: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_idx": int(self.scene_idx),
            "edits_proposed": int(self.edits_proposed),
            "edits_applied": int(self.edits_applied),
            "edits_skipped": int(self.edits_skipped),
            "edits_rejected_unknown_line": int(self.edits_rejected_unknown_line),
            "edits_rejected_wrong_scope": int(self.edits_rejected_wrong_scope),
            "edits_rejected_length_drift": int(self.edits_rejected_length_drift),
            "synopsis": self.synopsis,
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
        }


@dataclass
class Phase4SceneCoherenceReport:
    scenes_scanned: int = 0
    scenes_examined: int = 0
    scenes_failed: int = 0
    total_edits_applied: int = 0
    scene_results: list = field(default_factory=list)  # ScenePassResult.to_dict()
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "scenes_scanned": int(self.scenes_scanned),
            "scenes_examined": int(self.scenes_examined),
            "scenes_failed": int(self.scenes_failed),
            "total_edits_applied": int(self.total_edits_applied),
            "scene_results": list(self.scene_results),
            "skipped": bool(self.skipped),
            "skipped_reason": self.skipped_reason,
        }


# ---------------------------------------------------------------------------
# Scene processor
# ---------------------------------------------------------------------------


def _compute_scene_edit_cap(scene_lines: list) -> int:
    """Per-scene edit cap = min(3, scene_lines // 2). ADR section 4."""
    voiced = sum(
        1 for ln in scene_lines
        if isinstance(ln, dict)
        and (ln.get("speaker_role") or "").strip().lower()
        in ("character", "announcer")
    )
    return min(3, max(0, voiced // 2))


def _build_scene_reasoning_prompt(ledger_data: dict, scene: dict) -> str:
    from . import _otr_lfc_context as _LFC_CTX

    parts: list[str] = []
    parts.append(SCENE_COHERENCE_REASONING_SYSTEM_PROMPT.strip())
    parts.append("")
    parts.append("=" * 60)
    parts.append("MACRO-BIBLE:")
    parts.append(_LFC_CTX.build_macro_bible(ledger_data))
    parts.append("")
    parts.append("=" * 60)
    parts.append("MICRO-RECAP FOR THIS SCENE:")
    parts.append(_LFC_CTX.build_micro_recap(
        ledger_data, scope="scene", target=scene,
    ))
    parts.append("")
    parts.append("=" * 60)
    parts.append("LINES IN THIS SCENE (verbatim):")
    for ln in scene.get("lines", []) or []:
        if not isinstance(ln, dict):
            continue
        line_id = ln.get("line_id", "")
        role = ln.get("speaker_role", "")
        cid = ln.get("char_id", "")
        text = (ln.get("text") or "").strip()
        parts.append(f"  {line_id} | {role} | {cid}: {text}")
    parts.append("")
    cap = _compute_scene_edit_cap(scene.get("lines", []) or [])
    parts.append(
        f"You may propose AT MOST {cap} REVISE / SKIP edits in this "
        f"scene. End with a SCENE SYNOPSIS paragraph for Phase 6 "
        f"(episode arc) to read. Free prose -- the next stage "
        f"translates to JSON."
    )
    return "\n".join(parts)


def _process_scene(
    reasoning_generate_fn: Callable,
    formatter_generate_fn: Optional[Callable],
    ledger_data: dict,
    scene: dict,
) -> ScenePassResult:
    """Run two_step_generate for ONE scene + apply accepted edits.

    Returns a ScenePassResult with edit counts + synopsis. On any
    LLM failure / parse failure / cap overrun, marks `failed=True`
    and returns -- the orchestrator advances to the next scene
    (ADR section 8 skip-and-continue).
    """
    from . import _otr_lfc_llm_helpers as _LFC_LLM

    res = ScenePassResult(scene_idx=int(scene.get("scene_idx", 0)))
    scene_lines = scene.get("lines", []) or []
    cap = _compute_scene_edit_cap(scene_lines)

    if not scene_lines:
        return res

    reasoning_prompt = _build_scene_reasoning_prompt(ledger_data, scene)
    result = _LFC_LLM.two_step_generate(
        reasoning_generate_fn,
        formatter_generate_fn,
        reasoning_prompt=reasoning_prompt,
        formatter_prompt_template=SCENE_COHERENCE_FORMATTER_TEMPLATE,
        schema=_LFC_LLM.ReviewerOutput,
        reasoning_temperature=0.7,
        formatter_temperature=0.1,
    )
    if result is None:
        res.failed = True
        res.failure_reason = "two_step_generate returned None"
        log.warning(
            "[LFC:phase_4] scene %d: two_step_generate failed -- "
            "dropping all edits, advancing",
            res.scene_idx,
        )
        return res

    res.synopsis = (result.rationale or "").strip()
    res.edits_proposed = len(result.edits)

    # Build scene-scoped line lookup.
    scene_line_ids = {
        ln.get("line_id", "") for ln in scene_lines
        if isinstance(ln, dict)
    }
    lines_by_id = {
        ln.get("line_id", ""): ln for ln in scene_lines
        if isinstance(ln, dict)
    }

    applied = 0
    for edit in result.edits:
        if applied >= cap:
            log.info(
                "[LFC:phase_4] scene %d: edit cap %d reached; "
                "dropping remaining %d edit(s)",
                res.scene_idx, cap,
                len(result.edits) - (applied + 1),
            )
            break
        # Wrong scope: edit targets a line outside this scene.
        if edit.line_id not in scene_line_ids:
            res.edits_rejected_wrong_scope += 1
            continue
        target = lines_by_id.get(edit.line_id)
        if target is None:
            res.edits_rejected_unknown_line += 1
            continue
        role = (target.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            res.edits_rejected_wrong_scope += 1
            continue

        revised = edit.revised or ""
        reason = edit.reason or ""

        # SKIP convention: revised="" + reason starting "SKIP:".
        if not revised and reason.startswith("SKIP:"):
            target["skip"] = True
            target["text"] = ""
            target["char_count"] = 0
            target["word_count"] = 0
            target["reviewer_skip_reason"] = reason
            res.edits_skipped += 1
            applied += 1
            continue

        if not revised:
            # Empty revise without SKIP convention -- treat as
            # rejection.
            res.edits_rejected_unknown_line += 1
            continue

        # Length-drift guard.
        orig = target.get("text") or ""
        orig_wc = max(1, sum(1 for _ in orig.split()))
        new_wc = max(1, sum(1 for _ in revised.split()))
        deviation = abs(new_wc - orig_wc) / orig_wc
        if deviation > EDIT_LENGTH_DRIFT_REJECT_THRESHOLD:
            res.edits_rejected_length_drift += 1
            continue

        target["text"] = revised
        target["char_count"] = len(revised)
        target["word_count"] = sum(1 for _ in revised.split())
        applied += 1

    res.edits_applied = applied
    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase_4_scene_coherence(
    reasoning_generate_fn: Callable,
    led,
    *,
    enable: bool = False,
    formatter_generate_fn: Optional[Callable] = None,
) -> Phase4SceneCoherenceReport:
    """LFC Phase 4: per-scene coherence.

    `enable=False` (the default) short-circuits to a skipped report.
    Returns the full report and stamps `meta.phase_4_scene_coherence_record`
    + `meta.scene_synopses` (one synopsis per scene that fired).
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = Phase4SceneCoherenceReport()
    if not enable:
        rep.skipped = True
        rep.skipped_reason = "enable=False"
        ledger_data.setdefault("meta", {})[
            "phase_4_scene_coherence_record"
        ] = rep.to_dict()
        log.debug("[LFC:phase_4] disabled (enable=False)")
        return rep

    from . import _otr_lfc_context as _LFC_CTX
    scenes = _LFC_CTX.scenes_from_lines(ledger_data)
    rep.scenes_scanned = len(scenes)
    if not scenes:
        ledger_data.setdefault("meta", {})[
            "phase_4_scene_coherence_record"
        ] = rep.to_dict()
        return rep

    synopses: list[str] = []
    for scene in scenes:
        res = _process_scene(
            reasoning_generate_fn,
            formatter_generate_fn,
            ledger_data,
            scene,
        )
        rep.scene_results.append(res.to_dict())
        rep.scenes_examined += 1
        if res.failed:
            rep.scenes_failed += 1
            synopses.append("")  # placeholder so indexes line up
            continue
        rep.total_edits_applied += res.edits_applied
        synopses.append(res.synopsis)

    # Cache per-scene synopses for Phase 6 (commit 10). Even empty
    # entries land so the indexing stays scene-aligned.
    meta = ledger_data.setdefault("meta", {})
    meta["scene_synopses"] = synopses
    meta["phase_4_scene_coherence_record"] = rep.to_dict()
    log.info(
        "[LFC:phase_4] scanned=%d examined=%d failed=%d edits_applied=%d",
        rep.scenes_scanned, rep.scenes_examined,
        rep.scenes_failed, rep.total_edits_applied,
    )
    return rep

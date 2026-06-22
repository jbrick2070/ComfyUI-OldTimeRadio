"""nodes/_otr_story_spine.py -- Wave 2 post-script orchestrator (env-gated).

This is the in-process WIRING that runs the story-spine post-script
passes inside OTR_LedgerScriptWriter.run(), mirroring how
run_story_brief_reflection is called (`_otr_story_brief.py:861`):
Stage 2.5 (conditional Radio Editor length pass), Stage 3 (Targeted
Story QA defect router), Stage 3.5 (beat-local micro-repair on the
flagged beats), the writer-LLM unload, and Stage 4 (deterministic
Ledger Scrub). A REJECT verdict aborts at the writer boundary: the spine
sets meta["story_verdict"]="REJECT", unloads, skips the scrub, and
returns -- the writer raises (the spine never does).

GATED, DEFAULT ON (opt-out). `enabled()` reads `OTR_ENABLE_STORY_SPINE`
and is True unless it is exactly "0". Out of the box the four passes run
-- a fresh install gets the critic + editor + scrub with no env setup,
which is the point of building them. Setting the flag to "0" is the
operator escape hatch: it restores the writer's byte-identical baseline
path (the unload block at the original call site), e.g. for a baseline
A/B or a fast bare-writer smoke. The headless suite stays green on either
setting (every pass is fail-soft and never raises).

In-process, not a node (D4/D5): no INPUT_TYPES, no widget, no broadcast
output, no workflow-JSON change. The model ids arrive as the writer's
in-process `resolved[...]` values (critic -> technical slot per D6;
editor -> creative slot). PD3 (workflow JSON): N/A; adds no node surface.

NEVER RAISES (Prime Directive 1, audio is king). Every pass is wrapped;
a failure stamps a status on `meta` and the run continues with the
script it already had. At most ONE beat-local micro-repair cycle runs
(spine invariant 5), and the deterministic scrub is always told
`repair_available=False` -- it is mechanical and last, never a repair
trigger.

Full power, fail-soft by construction:
  * recompose_fn wraps `_otr_line_composer.compose_line` on the creative
    slot (`_make_recompose_fn`), so RECOMPOSE_BEAT_SAME_INTENT actually
    regenerates a flat beat (same speaker + intent, under the editor's
    fence). Any error returns the original line, so a beat is never
    corrupted -- RECOMPOSE just degrades to KEEP on failure.
  * turn_beat_index / button_beat_index are mapped from the Stream A
    outline's `turning_point` / `button` into the editor's voiced-view
    index space by beat_id (`_map_arc_indices`), so Tier-2 removals never
    drop the arc beats. A mismatch yields None, which only relaxes that
    one protection (the structural + length + visual-noun guards still
    prevent any render-contract corruption).

UTF-8 no BOM. No em-dashes. 4-space indentation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger("OTR_StorySpine")

_ENV_FLAG = "OTR_ENABLE_STORY_SPINE"
_QA_ENV_FLAG = "OTR_ENABLE_STORY_QA"


def enabled() -> bool:
    """True iff the story-spine post-script passes are switched on.

    Default ON (opt-out): the story spine is the out-of-the-box pipeline,
    so a fresh install gets the critic + editor + scrub with no env setup.
    Only an explicit "0" disables it -- the operator escape hatch for a
    byte-identical-baseline run or a fast bare-writer smoke. Cheap, pure,
    never raises.
    """
    try:
        return os.environ.get(_ENV_FLAG, "1").strip() != "0"
    except Exception:  # noqa: BLE001 -- env read must never break a run
        return True


def qa_enabled() -> bool:
    """True iff the post-script Story QA / reject pass may run.

    DEFAULT OFF (opt-in). BUG-LOCAL-302 (operator directive 2026-06-01): the
    Story QA pass is a credit-billed LLM call whose only power is to REJECT --
    aborting a fully-written, renderable episode and discarding the tokens
    already spent writing it. So the QA LLM does NOT run out of the box: it is
    gated off entirely (no call, no tokens, no reject) unless a power user sets
    OTR_ENABLE_STORY_QA=1. Cheap, pure, never raises."""
    try:
        return os.environ.get(_QA_ENV_FLAG, "0").strip() == "1"
    except Exception:  # noqa: BLE001 -- env read must never break a run
        return False


_VOICED_ROLES = ("character", "announcer")


def _voiced_lines(led: Any) -> list:
    """The editor's voiced view: character + announcer ledger lines, in
    order. Mirrors _otr_radio_editor.VOICED_ROLES so beat_index maps
    1:1 between this module and the editor. Never raises."""
    try:
        data = getattr(led, "data", led)
        lines = data.get("lines") or []
        return [l for l in lines
                if isinstance(l, dict) and l.get("speaker_role") in _VOICED_ROLES]
    except Exception:  # noqa: BLE001
        return []


def _cast_name(cast: list, char_id: Any) -> Optional[str]:
    for row in cast or ():
        if isinstance(row, dict) and row.get("char_id") == char_id:
            return row.get("name")
    return None


def _make_recompose_fn(led: Any, creative_generate_fn: Callable[..., str]):
    """Build the editor's recompose seam: regenerate one flat beat via
    `_otr_line_composer.compose_line` on the creative slot, same speaker
    and intent, under the editor's RECOMPOSE fence.

    Signature handed to the editor: ``recompose_fn(beat_index,
    original_text, hint) -> str``. FAIL-SOFT BY CONSTRUCTION: any error
    (bad index, LineRequest mismatch, a raising slot fn) returns the
    ORIGINAL line, so RECOMPOSE degrades to KEEP and the beat is never
    corrupted. `beat_index` is the editor's voiced-view index.
    """
    def _recompose(beat_index: int, original_text: str, hint: str) -> str:
        try:
            try:
                from . import _otr_line_composer as _LC
            except ImportError:  # pragma: no cover - standalone / test load
                import _otr_line_composer as _LC  # type: ignore
            data = getattr(led, "data", led)
            cast = data.get("cast") or []
            voiced = _voiced_lines(led)
            if not (0 <= beat_index < len(voiced)):
                return original_text
            line = voiced[beat_index]
            speaker = (
                (_cast_name(cast, line.get("char_id")) or line.get("char_id") or "")
                .strip().upper()
            )
            if not speaker:
                return original_text
            intent = (line.get("beat_intent") or "continue the scene").strip()
            mood = (line.get("mood") or "neutral")
            try:
                target_words = int(line.get("target_words") or 20)
            except Exception:  # noqa: BLE001
                target_words = 20
            # preceding voiced lines (most-recent-last) as scene context.
            last_lines = []
            for prev in voiced[max(0, beat_index - 3):beat_index]:
                pname = (_cast_name(cast, prev.get("char_id"))
                         or prev.get("char_id") or "").upper()
                last_lines.append((pname, prev.get("text") or ""))
            try:
                roster = _LC.build_allowed_roster(cast)
            except Exception:  # noqa: BLE001
                roster = frozenset()
            req = _LC.LineRequest(
                speaker=speaker,
                intent=intent,
                mood=mood,
                target_words=target_words,
                canon_header="",
                last_lines=last_lines,
                allowed_roster=roster,
                speaker_role=(line.get("speaker_role") or "character"),
            )
            res = _LC.compose_line(
                creative_fn=creative_generate_fn,
                req=req,
                reroll_hint=(hint or None),
            )
            text = getattr(res, "text", None)
            return text if (text and str(text).strip()) else original_text
        except Exception:  # noqa: BLE001 -- recompose must never corrupt a beat
            return original_text

    return _recompose


def _recompose_announcer_tagline(
    led: Any,
    meta: dict,
    creative_generate_fn: Callable[..., str],
    resolved: Any,
    *,
    is_intro: bool,
    intro_text: str = "",
) -> str:
    """Recompose a TRUNCATED announcer open/close tagline via the DEDICATED
    announcer composer (A6), never the character reroll path -- the critic
    excludes announcer lines as locked structural content, so a character
    reroll cannot act on them. Returns the new text, or "" on any failure
    (the caller keeps the original). Never raises."""
    try:
        try:
            from . import _otr_line_composer as _LC
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_line_composer as _LC  # type: ignore
        news = (meta.get("news") or {}) if isinstance(meta, dict) else {}
        if not isinstance(news, dict):
            news = {}
        script_brief = str(news.get("script_brief") or "")
        news_close_brief = str(news.get("news_close_brief") or "")
        repo = resolved.get("creative_writing_model") if isinstance(resolved, dict) else None
        if is_intro:
            res = _LC.compose_announcer_intro(
                creative_fn=creative_generate_fn,
                script_brief=script_brief,
                creative_repo_id=repo,
            )
        else:
            res = _LC.compose_announcer_outro(
                creative_fn=creative_generate_fn,
                script_brief=script_brief,
                news_close_brief=news_close_brief,
                intro_text=intro_text,
                creative_repo_id=repo,
            )
        text = getattr(res, "text", None)
        return str(text) if (text and str(text).strip()) else ""
    except Exception:  # noqa: BLE001 -- never corrupt the tagline
        return ""


def _map_arc_indices(outline: Any, led: Any):
    """Map the Stream A outline turning_point/button (indices into ALL
    outline beats) into the editor's voiced-view index space (by
    beat_id). Returns ``(turn_idx, button_idx)``; either is None on any
    mismatch, which simply relaxes the editor's arc-protection without
    risking a wrong-beat lock. Never raises."""
    try:
        voiced = _voiced_lines(led)
        bid_to_idx: dict = {}
        for i, l in enumerate(voiced):
            bid = l.get("beat_id")
            if bid is not None and bid not in bid_to_idx:
                bid_to_idx[bid] = i

        def _idx(ref: Any) -> Optional[int]:
            if ref is None:
                return None
            try:
                beats = outline.beats
                bidx = ref.beat_index
                if not (0 <= bidx < len(beats)):
                    return None
                return bid_to_idx.get(beats[bidx].beat_id)
            except Exception:  # noqa: BLE001
                return None

        return (
            _idx(getattr(outline, "turning_point", None)),
            _idx(getattr(outline, "button", None)),
        )
    except Exception:  # noqa: BLE001
        return None, None


def _unload_writer_llm(meta: dict) -> None:
    """Tear down the writer LLM (same call as the writer's default
    unload block, D8). Runs after the LLM passes (critic + editor) and
    before the deterministic scrub, so no model is resident for the
    cascade. Never raises (PD1)."""
    try:
        try:
            from . import _otr_writer_vram as _OTRVRAM
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_writer_vram as _OTRVRAM  # type: ignore
        meta["writer_llm_unload"] = _OTRVRAM.unload_writer_llm_after_script()
    except Exception as exc:  # noqa: BLE001
        meta["writer_llm_unload"] = f"error:{type(exc).__name__}"
        log.warning("[OTR_StorySpine] writer LLM unload failed: %r", exc)


def run_post_script_spine(
    led: Any,
    meta: dict,
    outline: Any,
    *,
    creative_generate_fn: Callable[..., str],
    technical_generate_fn: Callable[..., str],
    resolved: dict,
    slot_scheduler: Any = None,
) -> None:
    """Run the post-script spine in process, in flow order: Stage 2.5
    (conditional length pass) -> Stage 3 (Story QA router) -> [REJECT
    abort] / Stage 3.5 (micro-repair on flagged beats) -> writer-LLM
    unload -> Stage 4 (deterministic scrub).

    Called ONLY when `enabled()` is True. Mutates `led` (the editor
    applies its plan; the scrub normalizes in place) and stamps status
    keys on `meta`. NEVER RAISES -- on any pass failure it records the
    error on `meta` and continues, and it always performs the writer-LLM
    unload so VRAM is released before the cascade.

    On a REJECT verdict it sets meta["story_verdict"]="REJECT" +
    meta["story_reject_reason"], unloads, SKIPS the scrub, and returns
    normally -- the writer raises at its boundary; the spine never does.

    Slot routing (D6): QA router -> technical slot; length pass +
    micro-repair -> creative slot.
    """
    meta["story_spine_enabled"] = True

    # --- Stage 2.5: conditional length normalization (creative slot) ---
    # The Radio Editor's length pass runs FIRST and ONLY when the draft is
    # out of spec (over the word band OR any line over the spoken cap); a
    # clean draft skips it with no LLM call (go-forward Sprint 2). It owns
    # episode length / pacing via render-safe per-beat edits (Tier-1 tighten
    # + Tier-2 count change with needs_render_realign). NEVER RAISES (PD1).
    # LLM slot: creative -- narrative length / pacing pass.
    try:
        from . import _otr_radio_editor as _ED
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_radio_editor as _ED  # type: ignore
    try:
        led_data = getattr(led, "data", led)
        if _ED.needs_length_normalization(led_data):
            _turn_idx, _button_idx = _map_arc_indices(outline, led)
            _recompose = _make_recompose_fn(led, creative_generate_fn)
            ctx = (
                slot_scheduler.helper_context("radio_editor")
                if slot_scheduler is not None
                else _nullcontext()
            )
            with ctx:
                _length_plan, length_report = _ED.normalize_length(
                    led_data,
                    editor_model=resolved["creative_writing_model"],
                    slot_fn=creative_generate_fn,
                    recompose_fn=_recompose,
                    turn_beat_index=_turn_idx,
                    button_beat_index=_button_idx,
                    apply=True,
                )
            meta["length_pass_report"] = _editor_summary(length_report)
        else:
            meta["length_pass_report"] = {"status": "SKIPPED_IN_SPEC"}
    except Exception as exc:  # noqa: BLE001 -- length pass must never break a run
        meta["length_pass_report"] = {"status": "ERROR",
                                      "error": type(exc).__name__}
        log.warning("[OTR_StorySpine] length normalization failed: %r", exc)

    # --- Stage 3: Targeted Story QA router (read-only, technical slot) --
    # BUG-LOCAL-302: OPT-IN, DEFAULT OFF (operator directive 2026-06-01). This
    # is a credit-billed LLM call whose only action is to REJECT -- aborting a
    # fully-written, renderable episode and wasting the tokens already spent
    # composing it. So the QA LLM is GATED OFF and does NOT even run out of the
    # box: no call, no tokens, no reject. A power user sets OTR_ENABLE_STORY_QA=1
    # to turn the critic + reject gate back on. When on it is model-agnostic:
    # cold context (final script only), skeptical framing, high REJECT bar, and
    # fail-OPEN to PASS inside run_story_qa (a QA crash ships the episode as-is
    # rather than aborting it). LLM slot: technical -- structured verdict.
    verdict = None
    if not qa_enabled():
        meta["story_qa_verdict"] = {
            "verdict": "SKIPPED",
            "reason": ("OTR_ENABLE_STORY_QA not set -- QA LLM gated off by "
                       "default (BUG-LOCAL-302): no call, no tokens, no reject."),
        }
    else:
        try:
            from . import _otr_creative_qa as _QA
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_creative_qa as _QA  # type: ignore
        try:
            ctx = (
                slot_scheduler.helper_context("creative_qa")
                if slot_scheduler is not None
                else _nullcontext()
            )
            with ctx:
                verdict = _QA.run_story_qa(
                    led,
                    technical_generate_fn,
                    critic_model_id=resolved["technical_model"],
                )
            meta["story_qa_verdict"] = _verdict_summary(verdict)
        except Exception as exc:  # noqa: BLE001 -- QA must never break a run
            meta["story_qa_verdict"] = {"verdict": "ERROR",
                                        "error": type(exc).__name__}
            log.warning("[OTR_StorySpine] story QA failed: %r", exc)

    _verdict_value = getattr(verdict, "verdict", None)

    # --- REJECT abort (go-forward Sprint 3, spine side) ----------------
    # A structural defect a one-line edit cannot fix. Set the signal on
    # meta, unload the writer LLM, SKIP the scrub, and return NORMALLY --
    # NEVER raise (PD1, the spine's never-raises contract). The writer
    # raises at its boundary on this signal. A QA crash (verdict None or a
    # fail-open PASS) does NOT reach here, so a crash stays fail-soft.
    if _verdict_value == "REJECT":
        meta["story_verdict"] = "REJECT"
        meta["story_reject_reason"] = (
            getattr(verdict, "reason", "") or "story rejected"
        )
        log.warning("[OTR_StorySpine] story QA REJECT: %s",
                    meta["story_reject_reason"])
        _unload_writer_llm(meta)
        meta["story_spine_status"] = "ok_reject"
        return

    # --- Stage 3.5: beat-local micro-repair (creative slot) ------------
    # Only on MICRO_REPAIR_NEEDED with flagged beats; ONE cycle, flagged
    # beats only (spine invariant 5; the editor's scoped validator fences
    # it). flagged_beats are voiced-view indices, the SAME space the editor
    # uses (the QA router judges the voiced view). Arc indices + recompose
    # are recomputed here because the Stage 2.5 length pass may have
    # re-indexed beats. LLM slot: creative -- narrative editing.
    flagged = list(getattr(verdict, "flagged_beats", None) or [])
    if _verdict_value == "MICRO_REPAIR_NEEDED" and flagged:
        try:
            from . import _otr_radio_editor as _ED
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_radio_editor as _ED  # type: ignore
        try:
            led_data = getattr(led, "data", led)
            _turn_idx, _button_idx = _map_arc_indices(outline, led)
            _recompose = _make_recompose_fn(led, creative_generate_fn)
            ctx = (
                slot_scheduler.helper_context("radio_editor")
                if slot_scheduler is not None
                else _nullcontext()
            )
            with ctx:
                _mr_plan, mr_report = _ED.micro_repair(
                    led_data,
                    flagged,
                    editor_model=resolved["creative_writing_model"],
                    slot_fn=creative_generate_fn,
                    recompose_fn=_recompose,
                    turn_beat_index=_turn_idx,
                    button_beat_index=_button_idx,
                    apply=True,
                )
            meta["micro_repair_report"] = _editor_summary(mr_report)
        except Exception as exc:  # noqa: BLE001 -- micro-repair must never break a run
            meta["micro_repair_report"] = {"status": "ERROR",
                                           "error": type(exc).__name__}
            log.warning("[OTR_StorySpine] micro-repair failed: %r", exc)

    # --- Stage 3.6: deterministic mechanical anti-loop / dedupe (A4) -----
    # Story-quality Phase 1, UNCONDITIONAL repair-target source. The story
    # critic returns StoryCriticReport.clean() identically whether it
    # succeeded, found nothing, or silently failed, so a deterministic floor
    # is the only thing that guarantees real loop/dup defects get repaired.
    # CHARACTER near-duplicate + "What if...?" loop targets are recomposed
    # here via the creative slot (the one repair owner); the announcer
    # open/close taglines are EXEMPT and owned by the announcer composer
    # (A6). Runs BEFORE the writer-LLM unload so the creative slot is still
    # resident. No-LLM detection; recompose is the same fail-soft seam the
    # editor uses (any error degrades to KEEP, never corrupts a beat).
    try:
        from . import _otr_anti_loop as _AL
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_anti_loop as _AL  # type: ignore
    try:
        _al_recompose = _make_recompose_fn(led, creative_generate_fn)
        ctx = (
            slot_scheduler.helper_context("anti_loop")
            if slot_scheduler is not None
            else _nullcontext()
        )
        with ctx:
            _al_summary = _AL.repair_character_anti_loops(
                led, _voiced_lines(led), _al_recompose,
            )
        meta["anti_loop_report"] = _al_summary
    except Exception as exc:  # noqa: BLE001 -- anti-loop must never break a run
        meta["anti_loop_report"] = {"status": f"ERROR:{type(exc).__name__}"}
        log.warning("[OTR_StorySpine] anti-loop repair failed: %r", exc)

    # --- Stage 3.7: clean delivery -- scrub vs recompose (A6) -----------
    # Story-quality Phase 1. DETERMINISTIC SCRUB of spoken CHARACTER lines
    # (parenthetical stage-directions + the speaker's own-name vocative); and
    # TRUNCATION REPAIR BY RECOMPOSE (never token-surgery): a truncated
    # character line routes to the spine recompose seam, a truncated announcer
    # OPEN/CLOSE tagline routes to the DEDICATED announcer composer (the critic
    # excludes announcer lines, so a character reroll cannot act on them).
    # Runs before the writer-LLM unload so the creative slot is resident.
    # Fail-soft: any error degrades to KEEP, never corrupts a beat.
    try:
        from . import _otr_line_hygiene as _HY
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_line_hygiene as _HY  # type: ignore
    try:
        _hy_data = getattr(led, "data", led)
        _hy_cast = _hy_data.get("cast") or []
        _hy_voiced = _voiced_lines(led)
        _hy_recompose = _make_recompose_fn(led, creative_generate_fn)
        _hy_ann_idx = [i for i, ln in enumerate(_hy_voiced)
                       if ln.get("speaker_role") == "announcer"]
        _hy_first_ann = _hy_ann_idx[0] if _hy_ann_idx else -1
        _hy_last_ann = _hy_ann_idx[-1] if _hy_ann_idx else -1
        _hy_report = {"scrubbed": 0, "char_recomposed": 0,
                      "announcer_recomposed": 0, "narration_recomposed": 0,
                      "stage_direction_recomposed": 0}
        _hy_ctx = (
            slot_scheduler.helper_context("delivery_hygiene")
            if slot_scheduler is not None else _nullcontext()
        )
        with _hy_ctx:
            for _hy_i, _hy_line in enumerate(_hy_voiced):
                _hy_role = _hy_line.get("speaker_role")
                _hy_orig = str(_hy_line.get("text") or "")
                if _hy_role == "character":
                    _hy_name = (_cast_name(_hy_cast, _hy_line.get("char_id"))
                                or _hy_line.get("char_id") or "")
                    _hy_clean = _HY.clean_spoken_character_line(
                        _hy_orig, _hy_name)
                    if _hy_clean != _hy_orig and _hy_clean.strip():
                        _hy_line["text"] = _hy_clean
                        _hy_report["scrubbed"] += 1
                        _hy_orig = _hy_clean
                    # ROOT-CAUSE repair (2026-06-22): a character beat with NO
                    # spoken content -- entirely a stage direction like
                    # "(pauses, then flips the switch)" -- is KEPT by the scrub
                    # (it would otherwise empty the line), so it would leak to
                    # the voice worker and crash/silence it. RECOMPOSE it into
                    # real dialogue via the same seam truncation uses. Fail-soft:
                    # if the recompose still has no spoken content, keep the
                    # original (the per-line voice silence guard is the net).
                    if _HY.is_stage_direction_only(_hy_orig):
                        log.info(
                            "[OTR_StorySpine] stage-direction-only line %d "
                            "(%s); recomposing into spoken dialogue.",
                            _hy_i, _hy_name,
                        )
                        _hy_sd = _hy_recompose(
                            _hy_i, _hy_orig,
                            "this beat has NO spoken dialogue -- it is only a "
                            "stage direction; write the actual words the "
                            "character SAYS OUT LOUD here, with no parentheses "
                            "and no action description",
                        )
                        if (_hy_sd and str(_hy_sd).strip()
                                and not _HY.is_stage_direction_only(str(_hy_sd))):
                            _hy_line["text"] = str(_hy_sd)
                            _hy_orig = str(_hy_sd)
                            _hy_report["stage_direction_recomposed"] += 1
                    if _HY.is_truncated(_hy_orig):
                        _hy_new = _hy_recompose(
                            _hy_i, _hy_orig,
                            "finish the sentence; do not cut off mid-thought "
                            "or end on a dangling word",
                        )
                        if (_hy_new and str(_hy_new).strip()
                                and str(_hy_new) != _hy_orig):
                            _hy_line["text"] = str(_hy_new)
                            _hy_report["char_recomposed"] += 1
                    # F7 (story-engine v1): narration / self-address repair.
                    # One recompose attempt via the SAME seam; LOUD marker;
                    # fallback to the original (no change) on empty/identical.
                    _hy_cur = str(_hy_line.get("text") or "")
                    if _HY.detect_narration_self_address(_hy_cur, _hy_name):
                        log.info(
                            "[OTR_StorySpine] F7 narration/self-address at "
                            "line %d (%s); recomposing once.",
                            _hy_i, _hy_name,
                        )
                        _hy_nn = _hy_recompose(
                            _hy_i, _hy_cur,
                            "speak this line in the first person as the "
                            "character; do not narrate your own actions in "
                            "the third person and do not say your own name",
                        )
                        if (_hy_nn and str(_hy_nn).strip()
                                and str(_hy_nn) != _hy_cur):
                            _hy_line["text"] = str(_hy_nn)
                            _hy_report["narration_recomposed"] += 1
                elif _hy_role == "announcer" and _hy_i in (
                        _hy_first_ann, _hy_last_ann):
                    if _HY.is_truncated(_hy_orig):
                        _hy_fixed = _recompose_announcer_tagline(
                            led, meta, creative_generate_fn, resolved,
                            is_intro=(_hy_i == _hy_first_ann),
                            intro_text=str(
                                _hy_voiced[_hy_first_ann].get("text") or "")
                            if _hy_first_ann >= 0 else "",
                        )
                        if (_hy_fixed and _hy_fixed.strip()
                                and _hy_fixed != _hy_orig):
                            _hy_line["text"] = _hy_fixed
                            _hy_report["announcer_recomposed"] += 1
        meta["delivery_hygiene_report"] = _hy_report
    except Exception as exc:  # noqa: BLE001 -- hygiene must never break a run
        meta["delivery_hygiene_report"] = {
            "status": f"ERROR:{type(exc).__name__}"}
        log.warning("[OTR_StorySpine] delivery hygiene failed: %r", exc)

    # --- Writer-LLM unload (D8): after the LLM passes, before scrub -----
    _unload_writer_llm(meta)

    # --- Stage 4: deterministic Ledger Scrub (no LLM, LAST) ------------
    # repair_available=False: any beat-local micro-repair already ran
    # upstream, so the scrub never triggers a repair -- it is mechanical,
    # fail-closed, and the last word on the ledger (go-forward Sprint 4).
    try:
        from . import _otr_ledger_scrub as _SCRUB
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_ledger_scrub as _SCRUB  # type: ignore
    try:
        led_data = getattr(led, "data", led)
        result = _SCRUB.scrub_ledger(led_data, repair_available=False)
        meta["ledger_scrub_status"] = getattr(result, "status", "UNKNOWN")
        if getattr(result, "repair_consumed", False):
            meta["ledger_scrub_repair_consumed"] = True
    except Exception as exc:  # noqa: BLE001 -- scrub must never break a run
        meta["ledger_scrub_status"] = f"ERROR:{type(exc).__name__}"
        log.warning("[OTR_StorySpine] ledger scrub failed: %r", exc)

    meta["story_spine_status"] = "ok"


def _verdict_summary(verdict: Any) -> dict:
    """Compact, JSON-safe view of the story QA router verdict for meta."""
    keys = (
        "verdict", "flagged_beats", "reason", "dead_ending", "broken_turn",
        "flat_contrast", "unclear_grounding", "chopped_dialogue",
        "pacing_failure",
    )
    out: dict = {}
    for k in keys:
        try:
            out[k] = getattr(verdict, k)
        except Exception:  # noqa: BLE001
            pass
    return out


def _editor_summary(report: Any) -> dict:
    """Compact, JSON-safe view of the editor report for meta."""
    if isinstance(report, dict):
        keep = {}
        for k in ("status", "applied", "projected_word_total",
                  "actual_word_total", "needs_render_realign",
                  "tier1_edits", "tier2_edits", "repaired"):
            if k in report:
                keep[k] = report[k]
        return keep or {"status": "applied"}
    return {"status": "applied"}


class _nullcontext:
    """Minimal no-op context manager (slot_scheduler-absent fallback;
    avoids a contextlib import just for this)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Self-test: flag default-off no-op; flag-on runs the four passes with
# stubbed generate fns + a minimal ledger; never raises; ledger stays
# structurally valid. Zero network / GPU. Prints "SELF-TEST PASS: N/N".
# ---------------------------------------------------------------------------


def _selftest() -> int:
    import json as _json

    passed = 0
    total = 0

    class _Ledger:
        def __init__(self, data):
            self.data = data

    def _min_ledger():
        return _Ledger({
            "cast": [
                {"char_id": "c01", "name": "ANNOUNCER",
                 "voice_preset": "kokoro/af_heart", "speaker_role": "announcer"},
                {"char_id": "c02", "name": "ALICE",
                 "voice_preset": "v2/en_speaker_1", "speaker_role": "character"},
                {"char_id": "c03", "name": "BORIS",
                 "voice_preset": "v2/en_speaker_2", "speaker_role": "character"},
            ],
            "lines": [
                {"line_id": "L001", "beat_id": "b001", "char_id": "c01",
                 "speaker_role": "announcer", "text": "Welcome to the broadcast.",
                 "word_count": 4},
                {"line_id": "L002", "beat_id": "b002", "char_id": "c02",
                 "speaker_role": "character", "text": "The signal is getting stronger.",
                 "word_count": 5},
                {"line_id": "L003", "beat_id": "b003", "char_id": "c03",
                 "speaker_role": "character", "text": "Then we answer it tonight.",
                 "word_count": 5},
                {"line_id": "L004", "beat_id": "b004", "char_id": "c01",
                 "speaker_role": "announcer", "text": "Stay tuned.", "word_count": 2},
            ],
            "meta": {},
        })

    def _stub_generate(verdict_json):
        def _fn(messages, *, temperature, max_new_tokens, stop=None):
            return verdict_json
        return _fn

    pass_verdict = _json.dumps({
        "verdict": "PASS", "flagged_beats": [], "reason": "clean",
        "dead_ending": False, "broken_turn": False, "flat_contrast": False,
        "unclear_grounding": False, "chopped_dialogue": False,
        "pacing_failure": False,
    })

    # A no-op KEEP RadioEditPlan, in band -- the creative slot returns this
    # when the Stage 2.5 length pass runs on the short _min_ledger (which is
    # under band, so the pass fires); it validates + applies as a no-op.
    keep_plan_json = _json.dumps({
        "edits": [{"beat_index": 0, "action": "KEEP"}],
        "projected_word_total": 350,
    })

    resolved = {"technical_model": "test/tech", "creative_writing_model": "test/creative"}

    # Test 1: flag UNSET -> default ON (out-of-the-box best run).
    total += 1
    os.environ.pop(_ENV_FLAG, None)
    if enabled():
        passed += 1
        print("  [PASS] flag unset -> enabled() True (default on)")
    else:
        print("  [FAIL] flag unset should be enabled (default on)")

    # Test 2: flag "0" -> the explicit opt-out escape hatch.
    total += 1
    os.environ[_ENV_FLAG] = "0"
    if not enabled():
        passed += 1
        print("  [PASS] flag '0' -> enabled() False (opt-out)")
    else:
        print("  [FAIL] flag '0' should be disabled")

    # Test 3: flag "1" -> on.
    total += 1
    os.environ[_ENV_FLAG] = "1"
    if enabled():
        passed += 1
        print("  [PASS] flag '1' -> enabled() True")
    else:
        print("  [FAIL] flag '1' should be enabled")

    # BUG-LOCAL-302: the Story QA pass is opt-in (default off). Tests 4/5/9/10
    # exercise it, so turn it on for them; Test 11 covers the default-off skip.
    os.environ[_QA_ENV_FLAG] = "1"

    # Test 4: full PASS path runs, stamps meta, never raises, ledger valid.
    total += 1
    try:
        led = _min_ledger()
        meta = led.data["meta"]
        run_post_script_spine(
            led, meta, outline=None,
            creative_generate_fn=_stub_generate(keep_plan_json),
            technical_generate_fn=_stub_generate(pass_verdict),
            resolved=resolved, slot_scheduler=None,
        )
        ok = (
            meta.get("story_spine_status") == "ok"
            and meta.get("story_spine_enabled") is True
            and "length_pass_report" in meta
            and "story_qa_verdict" in meta
            and "ledger_scrub_status" in meta
            and "writer_llm_unload" in meta
            and isinstance(led.data["lines"], list)
            and len(led.data["lines"]) == 4
        )
        # JSON round-trip proves the stamped meta is serializable.
        _json.dumps(meta)
        if ok:
            passed += 1
            print("  [PASS] PASS verdict -> 4 passes ran, meta stamped, "
                  "ledger intact")
        else:
            print(f"  [FAIL] PASS path meta = {meta}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] PASS path raised: {exc!r}")

    # Test 5: a raising critic slot fn must NOT break the run (PD1).
    total += 1
    try:
        led = _min_ledger()
        meta = led.data["meta"]

        def _boom(messages, *, temperature, max_new_tokens, stop=None):
            raise RuntimeError("loader OOM")

        run_post_script_spine(
            led, meta, outline=None,
            creative_generate_fn=_boom, technical_generate_fn=_boom,
            resolved=resolved, slot_scheduler=None,
        )
        # The router is fail-OPEN: a raising slot fn is caught inside
        # run_story_qa and returned as verdict="PASS", so the spine ships
        # the episode as-is (no micro-repair, no abort), the scrub still
        # runs, and the run stays intact (PD1 -- a QA crash never breaks it).
        if (meta.get("story_spine_status") == "ok"
                and meta.get("story_qa_verdict", {}).get("verdict")
                in ("PASS", "ERROR")
                and "ledger_scrub_status" in meta
                and "micro_repair_report" not in meta
                and "story_verdict" not in meta):
            passed += 1
            print("  [PASS] raising critic -> fail-open PASS (no repair, no "
                  "abort), scrub still ran, run intact")
        else:
            print(f"  [FAIL] raising-critic meta = {meta}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] raising critic propagated: {exc!r}")

    # Test 6: _map_arc_indices maps the outline arc refs by beat_id into
    # the voiced view, and is None-safe.
    total += 1
    try:
        class _Ref:
            def __init__(self, bi):
                self.beat_index = bi

        class _B:
            def __init__(self, bid):
                self.beat_id = bid

        class _OL:
            beats = [_B("b001"), _B("b002"), _B("b003"), _B("b004")]
            turning_point = _Ref(2)   # b003
            button = _Ref(3)          # b004

        led6 = _min_ledger()
        t, b = _map_arc_indices(_OL(), led6)
        if t == 2 and b == 3 and _map_arc_indices(None, led6) == (None, None):
            passed += 1
            print("  [PASS] _map_arc_indices maps by beat_id; None-safe")
        else:
            print(f"  [FAIL] arc map -> t={t} b={b}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] arc map raised: {exc!r}")

    # Test 7: _make_recompose_fn is fail-soft -- a bad index or a raising
    # creative fn returns the ORIGINAL line, never raises, never corrupts.
    total += 1
    try:
        led7 = _min_ledger()

        def _raise_fn(messages, *, temperature, max_new_tokens, stop=None):
            raise RuntimeError("no model")

        rc = _make_recompose_fn(led7, _raise_fn)
        out_bad = rc(999, "ORIGINAL", "tighten")
        out_raise = rc(1, "ORIGINAL", "tighten")
        if out_bad == "ORIGINAL" and out_raise == "ORIGINAL":
            passed += 1
            print("  [PASS] recompose fail-soft: bad index + raising fn -> original")
        else:
            print(f"  [FAIL] recompose bad={out_bad!r} raise={out_raise!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] recompose raised: {exc!r}")

    # Test 8: Stage 2.5 wiring -- an in-spec draft SKIPS the length pass
    # with NO creative-slot LLM call (the conditional wiring), and the run
    # still completes and stamps meta.
    total += 1
    try:
        os.environ[_ENV_FLAG] = "1"
        _filler = " ".join(["word"] * 28)
        _in_spec_lines = [
            {"line_id": "L%02d" % k, "beat_id": "b%02d" % k, "char_id": "c02",
             "speaker_role": "character", "text": _filler, "word_count": 28}
            for k in range(12)
        ]
        led8 = _Ledger({
            "cast": [{"char_id": "c02", "name": "ALICE",
                      "voice_preset": "v2/en_speaker_1",
                      "speaker_role": "character"}],
            "lines": _in_spec_lines, "meta": {},
        })
        meta8 = led8.data["meta"]

        def _boom_creative(messages, *, temperature, max_new_tokens, stop=None):
            raise AssertionError(
                "length pass must not call the creative LLM on an in-spec draft")

        run_post_script_spine(
            led8, meta8, outline=None,
            creative_generate_fn=_boom_creative,
            technical_generate_fn=_stub_generate(pass_verdict),
            resolved=resolved, slot_scheduler=None,
        )
        if (meta8.get("story_spine_status") == "ok"
                and meta8.get("length_pass_report", {}).get("status")
                == "SKIPPED_IN_SPEC"):
            passed += 1
            print("  [PASS] in-spec draft skips the length pass (no LLM call)")
        else:
            print(f"  [FAIL] skip-path length_pass_report = "
                  f"{meta8.get('length_pass_report')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] skip-path test raised: {exc!r}")

    # Test 9: MICRO_REPAIR_NEEDED verdict -> beat-local micro-repair runs
    # on the flagged beats, the scrub still runs, no REJECT signal.
    total += 1
    try:
        os.environ[_ENV_FLAG] = "1"
        led9 = _min_ledger()
        meta9 = led9.data["meta"]
        qa_micro_json = _json.dumps({
            "verdict": "MICRO_REPAIR_NEEDED", "flagged_beats": [1],
            "reason": "beat 1 reads chopped", "dead_ending": False,
            "broken_turn": False, "flat_contrast": False,
            "unclear_grounding": False, "chopped_dialogue": True,
            "pacing_failure": False,
        })
        # A KEEP on the flagged beat 1 -- valid for BOTH the Stage 2.5
        # length pass (any beat) AND the scoped micro-repair (flagged only).
        keep_beat1_json = _json.dumps({
            "edits": [{"beat_index": 1, "action": "KEEP"}],
            "projected_word_total": 350,
        })
        run_post_script_spine(
            led9, meta9, outline=None,
            creative_generate_fn=_stub_generate(keep_beat1_json),
            technical_generate_fn=_stub_generate(qa_micro_json),
            resolved=resolved, slot_scheduler=None,
        )
        if (meta9.get("story_spine_status") == "ok"
                and meta9.get("story_qa_verdict", {}).get("verdict")
                == "MICRO_REPAIR_NEEDED"
                and "micro_repair_report" in meta9
                and "ledger_scrub_status" in meta9
                and "story_verdict" not in meta9):
            passed += 1
            print("  [PASS] MICRO_REPAIR_NEEDED -> micro-repair ran, scrub ran, "
                  "no abort")
        else:
            print(f"  [FAIL] micro-repair path meta = {meta9}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] micro-repair path raised: {exc!r}")

    # Test 10: REJECT verdict -> abort at the writer boundary. The spine
    # sets the signal, unloads, SKIPS the scrub, returns ok_reject (never
    # raises). The writer is what raises on this signal.
    total += 1
    try:
        os.environ[_ENV_FLAG] = "1"
        led10 = _min_ledger()
        meta10 = led10.data["meta"]
        qa_reject_json = _json.dumps({
            "verdict": "REJECT", "flagged_beats": [], "reason": "dead ending",
            "dead_ending": True, "broken_turn": False, "flat_contrast": False,
            "unclear_grounding": False, "chopped_dialogue": False,
            "pacing_failure": False,
        })
        keep_beat1_json = _json.dumps({
            "edits": [{"beat_index": 1, "action": "KEEP"}],
            "projected_word_total": 350,
        })
        run_post_script_spine(
            led10, meta10, outline=None,
            creative_generate_fn=_stub_generate(keep_beat1_json),
            technical_generate_fn=_stub_generate(qa_reject_json),
            resolved=resolved, slot_scheduler=None,
        )
        if (meta10.get("story_verdict") == "REJECT"
                and meta10.get("story_reject_reason")
                and meta10.get("story_spine_status") == "ok_reject"
                and "ledger_scrub_status" not in meta10
                and "writer_llm_unload" in meta10):
            passed += 1
            print("  [PASS] REJECT -> signal set, unloaded, scrub skipped, "
                  "returned (no raise)")
        else:
            print(f"  [FAIL] reject path meta = {meta10}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] reject path raised: {exc!r}")

    # Test 11: QA gated OFF by default (BUG-LOCAL-302). The QA LLM must NOT be
    # called (a raising technical fn proves it would have), the verdict is
    # SKIPPED, there is no REJECT signal, and the run still completes + scrubs.
    total += 1
    try:
        os.environ[_ENV_FLAG] = "1"
        os.environ.pop(_QA_ENV_FLAG, None)  # default: QA off
        led11 = _min_ledger()
        meta11 = led11.data["meta"]

        def _qa_must_not_run(messages, *, temperature, max_new_tokens, stop=None):
            raise AssertionError(
                "QA LLM must not be called when OTR_ENABLE_STORY_QA is unset")

        keep_plan = _json.dumps({
            "edits": [{"beat_index": 0, "action": "KEEP"}],
            "projected_word_total": 350,
        })
        run_post_script_spine(
            led11, meta11, outline=None,
            creative_generate_fn=_stub_generate(keep_plan),
            technical_generate_fn=_qa_must_not_run,
            resolved=resolved, slot_scheduler=None,
        )
        if (meta11.get("story_spine_status") == "ok"
                and meta11.get("story_qa_verdict", {}).get("verdict") == "SKIPPED"
                and "story_verdict" not in meta11
                and "ledger_scrub_status" in meta11):
            passed += 1
            print("  [PASS] QA gated off by default -> no QA call, SKIPPED, "
                  "no reject, scrub ran")
        else:
            print(f"  [FAIL] QA-default-off meta = {meta11}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] QA-default-off raised: {exc!r}")

    os.environ.pop(_ENV_FLAG, None)
    os.environ.pop(_QA_ENV_FLAG, None)
    print(f"SELF-TEST PASS: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

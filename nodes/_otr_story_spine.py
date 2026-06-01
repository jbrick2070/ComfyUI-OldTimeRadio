"""nodes/_otr_story_spine.py -- Wave 2 post-script orchestrator (env-gated).

This is the in-process WIRING that runs the story-spine post-script
passes -- Stage 3 (Creative QA critic), Stage 3.5 (the single Radio
Editor repair), the writer-LLM unload, and Stage 4 (deterministic
Ledger Scrub) -- inside OTR_LedgerScriptWriter.run(), mirroring how
run_story_brief_reflection is called (`_otr_story_brief.py:861`).

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
script it already had. The single shared repair (spine invariant 5) is
tracked here: a REPAIR_ONCE verdict spends it on the editor; the scrub
is then told `repair_available=False`.

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
    """Run Stage 3 (critic) -> 3.5 (single editor repair) -> unload ->
    Stage 4 (scrub) in process, in that order.

    Called ONLY when `enabled()` is True. Mutates `led` (the editor
    applies its plan; the scrub normalizes in place) and stamps status
    keys on `meta`. NEVER RAISES -- on any pass failure it records the
    error on `meta` and continues, and it always performs the writer-LLM
    unload so VRAM is released before the cascade.

    Slot routing (D6): critic -> technical slot; editor -> creative slot.
    """
    meta["story_spine_enabled"] = True
    repair_used = False

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

    # --- Stage 3: Creative QA critic (read-only, technical slot) -------
    # LLM slot: technical -- structured categorical verdict.
    verdict = None
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
            verdict = _QA.run_creative_qa(
                led,
                technical_generate_fn,
                critic_model_id=resolved["technical_model"],
            )
        meta["creative_qa_verdict"] = _verdict_summary(verdict)
    except Exception as exc:  # noqa: BLE001 -- critic must never break a run
        meta["creative_qa_verdict"] = {"verdict": "ERROR",
                                       "error": type(exc).__name__}
        log.warning("[OTR_StorySpine] creative QA failed: %r", exc)

    # --- Stage 3.5: the single Radio Editor repair (creative slot) -----
    # Only on a recoverable REPAIR_ONCE verdict; one cycle, then stop
    # (spine invariant 5). LLM slot: creative -- narrative editing.
    if verdict is not None and getattr(verdict, "verdict", None) == "REPAIR_ONCE":
        try:
            from . import _otr_radio_editor as _ED
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_radio_editor as _ED  # type: ignore
        try:
            led_data = getattr(led, "data", led)
            ctx = (
                slot_scheduler.helper_context("radio_editor")
                if slot_scheduler is not None
                else _nullcontext()
            )
            _turn_idx, _button_idx = _map_arc_indices(outline, led)
            _recompose = _make_recompose_fn(led, creative_generate_fn)
            with ctx:
                _plan, report = _ED.run_radio_editor(
                    led_data,
                    editor_model=resolved["creative_writing_model"],
                    slot_fn=creative_generate_fn,
                    recompose_fn=_recompose,
                    turn_beat_index=_turn_idx,
                    button_beat_index=_button_idx,
                    apply=True,
                )
            repair_used = True
            meta["radio_editor_report"] = _editor_summary(report)
        except Exception as exc:  # noqa: BLE001 -- editor must never break a run
            meta["radio_editor_report"] = {"status": "ERROR",
                                           "error": type(exc).__name__}
            log.warning("[OTR_StorySpine] radio editor repair failed: %r", exc)

    # --- Writer-LLM unload (D8): after the LLM passes, before scrub -----
    _unload_writer_llm(meta)

    # --- Stage 4: deterministic Ledger Scrub (no LLM) ------------------
    try:
        from . import _otr_ledger_scrub as _SCRUB
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_ledger_scrub as _SCRUB  # type: ignore
    try:
        led_data = getattr(led, "data", led)
        result = _SCRUB.scrub_ledger(led_data, repair_available=not repair_used)
        meta["ledger_scrub_status"] = getattr(result, "status", "UNKNOWN")
        if getattr(result, "repair_consumed", False):
            meta["ledger_scrub_repair_consumed"] = True
    except Exception as exc:  # noqa: BLE001 -- scrub must never break a run
        meta["ledger_scrub_status"] = f"ERROR:{type(exc).__name__}"
        log.warning("[OTR_StorySpine] ledger scrub failed: %r", exc)

    meta["story_spine_status"] = "ok"


def _verdict_summary(verdict: Any) -> dict:
    """Compact, JSON-safe view of the critic verdict for meta."""
    keys = (
        "verdict", "has_turn", "ending_earned", "grounded_in_premise",
        "voices_distinct", "weakest_beat_index", "sfw_ok",
        "overlong_line_indices", "cast_name_leak_indices",
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
        "has_turn": True, "turn_beat_index": 0, "ending_earned": True,
        "ending_note": "lands", "grounded_in_premise": True,
        "voices_distinct": True, "weakest_beat_index": None,
        "weakest_problem": "", "overlong_line_indices": [],
        "cast_name_leak_indices": [], "sfw_ok": True, "verdict": "PASS",
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
            and "creative_qa_verdict" in meta
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
        # The critic is itself fail-closed: a raising slot fn is caught
        # inside run_creative_qa and returned as verdict="FAIL", so the
        # orchestrator sees FAIL (editor must NOT run), not "ERROR".
        if (meta.get("story_spine_status") == "ok"
                and meta.get("creative_qa_verdict", {}).get("verdict")
                in ("FAIL", "ERROR")
                and "ledger_scrub_status" in meta
                and "radio_editor_report" not in meta):
            passed += 1
            print("  [PASS] raising critic -> fail-closed (no editor), "
                  "scrub still ran, run intact")
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

    os.environ.pop(_ENV_FLAG, None)
    print(f"SELF-TEST PASS: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

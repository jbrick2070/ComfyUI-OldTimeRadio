"""nodes/_otr_story_spine.py -- Wave 2 post-script orchestrator (env-gated).

This is the in-process WIRING that runs the story-spine post-script
passes -- Stage 3 (Creative QA critic), Stage 3.5 (the single Radio
Editor repair), the writer-LLM unload, and Stage 4 (deterministic
Ledger Scrub) -- inside OTR_LedgerScriptWriter.run(), mirroring how
run_story_brief_reflection is called (`_otr_story_brief.py:861`).

GATED, DEFAULT OFF. `enabled()` reads `OTR_ENABLE_STORY_SPINE` and is
False unless it is exactly "1". When the flag is off the writer takes
its existing path (the unload block at the original call site), so the
default pipeline is byte-identical and every headless regression stays
green. When the flag is on -- an operator's deliberate GPU validation
run -- the four passes run here. This is the same opt-in/default-off
pattern the OpenRouter remote-LLM feature shipped with.

In-process, not a node (D4/D5): no INPUT_TYPES, no widget, no broadcast
output, no workflow-JSON change. The model ids arrive as the writer's
in-process `resolved[...]` values (critic -> technical slot per D6;
editor -> creative slot). PD3 (workflow JSON): N/A; adds no node surface.

NEVER RAISES (Prime Directive 1, audio is king). Every pass is wrapped;
a failure stamps a status on `meta` and the run continues with the
script it already had. The single shared repair (spine invariant 5) is
tracked here: a REPAIR_ONCE verdict spends it on the editor; the scrub
is then told `repair_available=False`.

Two refinements are deliberately deferred to the GPU-validation
follow-up and are SAFE no-ops until then (both contained by the
default-off flag):
  * recompose_fn is a no-op (returns the original line). The editor's
    seven other actions (KEEP / SHORTEN_LINE / CLEAN_PUNCTUATION /
    CUT_LINE / REMOVE_REDUNDANT_BEAT / SPLIT_LINE / MERGE_SHORT_LINES)
    all run; only RECOMPOSE_BEAT_SAME_INTENT degrades to KEEP rather
    than risk a mis-built LineRequest -- a flat beat stays as-is, never
    corrupted. Production recompose wraps `_otr_line_composer.compose_line`.
  * turn_beat_index / button_beat_index are passed None (the editor's
    structural + length + visual-noun guards still prevent any render-
    contract corruption; only the "Tier-2 removals never drop the arc
    beat" check is relaxed). The Stream A outline already carries
    `turning_point` / `button`; mapping their all-beats index into the
    editor's voiced-view index space is the follow-up.

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

    Default OFF: only an exact "1" enables it. Any other value (unset,
    "0", "", "true", "yes") leaves the writer on its byte-identical
    default path. Cheap, pure, never raises.
    """
    try:
        return os.environ.get(_ENV_FLAG, "0").strip() == "1"
    except Exception:  # noqa: BLE001 -- env read must never break a run
        return False


def _noop_recompose(beat_index: int, original_text: str, hint: str) -> str:
    """SAFE placeholder recompose seam (see module docstring).

    Returns the original line unchanged, so RECOMPOSE_BEAT_SAME_INTENT
    degrades to KEEP -- the beat is never corrupted. Production wiring
    replaces this with a `_otr_line_composer.compose_line` wrapper.
    """
    return original_text


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
            with ctx:
                _plan, report = _ED.run_radio_editor(
                    led_data,
                    editor_model=resolved["creative_writing_model"],
                    slot_fn=creative_generate_fn,
                    recompose_fn=_noop_recompose,
                    turn_beat_index=None,
                    button_beat_index=None,
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

    resolved = {"technical_model": "test/tech", "creative_writing_model": "test/creative"}

    # Test 1: flag OFF -> enabled() False.
    total += 1
    os.environ.pop(_ENV_FLAG, None)
    if not enabled():
        passed += 1
        print("  [PASS] flag unset -> enabled() False")
    else:
        print("  [FAIL] flag unset should be disabled")

    # Test 2: flag "0" -> still off.
    total += 1
    os.environ[_ENV_FLAG] = "0"
    if not enabled():
        passed += 1
        print("  [PASS] flag '0' -> enabled() False")
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
            creative_generate_fn=_stub_generate(pass_verdict),
            technical_generate_fn=_stub_generate(pass_verdict),
            resolved=resolved, slot_scheduler=None,
        )
        ok = (
            meta.get("story_spine_status") == "ok"
            and meta.get("story_spine_enabled") is True
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

    os.environ.pop(_ENV_FLAG, None)
    print(f"SELF-TEST PASS: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

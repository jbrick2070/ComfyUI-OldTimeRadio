"""tests/test_lfc_phase_3_polish_in_cascade.py

LFC sprint Commit 4 -- regression coverage for Phase 3 (per-line
polish) inside the cascade orchestrator. The composer-side polish
flow (compose_line + polish_line) is tested separately in
test_lfc_polish_fixes.py and test_phase1_composer_prompt.py. This
file pins:

  * Phase 3 disabled by default -- no-op, no mutations.
  * Phase 3 enabled -- only leaky lines are touched.
  * Phase 3 enabled -- announcer beats are skipped by default
    (polish_announcer_beats=False).
  * Phase 3 enabled with polish_announcer_beats=True -- announcer
    leaks are polished.
  * Mutation policy: text + word_count + char_count update in
    lockstep; skip / char_id / speaker_role untouched.
  * Refusal output keeps pre-polish text.
  * Polish-still-leaky output keeps pre-polish text.
  * polish_generate_fn is routed through when provided.
  * meta.phase_3_polish_record captures examined / applied /
    rejected counts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(lines, cast=None):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_phase3_test",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "voice_card": "ALICE (calm engineer)"},
                {"char_id": "c02", "name": "BOB",
                 "voice_card": "BOB (skeptical analyst)"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "voice_card": "ANNOUNCER (warm broadcast)"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "music": [], "clips": [],
            "meta": {"episode_title": "phase 3 test",
                     "style": "mission_control_procedural"},
        },
        episode_id="ep_phase3_test",
    )


def _line(line_id, char_id, role, text, *, skip=False, beat_intent=""):
    return {
        "line_id": line_id, "char_id": char_id,
        "speaker_role": role, "text": text,
        "skip": skip,
        "char_count": len(text), "word_count": len(text.split()),
        "beat_intent": beat_intent,
    }


def _polish_fn(replies):
    """Return a generate_fn-shaped callable that returns replies in
    order. Each reply is the raw LLM output (will be stripped /
    rejected by polish_line).
    """
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens, stop=None):
        i = state["i"]
        state["i"] += 1
        return replies[i] if i < len(replies) else (replies[-1] if replies else "")

    return fn


# ---------------------------------------------------------------------------
# Disabled-by-default contract
# ---------------------------------------------------------------------------


class TestPhase3DisabledByDefault:
    def test_disabled_does_not_call_generate_fn(self):
        called = {"n": 0}

        def boom(messages, *, temperature, max_new_tokens, stop=None):
            called["n"] += 1
            return "should never be called"

        # Leaky line that would normally trip polish.
        led = _ledger([
            _line("l001", "c01", "character",
                  '"That can\'t be right," she whispered.'),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(boom, led, enable=False)
        assert called["n"] == 0
        assert rep.edits_applied == 0
        assert led.data["lines"][0]["text"] == (
            '"That can\'t be right," she whispered.'
        )

    def test_disabled_stamps_zero_record(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        _LFC_ORCH._phase_3_per_line_polish(
            lambda *a, **k: "", led, enable=False,
        )
        rec = led.data["meta"]["phase_3_polish_record"]
        assert rec["lines_scanned"] == 0
        assert rec["edits_applied"] == 0
        assert rec["lines_examined"] == 0


# ---------------------------------------------------------------------------
# Enabled -- mutation policy
# ---------------------------------------------------------------------------


class TestPhase3EnabledMutations:
    def test_clean_line_not_examined(self):
        called = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            called["n"] += 1
            return "should not fire"

        led = _ledger([
            _line("l001", "c01", "character", "Then we should go."),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(fn, led, enable=True)
        assert called["n"] == 0
        assert rep.edits_examined if hasattr(rep, "edits_examined") else (
            rep.lines_examined == 0
        )

    def test_leaky_line_polished_and_text_replaced(self):
        led = _ledger([
            _line("l001", "c01", "character",
                  '"It was rigged," she said.'),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn(["I told you it was rigged."]),
            led, enable=True,
        )
        assert rep.lines_examined == 1
        assert rep.edits_applied == 1
        assert led.data["lines"][0]["text"] == "I told you it was rigged."
        # Word + char counts re-stamped in lockstep.
        assert led.data["lines"][0]["word_count"] == 6
        assert led.data["lines"][0]["char_count"] == len(
            "I told you it was rigged."
        )

    def test_announcer_skipped_by_default(self):
        called = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            called["n"] += 1
            return "should not fire"

        led = _ledger([
            _line("l001", "announcer", "announcer",
                  "Tonight, [pauses dramatically] on Signal Lost."),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(fn, led, enable=True)
        assert called["n"] == 0
        assert rep.lines_examined == 0
        # Text unchanged.
        assert "[pauses dramatically]" in led.data["lines"][0]["text"]

    def test_announcer_polished_when_opt_in(self):
        led = _ledger([
            _line("l001", "announcer", "announcer",
                  "Tonight, [pauses] on Signal Lost."),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn(["Tonight, on Signal Lost."]),
            led, enable=True, polish_announcer_beats=True,
        )
        assert rep.lines_examined == 1
        assert rep.edits_applied == 1
        assert "[pauses]" not in led.data["lines"][0]["text"]

    def test_skipped_lines_not_polished(self):
        called = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            called["n"] += 1
            return "should not fire"

        led = _ledger([
            _line("l001", "c01", "character",
                  '"Leaky," she said.', skip=True),
        ])
        # Even though the text trips needs_polish, skip=True means
        # the line is already muted -- polish_line wouldn't help.
        _LFC_ORCH._phase_3_per_line_polish(fn, led, enable=True)
        assert called["n"] == 0

    def test_non_voiced_lines_not_polished(self):
        called = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens, stop=None):
            called["n"] += 1
            return "x"

        # sfx / music beats are by-definition non-dialogue. Even
        # leaky-looking text isn't a polish candidate.
        led = _ledger([
            _line("l001", "sfx", "sfx", "[door slam] *crunch*"),
            _line("l002", "music", "music_open", "(rising horn theme)"),
        ])
        _LFC_ORCH._phase_3_per_line_polish(fn, led, enable=True)
        assert called["n"] == 0

    def test_word_char_counts_lockstep(self):
        led = _ledger([
            _line("l001", "c01", "character",
                  '*beat* He said: "I knew it."'),
        ])
        replacement = "I knew it all along."
        _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn([replacement]),
            led, enable=True,
        )
        assert led.data["lines"][0]["text"] == replacement
        assert led.data["lines"][0]["char_count"] == len(replacement)
        assert led.data["lines"][0]["word_count"] == 5


# ---------------------------------------------------------------------------
# Rejection policy
# ---------------------------------------------------------------------------


class TestPhase3Rejection:
    def test_refusal_keeps_original_text(self):
        original = '"I cannot wait," she whispered.'
        led = _ledger([_line("l001", "c01", "character", original)])
        rep = _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn(["I cannot rewrite this."]),
            led, enable=True,
        )
        assert led.data["lines"][0]["text"] == original
        assert rep.edits_applied == 0
        assert rep.edits_rejected_refusal >= 1

    def test_still_leaky_keeps_original_text(self):
        original = '"It was him," she said.'
        led = _ledger([_line("l001", "c01", "character", original)])
        # Polish output still trips the leak gate. The "he whispered"
        # narration-verb survives strip_line_formatting and trips
        # needs_polish again.
        rep = _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn(["It was him, he whispered."]),
            led, enable=True,
        )
        assert led.data["lines"][0]["text"] == original
        assert rep.edits_applied == 0
        assert rep.edits_rejected_still_leaky >= 1

    def test_polish_raise_recorded_as_failure(self):
        def boom(messages, *, temperature, max_new_tokens, stop=None):
            raise RuntimeError("simulated LLM crash")

        led = _ledger([
            _line("l001", "c01", "character", '"Leaky," she said.'),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(boom, led, enable=True)
        # polish_line itself catches the raise and returns the
        # original; phase_3 treats that as a rejection-refusal-style
        # outcome (text unchanged).
        assert led.data["lines"][0]["text"] == '"Leaky," she said.'
        assert rep.edits_applied == 0


# ---------------------------------------------------------------------------
# polish_generate_fn routing
# ---------------------------------------------------------------------------


class TestPhase3PolishGenerateFnRouting:
    def test_polish_generate_fn_preferred(self):
        calls = {"main": 0, "polish": 0}

        def main_fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["main"] += 1
            return "main"

        def polish_fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["polish"] += 1
            return "polished output."

        led = _ledger([
            _line("l001", "c01", "character",
                  '"Leaky," she said.'),
        ])
        _LFC_ORCH._phase_3_per_line_polish(
            main_fn, led, enable=True, polish_generate_fn=polish_fn,
        )
        assert calls["main"] == 0
        assert calls["polish"] == 1
        assert led.data["lines"][0]["text"] == "polished output."


# ---------------------------------------------------------------------------
# Phase-record stamping (meta.phase_3_polish_record)
# ---------------------------------------------------------------------------


class TestPhase3RecordStamping:
    def test_record_carries_full_counts(self):
        led = _ledger([
            _line("l001", "c01", "character", "Clean line."),
            _line("l002", "c01", "character", '"Leaky," she said.'),
            _line("l003", "c02", "character", '[*sighs*] Yes.'),
            _line("l004", "announcer", "announcer",
                  "Tonight [whispers] on the show."),
        ])
        rep = _LFC_ORCH._phase_3_per_line_polish(
            _polish_fn([
                "I told you so.",       # accepts
                "I cannot rewrite this.",  # refusal -> reject
            ]),
            led, enable=True,
        )
        rec = led.data["meta"]["phase_3_polish_record"]
        assert rec["lines_scanned"] == 4
        # Three voiced beats, one is clean, one is announcer (skipped
        # by default) so two were examined.
        assert rec["lines_examined"] == 2
        assert rec["edits_applied"] == 1
        # Refusal recorded.
        assert rec["edits_rejected_refusal"] >= 1
        assert rep.lines_scanned == rec["lines_scanned"]


# ---------------------------------------------------------------------------
# Cascade integration -- enable flag flows through run_freeze_cascade
# ---------------------------------------------------------------------------


class TestCascadeIntegrationFlag:
    def test_cascade_default_runs_phase_3_disabled(self):
        # Default cascade call -- enable_phase_3_polish defaults to
        # False. The cascade still STAMPS phase_3_polish_record so
        # soak diagnostics see the no-op trace, but no LLM calls fire.
        from unittest.mock import patch
        from nodes import _otr_ledger_reviewer as _OTRLR

        called = {"polish": 0}

        def boom_polish(messages, *, temperature, max_new_tokens, stop=None):
            called["polish"] += 1
            return "should never fire"

        led = _ledger([
            _line("l001", "c01", "character",
                  '"Leaky," she said.'),
        ])

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(boom_polish, led)

        assert called["polish"] == 0
        # Record stamped, phase ran as no-op.
        assert "phase_3_polish_record" in led.data["meta"]
        # Phase 3 entry present in cleanup_passes bucket
        # (B6 split, 2026-05-12).
        passes = led.data["meta"]["cleanup_passes"]
        names = [p["phase_name"] for p in passes]
        assert "phase_3_per_line_polish" in names

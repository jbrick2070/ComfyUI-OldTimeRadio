"""tests/test_lfc_phase_5_voice_drift.py

LFC sprint Commit 9 -- regression coverage for Phase 5 per-speaker
voice drift detection + targeted rewrites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_phase_5_voice_drift as _P5  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line(line_id, char_id, text, role="character"):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger(lines, cast=None):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_p5_test",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "traits": "calm engineer"},
                {"char_id": "c02", "name": "BOB",
                 "traits": "skeptical analyst"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "traits": "warm broadcast"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "sfx": [], "music": [], "clips": [],
            "meta": {
                "episode_title": "p5 test",
                "style": "mission_control_procedural",
            },
        },
        episode_id="ep_p5_test",
    )


def _two_step_replies(reasoning_text, formatter_json):
    """Build a generate_fn that returns reasoning_text on call 1 and
    formatter_json on call 2 (the two-step contract)."""
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        if i == 0:
            return reasoning_text
        return formatter_json

    return fn


# ---------------------------------------------------------------------------
# Disabled-by-default
# ---------------------------------------------------------------------------


class TestPhase5Disabled:
    def test_disabled_skips_with_reason(self):
        led = _ledger([
            _line("l001", "c01", "Hello there general kenobi."),
        ])

        def boom(*a, **k):
            raise AssertionError("should never fire when disabled")

        rep = _P5.phase_5_voice_drift(boom, led, enable=False)
        assert rep.skipped is True
        assert rep.skipped_reason == "enable=False"
        assert rep.edits_applied == 0
        meta = led.data["meta"]["phase_5_voice_drift_record"]
        assert meta["skipped"] is True


# ---------------------------------------------------------------------------
# Flagging behaviour
# ---------------------------------------------------------------------------


class TestPhase5Flagging:
    def test_no_flagged_lines_no_llm_call(self):
        # Build a ledger where every line is roughly the same length
        # and reasonably diverse -- no drift.
        led = _ledger([
            _line("l001", "c01", "Hello there general kenobi."),
            _line("l002", "c01", "How are you today my friend."),
            _line("l003", "c01", "We must move quickly indeed."),
        ])

        def boom(*a, **k):
            raise AssertionError("no LLM call expected (no flagged lines)")

        rep = _P5.phase_5_voice_drift(boom, led, enable=True)
        assert rep.skipped is False
        assert len(rep.flagged_lines) == 0
        assert rep.edits_proposed == 0
        assert rep.edits_applied == 0

    def test_flagged_line_triggers_llm_call(self):
        # Mean = ~10 words; one outlier line at 30 words is flagged.
        # Revised text stays inside the +/-50% length-drift band
        # (15-45 words) so the apply-time guard accepts it.
        led = _ledger([
            _line("l001", "c01", "Alpha beta gamma delta epsilon zeta eta theta iota kappa."),
            _line("l002", "c01", "Lambda mu nu xi omicron pi rho sigma tau upsilon."),
            _line("l003", "c01", "Phi chi psi omega aleph beth gimel daleth he waw."),
            _line("l004", "c01",
                  "One two three four five six seven eight nine ten "
                  "eleven twelve thirteen fourteen fifteen sixteen "
                  "seventeen eighteen nineteen twenty twenty-one "
                  "twenty-two twenty-three twenty-four twenty-five "
                  "twenty-six twenty-seven twenty-eight twenty-nine thirty."),
        ])

        revised = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron."
        )
        fake = _two_step_replies(
            reasoning_text=f"REVISE l004 with: {revised}",
            formatter_json=json.dumps({
                "edits": [{
                    "line_id": "l004",
                    "original": "long original text",
                    "revised": revised,
                    "reason": "match cadence",
                }],
                "rationale": "shortened to match cadence",
            }),
        )
        rep = _P5.phase_5_voice_drift(fake, led, enable=True)
        assert len(rep.flagged_lines) >= 1
        assert rep.edits_applied == 1
        line4 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l004"
        )
        assert line4["text"] == revised
        assert line4["word_count"] == 15
        assert line4["char_count"] == len(revised)


# ---------------------------------------------------------------------------
# Edit acceptance rules
# ---------------------------------------------------------------------------


class TestPhase5EditAcceptance:
    def test_edit_on_unflagged_line_rejected(self):
        # Build a ledger where l001-l003 are 9-word lines + l004 is
        # the 30-word outlier. Mean ~= 14.25. Deviation for the
        # 9-word lines ~= 37% which is under the 40% threshold so
        # they are NOT flagged; only l004 is flagged. The LLM
        # proposes an edit on the unflagged l001 -- the
        # orchestrator must reject.
        led = _ledger([
            _line("l001", "c01",
                  "alpha beta gamma delta epsilon zeta eta theta iota."),
            _line("l002", "c01",
                  "lambda mu nu xi omicron pi rho sigma tau."),
            _line("l003", "c01",
                  "phi chi psi omega aleph beth gimel daleth he."),
            _line("l004", "c01", "x " * 30),  # outlier -> flagged
        ])

        fake = _two_step_replies(
            reasoning_text="LEAVE all flagged lines.",
            formatter_json=json.dumps({
                "edits": [{
                    "line_id": "l001",  # NOT flagged
                    "original": "Alpha beta.",
                    "revised": "Alpha beta gamma delta.",
                    "reason": "lengthen for cadence",
                }],
                "rationale": "",
            }),
        )
        rep = _P5.phase_5_voice_drift(fake, led, enable=True)
        assert rep.edits_rejected_not_flagged == 1
        assert rep.edits_applied == 0
        # l001 unchanged from the fixture (9-word original).
        line1 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l001"
        )
        assert line1["text"] == (
            "alpha beta gamma delta epsilon zeta eta theta iota."
        )

    def test_edit_exceeding_length_drift_rejected(self):
        led = _ledger([
            _line("l001", "c01", "Alpha beta."),
            _line("l002", "c01", "Gamma delta."),
            _line("l003", "c01", "Epsilon zeta."),
            _line("l004", "c01", "x " * 30),  # flagged outlier
        ])

        # Propose to triple the flagged line's length (>50% drift).
        long_text = " ".join(f"w{i}" for i in range(90))
        fake = _two_step_replies(
            reasoning_text="REVISE l004 with much more text.",
            formatter_json=json.dumps({
                "edits": [{
                    "line_id": "l004",
                    "original": "outlier",
                    "revised": long_text,
                    "reason": "expand the line",
                }],
                "rationale": "",
            }),
        )
        rep = _P5.phase_5_voice_drift(fake, led, enable=True)
        assert rep.edits_rejected_length_drift == 1
        assert rep.edits_applied == 0

    def test_edit_on_unknown_line_id_rejected(self):
        led = _ledger([
            _line("l001", "c01", "Alpha beta."),
            _line("l002", "c01", "Gamma delta."),
            _line("l003", "c01", "Epsilon zeta."),
            _line("l004", "c01", "x " * 30),
        ])
        fake = _two_step_replies(
            reasoning_text="REVISE l_ghost.",
            formatter_json=json.dumps({
                "edits": [{
                    "line_id": "l_ghost",
                    "original": "made up",
                    "revised": "edited ghost line",
                    "reason": "",
                }],
                "rationale": "",
            }),
        )
        rep = _P5.phase_5_voice_drift(fake, led, enable=True)
        # An unknown line is also implicitly not-flagged so the
        # rejection lands on the not_flagged counter (the first
        # gate the orchestrator hits).
        assert (
            rep.edits_rejected_not_flagged
            + rep.edits_rejected_unknown_line
        ) >= 1
        assert rep.edits_applied == 0


# ---------------------------------------------------------------------------
# LLM failure modes
# ---------------------------------------------------------------------------


class TestPhase5LLMFailureModes:
    def test_two_step_returns_none_drops_edits(self):
        led = _ledger([
            _line("l001", "c01", "Alpha beta."),
            _line("l002", "c01", "Gamma delta."),
            _line("l003", "c01", "Epsilon zeta."),
            _line("l004", "c01", "x " * 30),
        ])

        # Force two_step_generate to return None by having the
        # formatter return unparseable garbage.
        fake = _two_step_replies(
            reasoning_text="REVISE l004.",
            formatter_json="not json at all",
        )
        rep = _P5.phase_5_voice_drift(fake, led, enable=True)
        # No edits applied; flagged_lines still recorded for soak.
        assert len(rep.flagged_lines) >= 1
        assert rep.edits_proposed == 0
        assert rep.edits_applied == 0

    def test_reasoning_pass_raises_drops_edits(self):
        led = _ledger([
            _line("l001", "c01", "Alpha beta."),
            _line("l002", "c01", "Gamma delta."),
            _line("l003", "c01", "Epsilon zeta."),
            _line("l004", "c01", "x " * 30),
        ])

        def fail_first(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated OOM")

        rep = _P5.phase_5_voice_drift(fail_first, led, enable=True)
        assert rep.edits_applied == 0


# ---------------------------------------------------------------------------
# Cascade wiring
# ---------------------------------------------------------------------------


class TestPhase5CascadeWiring:
    def test_cascade_default_off_no_llm_call(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger([
            _line("l001", "c01", "x " * 30),
            _line("l002", "c01", "alpha beta."),
            _line("l003", "c01", "gamma delta."),
            _line("l004", "c01", "epsilon zeta."),
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
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        rec = led.data["meta"]["phase_5_voice_drift_record"]
        assert rec["skipped"] is True
        # Phase 5 record present in cleanup_passes bucket
        # (B6 split, 2026-05-12).
        names = [p["phase_name"]
                 for p in led.data["meta"]["cleanup_passes"]]
        assert "phase_5_voice_drift" in names

    def test_cascade_enabled_routes_to_real_phase_5(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        # Mean ~= 10; l004 is the 30-word outlier. Revised stays in
        # the +/-50% length-drift band (15 words).
        led = _ledger([
            _line("l001", "c01",
                  "alpha beta gamma delta epsilon zeta eta theta "
                  "iota kappa."),
            _line("l002", "c01",
                  "lambda mu nu xi omicron pi rho sigma tau upsilon."),
            _line("l003", "c01",
                  "phi chi psi omega aleph beth gimel daleth he waw."),
            _line("l004", "c01",
                  " ".join(f"w{i}" for i in range(30))),
        ])

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        revised = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron."
        )
        replies = [
            "REVISE l004",
            json.dumps({
                "edits": [{
                    "line_id": "l004",
                    "original": "long",
                    "revised": revised,
                    "reason": "match cadence",
                }],
                "rationale": "",
            }),
        ]
        state = {"i": 0}

        def fake_llm(*a, **k):
            i = state["i"]
            state["i"] += 1
            return replies[i] if i < len(replies) else replies[-1]

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(
                fake_llm, led,
                enable_phase_5_voice_drift=True,
            )

        rec = led.data["meta"]["phase_5_voice_drift_record"]
        assert rec["skipped"] is False
        assert rec["edits_applied"] == 1

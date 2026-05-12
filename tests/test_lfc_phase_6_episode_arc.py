"""tests/test_lfc_phase_6_episode_arc.py

LFC sprint Commit 10 -- regression coverage for Phase 6 episode-arc
audit with Editor-note scaffold + scene-synopses fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_phase_6_episode_arc as _P6  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line(line_id, char_id, role, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger(*, with_scene_synopses=True, lines=None, cast=None):
    if lines is None:
        lines = [
            _line("l001", "c01", "character",
                  "Bob, the readings are off the chart again."),
            _line("l002", "c02", "character",
                  "Then someone tampered with the calibration system."),
            _line("l003", "c01", "character",
                  "Look closer at the trace timestamps before lunch."),
            _line("l004", "c02", "character",
                  "That can't be right; they all overlap somehow."),
            _line("l005", "c01", "character",
                  "I'll start the diagnostic right now then."),
            _line("l006", "c02", "character",
                  "And I'll alert the team about it."),
        ]

    meta = {
        "episode_title": "p6 test",
        "style": "mission_control_procedural",
    }
    if with_scene_synopses:
        meta["scene_synopses"] = [
            "Scene 0: Alice and Bob detect a calibration anomaly.",
            "Scene 1: They trace the cause and split the response.",
        ]

    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_p6_test",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "traits": "calm engineer"},
                {"char_id": "c02", "name": "BOB",
                 "traits": "skeptical analyst"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "sfx": [], "music": [], "clips": [],
            "meta": meta,
        },
        episode_id="ep_p6_test",
    )


def _two_step(reasoning_text, formatter_json):
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


class TestPhase6Disabled:
    def test_disabled_skips(self):
        led = _ledger()

        def boom(*a, **k):
            raise AssertionError("should not fire when disabled")

        rep = _P6.phase_6_episode_arc(boom, led, enable=False)
        assert rep.skipped is True
        assert rep.skipped_reason == "enable=False"
        rec = led.data["meta"]["phase_6_episode_arc_record"]
        assert rec["skipped"] is True


# ---------------------------------------------------------------------------
# Edit cap math
# ---------------------------------------------------------------------------


class TestPhase6EditCap:
    @pytest.mark.parametrize("voiced,expected", [
        (0, 3),    # floor 3
        (3, 3),    # floor 3
        (9, 3),    # 9 // 3 = 3
        (12, 4),
        (18, 6),
        (24, 8),
        (30, 8),   # ceiling 8
        (100, 8),  # ceiling 8
    ])
    def test_compute_arc_edit_cap(self, voiced, expected):
        assert _P6.compute_arc_edit_cap(voiced) == expected


# ---------------------------------------------------------------------------
# Synopses fallback
# ---------------------------------------------------------------------------


class TestPhase6SynopsesFallback:
    def test_uses_cached_synopses_when_present(self):
        led = _ledger(with_scene_synopses=True)
        seen = {"prompt": None}

        def capture(messages, *, temperature, max_new_tokens):
            seen["prompt"] = messages[0]["content"]
            return "no notes"

        # Run reasoning + formatter both -- formatter returns no notes.
        state = {"i": 0}

        def fn(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            if i == 0:
                seen["prompt"] = messages[0]["content"]
                return "no notes"
            return json.dumps({"notes": []})

        rep = _P6.phase_6_episode_arc(fn, led, enable=True)
        assert rep.used_synopses_fallback is False
        # The cached synopsis text is in the reasoning prompt.
        assert "SCENE SYNOPSES (from Phase 4)" in seen["prompt"]
        assert "calibration anomaly" in seen["prompt"]

    def test_falls_back_when_no_synopses(self):
        led = _ledger(with_scene_synopses=False)
        seen = {"prompt": None}

        state = {"i": 0}

        def fn(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            if i == 0:
                seen["prompt"] = messages[0]["content"]
                return "no notes"
            return json.dumps({"notes": []})

        rep = _P6.phase_6_episode_arc(fn, led, enable=True)
        assert rep.used_synopses_fallback is True
        assert "SCENE SYNOPSES NOT AVAILABLE" in seen["prompt"]


# ---------------------------------------------------------------------------
# Apply edits
# ---------------------------------------------------------------------------


class TestPhase6ApplyEdits:
    def test_editor_note_with_suggested_edit_applied(self):
        led = _ledger()
        # ~8-word lines; revised stays within +/-50% length drift.
        revised = "Bob, the readings spike to twelve thousand again."
        fake = _two_step(
            reasoning_text=(
                "SCOPE: scene_0\nNOTE: tighten line l001.\n"
                f"SUGGESTED EDITS: line_id=l001 revise: {revised!r}"
            ),
            formatter_json=json.dumps({
                "notes": [{
                    "scope": "scene_0",
                    "note": "tighten line l001",
                    "suggested_edits": [{
                        "line_id": "l001",
                        "original": "Bob, the readings are off the chart again.",
                        "revised": revised,
                        "reason": "tighter delivery",
                    }],
                }],
            }),
        )
        rep = _P6.phase_6_episode_arc(fake, led, enable=True)
        assert rep.edits_applied == 1
        line1 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l001"
        )
        assert line1["text"] == revised

    def test_length_drift_edit_rejected(self):
        led = _ledger()
        # 50-word revise on 8-word original -> length drift.
        long_revise = " ".join(f"w{i}" for i in range(50))
        fake = _two_step(
            reasoning_text="REVISE l001",
            formatter_json=json.dumps({
                "notes": [{
                    "scope": "scene_0",
                    "note": "expand",
                    "suggested_edits": [{
                        "line_id": "l001",
                        "original": "x",
                        "revised": long_revise,
                        "reason": "expand",
                    }],
                }],
            }),
        )
        rep = _P6.phase_6_episode_arc(fake, led, enable=True)
        assert rep.edits_rejected_length_drift == 1
        assert rep.edits_applied == 0

    def test_unknown_line_id_rejected(self):
        led = _ledger()
        fake = _two_step(
            reasoning_text="REVISE l_ghost",
            formatter_json=json.dumps({
                "notes": [{
                    "scope": "episode",
                    "note": "fix ghost",
                    "suggested_edits": [{
                        "line_id": "l_ghost",
                        "original": "x",
                        "revised": "y",
                        "reason": "missing",
                    }],
                }],
            }),
        )
        rep = _P6.phase_6_episode_arc(fake, led, enable=True)
        assert rep.edits_rejected_unknown_line == 1
        assert rep.edits_applied == 0

    def test_cap_exceeded_records_rejection(self):
        led = _ledger()
        # 6 voiced beats -> cap = min(8, max(3, 6//3)) = 3.
        # Propose 5 edits; only 3 should apply, 2 rejected.
        revised_pool = [
            "Bob the readings spike to twelve thousand again.",
            "Then someone tampered with our calibration system here.",
            "Look closer at the trace timestamps before noon.",
            "That can't be right; they overlap somehow now.",
            "I'll start the diagnostic right now then.",
        ]
        notes = [{
            "scope": f"scene_{i//3}",
            "note": f"note {i}",
            "suggested_edits": [{
                "line_id": f"l00{i+1}",
                "original": "x",
                "revised": revised_pool[i],
                "reason": "tighten",
            }],
        } for i in range(5)]
        fake = _two_step(
            reasoning_text="five notes",
            formatter_json=json.dumps({"notes": notes}),
        )
        rep = _P6.phase_6_episode_arc(fake, led, enable=True)
        assert rep.edit_cap == 3
        assert rep.edits_applied == 3
        assert rep.edits_rejected_cap_exceeded == 2


# ---------------------------------------------------------------------------
# Budget stamping
# ---------------------------------------------------------------------------


class TestPhase6BudgetRemaining:
    def test_budget_remaining_stamped(self):
        led = _ledger()
        fake = _two_step(
            reasoning_text="no notes",
            formatter_json=json.dumps({"notes": []}),
        )
        _P6.phase_6_episode_arc(fake, led, enable=True)
        # 6 voiced beats -> cap = 3. Zero applied -> budget 3.
        assert led.data["meta"]["doctor_edit_budget_remaining"] == 3


# ---------------------------------------------------------------------------
# LLM failure
# ---------------------------------------------------------------------------


class TestPhase6Failure:
    def test_two_step_none_marks_failed(self):
        led = _ledger()
        # Send enough garbage to exhaust parse_with_repair (3 attempts).
        state = {"i": 0}
        replies = [
            "scene 0 prose",
            "not json",
            "still not json",
            "still definitely not json",
            "absolutely not json",
        ]

        def fn(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            return replies[i] if i < len(replies) else replies[-1]

        rep = _P6.phase_6_episode_arc(fn, led, enable=True)
        assert rep.failed is True
        assert rep.edits_applied == 0


# ---------------------------------------------------------------------------
# Cascade wiring
# ---------------------------------------------------------------------------


class TestPhase6CascadeWiring:
    def test_cascade_default_off(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger()

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

        rec = led.data["meta"]["phase_6_episode_arc_record"]
        assert rec["skipped"] is True
        # Phase 6 record present in cleanup_passes bucket
        # (B6 split, 2026-05-12).
        names = [p["phase_name"]
                 for p in led.data["meta"]["cleanup_passes"]]
        assert "phase_6_episode_arc" in names

    def test_cascade_enabled_routes_to_real_phase_6(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger()

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        revised = "Bob, the readings spike to twelve thousand again."
        replies = [
            "REVISE l001",
            json.dumps({
                "notes": [{
                    "scope": "scene_0",
                    "note": "tighten",
                    "suggested_edits": [{
                        "line_id": "l001",
                        "original": "x",
                        "revised": revised,
                        "reason": "tighten",
                    }],
                }],
            }),
        ]
        state = {"i": 0}

        def fake_llm(messages, **kw):
            i = state["i"]
            state["i"] += 1
            return replies[i] if i < len(replies) else replies[-1]

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(
                fake_llm, led,
                enable_phase_6_episode_arc=True,
            )
        rec = led.data["meta"]["phase_6_episode_arc_record"]
        assert rec["skipped"] is False
        assert rec["edits_applied"] == 1
        # doctor_edit_budget_remaining stamped.
        assert "doctor_edit_budget_remaining" in led.data["meta"]

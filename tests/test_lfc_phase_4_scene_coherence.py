"""tests/test_lfc_phase_4_scene_coherence.py

LFC sprint Commit 8 -- regression coverage for Phase 4 per-scene
coherence + audio-first directives + scene-synopsis caching for
Phase 6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_phase_4_scene_coherence as _P4  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line(line_id, char_id, role, text, *, skip=False):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": skip,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger_two_scenes():
    """Two scenes separated by a music_inter divider."""
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_p4_test",
            "cast": [
                {"char_id": "c01", "name": "ALICE",
                 "traits": "calm engineer"},
                {"char_id": "c02", "name": "BOB",
                 "traits": "skeptical analyst"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": [
                _line("l001", "c01", "character",
                      "Bob, the readings are off the chart."),
                _line("l002", "c02", "character",
                      "Then someone tampered with the calibration."),
                _line("l003", "", "music_inter",
                      "Interstitial music swell."),
                _line("l004", "c01", "character",
                      "Look closer at the trace timestamps."),
                _line("l005", "c02", "character",
                      "That can't be right -- they overlap."),
            ],
            "sfx": [], "music": [], "clips": [],
            "meta": {
                "episode_title": "p4 test",
                "style": "mission_control_procedural",
            },
        },
        episode_id="ep_p4_test",
    )


def _two_step_per_scene(scene_replies):
    """Build a generate_fn that returns reasoning + formatter pairs
    in sequence, one pair per scene. Each scene's pair is
    [reasoning_text, formatter_json].
    """
    flat: list[str] = []
    for pair in scene_replies:
        flat.extend(pair)
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        return flat[i] if i < len(flat) else flat[-1]

    return fn


# ---------------------------------------------------------------------------
# Disabled-by-default
# ---------------------------------------------------------------------------


class TestPhase4Disabled:
    def test_disabled_skips_with_reason(self):
        led = _ledger_two_scenes()

        def boom(*a, **k):
            raise AssertionError("should not fire when disabled")

        rep = _P4.phase_4_scene_coherence(boom, led, enable=False)
        assert rep.skipped is True
        assert rep.skipped_reason == "enable=False"
        rec = led.data["meta"]["phase_4_scene_coherence_record"]
        assert rec["skipped"] is True


# ---------------------------------------------------------------------------
# Scene-bounded behaviour
# ---------------------------------------------------------------------------


class TestPhase4SceneBounded:
    def test_iterates_scenes_in_order(self):
        led = _ledger_two_scenes()
        # Two scenes -> 2 LLM call pairs.
        fake = _two_step_per_scene([
            ["LEAVE all. SCENE SYNOPSIS: Setup of the mystery.",
             json.dumps({"edits": [],
                          "rationale": "Setup of the mystery."})],
            ["LEAVE all. SCENE SYNOPSIS: Investigators find the trail.",
             json.dumps({"edits": [],
                          "rationale": "Investigators find the trail."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        assert rep.scenes_scanned == 2
        assert rep.scenes_examined == 2
        assert rep.scenes_failed == 0

    def test_scene_synopses_cached_for_phase_6(self):
        led = _ledger_two_scenes()
        fake = _two_step_per_scene([
            ["LEAVE. SCENE SYNOPSIS: First scene synopsis here.",
             json.dumps({"edits": [],
                          "rationale": "First scene synopsis here."})],
            ["LEAVE. SCENE SYNOPSIS: Second scene synopsis here.",
             json.dumps({"edits": [],
                          "rationale": "Second scene synopsis here."})],
        ])
        _P4.phase_4_scene_coherence(fake, led, enable=True)
        synopses = led.data["meta"]["scene_synopses"]
        assert len(synopses) == 2
        assert synopses[0] == "First scene synopsis here."
        assert synopses[1] == "Second scene synopsis here."

    def test_edit_within_scene_applied(self):
        led = _ledger_two_scenes()
        fake = _two_step_per_scene([
            ["REVISE l001. SCENE SYNOPSIS: tightened.",
             json.dumps({
                 "edits": [{
                     "line_id": "l001",
                     "original": "Bob, the readings are off the chart.",
                     "revised": "Bob, look -- the readings spiked.",
                     "reason": "tighter pacing",
                 }],
                 "rationale": "tightened scene 0.",
             })],
            ["LEAVE. SCENE SYNOPSIS: ok.",
             json.dumps({"edits": [], "rationale": "ok."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        assert rep.total_edits_applied == 1
        line1 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l001"
        )
        assert line1["text"] == "Bob, look -- the readings spiked."

    def test_edit_targeting_wrong_scene_rejected(self):
        # Scene 0's LLM call proposes an edit on l004 (a scene-1
        # line). The orchestrator must reject it.
        led = _ledger_two_scenes()
        fake = _two_step_per_scene([
            ["REVISE l004. SCENE SYNOPSIS: ok.",
             json.dumps({
                 "edits": [{
                     "line_id": "l004",  # in scene 1, not scene 0
                     "original": "x",
                     "revised": "y",
                     "reason": "wrong scene",
                 }],
                 "rationale": "scene 0 synopsis.",
             })],
            ["LEAVE. SCENE SYNOPSIS: ok.",
             json.dumps({"edits": [], "rationale": "scene 1 synopsis."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        scene0 = rep.scene_results[0]
        assert scene0["edits_rejected_wrong_scope"] >= 1
        # l004 unchanged.
        line4 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l004"
        )
        assert line4["text"] == "Look closer at the trace timestamps."


# ---------------------------------------------------------------------------
# Edit caps + length-drift guards
# ---------------------------------------------------------------------------


class TestPhase4EditCaps:
    def test_length_drift_edit_rejected(self):
        led = _ledger_two_scenes()
        # Scene 0 line l001 has ~7 words; propose a 50-word revision.
        long_revise = " ".join(f"w{i}" for i in range(50))
        fake = _two_step_per_scene([
            ["REVISE l001. SCENE SYNOPSIS: tightened.",
             json.dumps({
                 "edits": [{
                     "line_id": "l001",
                     "original": "x",
                     "revised": long_revise,
                     "reason": "expand",
                 }],
                 "rationale": "scene 0 synopsis.",
             })],
            ["LEAVE. SCENE SYNOPSIS: ok.",
             json.dumps({"edits": [], "rationale": "scene 1 synopsis."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        scene0 = rep.scene_results[0]
        assert scene0["edits_rejected_length_drift"] >= 1
        # l001 untouched.
        line1 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l001"
        )
        assert line1["text"].startswith("Bob, the readings")

    def test_skip_action_clears_text(self):
        led = _ledger_two_scenes()
        fake = _two_step_per_scene([
            ["SKIP l001. SCENE SYNOPSIS: muted.",
             json.dumps({
                 "edits": [{
                     "line_id": "l001",
                     "original": "x",
                     "revised": "",
                     "reason": "SKIP: redundant exposition",
                 }],
                 "rationale": "scene 0 synopsis.",
             })],
            ["LEAVE. SCENE SYNOPSIS: ok.",
             json.dumps({"edits": [], "rationale": "scene 1 synopsis."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        line1 = next(
            ln for ln in led.data["lines"] if ln["line_id"] == "l001"
        )
        assert line1["skip"] is True
        assert line1["text"] == ""

    def test_edit_cap_capped_at_three(self):
        # 4-line scene -> cap = min(3, 4//2) = 2. Propose 5 edits;
        # only 2 should apply.
        led = _ledger_two_scenes()
        # Replace scene 0 with 4 short voiced lines so cap=2.
        led.data["lines"] = [
            _line("s001", "c01", "character", "Hello."),
            _line("s002", "c02", "character", "Hi."),
            _line("s003", "c01", "character", "How are you."),
            _line("s004", "c02", "character", "Fine, thanks."),
        ]
        edits = [
            {"line_id": f"s00{i}",
             "original": "x", "revised": f"revised {i}",
             "reason": "tighten"} for i in range(1, 6)
        ]
        # Only one scene now.
        fake = _two_step_per_scene([
            ["REVISE everything. SCENE SYNOPSIS: rewritten.",
             json.dumps({"edits": edits, "rationale": "synopsis."})],
        ])
        rep = _P4.phase_4_scene_coherence(fake, led, enable=True)
        assert rep.total_edits_applied == 2


# ---------------------------------------------------------------------------
# Failure cascade -- skip-and-continue
# ---------------------------------------------------------------------------


class TestPhase4FailureCascade:
    def test_one_scene_failure_does_not_kill_phase(self):
        # Scene 0's formatter returns garbage AND every repair
        # attempt also returns garbage -> two_step returns None.
        # parse_with_repair has 3 max attempts; we feed 4 garbage
        # strings so the repair budget exhausts inside scene 0
        # and scene 1's replies remain untouched.
        led = _ledger_two_scenes()
        flat = [
            "scene 0 prose",
            "not json",
            "still not json",
            "still definitely not json",
            "absolutely not json",
            "LEAVE. SCENE SYNOPSIS: scene 1 ok.",
            json.dumps({"edits": [], "rationale": "scene 1 ok."}),
        ]
        state = {"i": 0}

        def fn(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            return flat[i] if i < len(flat) else flat[-1]

        rep = _P4.phase_4_scene_coherence(fn, led, enable=True)
        assert rep.scenes_examined == 2
        assert rep.scenes_failed == 1
        # Scene 0 failed; scene 1 succeeded.
        assert rep.scene_results[0]["failed"] is True
        assert rep.scene_results[1]["failed"] is False

    def test_reasoning_raise_marks_scene_failed(self):
        led = _ledger_two_scenes()
        state = {"i": 0}

        def fn(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            if i == 0:
                raise RuntimeError("simulated OOM on scene 0 reasoning")
            # Scene 1's reasoning + formatter both ok.
            if i == 1:
                return "LEAVE. SCENE SYNOPSIS: scene 1 ok."
            return json.dumps({"edits": [], "rationale": "scene 1 ok."})

        rep = _P4.phase_4_scene_coherence(fn, led, enable=True)
        assert rep.scenes_failed >= 1
        assert rep.scenes_examined == 2


# ---------------------------------------------------------------------------
# Cascade wiring
# ---------------------------------------------------------------------------


class TestPhase4CascadeWiring:
    def test_cascade_default_off_no_llm_call(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger_two_scenes()

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

        rec = led.data["meta"]["phase_4_scene_coherence_record"]
        assert rec["skipped"] is True
        # Phase record landed in cleanup_passes.
        names = [
            p["phase_name"] for p in led.data["meta"]["cleanup_passes"]
        ]
        assert "phase_4_per_scene_coherence" in names

    def test_cascade_enabled_caches_synopses(self):
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger_two_scenes()
        fake = _two_step_per_scene([
            ["LEAVE. SCENE SYNOPSIS: scene 0 wired.",
             json.dumps({"edits": [],
                          "rationale": "scene 0 wired."})],
            ["LEAVE. SCENE SYNOPSIS: scene 1 wired.",
             json.dumps({"edits": [],
                          "rationale": "scene 1 wired."})],
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
            _LFC_ORCH.run_freeze_cascade(
                fake, led,
                enable_phase_4_scene_coherence=True,
            )

        synopses = led.data["meta"]["scene_synopses"]
        assert "scene 0 wired" in synopses[0]
        assert "scene 1 wired" in synopses[1]

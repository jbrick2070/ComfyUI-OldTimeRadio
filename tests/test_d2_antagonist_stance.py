"""D2 (2026-06-22, story-quality lift) -- antagonist stance.

GENERATION lever (a JSON-free stance-consistency rider in the per-beat outline
prompt) + DETECTION telemetry (a typed StanceIssue on the critic report).
TELEMETRY ONLY: a StanceIssue must NOT add a reroll target, must NOT change the
freeze verdict, and "stance" must NOT enter FailedDimension (R3 grounding proved
no cross-run repair channel survives a JSON-frozen rerun). Pure / CPU. UTF-8 no
BOM, SFW.
"""
from __future__ import annotations

import sys
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_critic as SC  # noqa: E402
from nodes import _otr_outline as OUT  # noqa: E402
from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402
from nodes import _otr_ledger_reviewer as _OTRLR  # noqa: E402


# ---------------------------------------------------------------------------
# Generation lever: the per-beat outline prompt carries the stance rider
# ---------------------------------------------------------------------------

class TestOutlineStanceRider:
    def _macro(self):
        return OUT._MacroShape(
            title="Chandra's Echo",
            premise="An observatory catches a signal that someone wants buried.",
            setting="A mountaintop radio observatory",
            time_of_day="night",
        )

    def _req(self):
        from nodes import _otr_episode_budget as EB
        budget = EB.compute_episode_budget(
            target_words=400, act_count=EB.default_act_count(400),
            include_act_breaks=True, num_characters=2,
        )
        return OUT.OutlineRequest(
            news_seed="A deep-space signal is detected.",
            style="hard sci-fi procedural",
            target_words=400,
            character_cast=("MALI", "MANFRED"),
            budget=budget,
        )

    def test_beat_prompt_has_stance_consistency_rider(self):
        prompt = OUT._build_beat_user_prompt(
            self._req(), self._macro(), "rising_action", "MANFRED", (1, 4),
        )
        assert "KEEP STANCE CONSISTENT" in prompt
        assert "turn" in prompt.lower()


# ---------------------------------------------------------------------------
# StanceIssue model -- round-trip + defaults
# ---------------------------------------------------------------------------

class TestStanceIssueModel:
    def test_round_trip(self):
        si = SC.StanceIssue(
            character_id="c02", target="Mali / the broadcast",
            prior_stance="leaks her work to the press",
            new_stance="defends her against the AI",
            missing_turn_beat="no beat shows why Manfred switches sides",
            line_ids=["b011", "b017"], severity="high",
        )
        dumped = si.model_dump()
        again = SC.StanceIssue.model_validate(dumped)
        assert again == si
        assert again.line_ids == ["b011", "b017"]

    def test_defaults_partial(self):
        si = SC.StanceIssue(character_id="c02")
        assert si.line_ids == [] and si.target == "" and si.severity == ""


class TestReportCarriesStanceIssues:
    def test_report_round_trip_with_stance(self):
        rep = SC.StoryCriticReport(
            stance_issues=[SC.StanceIssue(character_id="c02", new_stance="flips")]
        )
        dumped = rep.model_dump()
        assert "stance_issues" in dumped
        again = SC.StoryCriticReport.model_validate(dumped)
        assert len(again.stance_issues) == 1
        # telemetry only -- no reroll target was created from the stance issue
        assert again.reroll_targets == []
        assert again.arc_verdict == "strong"

    def test_clean_has_empty_stance_issues(self):
        rep = SC.StoryCriticReport.clean()
        assert rep.stance_issues == []

    def test_legacy_report_without_stance_field_validates(self):
        # an older stored meta.story_critic_report has no stance_issues key
        legacy = {
            "continuity_issues": [], "voice_drift": [], "flat_lines": [],
            "arc_verdict": "uneven", "reroll_targets": [], "render_priority": [],
        }
        rep = SC.StoryCriticReport.model_validate(legacy)
        assert rep.stance_issues == []


# ---------------------------------------------------------------------------
# Locked constraint: "stance" is NOT a FailedDimension
# ---------------------------------------------------------------------------

class TestStanceNotAFailedDimension:
    def test_stance_absent_from_failed_dimension(self):
        allowed = set(typing.get_args(SC.FailedDimension))
        assert "stance" not in allowed
        assert allowed == {
            "knowledge", "pressure", "relationship", "decision",
            "obstacle", "tension", "unspecified",
        }


# ---------------------------------------------------------------------------
# Telemetry only: a stance-only critic report leaves reroll + freeze unchanged
# ---------------------------------------------------------------------------

def _ledger_obj(data):
    obj = SimpleNamespace(data=data, episode_id=data.get("episode_id", "ep"))
    obj.save = lambda: None  # type: ignore[attr-defined]
    return obj


def _clean_ledger_data():
    from nodes import _otr_ledger_freeze as _LFC
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_d2",
        "cast": [
            {"char_id": "c01", "name": "MALI", "traits": "calm",
             "voice_preset": "v2/en_speaker_6"},
            {"char_id": "c02", "name": "MANFRED", "traits": "smug",
             "voice_preset": "v2/en_speaker_2"},
            {"char_id": "announcer", "name": "ANNOUNCER", "traits": "warm",
             "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": f"b00{i}"} for i in range(1, 5)],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Gather round, listeners.",
             "word_count": 3, "char_count": 24, "skip": False},
            {"line_id": "b002", "beat_id": "b002", "speaker_role": "character",
             "char_id": "c01", "text": "I have isolated the frequency at last.",
             "word_count": 7, "char_count": 37, "skip": False},
            {"line_id": "b003", "beat_id": "b003", "speaker_role": "character",
             "char_id": "c02", "text": "Then someone tampered with the dish.",
             "word_count": 6, "char_count": 35, "skip": False},
            {"line_id": "b004", "beat_id": "b004", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Stay tuned.",
             "word_count": 2, "char_count": 11, "skip": False},
        ],
        "music": [], "clips": [],
        "meta": {"episode_title": "Chandra's Echo", "style": "observatory"},
    }


class TestStanceTelemetryDoesNotGate:
    def _run(self, critic_report):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits", pre_audit_violations=0,
                pre_audit_repairs_applied=0, doctor_edits_proposed=0,
                doctor_edits_applied=0, post_audit_violations=0,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._OTRSC, "run_story_critic",
                          return_value=critic_report):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)
        return led, disp

    def test_stance_only_report_leaves_verdict_and_reroll_unchanged(self):
        # baseline: a fully clean report
        _led_a, disp_a = self._run(SC.StoryCriticReport.clean())
        # same report PLUS a stance issue (no reroll targets)
        rep = SC.StoryCriticReport(
            stance_issues=[SC.StanceIssue(
                character_id="c02", target="Mali",
                prior_stance="hostile", new_stance="protective",
                missing_turn_beat="none", line_ids=["b003"], severity="high",
            )]
        )
        led_b, disp_b = self._run(rep)
        # the stance issue changed nothing about the verdict or the reroll
        assert disp_b.verdict == disp_a.verdict
        assert getattr(disp_b, "lines_rerolled", 0) == 0
        # the stance telemetry rode through onto meta.story_critic_report
        stamped = led_b.data["meta"]["story_critic_report"]
        assert stamped["stance_issues"] and \
            stamped["stance_issues"][0]["character_id"] == "c02"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

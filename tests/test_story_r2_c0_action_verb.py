"""Story-Quality R2 C0 -- non-default wants classifier + action-verb beat intents.

NOTE (grounded): the opposed wants are derived AFTER the outline in the writer,
so the outline beats cannot be grounded in them. C0 ships the action-under-
pressure prompt constraint (always-on) + the wants_are_default classifier (a scan
signal) + intent_is_action_under_pressure (a measurement helper). Pure / CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_dramatic_state import (  # noqa: E402
    _DEFAULT_A_WANTS, _DEFAULT_B_WANTS, wants_are_default,
)
from nodes._otr_outline import (  # noqa: E402
    OutlineRequest, _MacroShape, _build_beat_user_prompt,
    intent_is_action_under_pressure,
)


class TestWantsAreDefault:
    def test_default_detected(self):
        s = SimpleNamespace(
            character_a_wants=f"CHRIS: {_DEFAULT_A_WANTS}",
            character_b_wants=f"SKIP: {_DEFAULT_B_WANTS}",
        )
        assert wants_are_default(s) is True

    def test_news_derived_not_default(self):
        s = SimpleNamespace(
            character_a_wants="CHRIS: launch Link and grab Swift with three arms",
            character_b_wants="SKIP: scrub the mission and announce the loss",
        )
        assert wants_are_default(s) is False

    def test_defensive(self):
        assert wants_are_default(None) is False
        assert wants_are_default(SimpleNamespace()) is False


class TestIntentActionUnderPressure:
    @pytest.mark.parametrize("intent", [
        "Chris reveals the breach to the whole team",
        "Skip refuses to sign the waiver",
        "Kelly accuses the contractor of cutting corners",
        "Chris chooses to ground the launch",
        "Nora conceals the failure from the board",
    ])
    def test_action_intents(self, intent):
        assert intent_is_action_under_pressure(intent) is True

    @pytest.mark.parametrize("intent", [
        "The lab is quiet and the lights hum softly",
        "A calm moment of reflection on the console",
    ])
    def test_descriptive_intents(self, intent):
        assert intent_is_action_under_pressure(intent) is False


class TestBeatPromptCarriesConstraint:
    def test_prompt_has_action_constraint(self):
        from nodes._otr_episode_budget import compute_episode_budget
        budget = compute_episode_budget(
            target_words=300, act_count=3, include_act_breaks=True,
            num_characters=2,
        )
        req = OutlineRequest(
            news_seed="a falling telescope", style="noir thriller",
            target_words=300, character_cast=("ALICE", "BOB"), budget=budget,
        )
        macro = _MacroShape(
            title="The Signal", premise="A lab hears a falling telescope.",
            setting="a radio observatory", time_of_day="night",
            central_tension="Will they save it?",
        )
        prompt = _build_beat_user_prompt(req, macro, "setup", "ALICE", (0, 3))
        assert "ACTION UNDER PRESSURE" in prompt
        assert "reveal" in prompt and "refuse" in prompt


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

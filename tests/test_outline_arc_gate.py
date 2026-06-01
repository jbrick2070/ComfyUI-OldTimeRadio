"""Story-spine Stream A -- outline arc gate.

Locks in the deterministic arc-ref stamping (`_climactic_phase` /
`_derive_arc_refs`) and the `Outline._arc_refs_coherent` validator.
Pure Python; no GPU, no LLM, no network. The combiner is the only
production path that builds an Outline, so these guard the contract that
every shipped outline carries a turn + a payoff (and that legacy /
back-compat outlines without arc refs still validate).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_outline import (  # noqa: E402
    Beat,
    Outline,
    TurnRef,
    ButtonRef,
    _MacroShape,
    _climactic_phase,
    _derive_arc_refs,
)


def _beat(idx, role, phase, intent="advance the plot here"):
    return Beat(
        beat_id=f"b{idx:03d}",
        speaker="ANNOUNCER" if role == "announcer" else "ALICE",
        speaker_role=role,
        intent=intent,
        target_words=15,
        mood="tense",
        arc_phase=phase,
    )


def _sample_beats():
    # announcer-open, 4 character beats across phases, announcer-close
    return [
        _beat(1, "announcer", "setup", "open the episode"),
        _beat(2, "character", "setup", "establish the lab and the signal"),
        _beat(3, "character", "complication", "the signal turns hostile"),
        _beat(4, "character", "climax", "the team must choose who to save"),
        _beat(5, "character", "resolution", "the cost of the choice lands"),
        _beat(6, "announcer", "resolution", "close the broadcast"),
    ]


def _macro(central="Will they answer the signal?"):
    return _MacroShape(
        title="The Signal",
        premise="A lab hears something that should not exist.",
        setting="a radio observatory",
        time_of_day="pre-dawn",
        central_tension=central,
    )


class TestClimacticPhase:
    def test_prefers_explicit_climax(self):
        assert _climactic_phase(
            ("setup", "complication", "climax", "resolution")
        ) == "climax"

    def test_falls_back_to_penultimate(self):
        assert _climactic_phase(("setup", "rising", "falling", "coda")) == "falling"

    def test_single_phase(self):
        assert _climactic_phase(("only",)) == "only"

    def test_empty(self):
        assert _climactic_phase(()) is None


class TestDeriveArcRefs:
    _phases = ("setup", "complication", "climax", "resolution")

    def test_button_is_last_character_beat_and_turn_on_climax(self):
        beats = _sample_beats()
        central, turn, button = _derive_arc_refs(beats, _macro(), self._phases)
        assert central == "Will they answer the signal?"
        assert turn is not None and button is not None
        assert button.beat_index == 4          # last character beat (b005)
        assert turn.beat_index == 3            # climax-phase character beat
        assert turn.beat_index < button.beat_index
        assert turn.what_changes == beats[3].intent
        assert button.payoff == beats[4].intent

    def test_central_tension_falls_back_to_premise(self):
        beats = _sample_beats()
        central, _turn, _button = _derive_arc_refs(
            beats, _macro(central=""), self._phases
        )
        assert central == "A lab hears something that should not exist."

    def test_degenerate_outline_returns_none_refs(self):
        beats = [
            _beat(1, "announcer", "setup"),
            _beat(2, "character", "setup"),
            _beat(3, "announcer", "resolution"),
            _beat(4, "announcer", "resolution"),
        ]
        central, turn, button = _derive_arc_refs(
            beats, _macro(central=""), ("setup", "resolution")
        )
        assert turn is None and button is None
        assert central  # still derives a non-empty tension string

    def test_no_explicit_climax_uses_penultimate(self):
        beats = _sample_beats()
        # remap phases so none is named "climax"
        for b, ph in zip(beats, ("a", "a", "b", "c", "d", "d")):
            b.arc_phase = ph
        _central, turn, button = _derive_arc_refs(
            beats, _macro(), ("a", "b", "c", "d")
        )
        # penultimate phase is "c" -> first character beat in "c" is index 4,
        # but that is the last character beat (button); turn is kept strictly
        # below button via the fallback.
        assert turn.beat_index < button.beat_index


class TestArcRefsCoherentValidator:
    def _ok_outline(self, **overrides):
        kwargs = dict(
            title="The Signal",
            premise="A lab hears something that should not exist.",
            setting="a radio observatory",
            time_of_day="pre-dawn",
            beats=_sample_beats(),
            central_tension="Will they answer the signal?",
            turning_point=TurnRef(beat_index=3, what_changes="the team must choose"),
            button=ButtonRef(beat_index=4, payoff="the cost lands"),
        )
        kwargs.update(overrides)
        return Outline(**kwargs)

    def test_valid_arc_passes(self):
        o = self._ok_outline()
        assert o.turning_point.beat_index == 3
        assert o.button.beat_index == 4

    def test_legacy_outline_without_arc_passes(self):
        o = Outline(
            title="The Signal",
            premise="A lab hears something that should not exist.",
            setting="a radio observatory",
            time_of_day="pre-dawn",
            beats=_sample_beats(),
        )
        assert o.turning_point is None and o.button is None
        assert o.central_tension == ""

    def test_turn_after_button_rejected(self):
        with pytest.raises(ValueError):
            self._ok_outline(
                turning_point=TurnRef(beat_index=4, what_changes="x"),
                button=ButtonRef(beat_index=3, payoff="y"),
            )

    def test_out_of_range_index_rejected(self):
        with pytest.raises(ValueError):
            self._ok_outline(button=ButtonRef(beat_index=99, payoff="y"))

    def test_button_in_front_half_rejected(self):
        with pytest.raises(ValueError):
            self._ok_outline(
                turning_point=TurnRef(beat_index=1, what_changes="x"),
                button=ButtonRef(beat_index=2, payoff="y"),
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""The outline's beat ceiling holds every outline the generator accepts.

Plan item 3 (2026-10-14). The writer asks four voiced beats per act -- 36 at
seven acts with act breaks -- but a phase skeleton may return 1-10 beats and
`_phase_check` accepts a count that differs from the request. The final
`Outline` used to cap at 36, so one five-beat phase in a 7-act episode raised a
raw ValidationError after every macro, phase and beat call had been paid for.
The ceiling is now the bounded producers' own maximum, 7 * 10 + 6 + 2 = 78.

Pure Python, no model: generate_fn is a stage-aware stub.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_episode_budget import (  # noqa: E402
    MAX_ACT_COUNT, InvalidEpisodeBudgetError, compute_episode_budget,
)
from nodes._otr_outline import (  # noqa: E402
    OUTLINE_BEATS_MAX, Outline, OutlineRequest, generate_outline,
)

_CAST = ("ALICE", "BOB")
_MACRO = json.dumps({
    "title": "Capacity", "premise": "A premise about a faint signal.",
    "setting": "An observatory", "time_of_day": "midnight",
})
_BEAT = json.dumps({"intent": "advance the scene", "mood": "tense"})


def _stage(messages):
    system = messages[0]["content"]
    if "You plan one phase" in system:
        return "phase"
    if "You flesh out one beat" in system:
        return "beat"
    return "macro"


def _gen(phase_sizes, calls):
    """phase_sizes[i](requested) -> how many beats phase i returns."""
    phase_no = {"i": 0}

    def gen(messages, *, temperature, max_new_tokens):
        stage = _stage(messages)
        calls.append(stage)
        if stage == "phase":
            m = re.search(r"Beats to plan in this phase: (\d+)", messages[1]["content"])
            requested = int(m.group(1)) if m else 1
            n = phase_sizes(phase_no["i"], requested)
            phase_no["i"] += 1
            speakers = [_CAST[k % 2] for k in range(n)]
            return json.dumps({"beats": [{"speaker": s} for s in speakers]})
        if stage == "beat":
            return _BEAT
        return _MACRO
    return gen


def _request(acts):
    return OutlineRequest(news_seed="a real science seed", style="noir",
                          character_cast=_CAST,
                          budget=compute_episode_budget(acts, True, len(_CAST)))


def _run(acts, phase_sizes):
    calls = []
    outline = generate_outline(_gen(phase_sizes, calls), _request(acts))
    return outline, calls


def _roles(outline):
    roles = [b.speaker_role for b in outline.beats]
    return (roles.count("character"), roles.count("announcer"),
            roles.count("music_inter"))


def test_the_ceiling_is_the_bounded_producers_maximum():
    assert MAX_ACT_COUNT == 7
    assert OUTLINE_BEATS_MAX == 78


def test_a_seven_act_outline_with_one_long_phase_is_accepted_without_repair():
    outline, calls = _run(7, lambda i, req: req + 1 if i == 0 else req)
    assert len(outline.beats) == 37
    assert _roles(outline) == (29, 2, 6)
    # One macro, one call per phase, one detail call per voiced beat -- the
    # extra beat gets its own detail call and nothing is retried or repaired.
    assert calls.count("macro") == 1
    assert calls.count("phase") == 7
    assert calls.count("beat") == 29


def test_the_nominal_seven_act_and_one_act_shapes_are_unchanged():
    outline, calls = _run(7, lambda i, req: req)
    assert len(outline.beats) == 36
    assert calls.count("phase") == 7
    one, _ = _run(1, lambda i, req: req)
    assert _roles(one)[2] == 0           # no interstitial in one act


def test_the_maximum_is_accepted_with_unique_ids_and_owned_cast():
    outline, calls = _run(7, lambda i, req: 10)
    assert len(outline.beats) == OUTLINE_BEATS_MAX
    assert _roles(outline) == (70, 2, 6)
    ids = [b.beat_id for b in outline.beats]
    assert len(set(ids)) == len(ids)
    assert {b.speaker for b in outline.beats
            if b.speaker_role == "character"} <= set(_CAST)
    assert calls.count("phase") == 7     # no phase was retried
    assert calls.count("beat") == 70


def test_shorter_episodes_with_long_phases_are_accepted_too():
    """The old cap of 36 also rejected 4- to 6-act outlines whose phases came
    back long -- the shipped otr_cloud_low_5act graph is 5 acts."""
    from nodes._otr_episode_budget import outline_beats_max
    outline, calls = _run(5, lambda i, req: 7)
    assert len(outline.beats) == 5 * 7 + 4 + 2 == 41
    assert calls.count("phase") == 5
    for acts in range(1, MAX_ACT_COUNT + 1):
        top, _ = _run(acts, lambda i, req: 10)
        assert len(top.beats) == outline_beats_max(acts)


def _beat(i):
    return {"beat_id": "b%03d" % i, "speaker": "ALICE", "speaker_role": "character",
            "intent": "advance", "mood": "tense", "arc_phase": "setup"}


def _outline(n):
    return {"title": "T", "premise": "A premise.", "setting": "A lab",
            "time_of_day": "midnight", "beats": [_beat(i) for i in range(1, n + 1)]}


def test_a_direct_outline_follows_the_same_capacity_contract():
    Outline.model_validate(_outline(OUTLINE_BEATS_MAX))
    with pytest.raises(ValidationError):
        Outline.model_validate(_outline(OUTLINE_BEATS_MAX + 1))


def test_eight_acts_still_refuse_before_generation():
    with pytest.raises(InvalidEpisodeBudgetError):
        compute_episode_budget(8, True, len(_CAST))

"""If the operator asks for 7 acts, the spine has 7 acts in it.

OPERATOR RULING 2026-08-15: *"we need to remove hard schema ceilings for
anything -- if I ask for 7 acts it needs to generate a spine of 7 acts"*, and
*"run through 7 act passes including 4 beats per each 7"*.

What this replaced: `_RADIO_SCORE_MAX_SCENES = 3` and
`_RADIO_SCORE_MAX_BEATS_PER_SCENE = 4`, hardcoded, giving a hard 12-beat
ceiling on a WHOLE episode at ANY act count. The score's scene is the codex
lane's act-sized unit, so a 7-act pick could not produce a 7-act spine -- and
because the lane decodes under a grammar built from that schema, the cap
truncated during generation instead of refusing loudly. A number typed into a
schema was outranking the operator's own pick.

The caps are now DERIVED from the topology that sizes the request, so the
schema always moves with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_scifi_codex as codex  # noqa: E402
from nodes._otr_episode_budget import (  # noqa: E402
    BEATS_PER_ACT,
    MAX_ACT_COUNT,
    MIN_ACT_COUNT,
    voiced_beat_count,
)


ACT_COUNTS = list(range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1))


@pytest.mark.parametrize("acts", ACT_COUNTS)
def test_a_ceiling_never_shrinks_the_ask(acts):
    """Nothing here says what an episode came back with. That is not ours.

    OPERATOR DIRECTIVE 2026-08-15: *"no chasing beats"*, and *"if it writes 1
    act [when] I request 7, fine."* An act count and a beat count are both
    REQUESTS. A model that returns less is not a failure, and no test in this
    repo may assert what it returned.

    What this checks is the other direction entirely, and it is the thing the
    operator actually asked for: that OUR OWN schema does not quietly reduce
    the request before the model ever sees it. A ceiling that clips the ask is
    us deciding the shape; a model that writes short is the model writing.
    """
    requested = voiced_beat_count(acts)
    asked = codex._codex_target_beat_count(acts, cast_count=3)
    assert asked >= requested, (
        f"{acts} acts asks for {requested} beats and the lane reduced the ask "
        f"to {asked} -- a schema ceiling is shaping the episode instead of "
        f"the act count"
    )


@pytest.mark.parametrize("acts", ACT_COUNTS)
def test_the_scene_schema_can_hold_one_scene_per_act(acts):
    """A scene is the act-sized unit; 7 acts needs room for 7 of them."""
    assert codex._RADIO_SCORE_MAX_SCENES >= acts


def test_the_beat_schema_can_hold_a_full_act_of_beats():
    assert codex._RADIO_SCORE_MAX_BEATS_PER_SCENE >= BEATS_PER_ACT


def test_the_largest_legal_request_is_nowhere_near_the_ceiling():
    """A cap must never sit AT the number a request produces (Bible 12.102).

    Equality is the failure mode this is written against: a ceiling that
    exactly matches the largest request looks like it fits, and then trims the
    very first story that tries to use all of it.
    """
    largest = voiced_beat_count(MAX_ACT_COUNT)
    assert codex._RADIO_SCORE_MAX_BEATS > largest
    assert codex._RADIO_SCORE_MAX_SCENES > MAX_ACT_COUNT


def test_the_caps_are_derived_not_typed():
    """Raising MAX_ACT_COUNT must carry the schema with it.

    The defect was a hardcoded number drifting out of step with the topology.
    Deriving them is what stops that from recurring, so it is the derivation
    itself that gets pinned, not the current values.
    """
    assert codex._RADIO_SCORE_MAX_SCENES == MAX_ACT_COUNT * codex._SCHEMA_HEADROOM
    assert (
        codex._RADIO_SCORE_MAX_BEATS_PER_SCENE
        == BEATS_PER_ACT * codex._SCHEMA_HEADROOM
    )
    assert codex._SCHEMA_HEADROOM >= 2, (
        "headroom below 2 puts the backstop within reach of a real request"
    )


def test_the_backstop_still_exists():
    """Removing the truncation is not the same as removing the guard.

    Runaway guards stay code-side (standing rule). A backstop at several
    times the largest legal request cannot shape an episode and still stops a
    decode that has stopped making sense.
    """
    assert codex._RADIO_SCORE_MAX_BEATS < 10_000

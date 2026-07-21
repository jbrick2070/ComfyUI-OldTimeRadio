"""Chunk 3 -- bare stage-direction reroll wiring in compose_line.

A draft whose text leads with a bare stage direction triggers ONE reroll (the
detector runs on the post-draft text); a clean draft does not; the guard caps it
at one level (the freeze floor is the backstop). Pure / CPU (mock creative_fn).
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import LineRequest, compose_line  # noqa: E402

_CLEAN = "Look, Pinky, we have been through this whole thing already, alright."
_DIRTY = ("twirls his pen nervously Look, Pinky, we have been through this "
          "whole thing already.")


def _req(reroll_hint=""):
    return LineRequest(
        speaker="ALICE", intent="reveal", mood="tense", target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
        reroll_hint=reroll_hint,
    )


def test_reroll_fires_on_bare_stage_direction():
    calls = {"n": 0}

    def mock(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        return _DIRTY if calls["n"] == 1 else _CLEAN

    res = compose_line(creative_fn=mock, req=_req())
    assert calls["n"] == 2            # exactly one reroll
    assert res.text == _CLEAN


def test_no_reroll_on_clean_draft():
    calls = {"n": 0}

    def mock(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        return _CLEAN

    res = compose_line(creative_fn=mock, req=_req())
    assert calls["n"] == 1            # no reroll
    assert res.text == _CLEAN


def test_guard_caps_at_one_reroll():
    # Every authored rung leaks -> the bounded ladder reaches its floor. The
    # stage action never ships and the episode remains renderable.
    calls = {"n": 0}

    def mock(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        return _DIRTY

    res = compose_line(creative_fn=mock, req=_req())
    assert calls["n"] == 4  # draft stage retry + quality A + lower-temp B
    assert res.text.startswith("Look, Pinky")
    assert "twirls his pen" not in res.text
    assert (
        "hygiene_repaired_after_reroll:stage_direction:deterministic_floor"
        in res.compose_flags
    )


def test_reroll_prompt_carries_the_hint_and_existing_critic_hint():
    seen = {"reroll_msg": None}
    calls = {"n": 0}

    def mock(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        if calls["n"] == 2:  # the reroll call
            seen["reroll_msg"] = "\n".join(
                m.get("content", "") for m in messages)
        return _DIRTY if calls["n"] == 1 else _CLEAN

    compose_line(creative_fn=mock, req=_req(reroll_hint="critic: make it punchier"))
    assert seen["reroll_msg"] is not None
    # both the existing critic hint and the stage-direction directive appear
    assert "punchier" in seen["reroll_msg"]
    assert "spoken words" in seen["reroll_msg"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

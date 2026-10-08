"""The per-line composer carries a source passage when it is handed one.

The only supplier is the cast-coverage repair: its fidelity graft gives a
repaired line the author's own first speech for that character. Every other
line is composed without one, and its prompt must not move.
"""
from __future__ import annotations

from nodes import _otr_line_composer as lc

# Shaped like the repair's own block (_otr_cast_coverage_repair._source_block_for).
_BLOCK = (
    "SOURCE (verbatim, Twelfth Night, Act 2, Scene 5, spoken by MARIA):\n"
    "Get you all three into the boxtree."
)


# ---------------------------------------------------------------------------
# the per-line route
# ---------------------------------------------------------------------------

def _line_request(**kw) -> lc.LineRequest:
    base = dict(speaker="EDITOR", intent="press him", mood="wry",
                canon_header="CANON", last_lines=[])
    base.update(kw)
    return lc.LineRequest(**base)


def test_the_line_prompt_carries_the_source_when_given_one():
    prompt = lc._build_user_prompt(_line_request(source_block=_BLOCK))
    assert _BLOCK in prompt
    assert "The passage below is the SOURCE this scene adapts." in prompt


def test_the_line_prompt_is_unchanged_without_a_source():
    a = lc._build_user_prompt(_line_request())
    b = lc._build_user_prompt(_line_request(source_block=""))
    assert a == b
    assert "The passage below is the SOURCE" not in a


def test_the_line_request_default_is_no_source():
    assert _line_request().source_block == ""


def test_the_source_block_sits_immediately_above_the_write_instruction():
    # A per-line prompt runs many hundreds of tokens; source constraints far
    # above the generation point compete badly with everything between.
    prompt = lc._build_user_prompt(_line_request(
        source_block=_BLOCK,
        style_descriptor="wry", theme="regret", outline_spine="spine",
        current_beat_block="beat", continuity_slice="CONTINUITY: x",
        position="mid", all_voice_cards="cards",
    ))
    last = _BLOCK.splitlines()[-1]
    assert prompt.index(last) < prompt.index("WRITE LINE")
    between = prompt.split(last)[-1].split("WRITE LINE")[0]
    # Nothing of substance separates the passage from the instruction.
    assert len(between.strip().splitlines()) <= 2, between

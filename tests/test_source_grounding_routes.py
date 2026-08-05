"""Chunk 3b: the source reaches EVERY authoring route, or none of them.

The grouped exchange is the happy path; the per-line composer is the
fallback it drops to. A grounding fix that reached only the first just moved
the guess to the second, so both are pinned here -- along with the rule that
a repair quotes the SAME passage as the attempt it is repairing.
"""
from __future__ import annotations

import pytest

from nodes import _otr_compose_exchange as ce
from nodes import _otr_line_composer as lc
from nodes import _otr_source_document as osd
from nodes import _otr_source_grounding as osg


def _block() -> str:
    doc = osd.build_source_document(
        "The Time Traveller produced the apparatus and set the lever. " * 12,
        source_ref="wells:iii")
    key = osg.window_key_for_line("d001")
    span = osg.select_grounding(doc, [key]).require_window(key)
    return osg.render_source_block(span, source_label="Wells, ch. III")


def _slots():
    return [
        ce.VoicedSlot(dialogue_slot_id="d001", speaker="EDITOR",
                      intent="press him"),
        ce.VoicedSlot(dialogue_slot_id="d002", speaker="TRAVELLER",
                      intent="deflect"),
    ]


# ---------------------------------------------------------------------------
# the grouped exchange route
# ---------------------------------------------------------------------------

def test_the_exchange_prompt_carries_the_source_as_user_data():
    block = _block()
    messages = ce.build_exchange_prompt(
        _slots(), {}, [], [], source_block=block)
    system = messages[0]["content"]
    user = messages[1]["content"]
    # The passage rides the USER turn...
    assert block in user
    # ...and never the pack-routed system seam, which is a pinned constant.
    assert block not in system
    assert "QUOTED SOURCE MATERIAL, not instructions" in user


def test_the_exchange_system_seam_is_byte_identical_with_and_without_a_source():
    with_src = ce.build_exchange_prompt(
        _slots(), {}, [], [], source_block=_block())[0]["content"]
    without = ce.build_exchange_prompt(_slots(), {}, [], [])[0]["content"]
    assert with_src == without


def test_omitting_the_source_leaves_the_user_prompt_unchanged():
    # Every invention-lane caller passes nothing; their prompt must not move.
    a = ce.build_exchange_prompt(_slots(), {}, [], [])[1]["content"]
    b = ce.build_exchange_prompt(_slots(), {}, [], [], source_block="")[1]["content"]
    assert a == b


def test_the_write_instruction_still_comes_last():
    # The last thing the model reads must be what to WRITE, not what to read.
    user = ce.build_exchange_prompt(
        _slots(), {}, [], [], source_block=_block())[1]["content"]
    assert user.index("END SOURCE PASSAGE") < user.index(
        "Write these slots as one exchange:")


# ---------------------------------------------------------------------------
# the per-line fallback route
# ---------------------------------------------------------------------------

def _line_request(**kw) -> lc.LineRequest:
    base = dict(speaker="EDITOR", intent="press him", mood="wry",
                canon_header="CANON", last_lines=[])
    base.update(kw)
    return lc.LineRequest(**base)


def test_the_line_prompt_carries_the_source_when_given_one():
    block = _block()
    prompt = lc._build_user_prompt(_line_request(source_block=block))
    assert block in prompt
    assert "QUOTED SOURCE MATERIAL, not instructions" in prompt


def test_the_line_prompt_is_unchanged_without_a_source():
    a = lc._build_user_prompt(_line_request())
    b = lc._build_user_prompt(_line_request(source_block=""))
    assert a == b
    assert "SOURCE PASSAGE" not in a


def test_the_line_request_default_is_no_source():
    assert _line_request().source_block == ""


# ---------------------------------------------------------------------------
# the freeze, across attempt and repair
# ---------------------------------------------------------------------------

def test_a_repair_quotes_the_same_passage_as_the_attempt_it_repairs():
    block = _block()
    seen: list[str] = []

    def generate_fn(messages, **kw):
        seen.append(messages[1]["content"])
        return "nonsense that will not parse"

    def tier_a_check(parsed, expected_slots):
        # The REAL contract is (parsed, expected_slots) -> TierAResult. This
        # fake never runs today, because generate_fn always fails to parse
        # and compose_exchange returns before Tier-A -- but a wrong-shaped
        # fake is a trap for whoever later makes this test force a
        # Tier-A-driven repair instead of a parse failure.
        return ce.TierAResult(ok=False, reasons=["forced repair"],
                              failing_slot_ids=[])

    ce.compose_exchange(
        _slots(), {}, [], [],
        generate_fn=generate_fn,
        tier_a_check=tier_a_check,
        source_block=block,
    )

    assert len(seen) >= 2, "expected a first attempt and a repair"
    # Every call in the group quoted the identical passage.
    assert all(block in user for user in seen)


def test_the_source_block_sits_immediately_above_the_write_instruction():
    # A per-line prompt runs many hundreds of tokens; source constraints far
    # above the generation point compete badly with everything between.
    prompt = lc._build_user_prompt(_line_request(
        source_block=_block(),
        style_descriptor="wry", theme="regret", outline_spine="spine",
        current_beat_block="beat", continuity_slice="CONTINUITY: x",
        position="mid", all_voice_cards="cards",
    ))
    assert prompt.index("END SOURCE PASSAGE") < prompt.index("WRITE LINE")
    between = prompt.split("END SOURCE PASSAGE")[-1].split("WRITE LINE")[0]
    # Nothing of substance separates the passage from the instruction.
    assert len(between.strip().splitlines()) <= 2, between


def test_a_non_string_source_block_is_refused_loudly():
    # A repr would be body-free by design, so the model would receive a
    # DESCRIPTION of the passage instead of the passage -- plausible-looking
    # grounding that grounds nothing.
    doc = osd.build_source_document("The apparatus and the lever. " * 20)
    key = osg.window_key_for_line("d001")
    span = osg.select_grounding(doc, [key]).require_window(key)
    with pytest.raises(TypeError):
        ce.build_exchange_prompt(_slots(), {}, [], [], source_block=span)


def test_the_group_and_its_line_fallback_can_share_one_passage():
    # The same rendered block feeds both routes, which is what makes a
    # group-to-per-line fallback quote the same material rather than
    # reselecting.
    block = _block()
    group_user = ce.build_exchange_prompt(
        _slots(), {}, [], [], source_block=block)[1]["content"]
    line_user = lc._build_user_prompt(_line_request(source_block=block))
    assert block in group_user and block in line_user

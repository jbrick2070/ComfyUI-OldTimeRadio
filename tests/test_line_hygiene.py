"""Tests for nodes/_otr_line_hygiene.py -- story-quality Phase 1, A6
(clean delivery: deterministic scrub + truncation detection)."""

from nodes import _otr_line_hygiene as HY


def test_scrub_closed_parenthetical():
    assert HY.scrub_parentheticals(
        "We must go (she whispers) before dawn.") == "We must go before dawn."


def test_scrub_dangling_open_parenthetical():
    # leg_0013: "(she translates a passage," -- an unclosed stage direction.
    out = HY.scrub_parentheticals(
        "The tablet reads clearly (she translates a passage,")
    assert "(" not in out
    assert out.startswith("The tablet reads clearly")


def test_scrub_parenthetical_keeps_line_if_emptied():
    assert HY.scrub_parentheticals("(only a stage direction)") == \
        "(only a stage direction)"


# ---------------------------------------------------------------------------
# is_stage_direction_only (2026-06-22 root-cause repair): a character line with
# NO spoken content -- entirely a stage direction -- is detected so the spine
# RECOMPOSES it into real dialogue (the scrub keeps it; TTS would empty it).
# ---------------------------------------------------------------------------
def test_stage_direction_only_flags_parenthetical_only():
    assert HY.is_stage_direction_only(
        "(pauses, then flips the launch sequencer switch, ignoring Stone)")
    assert HY.is_stage_direction_only("(beat)")
    assert HY.is_stage_direction_only("(softly)")


def test_stage_direction_only_false_for_real_dialogue():
    assert not HY.is_stage_direction_only("Lockdown, now! No one in or out.")
    # a line with a stage direction PLUS real words is not stage-direction-only
    assert not HY.is_stage_direction_only("(softly) We have to move.")


def test_stage_direction_only_false_for_empty():
    # an already-empty line is a different (mechanical) defect, not this one
    assert not HY.is_stage_direction_only("")
    assert not HY.is_stage_direction_only("   ")
    assert not HY.is_stage_direction_only(None)


def test_scrub_self_vocative_leading_and_trailing():
    assert HY.scrub_self_vocative("Edna, we have to leave now.", "EDNA") == \
        "we have to leave now."
    assert HY.scrub_self_vocative("We have to leave now, Edna.", "Edna") == \
        "We have to leave now."


def test_scrub_self_vocative_ignores_other_names_and_inline():
    # not the speaker's own name -> untouched
    assert HY.scrub_self_vocative("Marcus, hurry.", "EDNA") == "Marcus, hurry."
    # in-line occurrence of own name -> untouched (only leading/trailing)
    assert HY.scrub_self_vocative("They called me Edna for years.", "Edna") == \
        "They called me Edna for years."


def test_clean_combines_paren_and_vocative():
    out = HY.clean_spoken_character_line(
        "Edna, the reading (she gasps) is impossible.", "EDNA")
    assert out == "the reading is impossible."


def test_is_truncated_signals():
    assert HY.is_truncated("We have to reach the reactor and")  # dangling conj
    assert HY.is_truncated("She turned to him,")               # trailing comma
    assert HY.is_truncated("The truth is (")                   # dangling paren
    assert HY.is_truncated("The. Stay with us.")               # lone stopword
    assert HY.is_truncated("It was the best of")               # trailing prep


def test_is_truncated_false_for_clean_lines():
    assert not HY.is_truncated("We have to reach the reactor before midnight.")
    assert not HY.is_truncated("Why would they do that?")
    assert not HY.is_truncated("No.")
    assert not HY.is_truncated('"Run," she said.')
    assert not HY.is_truncated("")


def test_functions_never_raise_on_none():
    assert HY.scrub_parentheticals(None) == ""
    assert HY.scrub_self_vocative(None, None) == ""
    assert HY.is_truncated(None) is False

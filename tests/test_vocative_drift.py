"""tests/test_vocative_drift.py -- BUG-LOCAL-233 vocative-drift pass.

A character line sometimes addresses the narration role label
"ANNOUNCER" vocatively ("It wasn't just geology, ANNOUNCER."). The
phantom-name gate whitelists every roster name, so "ANNOUNCER" -- always
on the roster -- was never flagged and the drift reached text_for_tts.

Coverage:

  A2  _format_last_lines renders announcer window entries as
      [narration], so the composing LLM is not shown an addressable
      "ANNOUNCER" speaker label above the WRITE LINE block.
  B   compose_line stamps a vocative_drift:ANNOUNCER compose-flag.
  C   strip_announcer_vocative removes the comma/boundary-anchored
      address; a plain noun reference and real cast names survive.

Pure-Python. No GPU. No LLM (compose_line is driven by a mock fn).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    _format_last_lines,
    compose_line,
    strip_announcer_vocative,
)


# ---------------------------------------------------------------------------
# C -- strip_announcer_vocative
# ---------------------------------------------------------------------------


class TestStripAnnouncerVocative:

    def test_trailing_vocative_before_period(self):
        out, n = strip_announcer_vocative(
            "It wasn't just geology, ANNOUNCER. It felt ancient."
        )
        assert out == "It wasn't just geology. It felt ancient."
        assert n == 1

    def test_trailing_vocative_at_end_of_line(self):
        out, n = strip_announcer_vocative(
            "We found evidence of complex structuring, ANNOUNCER."
        )
        assert out == "We found evidence of complex structuring."
        assert n == 1

    def test_mid_sentence_vocative(self):
        out, n = strip_announcer_vocative(
            "We're close, ANNOUNCER, very close now."
        )
        assert out == "We're close, very close now."
        assert n == 1

    def test_mid_sentence_vocative_semicolon_delimiter(self):
        # BUG-LOCAL-233 b003: a mid-sentence vocative can close on a
        # semicolon, not just a comma. Episode
        # signal_lost_singing_asteroid_language_20260521_120244 line
        # b003 reached the ledger unstripped because the MID regex
        # accepted only a comma as the closing delimiter.
        out, n = strip_announcer_vocative(
            "This isn't some old rock, ANNOUNCER; this thing has a cadence."
        )
        assert out == "This isn't some old rock; this thing has a cadence."
        assert n == 1

    def test_mid_sentence_vocative_colon_delimiter(self):
        out, n = strip_announcer_vocative(
            "We're nearly there, ANNOUNCER: the lock is holding."
        )
        assert out == "We're nearly there: the lock is holding."
        assert n == 1

    def test_leading_vocative_recapitalizes_next_word(self):
        out, n = strip_announcer_vocative("ANNOUNCER, we did it.")
        assert out == "We did it."
        assert n == 1

    def test_leading_vocative_after_sentence_boundary(self):
        out, n = strip_announcer_vocative("I knew it. ANNOUNCER, look here.")
        assert out == "I knew it. Look here."
        assert n == 1

    def test_case_insensitive(self):
        out, n = strip_announcer_vocative("It's over, announcer.")
        assert out == "It's over."
        assert n == 1

    def test_multiple_addresses_in_one_line(self):
        out, n = strip_announcer_vocative(
            "ANNOUNCER, listen. We are close, ANNOUNCER."
        )
        assert "ANNOUNCER" not in out
        assert "announcer" not in out.lower()
        assert n == 2

    def test_plain_noun_reference_is_left_untouched(self):
        # "the announcer" carries no comma/boundary delimiter -- it is a
        # noun reference, not a vocative address.
        text = "I could hear the announcer's voice in the static."
        out, n = strip_announcer_vocative(text)
        assert out == text
        assert n == 0

    def test_real_cast_name_address_is_left_untouched(self):
        # Characters addressing each other by name is normal dialogue.
        # The strip targets only the "ANNOUNCER" label.
        text = "Hold on, LEMMY. We are not done yet."
        out, n = strip_announcer_vocative(text)
        assert out == text
        assert n == 0

    def test_no_announcer_token_is_a_fast_noop(self):
        out, n = strip_announcer_vocative("Alice, run for the door!")
        assert out == "Alice, run for the door!"
        assert n == 0

    def test_empty_text(self):
        assert strip_announcer_vocative("") == ("", 0)


# ---------------------------------------------------------------------------
# A2 -- _format_last_lines relabels the announcer as narration
# ---------------------------------------------------------------------------


class TestLastSpokenNarrationLabel:

    def test_announcer_entry_renders_as_narration(self):
        out = _format_last_lines([
            ("LEMMY", "We're close."),
            ("ANNOUNCER", "Far across the void, a signal stirs."),
        ])
        assert "[LEMMY]: We're close." in out
        assert "[narration]: Far across the void, a signal stirs." in out
        assert "[ANNOUNCER]:" not in out

    def test_character_labels_unchanged(self):
        out = _format_last_lines([("ALICE", "Hello there.")])
        assert out == "[ALICE]: Hello there."

    def test_build_user_prompt_last_spoken_has_no_announcer_label(self):
        req = LineRequest(
            speaker="LEMMY", intent="reveal", mood="awe", target_words=10,
            canon_header="x",
            last_lines=[("ANNOUNCER", "And the Psyche mission circles on.")],
        )
        prompt = _build_user_prompt(req)
        assert "[narration]: And the Psyche mission circles on." in prompt
        assert "[ANNOUNCER]:" not in prompt


# ---------------------------------------------------------------------------
# B + C -- compose_line wires the vocative gate
# ---------------------------------------------------------------------------


class TestComposeLineVocativeGate:

    def test_character_line_is_stripped_and_flagged(self):
        def mock(messages, *, temperature, max_new_tokens, stop=None):
            return "It wasn't just geology, ANNOUNCER. It felt ancient."
        req = LineRequest(
            speaker="LEMMY", intent="reveal", mood="awe", target_words=8,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset({"LEMMY", "ANNOUNCER"}),
        )
        res = compose_line(creative_fn=mock, technical_fn=mock, req=req)
        assert res.text == "It wasn't just geology. It felt ancient."
        assert "ANNOUNCER" not in res.text
        assert "vocative_drift:ANNOUNCER" in res.compose_flags

    def test_announcer_speaker_is_exempt(self):
        # An ANNOUNCER-speaker line may legitimately reference its own
        # role -- the gate skips it.
        def mock(messages, *, temperature, max_new_tokens, stop=None):
            return "This, ANNOUNCER, is the signal you have waited for."
        req = LineRequest(
            speaker="ANNOUNCER", intent="open", mood="grand", target_words=9,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset({"LEMMY", "ANNOUNCER"}),
        )
        res = compose_line(creative_fn=mock, technical_fn=mock, req=req)
        assert "ANNOUNCER" in res.text
        assert "vocative_drift:ANNOUNCER" not in res.compose_flags

    def test_clean_character_line_gets_no_vocative_flag(self):
        def mock(messages, *, temperature, max_new_tokens, stop=None):
            return "Hold the line, we still have a chance now."
        req = LineRequest(
            speaker="LEMMY", intent="reveal", mood="tense", target_words=10,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset({"LEMMY", "ANNOUNCER"}),
        )
        res = compose_line(creative_fn=mock, technical_fn=mock, req=req)
        assert res.text == "Hold the line, we still have a chance now."
        assert not any(
            f.startswith("vocative_drift") for f in res.compose_flags
        )

    def test_semicolon_address_is_stripped_and_flagged(self):
        # BUG-LOCAL-233 b003 at the gate layer: the character line
        # reached the ledger with empty compose_flags. The gate must
        # now both strip the ";" address and stamp the drift flag.
        def mock(messages, *, temperature, max_new_tokens, stop=None):
            return (
                "This isn't some old rock, ANNOUNCER; "
                "this thing has a cadence."
            )
        req = LineRequest(
            speaker="LEMMY", intent="reveal", mood="awe", target_words=11,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset({"LEMMY", "ANNOUNCER"}),
        )
        res = compose_line(creative_fn=mock, technical_fn=mock, req=req)
        assert "ANNOUNCER" not in res.text
        assert res.text == "This isn't some old rock; this thing has a cadence."
        assert "vocative_drift:ANNOUNCER" in res.compose_flags


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

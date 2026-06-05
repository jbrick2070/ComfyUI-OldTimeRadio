"""BUG-LOCAL-295: a multi-word ALL-CAPS OTHER-roster name leaking into line
body is scrubbed (inside a *...* stage direction) or retried (bare noun slot);
single-word drama, vocatives, non-roster names, and clean lines are untouched.
The roster is the episode's ACTUAL drawn cast (req.allowed_people), never a
fixed list."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

lc = pytest.importorskip("_otr_line_composer")


class _Req:
    def __init__(self, speaker, people):
        self.speaker = speaker
        self.allowed_people = people


def test_bug295_roster_leak_names_filters_speaker_and_singleword():
    names = lc._roster_leak_names(
        _Req("HIRO PIERCE", ["HIRO PIERCE", "ERIN SPENDER", "MAEVE", "ANNOUNCER"])
    )
    assert "ERIN SPENDER" in names      # multi-word OTHER character -> leak set
    assert "HIRO PIERCE" not in names   # the speaker's own name -> excluded
    assert "MAEVE" not in names         # single word -> legitimate drama
    assert "ANNOUNCER" not in names     # single word


def test_bug295_scrub_inside_asterisks():
    out, retry = lc._scrub_or_flag_roster_leak(
        "*ERIN SPENDER the monkeys' enclosure*", ["ERIN SPENDER"]
    )
    assert "ERIN SPENDER" not in out
    assert "monkeys'" in out
    assert retry is None  # scrubbed in place, no retry needed


def test_bug295_bare_noun_slot_leak_flags_retry():
    out, retry = lc._scrub_or_flag_roster_leak(
        "safe in the ERIN SPENDER", ["ERIN SPENDER"]
    )
    assert retry == "ERIN SPENDER"  # noun-slot leak (scrub would break grammar)


def test_bug295_vocative_not_flagged():
    out, retry = lc._scrub_or_flag_roster_leak(
        "Get back here, ERIN SPENDER!", ["ERIN SPENDER"]
    )
    assert retry is None  # vocative (after a comma) is legitimate address


def test_bug295_single_word_drama_untouched():
    out, retry = lc._scrub_or_flag_roster_leak(
        "Who did this? Maeve.", ["ERIN SPENDER"]
    )
    assert retry is None
    assert out == "Who did this? Maeve."


def test_bug295_clean_line_unchanged():
    line = "The lighthouse stood against the dawn."
    out, retry = lc._scrub_or_flag_roster_leak(line, ["ERIN SPENDER"])
    assert retry is None
    assert out == line


def test_bug295_no_leak_names_is_inert():
    line = "safe in the ERIN SPENDER"
    out, retry = lc._scrub_or_flag_roster_leak(line, [])
    assert retry is None
    assert out == line

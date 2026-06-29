"""Story-quality C5 (S4) -- cliche exact-span replacement locks.

`repair_cliche_span` swaps the FIRST worn-cliche span for a curated plain phrase,
preserving a leading capital and a pronoun-carrying form's pronoun, and clears the
phrase from `flag_cliche` / `find_cliche_phrase`. Clean text is returned unchanged
(byte-identical), and a SECOND distinct cliche survives the single-span repair so
the composer can stamp `cliche_shipped_after_reroll`. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from nodes._otr_line_hygiene import (
    find_cliche_phrase,
    flag_cliche,
    repair_cliche_span,
)


def test_clean_text_unchanged():
    s = "The crew sealed the breach and moved on."
    assert repair_cliche_span(s) == s


def test_empty_and_none_safe():
    assert repair_cliche_span("") == ""
    assert repair_cliche_span(None) == ""


def test_swaps_known_cliche_and_clears_flag():
    out = repair_cliche_span("We are running out of time.")
    assert "running out of time" not in out.lower()
    assert flag_cliche(out)[0] is False
    assert out == "We are almost out of time."


def test_preserves_leading_capital():
    out = repair_cliche_span("Against all odds, we made it.")
    assert out[0].isupper()
    assert out.startswith("With everything against us")


def test_preserves_pronoun_form():
    out = repair_cliche_span("You're playing with fire, Chen.")
    assert out.startswith("You're ")
    assert "playing with fire" not in out.lower()


def test_first_span_only_second_survives():
    # Two distinct cliches: the FIRST is repaired; the second is left so the
    # composer stamps cliche_shipped_after_reroll rather than mass-rewriting.
    out = repair_cliche_span(
        "This changes everything, and there's no turning back.")
    assert "this changes everything" not in out.lower()
    assert find_cliche_phrase(out) != ""


def test_idempotent_on_repaired():
    once = repair_cliche_span("It hangs in the balance.")
    assert repair_cliche_span(once) == once
    assert flag_cliche(once)[0] is False


def test_every_cliche_pattern_has_a_repair():
    # Every _CLICHE_RES phrase the gate can flag must have a curated swap, so a
    # flagged line is always repairable (the stamp is a defensive backstop).
    samples = [
        "You're playing with fire.",
        "This changes everything.",
        "We're not leaving anything to chance.",
        "Leaving nothing to chance now.",
        "There's no turning back.",
        "Against all odds.",
        "It hangs in the balance.",
        "Over my dead body.",
        "Not on my watch.",
        "That is best left buried.",
        "We are running out of time.",
        "Before it's too late.",
        "Safety first.",
        "It all goes up in smoke.",
        "It will go up in smoke.",
    ]
    for s in samples:
        assert flag_cliche(s)[0] is True, f"sample not a cliche: {s}"
        assert flag_cliche(repair_cliche_span(s))[0] is False, (
            f"repair did not clear: {s}")

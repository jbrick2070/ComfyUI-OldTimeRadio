"""D3 -- the terminal delimiter, and the diagnostic that never named it.

THE DEFECT (PBUG-20260815-03). `_RE_END` demanded `END.` with a period. A model
that wrote a bare `END` fell past it, past `_RE_SPEAKER` (which needs a colon),
onto `BAD_LINE_SHAPE` -- and because `on_end` never fired, the end-of-text check
added `MISSING_END` as well. Two reported defects, one missing character.
`scifi_news_pro` died on it at 3.3 minutes.

The ladder then burned all four rungs re-emitting the same bare `END`, and that
half is the more interesting one: the retry plumbing was correct -- each rung
really did carry the rejected draft plus its defects forward. What it carried
was useless. `BAD_LINE_SHAPE` and `MISSING_END` say what is WRONG and never once
state the one-character fix. A model cannot infer "add a period" from "your
line's shape is bad" plus "the terminal marker is missing"; those are two
symptoms of one omission, not two facts to reason from.

So this file pins BOTH halves: the grammar accepts what a model actually writes,
and the repair note states the required shape. Fixing only the grammar leaves
the next delimiter defect just as unrepairable.

CPU-only: regex and string assembly. No model, no GPU.
"""

from __future__ import annotations

import pytest

from nodes import _otr_scifi_news_pro_markup as markup
from nodes import _otr_scifi_news_pro as scifi_news_pro


# --------------------------------------------------------------------------- #
# the grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line", [
    "END",          # the form the model actually wrote, and the whole defect
    "END.",         # the only form the old regex accepted
    "[END]",        # the lane brackets transport elsewhere
    "[END.]",
    "end",          # the classifier is case-insensitive and always was
    "End.",
    "END   ",       # trailing whitespace is transport, not content
])
def test_every_accepted_form_resolves_to_the_delimiter(line):
    assert markup._RE_END.match(line), f"{line!r} should be a terminal marker"


@pytest.mark.parametrize("line", [
    "[END",             # unpaired bracket
    "END]",             # unpaired the other way
    "END. Fade out.",   # trailing content
    "END: the last word",
    "THE END",          # content-bearing variant
    "ENDING",
    "[[END]]",
    "END .",            # a space before the period is not one of the forms
    "And so we END",
])
def test_a_near_miss_stays_a_LOUD_defect(line):
    """Widening a terminal delimiter into "anything containing END" is how a
    structural marker stops being structural. The parser has to be able to tell
    the end of the script from a line of dialogue about endings."""
    assert not markup._RE_END.match(line), (
        f"{line!r} must NOT be accepted as a terminal marker")


def test_a_bold_wrapped_END_arrives_as_a_bare_END():
    """Shape 4 unwraps balanced emphasis around transport, so `**END**` reaches
    the classifier as `END` -- which is only useful now that bare `END` is a
    form the classifier accepts."""
    assert markup._is_transport("END")
    assert markup._RE_END in markup._TRANSPORT_CLASSIFIERS


# --------------------------------------------------------------------------- #
# the diagnostic -- the half that burned the four rungs
# --------------------------------------------------------------------------- #
def _Defect(code_name: str, detail: str = ""):
    """A REAL `ParseDefect`, not a stand-in.

    The first cut of this file invented a stub with a `.kind` string attribute.
    Production sets `.code` to a `NewsProParseDefect` ENUM -- so the helper under
    test was written to match the stub, could never have fired on a real defect,
    and the suite would have been green over dead code. Build the real record
    and the test cannot drift from what the ladder actually passes.
    """
    return markup.ParseDefect(
        code=getattr(markup.NewsProParseDefect, code_name), detail=detail)


class TestTheRepairNoteStatesTheRequiredShape:
    def test_MISSING_END_gets_the_accepted_literals_verbatim(self):
        note = scifi_news_pro._end_delimiter_repair_note([_Defect("MISSING_END")])
        assert note, "a missing terminal marker must produce a repair rule"
        for form in scifi_news_pro._END_ACCEPTED_FORMS:
            assert f"'{form}'" in note, (
                f"the note never shows the accepted literal {form!r} -- naming "
                "the offence without the target is the whole defect")

    def test_a_BAD_LINE_SHAPE_carrying_END_also_triggers_it(self):
        """The live failure reported BOTH defects from one omission, and the
        bare `END` arrived as a BAD_LINE_SHAPE detail."""
        note = scifi_news_pro._end_delimiter_repair_note(
            [_Defect("BAD_LINE_SHAPE", "END")])
        assert "'END.'" in note

    def test_the_note_names_the_unpaired_bracket_cases(self):
        note = scifi_news_pro._end_delimiter_repair_note([_Defect("MISSING_END")])
        assert "[END" in note and "END]" in note

    def test_an_unrelated_defect_gets_NO_delimiter_lecture(self):
        """Self-silencing. A repair turn should carry the rules its own defects
        call for and nothing else -- padding every turn with every rule is how
        the one that matters gets lost."""
        assert scifi_news_pro._end_delimiter_repair_note(
            [_Defect("UNKNOWN_SPEAKER", "*SFX")]) == ""
        assert scifi_news_pro._end_delimiter_repair_note([]) == ""
        assert scifi_news_pro._end_delimiter_repair_note(None) == ""

    def test_a_BAD_LINE_SHAPE_about_something_else_is_not_hijacked(self):
        assert scifi_news_pro._end_delimiter_repair_note(
            [_Defect("BAD_LINE_SHAPE", "(he crosses to the window)")]) == ""

    def test_it_is_its_OWN_helper_not_a_branch_on_the_stage_direction_note(self):
        """Different defect data, different question. Folding two unrelated
        diagnoses into one function is how the next reader learns the wrong
        rule for their defect."""
        assert callable(scifi_news_pro._end_delimiter_repair_note)
        assert (scifi_news_pro._end_delimiter_repair_note
                is not scifi_news_pro._standalone_stage_direction_repair_note)
        # The stage-direction note must stay SILENT on a delimiter defect --
        # if it started answering this too, the split would be cosmetic.
        assert scifi_news_pro._standalone_stage_direction_repair_note(
            [_Defect("MISSING_END")], cast_names=()) == ""


def test_the_note_and_the_parser_cannot_drift():
    """A diagnostic that names a form its own validator rejects teaches the
    model a wrong target -- worse than saying nothing at all."""
    for form in scifi_news_pro._END_ACCEPTED_FORMS:
        assert markup._RE_END.match(form), (
            f"the repair note advertises {form!r} but the parser rejects it")

"""PBUG-20260812-03 -- the repair rule that could not fire on its own defect class.

FOUND BY A LIVE RUN. The `viz_green` leg of the 45-word every-visual-path
campaign (2026-08-12) died in `OTR_LedgerScriptWriter` after 3.0 minutes:

    [scifi_fable2] pass 'script' failed after 4 attempt(s):
    markup ladder exhausted; last defects:
    - UNKNOWN_SPEAKER: *SFX (line 25)
    - SKELETON_BREAK: character line (*SFX) after the last scene

THE MECHANISM. `_standalone_stage_direction_repair_note` exists to tell the
repair rung exactly how to fix an illegal stage-direction row. It fired only on
`BAD_LINE_SHAPE` whose detail opened with `(` or `[`. A stage direction written
WITH a colon -- `*SFX: a door slams` -- parses as a SPEAKER instead, so the
defect is `UNKNOWN_SPEAKER` and the detail opens with `*`. The note returned ""
and the repair rung got only the generic "repair the malformed FORMAT defects
below". The model re-emitted the same shape on all four attempts. That ending is
recorded twice already in `_otr_fable2_markup`'s own docstring; this is the
third.

WHAT THIS SUITE PINS, and every item was learned by getting it wrong first --
once from the QA pass, once from the kibitz panel (Codex + Antigravity,
`kibitz-runs/2026-08-12-writer-stage-direction-note/r2/`):

* **Fixtures come from the REAL PARSER, never hand-written.** The first version
  asserted on an invented row, `SKELETON_BREAK: [SOUND] after the last scene`,
  and passed -- while the real emission is `character line (*SFX) after the last
  scene`, which opens with "character" and could never match. A hand-written
  fixture proved a branch that could not execute.
* **The note takes TYPED `ParseDefect` objects.** `str(defect)` appends
  ` (line N)`, and re-parsing that string is what corrupted the token. The
  objects carry `code`, `detail` and `line_no` already.
* **`Fable2ParseDefect` is a PLAIN enum, not a `str` enum**, so codes are
  compared as MEMBERS. A string membership test would be False forever and the
  note would silently never fire -- the live defect, reintroduced by its own fix.
* **The two codes carry different data.** `UNKNOWN_SPEAKER.detail` is the bare
  label; `BAD_LINE_SHAPE.detail` is a line fragment (`line[:80]`). Roster
  resolution is meaningful only for the former.
* **A decorated REAL cast name gets a DIFFERENT rule, not silence.** Silence
  hands back the same generic instruction that already failed four attempts.
"""
from __future__ import annotations

import pytest

from nodes._otr_fable2_markup import (
    ANNOUNCER_NAME,
    Fable2ParseDefect,
    ParseDefect,
    parse_fable2_markup,
)
from nodes._otr_scifi_fable2 import _standalone_stage_direction_repair_note


CAST = ("Ada", "Bo")


def note(*defects, cast=CAST):
    return _standalone_stage_direction_repair_note(defects, cast_names=cast)


def script_with(*body_lines):
    """A minimal well-formed play with `body_lines` spliced into scene 1."""
    return "\n".join((
        "TITLE: The Test",
        "MUSIC: theme up",
        f"{ANNOUNCER_NAME}: Tonight, a test.",
        "SCENE 1: a room",
        "Ada: We begin.",
        *body_lines,
        "Bo: We end.",
        f"{ANNOUNCER_NAME}: That was a test.",
        "CODA: The end.",
        "MUSIC: theme down",
        "END.",
    ))


def real_defects(*body_lines, cast=CAST):
    """The typed defects the LIVE ladder hands the note."""
    _parsed, defects = parse_fable2_markup(script_with(*body_lines), cast)
    return defects


# ---------------------------------------------------------------------------
# Fixture integrity -- these guard every test below from passing vacuously
# ---------------------------------------------------------------------------
def test_the_fixture_helper_really_produces_typed_defects():
    defects = real_defects("*SFX: a door slams")
    assert defects, "the parser accepted the illegal row -- fixture is broken"
    assert all(isinstance(d, ParseDefect) for d in defects)
    assert any(d.code is Fable2ParseDefect.UNKNOWN_SPEAKER for d in defects)


def test_a_clean_script_raises_nothing_and_earns_no_note():
    assert real_defects() == ()
    assert note(*real_defects()) == ""


def test_the_defect_code_enum_is_NOT_a_string_enum():
    """Why codes are compared as members. If this ever becomes a `str` enum the
    member comparison still works -- but a future author must not "simplify" it
    back to strings on the assumption that it is one."""
    assert not isinstance(Fable2ParseDefect.UNKNOWN_SPEAKER, str)
    assert Fable2ParseDefect.UNKNOWN_SPEAKER != "UNKNOWN_SPEAKER"


# ---------------------------------------------------------------------------
# THE LIVE DEFECT
# ---------------------------------------------------------------------------
def test_THE_LIVE_DEFECT_now_gets_the_rule():
    defects = real_defects("*SFX: a door slams")
    got = note(*defects)
    assert got, ("the repair rung would still get no rule for the defect that "
                 "killed the viz_green leg: %s" % (defects,))
    assert "FORMAT REPAIR RULE" in got
    assert "*SFX" in got


def test_the_rule_names_the_speaker_label_case_explicitly():
    got = note(*real_defects("*SFX: a door slams"))
    assert "SPEAKER LABEL" in got
    assert "sound-effect speaker" in got


def test_the_rule_reports_the_line_number_it_was_given():
    """`line_no` travels as its own field now, instead of being smuggled inside
    a stringified detail."""
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, "*SFX", 25))
    assert "line 25" in got


@pytest.mark.parametrize("body", [
    "(a door slams)",            # no colon -> BAD_LINE_SHAPE
    "[a door slams]",
    "*SFX: a door slams",        # asterisk WITH a colon -> UNKNOWN_SPEAKER
    "[SOUND]: a door slams",
    "(NARRATOR): the night was long",
])
def test_every_real_stage_direction_shape_gets_the_rule(body):
    defects = real_defects(body)
    assert defects, body
    assert note(*defects), (body, defects)


def test_the_original_parenthetical_case_is_UNCHANGED():
    got = note(*real_defects("(a door slams)"))
    assert "FORMAT REPAIR RULE" in got
    assert "a door slams" in got


# ---------------------------------------------------------------------------
# A decorated REAL cast name -- the second rule
# ---------------------------------------------------------------------------
def test_a_REAL_cast_name_wearing_a_stray_marker_is_told_to_RESTORE_the_label():
    """THE PANEL'S BEST FINDING. `*Ada: Hello` looks exactly like `*SFX: ...`;
    only the roster separates them. Going silent was the first design and it was
    wrong -- silence returns the generic instruction that already failed four
    attempts. A decorated real name has a safe repair, so say it."""
    defects = real_defects("*Ada: We continue.")
    assert any(d.code is Fable2ParseDefect.UNKNOWN_SPEAKER for d in defects), (
        "fixture drift: the parser now resolves a stray-marker name, so this "
        "test no longer exercises the case it was written for -- re-derive it")
    got = note(*defects)
    assert "Restore the plain canonical label" in got
    assert "'Ada: '" in got
    assert "KEEP THE DIALOGUE EXACTLY AS WRITTEN" in got
    # And it must NOT carry the advice that deletes a character.
    assert "omit it when nonessential" not in got
    assert "there is no sound-effect speaker" not in got


def test_the_role_parenthetical_survives_the_decoration_strip():
    """`*Ada (Engineer)` must resolve to Ada, not to `Ada (Engineer`. The parser
    already understands trailing role parentheticals; mangling it here would
    defeat that and hand a real character the delete-me rule."""
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER,
                           "*Ada (Engineer)", 9))
    assert "Restore the plain canonical label" in got


def test_a_decorated_TYPO_name_still_gets_the_stage_direction_rule():
    """`*Adda` resolves to nothing on the roster, so it is treated as a stage
    direction -- which is the honest reading of an unknown decorated token."""
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, "*Adda", 9))
    assert "FORMAT REPAIR RULE" in got
    assert "Restore the plain canonical label" not in got


def test_an_UNDECORATED_misspelled_cast_name_gets_NO_advice_at_all():
    """The original promise, still kept: an ordinary unknown speaker is not this
    note's business, and inventing advice for it would lose a line."""
    defects = real_defects("Adda: We continue.")
    assert any(d.code is Fable2ParseDefect.UNKNOWN_SPEAKER for d in defects)
    assert note(*defects) == ""


def test_the_roster_check_is_case_and_space_insensitive():
    """Matches the parser's own `_speaker_key`, so the guard cannot be evaded by
    capitalization the roster does not use."""
    for label in ("(ada)", "(  Ada  )", "*ADA"):
        got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, label, 9))
        assert "Restore the plain canonical label" in got, label


def test_the_announcer_counts_as_roster_even_though_it_is_not_in_the_cast():
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER,
                           "(%s)" % ANNOUNCER_NAME, 9))
    assert "Restore the plain canonical label" in got


def test_a_protected_cast_defect_does_not_suppress_a_LATER_genuine_one():
    """A roster hit returns its own rule, so this asserts ordering explicitly:
    whichever matches FIRST wins, and neither silences the other."""
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, "*Adda", 4),
               ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, "*SFX", 25))
    assert "FORMAT REPAIR RULE" in got
    assert "*Adda" in got


@pytest.mark.parametrize("code", [
    Fable2ParseDefect.MISSING_END,
    Fable2ParseDefect.SKELETON_BREAK,
])
def test_an_unrelated_defect_code_is_silent(code):
    assert note(ParseDefect(code, "(something)", 3)) == ""


def test_an_empty_defect_list_is_silent():
    assert note() == ""


# ---------------------------------------------------------------------------
# BAD_LINE_SHAPE carries a LINE FRAGMENT, not a label
# ---------------------------------------------------------------------------
def test_a_BAD_LINE_SHAPE_detail_never_gets_a_roster_lookup():
    """Its detail is `line[:80]`, so a roster hit would be meaningless -- a line
    that merely BEGINS with a cast name is not a decorated label."""
    got = note(ParseDefect(Fable2ParseDefect.BAD_LINE_SHAPE,
                           "(Ada crosses to the window)", 12))
    assert "FORMAT REPAIR RULE" in got
    assert "Restore the plain canonical label" not in got


def test_the_prompt_does_not_claim_a_truncated_fragment_is_the_exact_row():
    """`BAD_LINE_SHAPE` details are cut at 80 chars, so the earlier wording --
    'the malformed source row is exactly ...' -- was false evidence handed to
    the repair rung."""
    got = note(ParseDefect(Fable2ParseDefect.BAD_LINE_SHAPE, "(" + "x" * 79, 12))
    assert "may be truncated" in got
    assert "is exactly" not in got


def test_an_UNKNOWN_SPEAKER_prompt_calls_the_detail_a_LABEL_not_a_row():
    got = note(ParseDefect(Fable2ParseDefect.UNKNOWN_SPEAKER, "*SFX", 25))
    assert "illegal speaker label" in got
    assert "is exactly" not in got


# ---------------------------------------------------------------------------
# SKELETON_BREAK -- documented as out of scope, and pinned so it stays honest
# ---------------------------------------------------------------------------
def test_a_real_SKELETON_BREAK_detail_cannot_open_with_a_marker():
    """Why `SKELETON_BREAK` is deliberately NOT in `_STAGE_DIRECTION_CODES`.
    Its detail is a descriptive sentence that merely CONTAINS the token. This is
    the assertion that would have caught the invented fixture."""
    defects = real_defects(f"{ANNOUNCER_NAME}: An outro.", "*SFX: a door slams")
    breaks = [d for d in defects if d.code is Fable2ParseDefect.SKELETON_BREAK]
    assert breaks, defects
    for defect in breaks:
        assert not str(defect.detail).lstrip().startswith(("(", "[", "*"))


def test_a_stage_direction_outside_a_scene_is_STILL_covered():
    """Because the same line always raises `UNKNOWN_SPEAKER` too -- which is why
    leaving `SKELETON_BREAK` out costs nothing."""
    defects = real_defects(f"{ANNOUNCER_NAME}: An outro.", "*SFX: a door slams")
    assert any(d.code is Fable2ParseDefect.SKELETON_BREAK for d in defects)
    assert note(*defects), defects

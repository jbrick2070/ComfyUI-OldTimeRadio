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
defects are `UNKNOWN_SPEAKER` / `SKELETON_BREAK` and the detail opens with `*`.
Three ways to miss the same rule, so the note returned "" and the repair rung
got only the generic "repair the malformed FORMAT defects below". The model
re-emitted the same shape on all four attempts and the ladder exhausted.

That ending is recorded twice already in `_otr_fable2_markup`'s own docstring --
a decorated label falling to the speaker catch-all, `UNKNOWN_SPEAKER` four
attempts running, the leg dying in the writer having never reached a video
engine. This is the third.

THE FIX IS PROMPT-ONLY and cannot make an invalid script valid: it widens which
defects the note fires on and adds a sentence naming the speaker-label case. It
changes no parser, no acceptance rule, and no schema.

WHAT MUST STAY NARROW: an `UNKNOWN_SPEAKER` for a MISSPELLED CAST NAME must not
get this note. Telling the model to fold a real character's line into a
neighbour, or to drop it, is worse advice than the failure it replaces -- so the
note still requires the offending TOKEN to look like a stage direction.
"""
from __future__ import annotations

import pytest

from nodes._otr_scifi_fable2 import _standalone_stage_direction_repair_note


def note(*rows):
    return _standalone_stage_direction_repair_note(tuple(rows))


# ---------------------------------------------------------------------------
# THE LIVE DEFECT -- the exact rows the failed leg produced
# ---------------------------------------------------------------------------
def test_the_exact_rows_from_the_failed_leg_now_get_the_rule():
    got = note("UNKNOWN_SPEAKER: *SFX (line 25)",
               "SKELETON_BREAK: character line (*SFX) after the last scene")
    assert got, ("the repair rung would get no rule for the defect that killed "
                 "the viz_green leg")
    assert "FORMAT REPAIR RULE" in got
    # It names the offending row back to the model, which is what makes a retry
    # a REPAIR rather than a regeneration.
    assert "*SFX" in got


def test_the_rule_names_the_speaker_label_case_explicitly():
    """The generic 'standalone row' wording did not cover a stage direction
    that carries a colon and therefore reads as a character."""
    got = note("UNKNOWN_SPEAKER: *SFX (line 25)")
    assert "SPEAKER LABEL" in got
    assert "sound-effect speaker" in got


# ---------------------------------------------------------------------------
# The shapes it must fire on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", [
    "BAD_LINE_SHAPE: (a door slams) (line 12)",      # the original case
    "BAD_LINE_SHAPE: [a door slams] (line 12)",
    "BAD_LINE_SHAPE: *a door slams* (line 12)",      # asterisk, no colon
    "UNKNOWN_SPEAKER: *SFX (line 25)",               # asterisk, WITH a colon
    "UNKNOWN_SPEAKER: (NARRATOR) (line 9)",
    "SKELETON_BREAK: [SOUND] after the last scene",
])
def test_every_stage_direction_shape_gets_the_rule(row):
    assert note(row), row


def test_the_original_parenthetical_case_is_UNCHANGED():
    """The widening must not have cost the case the note was written for."""
    got = note("BAD_LINE_SHAPE: (a door slams) (line 12)")
    assert "FORMAT REPAIR RULE" in got
    assert "(a door slams)" in got


# ---------------------------------------------------------------------------
# The shapes it must NOT fire on -- wrong advice is worse than none
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", [
    "UNKNOWN_SPEAKER: BO NI (line 4)",        # a misspelled/off-roster NAME
    "UNKNOWN_SPEAKER: ANNOUCNER (line 7)",    # a typo'd real label
    "SKELETON_BREAK: character line (ADA) after the last scene",
    "MISSING_END",
    "SHORT_SCENE: scene 2 has 3 words",
])
def test_a_real_name_defect_gets_NO_stage_direction_advice(row):
    """Telling the model to fold a real character's line into a neighbour, or
    to drop it, would lose a line the story needs."""
    assert note(row) == "", row


def test_an_empty_defect_list_is_silent():
    assert note() == ""


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def test_the_rule_is_returned_once_even_with_several_matching_rows():
    """The rung gets ONE rule, not one per defect -- the note returns on the
    first match by design."""
    got = note("UNKNOWN_SPEAKER: *SFX (line 25)",
               "UNKNOWN_SPEAKER: *MUSIC CUE (line 31)")
    assert got.count("FORMAT REPAIR RULE") == 1


def test_the_first_MATCHING_row_is_the_one_quoted():
    """A non-matching defect ahead of a matching one must not suppress it --
    the live leg's rows arrived in exactly that order."""
    got = note("SHORT_SCENE: scene 2 has 3 words",
               "UNKNOWN_SPEAKER: *SFX (line 25)")
    assert "*SFX" in got

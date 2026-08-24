"""PBUG-20260824-01 -- the lane that refused to produce an episode 60% of the time.

FOUND BY MEASUREMENT, not by review. An overnight loop (2026-08-24, ten passes
over all five writer banks through the canonical workflow) failed
`scifi_news_pro` on SIX of ten passes while every other bank failed zero. The
freshest capture, `tmp/_bankgate_scifi_news_pro.log`, died in `_pass_script`
after four attempts with seventeen defects.

WHAT THE CAPTURE ACTUALLY CONTAINED, and the first framing got this wrong. The
go-forward plan called it "the `**ANNOUNCER` markdown leak" (Bug Bible 12.132).
Markdown was real but it was a MINORITY: of the six rejected speaker tokens,
five carried a comma delivery tag, four were a shortened cast name, and only
three wore markdown. No single mechanism covered even half.

  UNKNOWN_SPEAKER: DR. CHEN                 <- short form of 'Dr. Haorong Chen'
  UNKNOWN_SPEAKER: **DR. CHEN**, urgent     <- markdown AND tag AND short form
  UNKNOWN_SPEAKER: ELI, to himself          <- tag only
  UNKNOWN_SPEAKER: ELI, whispering          <- tag only

AND A PERFECT MATCHER WOULD NOT HAVE SAVED THE LEG. Proved before any code was
written, by rebuilding the draft from its defect fingerprint and feeding the
real parser a roster in which every supplied label already resolved: five
`BAD_LINE_SHAPE` narration rows and four `SKELETON_BREAK`s survived. Two
further mechanisms were hiding behind the loud one:

* unlabelled prose action rows ("Eli opens his package."), which no matcher can
  fix because they name no speaker at all;
* a mid-scene ANNOUNCER row, which CLOSES the story frame -- so every character
  line after it was "after the last scene". The driver's first reading called
  those breaks derivative of the unknown speakers; the state machine says
  otherwise (`on_speaker` never changes state for an unresolved label), and
  both the codex and Fable r1 lanes caught it.

WHY IT BURNED ALL FOUR ATTEMPTS. `_standalone_stage_direction_repair_note`
returned on the FIRST matching defect. The first one here was the line-5 action
row, so every repair turn carried the fold-the-stage-direction rule and never
once mentioned the six broken labels. The model was never told what was
actually failing.

WHAT THIS SUITE PINS:
* one shared matcher (Bug Bible 12.132 verify-condition 3), used by the parser
  AND the repair note, so they can never disagree about who is a legal speaker;
* relaxed rungs that are CLOSED and structural -- never fuzzy, never
  edit-distance, and never resolving an ambiguous alias;
* the raw label survives into the defect, so decoration is still diagnosable;
* the repair channel names EVERY defect class it saw, not just the first;
* salvage delivers an episode rather than refusing one, and says so loudly.
"""
import pytest

from nodes._otr_scifi_news_pro_markup import (
    ANNOUNCER_NAME,
    NewsProParseDefect,
    ParseDefect,
    build_speaker_roster,
    parse_scifi_news_pro_markup,
)
from nodes._otr_scifi_news_pro import (
    _frame_order_repair_note,
    _resolves_to_cast,
    _standalone_stage_direction_repair_note,
)

D = NewsProParseDefect
CAST = ("Dr. Haorong Chen", "Eli Marsh")


# --- the real captured draft, rebuilt from its defect fingerprint -----------
# Lines 8/10/13/19 were CLEAN in the capture. 13 is the only clean line between
# the last in-scene defect (12) and the first "after the last scene" break
# (14), and only a RESOLVED ANNOUNCER row can make that transition -- which is
# how the mid-scene announcer was identified without the draft being logged.
PASS11_DRAFT = "\n".join([
    "TITLE: The Quiet Frequency",                                         # 1
    "MUSIC: low hum, tape hiss",                                          # 2
    "ANNOUNCER: Tonight, a signal nobody asked for.",                     # 3
    "SCENE 1: A cramped lab, after midnight",                             # 4
    "**DR. CHEN**, hunched over a microscope, speaks with a raspy voice.",  # 5
    "DR. CHEN: The sample is not decaying.",                              # 6
    "Eli, a stranger, enters, carrying a package.",                       # 7
    "ELI: You called me at three in the morning.",                        # 8
    "**DR. CHEN**, urgent: Look at the plate.",                           # 9
    "ELI: That is not possible.",                                         # 10
    "**DR. CHEN**, smiling weakly: I know.",                              # 11
    "**DR. CHEN** activates a console, typing frantically.",              # 12
    "ANNOUNCER: The lab went quiet that night.",                          # 13
    "**DR. CHEN**, whispering: It is still growing.",                     # 14
    "ELI opens his package, revealing a small, blinking device.",         # 15
    "ELI, to himself: They knew.",                                        # 16
    "**DR. CHEN** collapses.",                                            # 17
    "ELI, whispering: Doctor.",                                           # 18
    "ANNOUNCER: Some questions answer themselves.",                       # 19
    "ELI, whispering: Doctor, please.",                                   # 20
    "CODA: The frequency never stopped.",                                 # 21
    "MUSIC: closing tone",                                                # 22
    "END.",                                                               # 23
])


def _codes(defects):
    return [d.code for d in defects]


# ---------------------------------------------------------------------------
# The resolver: what must now resolve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label, expected", [
    ("Dr. Haorong Chen", "Dr. Haorong Chen"),      # exact still wins
    ("DR. CHEN", "Dr. Haorong Chen"),              # honorific + surname
    ("Chen", "Dr. Haorong Chen"),                  # bare surname
    ("Haorong", "Dr. Haorong Chen"),               # first name
    ("ELI, whispering", "Eli Marsh"),              # comma delivery tag
    ("**DR. CHEN**", "Dr. Haorong Chen"),          # markdown
    ("**DR. CHEN**, urgent", "Dr. Haorong Chen"),  # markdown AND tag
    ("**ANNOUNCER", ANNOUNCER_NAME),               # Bible 12.132's headline
    ("Chen (Virologist)", "Dr. Haorong Chen"),     # role parenthetical + alias
])
def test_the_captured_label_shapes_resolve(label, expected):
    assert build_speaker_roster(CAST).resolve(label)[0] == expected


@pytest.mark.parametrize("label", [
    "IVOR",                # near-miss of nobody; no fuzzy remap
    "STRANGER, urgent",    # a tag does not make a stranger legal
    "THOR",                # a genuinely invented character
    "**LUCAS**",           # decoration does not make one legal either
])
def test_a_genuinely_unknown_speaker_still_REFUSES(label):
    """The widening must not become blanket acceptance. `IVOR` against a cast
    holding `IVO` is pinned elsewhere as a hard no-remap; the same contract
    holds here for every relaxed rung."""
    assert build_speaker_roster(CAST).resolve(label)[0] is None


def test_an_AMBIGUOUS_alias_resolves_for_NOBODY():
    """THE COLLISION GUARD, and it degrades rather than guessing. Two Chens
    both claim 'Chen' and 'DR. CHEN', so NEITHER gets them and both fall back
    to exact-only. Silently merging two characters would mis-cast a voice and
    corrupt the ledger -- far worse than the refusal it replaces."""
    roster = build_speaker_roster(("Dr. Haorong Chen", "Dr. Wei Chen"))
    assert roster.resolve("Chen")[0] is None
    assert roster.resolve("DR. CHEN")[0] is None
    # Exact identity is untouched, and a still-unique alias still works.
    assert roster.resolve("Dr. Haorong Chen")[0] == "Dr. Haorong Chen"
    assert roster.resolve("Haorong")[0] == "Dr. Haorong Chen"


def test_a_COMMA_BEARING_canonical_name_matches_itself_first():
    """A comma is not always a delivery tag -- it is legal INSIDE a canonical
    roster label, and `DR. ORION NINE, SENIOR SIGNAL ANALYST` is a real one
    with its own passing test in the markup suite. Exact identity runs before
    any rung, so the tag-stripper never gets to mangle it."""
    cast = ("DR. ORION NINE, SENIOR SIGNAL ANALYST", "Elodie Vancourt")
    roster = build_speaker_roster(cast)
    name, how = roster.resolve("dr. orion nine, senior signal analyst")
    assert name == "DR. ORION NINE, SENIOR SIGNAL ANALYST"
    assert how == "exact"
    # The name proper is still reachable when the role clause is dropped.
    assert roster.resolve("DR. ORION NINE")[0] == cast[0]


def test_authored_aliases_are_additive_and_still_guarded():
    """The operator asked for a model's judgment on aliases rather than only
    token rules, because a nickname is not derivable. They arrive as DATA and
    are held to the SAME ambiguity guard."""
    roster = build_speaker_roster(CAST, {"Dr. Haorong Chen": ["Doc", "Hao"]})
    assert roster.resolve("Doc")[0] == "Dr. Haorong Chen"
    assert roster.resolve("**Doc**, urgent")[0] == "Dr. Haorong Chen"
    # A hostile or careless alias cannot capture another character.
    hostile = build_speaker_roster(CAST, {"Dr. Haorong Chen": ["ELI"]})
    assert hostile.resolve("ELI")[0] is None, (
        "an alias claimed by two characters must resolve for neither")
    assert hostile.resolve("Eli Marsh")[0] == "Eli Marsh"
    # An alias naming nobody on the roster is dropped, not trusted.
    junk = build_speaker_roster(CAST, {"Nobody At All": ["Ghost"]})
    assert junk.resolve("Ghost")[0] is None


def test_ONE_MATCHER_the_repair_note_agrees_with_the_parser():
    """Bug Bible 12.132 verify-condition 3. These lived in two modules as two
    hand-written ladders; `_resolves_to_cast`'s docstring claimed it imported
    the parser's rule while actually copying it. A parser that accepts a label
    the note calls illegal tells the model to DELETE A LINE the parser would
    have taken."""
    roster = build_speaker_roster(CAST)
    for label in ("DR. CHEN", "ELI, whispering", "**DR. CHEN**, urgent",
                  "IVOR", "STRANGER"):
        assert _resolves_to_cast(label, roster) is (
            roster.resolve(label)[0] is not None), label


# ---------------------------------------------------------------------------
# The parse: honest mode still refuses, and refuses for the RIGHT reasons
# ---------------------------------------------------------------------------
def test_the_captured_draft_loses_every_UNKNOWN_SPEAKER():
    """The nine speaker defects are gone. The rest survive on purpose: they
    are real defects the repair ladder exists to get fixed."""
    _parsed, defects = parse_scifi_news_pro_markup(PASS11_DRAFT, CAST)
    assert D.UNKNOWN_SPEAKER not in _codes(defects)
    assert D.CAST_MEMBER_SILENT not in _codes(defects), (
        "Dr. Chen was reported SILENT only because his lines never resolved; "
        "he speaks four times")
    assert D.BAD_LINE_SHAPE in _codes(defects)
    assert D.SKELETON_BREAK in _codes(defects)


def test_the_frame_break_is_NOT_derivative_of_the_unknown_speakers():
    """The driver's first reading called every SKELETON_BREAK derivative. It
    is not: an unresolved label never changes parser state. Removing ONLY the
    mid-scene ANNOUNCER collapses the breaks, while leaving the labels broken
    does not."""
    def breaks(text):
        _p, defects = parse_scifi_news_pro_markup(text, CAST)
        return [d for d in defects
                if d.code is D.SKELETON_BREAK
                and "after the last scene" in d.detail]

    # The draft as written closes the frame at line 13 and again at 19.
    assert len(breaks(PASS11_DRAFT)) == 4
    # Move ONLY the first announcer row back into the drama. The speaker
    # labels are untouched, so if the breaks were derivative of them nothing
    # would change -- instead they collapse to the one caused by line 19.
    without_mid_announcer = PASS11_DRAFT.replace(
        "ANNOUNCER: The lab went quiet that night.", "ELI: Doctor, sit down.")
    assert len(breaks(without_mid_announcer)) == 1
    # Move the second one too and they vanish entirely.
    frame_intact = without_mid_announcer.replace(
        "ANNOUNCER: Some questions answer themselves.",
        "ELI: Stay with me.")
    assert breaks(frame_intact) == [], (
        "the mid-scene ANNOUNCER rows are what close the frame, not the "
        "unresolved speaker labels")


# ---------------------------------------------------------------------------
# The repair channel: every class, not just the first
# ---------------------------------------------------------------------------
def test_the_repair_note_names_EVERY_defect_class_it_saw():
    """THE REASON FOUR ATTEMPTS BURNED. The note returned on the first match,
    which here was the line-5 action row, so the six broken speaker labels
    were never mentioned in any of the four repair turns."""
    _p, defects = parse_scifi_news_pro_markup(
        PASS11_DRAFT, CAST[:1] + ("Somebody Else",))
    got = _standalone_stage_direction_repair_note(
        defects, cast_names=CAST[:1] + ("Somebody Else",))
    assert "unlabelled row" in got, "the action rows must be named"
    assert "is not one of this episode's characters" in got, (
        "the unresolved labels must be named too, in the SAME turn")


def test_an_UNLABELLED_prose_row_finally_gets_advice():
    """`Eli opens his package.` opens with a letter, so it matched no branch
    and earned NOTHING. Five of them sat in the leg that died."""
    defects = (ParseDefect(D.BAD_LINE_SHAPE, "Eli opens his package.", 15),)
    got = _standalone_stage_direction_repair_note(defects, cast_names=CAST)
    assert "every nonblank row must begin with a legal label" in got
    assert "Eli opens his package." in got


def test_the_frame_order_note_fires_and_is_self_silencing():
    """The ANNOUNCER-as-narrator mechanism had no note at all, so the model
    could not know the announcer row was the problem rather than the drama."""
    defects = (ParseDefect(
        D.SKELETON_BREAK, "character line (Eli Marsh) after the last scene", 16),)
    got = _frame_order_repair_note(defects)
    assert "ANNOUNCER is the show's frame" in got
    assert "move the ANNOUNCER outro down" in got
    assert _frame_order_repair_note(()) == ""
    assert _frame_order_repair_note(
        (ParseDefect(D.MISSING_END, "", None),)) == ""


# ---------------------------------------------------------------------------
# Salvage: deliver the episode rather than refuse it
# ---------------------------------------------------------------------------
def test_SALVAGE_delivers_the_episode_that_died():
    """Operator, 2026-08-24: "accepts sometimes a wrong name populated but
    shouldn't kill the whole episode." THE LAW agrees -- an audit may improve
    a story, it may never fail one."""
    parsed, defects = parse_scifi_news_pro_markup(
        PASS11_DRAFT, CAST, salvage=True)
    assert defects == (), f"salvage must deliver, got {defects}"
    spoken = [ln.speaker for scene in parsed.scenes for ln in scene.lines]
    assert spoken.count("Dr. Haorong Chen") == 4
    assert spoken.count("Eli Marsh") == 5
    # The frame is kept: the announcer still speaks, and the drama survives.
    assert parsed.announcer_intro and parsed.announcer_outro
    assert parsed.coda and parsed.music_close
    # And every trade is on the record.
    assert len(parsed.dropped_rows) == 5
    assert any("resolved to" in n for n in parsed.normalizations)


def test_SALVAGE_is_OFF_by_default_so_the_honest_gate_never_moved():
    """Salvage is the ladder's last rung, not a looser parser. Nothing about
    the honest path changed, which is what keeps this a fix and not a shim."""
    _p, defects = parse_scifi_news_pro_markup(PASS11_DRAFT, CAST)
    assert defects, "the default parse must still refuse this draft"


def test_SALVAGE_still_REFUSES_a_draft_with_no_story_in_it():
    """It admits wrong names; it does not manufacture an episode. A draft
    whose every row is unlabelled has no drama to deliver."""
    empty = "\n".join([
        "TITLE: Nothing At All",
        "MUSIC: hum",
        "ANNOUNCER: Tonight, nothing.",
        "SCENE 1: a room",
        "Something happens offstage.",
        "ANNOUNCER: And that was nothing.",
        "CODA: Nothing ends.",
        "MUSIC: out",
        "END.",
    ])
    parsed, defects = parse_scifi_news_pro_markup(empty, CAST, salvage=True)
    assert parsed is None and defects
    assert any("salvage cannot proceed" in str(d.detail) for d in defects)


# ---------------------------------------------------------------------------
# NO SFX -- ANYWHERE, AND NEVER AT THE COST OF A LINE OF DIALOGUE
# ---------------------------------------------------------------------------
# Operator, 2026-08-24: "there should be no SFX", "we ripped out all SFX
# layers", "we need to fix so there are no SFX in the ledger" -- and the
# constraint that decides HOW: "they should not chunk off dialogue, we should
# just never do any SFX."
#
# This pipeline has no sound effects. The `[SFX: ...]` ledger token was removed
# 2026-07-01 and the SFX bed subsystem was ripped 2026-08-06, so a cue row can
# only ever become a character reading a stage direction aloud in their own
# voice, or a ledger row with nowhere to live.
@pytest.mark.parametrize("label", [
    "SFX", "sfx", "*SFX", "**SFX**", "[SFX]", "(sound)", "SOUND EFFECT",
    "FOLEY", "ENV", "AMBIENCE", "Atmos", "STINGER", "fx",
])
def test_a_SOUND_CUE_is_never_admitted_as_a_character(label):
    """Undecorated `SFX:` matters most here. A decoration-based rule would
    have caught `*SFX` and missed the bare form entirely -- and the bare form
    is the one that would have walked into the ledger with a voice."""
    roster = build_speaker_roster(CAST)
    assert roster.resolve(label)[0] is None
    draft = PASS11_DRAFT.replace(
        "ELI, whispering: Doctor, please.", f"{label}: a door slams")
    parsed, defects = parse_scifi_news_pro_markup(draft, CAST, salvage=True)
    assert defects == ()
    spoken = {ln.speaker for scene in parsed.scenes for ln in scene.lines}
    assert spoken == {"Dr. Haorong Chen", "Eli Marsh"}, (
        f"a sound cue reached the cast as {spoken - set(CAST)}")
    assert not parsed.adopted_speakers
    assert any("sound cue" in row for row in parsed.dropped_rows)


def test_dropping_a_CUE_never_costs_a_line_of_DIALOGUE():
    """THE OPERATOR'S CONSTRAINT, pinned. The first draft of this rule keyed on
    DECORATION and would have thrown away `(SOMEONE NEW): I have something to
    say.` -- brackets and all, dialogue included. Decoration is not evidence
    that something is not a person; a cue WORD is."""
    draft = PASS11_DRAFT.replace(
        "ELI, whispering: Doctor, please.",
        "(SOMEONE NEW): I have something to say.")
    parsed, defects = parse_scifi_news_pro_markup(draft, CAST, salvage=True)
    assert defects == ()
    lines = [ln.text for scene in parsed.scenes for ln in scene.lines]
    assert "I have something to say." in lines, (
        "an invented character's DIALOGUE must survive; only cues are dropped")
    assert parsed.adopted_speakers, "and they must be cast, not discarded"
    # AND THE NAME IS FIT TO PRINT. An adopted speaker becomes a real cast row
    # -- it reaches the voice deal, the captions and the credit roll -- so it
    # must not carry the model's punctuation with it.
    assert parsed.adopted_speakers == ("SOMEONE NEW",), (
        f"adopted name still decorated: {parsed.adopted_speakers}")
    spoken = {ln.speaker for scene in parsed.scenes for ln in scene.lines}
    assert "SOMEONE NEW" in spoken
    # The raw label stays traceable in the receipt.
    assert any("'(SOMEONE NEW)' ADOPTED as 'SOMEONE NEW'" in n
               for n in parsed.normalizations)


def test_the_repair_rule_NEVER_SHOWS_the_model_a_sound_cue_token():
    """Bug Bible 12.132: do not teach a model the forbidden form. This rule is
    RETURNED INTO THE WRITER'S PROMPT, and until 2026-08-24 it carried the
    worked example `'*SFX: a door slams'` -- the one place in the whole
    generation path that named a token from a subsystem ripped out twice."""
    got = _standalone_stage_direction_repair_note(
        (ParseDefect(D.UNKNOWN_SPEAKER, "*SFX", 25),), cast_names=CAST)
    assert got, "the rule must still fire"
    # The model may see its OWN offending row quoted back -- that is evidence,
    # and it is the only legitimate way a cue token appears in the prompt. So
    # strip the evidence sentence and assert the RULE text invents none.
    rule_text = got.split("The parser reported")[0]
    assert "SFX" not in rule_text.upper(), (
        f"the rule text still names a sound-cue token: {rule_text!r}")
    assert "a door slams" not in got
    assert "sound-effect speaker" not in got


def test_SALVAGE_adopts_an_invented_character_rather_than_refusing():
    """A speaker the roster cannot place is a character the model wrote. The
    producer deals them a voice; the artifact records the adoption."""
    draft = PASS11_DRAFT.replace(
        "ELI, whispering: Doctor, please.", "THORNE: I warned you both.")
    parsed, defects = parse_scifi_news_pro_markup(draft, CAST, salvage=True)
    assert defects == ()
    assert "THORNE" in parsed.adopted_speakers
    spoken = {ln.speaker for scene in parsed.scenes for ln in scene.lines}
    assert "THORNE" in spoken, (
        "an adopted speaker must reach the script, or the line is lost")

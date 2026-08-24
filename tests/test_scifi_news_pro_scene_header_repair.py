"""A bare SCENE header, and the repair note that never named the fix.

THE DEFECT (PBUG-20260824-02, found live by the fast-iteration scifi_news_pro
rate measurement started to retire PBUG-20260824-01). `_RE_SCENE` demands
`SCENE <n>: <setting>` -- a nonempty description after the colon. A model
that writes only `SCENE 1:` with nothing after it falls past every classifier
and lands as `BAD_LINE_SHAPE` carrying the bare header. Because `on_scene`
never fires, every character line that follows reads as "before SCENE 1",
cascading into a wall of SKELETON_BREAKs -- one leg died with 12 defects from
one missing clause, and salvage could not recover it because no scene ever
opened for a line to belong to.

Same shape as the bare-END bug (PBUG-20260815-03, see test_d3_end_delimiter.py):
the generic repair turn says WHAT is wrong and never states the fix, so a
retry with no targeted note just repeats the omission -- confirmed live, the
second attempt in the failing leg reproduced nearly the same defect list.

UNLIKE END, this is not a grammar widening. A scene's setting is real content
(it lands in `scenes[].setting` and feeds shot direction), so the fix teaches
the model to WRITE one rather than accepting its absence.

CPU-only: regex and string assembly. No model, no GPU.
"""

from __future__ import annotations

from nodes import _otr_scifi_news_pro_markup as markup
from nodes import _otr_scifi_news_pro as scifi_news_pro


# --------------------------------------------------------------------------- #
# the grammar -- unchanged, pinned so a future edit cannot silently widen it
# --------------------------------------------------------------------------- #
def test_a_bare_scene_header_is_rejected_by_the_grammar():
    """This fix teaches the model to avoid the shape, not the parser to
    accept it -- a setting description is real content the ledger reads."""
    assert not markup._RE_SCENE.match("SCENE 1:")


def test_a_scene_header_with_a_setting_is_accepted():
    m = markup._RE_SCENE.match("SCENE 1: a potting shed, before dawn")
    assert m
    assert m.group(1) == "1"
    assert m.group(2) == "a potting shed, before dawn"


# --------------------------------------------------------------------------- #
# the diagnostic -- the half that would otherwise burn every rung
# --------------------------------------------------------------------------- #
def _Defect(code_name: str, detail: str = ""):
    """A REAL `ParseDefect`, matching production's `.code`/`.detail` shape --
    see test_d3_end_delimiter.py's `_Defect` for why this is not a stub."""
    return markup.ParseDefect(
        code=getattr(markup.NewsProParseDefect, code_name), detail=detail)


class TestTheRepairNoteStatesTheRequiredShape:
    def test_a_bare_SCENE_1_triggers_the_note(self):
        note = scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE", "SCENE 1:")])
        assert note, "a bare SCENE header must produce a repair rule"
        assert "SCENE" in note and "setting" in note.lower()

    def test_a_bare_two_digit_scene_number_also_triggers_it(self):
        note = scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE", "SCENE 12:")])
        assert note

    def test_the_note_shows_a_grammar_valid_example(self):
        """The example inside the note must itself parse -- a repair rule
        that shows a form the parser rejects teaches the wrong target."""
        note = scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE", "SCENE 1:")])
        assert markup._RE_SCENE.match("SCENE 1: a potting shed, before dawn")
        assert "a potting shed, before dawn" in note

    def test_an_unrelated_defect_gets_NO_scene_lecture(self):
        """Self-silencing, same convention as every other note here."""
        assert scifi_news_pro._scene_header_repair_note(
            [_Defect("UNKNOWN_SPEAKER", "*SFX")]) == ""
        assert scifi_news_pro._scene_header_repair_note([]) == ""
        assert scifi_news_pro._scene_header_repair_note(None) == ""

    def test_a_scene_header_that_already_has_a_setting_is_not_flagged(self):
        """A BAD_LINE_SHAPE for a DIFFERENT reason must not be mistaken for
        the bare-header shape just because it starts with SCENE."""
        assert scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE",
                      "SCENE 1: a potting shed -- extra trailing junk")]) == ""

    def test_a_BAD_LINE_SHAPE_about_something_else_is_not_hijacked(self):
        assert scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE", "(he crosses to the window)")]) == ""
        assert scifi_news_pro._scene_header_repair_note(
            [_Defect("BAD_LINE_SHAPE", "END")]) == ""

    def test_it_is_its_OWN_helper_not_a_branch_on_the_end_delimiter_note(self):
        assert callable(scifi_news_pro._scene_header_repair_note)
        assert (scifi_news_pro._scene_header_repair_note
                is not scifi_news_pro._end_delimiter_repair_note)
        # The END note must stay SILENT on a scene-header defect -- if it
        # started answering this too, the split would be cosmetic.
        assert scifi_news_pro._end_delimiter_repair_note(
            [_Defect("BAD_LINE_SHAPE", "SCENE 1:")]) == ""

    def test_the_stage_direction_note_does_not_contradict_this_one(self):
        """QA CAUGHT THIS (2026-08-24), and it is the whole reason this test
        exists. `_standalone_stage_direction_repair_note`'s catch-all for an
        unlabelled row opening with a letter matches a bare 'SCENE 1:' too --
        without the guard it added "fold it in or drop it" to the SAME repair
        turn that this note tells to keep the line and add a setting. Both
        notes are concatenated together in `_run_markup_ladder`, so silence
        from the OTHER note is what makes this fix's instruction the only
        one the model reads."""
        defects = [_Defect("BAD_LINE_SHAPE", "SCENE 1:")]
        scene_note = scifi_news_pro._scene_header_repair_note(defects)
        stage_note = scifi_news_pro._standalone_stage_direction_repair_note(
            defects, cast_names=())
        assert scene_note, "the targeted note must still fire"
        assert stage_note == "", (
            "the stage-direction note must go SILENT on a bare scene header "
            "-- its 'fold it in or drop it' advice contradicts this note's "
            "'add a setting' advice, and both land in the same repair turn")

    def test_an_unlabelled_prose_row_still_gets_its_own_note(self):
        """The guard above must be scoped to the bare-scene shape only -- a
        genuine unlabelled action row must still be caught, or the fix for
        this bug quietly reopens the one it borrowed its branch from."""
        stage_note = scifi_news_pro._standalone_stage_direction_repair_note(
            [_Defect("BAD_LINE_SHAPE", "Eli opens his package.")],
            cast_names=())
        assert "fold" in stage_note.lower() or "drop" in stage_note.lower()


def test_the_note_and_the_parser_cannot_drift():
    """A diagnostic that shows an example its own validator rejects teaches
    the model a wrong target -- worse than saying nothing at all."""
    note = scifi_news_pro._scene_header_repair_note(
        [_Defect("BAD_LINE_SHAPE", "SCENE 1:")])
    assert markup._RE_SCENE.match("SCENE 1: a potting shed, before dawn"), (
        "the repair note's own example must be accepted by the real grammar")
    assert note  # keeps the note referenced so this test fails loudly if renamed


# --------------------------------------------------------------------------- #
# salvage -- the GENERAL backstop, independent of the specific malformed shape
# --------------------------------------------------------------------------- #
#
# Operator, 2026-08-24: "you never know what crazy stuff [a model will
# throw at the parser] -- like maybe a model will say SCENE [3] or some
# crazy stuff." The targeted repair note above only fires on the ONE shape
# actually observed live (a bare "SCENE <n>:"). Salvage must not depend on
# recognizing which specific way the header failed -- these tests use a
# DIFFERENT malformed shape than the one the note detects, to prove the
# rescue is genuinely shape-independent rather than accidentally narrow.
CAST = ("Dr. Haorong Chen", "Eli Marsh")


def _draft(scene_line: str) -> str:
    return "\n".join([
        "TITLE: The Quiet Frequency",
        "MUSIC: low hum, tape hiss",
        "ANNOUNCER: Tonight, a signal nobody asked for.",
        scene_line,
        "Dr. Haorong Chen: The readings do not make sense.",
        "Eli Marsh: Then we are reading them wrong.",
        "ANNOUNCER: The signal went quiet before dawn.",
        "CODA: Nobody slept that night.",
        "MUSIC: the hum returns, and out",
        "END.",
    ])


def test_a_scene_header_shape_nobody_specifically_handled_still_salvages():
    """`SCENE THREE` -- spelled out, in the operator's "you never know what
    crazy stuff a model will throw at the parser" spirit -- matches NEITHER
    the real grammar, NOR the bare-header note's regex, NOR (unlike a
    bracketed variant) the speaker catch-all. It is simply an unrecognized
    line, dropped in salvage same as any other -- and the rescue has to come
    from the REAL dialogue that follows it, not from recognizing this shape
    at all."""
    shape = "SCENE THREE"
    assert not markup._RE_SCENE.match(shape)
    assert not scifi_news_pro._RE_SCENE_HEADER_BARE.match(shape + ":")
    assert not markup._RE_SPEAKER.match(shape)

    parsed, defects = markup.parse_scifi_news_pro_markup(
        _draft(shape), CAST, salvage=True)
    assert defects == (), f"salvage must deliver, got {defects}"
    assert len(parsed.scenes) == 1
    assert parsed.scenes[0].n == 1
    spoken = [ln.speaker for ln in parsed.scenes[0].lines]
    assert spoken == ["Dr. Haorong Chen", "Eli Marsh"]


def test_a_bare_scene_header_also_salvages_via_the_general_path():
    """The shape the targeted note DOES handle must still salvage on its
    own if a retry never reaches it (e.g. the ladder is exhausted first)."""
    draft = _draft("SCENE 1:")
    parsed, defects = markup.parse_scifi_news_pro_markup(
        draft, CAST, salvage=True)
    assert defects == (), f"salvage must deliver, got {defects}"
    assert len(parsed.scenes) == 1
    assert parsed.scenes[0].setting == ""


def test_without_salvage_the_same_draft_still_refuses_loudly():
    """The honest (non-salvage) parse must NOT gain this leniency -- only
    the last-resort rung may rescue a missing scene header."""
    parsed, defects = markup.parse_scifi_news_pro_markup(
        _draft("SCENE THREE"), CAST, salvage=False)
    assert parsed is None
    assert any(d.code is markup.NewsProParseDefect.BAD_LINE_SHAPE
               for d in defects)


def test_salvage_still_refuses_when_nothing_is_actually_salvageable():
    """The pre-existing refusal must survive: a draft with no scene AND no
    resolvable dialogue is still an empty episode, not a rescued one."""
    empty = "\n".join([
        "TITLE: Nothing Happened",
        "MUSIC: low hum",
        "ANNOUNCER: Tonight, nothing at all.",
        "CODA: The end.",
        "MUSIC: out",
        "END.",
    ])
    parsed, defects = markup.parse_scifi_news_pro_markup(
        empty, CAST, salvage=True)
    assert parsed is None and defects

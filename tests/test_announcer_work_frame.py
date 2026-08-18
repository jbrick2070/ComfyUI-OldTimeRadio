"""PBUG-20260817-04 -- the announcer's WORK phrase is rendered, never composed.

THE DEFECT, from a published episode. The announcer was handed
``WORK: a scene from Nonsense Novels`` by a pack seam that literally says
*"Use ONLY the WORK title and the proper names in the cast list below; invent
none"* -- and said *"we gather for 'The Adventure of the Purloined Paper'"*, a
work that does not exist. The closing coda in the SAME episode named the source
correctly, because the coda is a TEMPLATE and the intro was not. Supplying the
fact is necessary and not sufficient; prompt wording had already been proven
unreliable on this seam (Bible ``12.103``, the Verona artifact).

So the work-title half stops being the model's to write, exactly as
``compose_news_coda`` already does with its Python-owned fact.

WHAT THESE TESTS PIN:
  * the shakespeare locator form, with act/scene as SPOKEN ORDINAL WORDS --
    "Act I" is a TTS hazard ("act eye") and digits render inconsistently;
  * the subtitle form, and that the subtitle is marked as OURS -- a bare colon
    puts an LLM title where radio announced real chapters, so a listener hears
    a chapter that does not exist;
  * that ``media_archive`` gets NO frame. ``work_title`` means the PUBLICATION
    there, and 56 of 98 live ledgers carry one -- an ungated read announces
    "a scene from Now See Hear!" on 57% of that lane, a worse defect than the
    one being fixed;
  * that the splice REPLACES sentence 1 (the seam's own work-naming sentence)
    and keeps the model's intrigue -- prepending ships a doubled "Tonight,";
  * and that everything degrades to the composed line. An audit may never fail
    an episode (operator law 2026-07-22).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_line_composer as LC  # noqa: E402


# --------------------------------------------------------------------------- #
# the three frame forms
# --------------------------------------------------------------------------- #
def test_shakespeare_gets_the_locator_form_in_spoken_ordinals():
    assert LC.build_work_frame(
        work_title="The Tempest", author="William Shakespeare",
        act=1, scene=2, episode_title="Shattered Histories",
    ) == "The Tempest, by William Shakespeare, Act One, Scene Two"


def test_the_locator_beats_the_subtitle():
    """A real pointer into the work outranks a title we invented."""
    frame = LC.build_work_frame(
        work_title="The Tempest", act=1, scene=2,
        episode_title="A Hot Take",
    )
    assert frame == "The Tempest, Act One, Scene Two"
    assert "Hot Take" not in frame


def test_public_domain_gets_the_subtitle_form_marked_as_ours():
    """`an episode we call` is load-bearing, not decoration."""
    frame = LC.build_work_frame(
        work_title="Nonsense Novels", author="Stephen Leacock",
        episode_title="The Blackwood Enigma",
    )
    assert frame == (
        'Nonsense Novels, by Stephen Leacock, '
        'an episode we call "The Blackwood Enigma"'
    )


def test_a_subtitle_restating_the_work_is_dropped():
    """"The Canterville Ghost, an episode we call The Canterville Ghost"."""
    assert LC.build_work_frame(
        work_title="The Canterville Ghost",
        episode_title="the canterville ghost!",
    ) == "The Canterville Ghost"


def test_no_work_title_means_no_frame():
    """The signal to leave the composed opening alone."""
    assert LC.build_work_frame(work_title="", episode_title="Whatever") == ""
    assert LC.build_work_frame(work_title="   ") == ""


@pytest.mark.parametrize("act, scene", [
    # No usable ACT at all -- an unusable SCENE alone still yields the
    # act-only locator, which `test_act_without_a_usable_scene...` pins.
    (None, None), ("", ""), ("one", "two"), (0, 2), (99, 1), (0, 0),
])
def test_unusable_act_scene_falls_through_rather_than_speaking_a_digit(
        act, scene):
    frame = LC.build_work_frame(
        work_title="Some Work", act=act, scene=scene,
        episode_title="Our Title",
    )
    assert "Act" not in frame
    assert frame == 'Some Work, an episode we call "Our Title"'


def test_act_without_a_usable_scene_still_names_the_act():
    assert LC.build_work_frame(
        work_title="King Lear", act=3, scene=None,
    ) == "King Lear, Act Three"


# --------------------------------------------------------------------------- #
# the lane gate -- the media_archive collision stays closed
# --------------------------------------------------------------------------- #
def test_media_archive_publication_never_becomes_a_frame():
    """The caller gates; an empty work_title must produce nothing.

    `work_title` holds the PUBLICATION on media_archive ("Now See Hear!"),
    which is why the writer passes "" for a non-adaptation lane.
    """
    assert LC.build_work_frame(
        work_title="", author="", episode_title="Some Episode",
    ) == ""


# --------------------------------------------------------------------------- #
# the splice
# --------------------------------------------------------------------------- #
def test_splice_replaces_sentence_one_and_keeps_the_intrigue():
    composed = (
        "Tonight, from the cluttered confines of an office, we gather for "
        "'The Adventure of the Purloined Paper', starring THE GREAT DETECTIVE. "
        "But what secret lies within that locked filing cabinet?"
    )
    out = LC.splice_work_frame(composed, "Nonsense Novels, by Stephen Leacock")

    assert out.startswith(
        "Tonight, a scene from Nonsense Novels, by Stephen Leacock.")
    assert "Purloined Paper" not in out, "the invented title survived"
    assert "But what secret lies within" in out, "the intrigue was discarded"
    assert out.count("Tonight,") == 1, "doubled framing"


def test_splice_with_a_single_sentence_yields_just_the_frame():
    out = LC.splice_work_frame(
        "Tonight we gather for 'The Invented Thing'.", "Real Work")
    assert out == "Tonight, a scene from Real Work."
    assert "Invented" not in out


def test_no_frame_leaves_the_composed_line_untouched():
    """THE LAW: degrade, never fail."""
    composed = "Tonight, a quiet room and a waiting telephone."
    assert LC.splice_work_frame(composed, "") == composed


def test_empty_composed_line_still_gets_the_frame():
    assert LC.splice_work_frame("", "Real Work") == (
        "Tonight, a scene from Real Work.")


@pytest.mark.parametrize("terminator", [".", "!", "?"])
def test_sentence_split_handles_every_terminator(terminator):
    composed = f"First sentence here{terminator} Second sentence survives."
    out = LC.splice_work_frame(composed, "W")
    assert out == "Tonight, a scene from W. Second sentence survives."


@pytest.mark.parametrize("honorific", ["Mr.", "Mrs.", "Dr.", "St.", "Capt."])
def test_an_honorific_does_not_end_sentence_one(honorific):
    """THE BUG THIS ALMOST SHIPPED WITH.

    A naive split on `[.!?]\\s+` fires inside "Mr. Holmes", so everything
    after the honorific becomes the "remainder" -- and the model's INVENTED
    TITLE survives into the line this function exists to clean. An announcer
    opening naming a period character is the common case, not an edge one.
    """
    composed = (
        f"Tonight, {honorific} Holmes enters the office and we gather for "
        "'The Adventure of the Purloined Paper'. But what waits inside?"
    )
    out = LC.splice_work_frame(composed, "Nonsense Novels")

    assert out == (
        "Tonight, a scene from Nonsense Novels. But what waits inside?")
    assert "Purloined" not in out, "the invented title survived the splice"


def test_splice_is_idempotent():
    """A second application must not double the frame sentence."""
    once = LC.splice_work_frame(
        "Tonight, something invented. The intrigue.", "Real Work")
    assert LC.splice_work_frame(once, "Real Work") == once
    assert once.count("a scene from") == 1


def test_a_single_sentence_containing_an_honorific_yields_just_the_frame():
    out = LC.splice_work_frame(
        "Tonight Mr. Holmes gathers for 'The Invented Thing'", "Real Work")
    assert out == "Tonight, a scene from Real Work."


def test_the_tail_can_actually_reach_the_composer():
    """THE DEAD-CODE PIN, and the reason it exists.

    `_run_writer_tail` is a SEPARATE METHOD, not a closure over `run()` --
    its docstring says it consumes only `ctx`. The J.7 block first reached
    for `_OTRLC`, which is bound inside `run()`, so it raised `NameError` on
    every episode and the block's own `except Exception` swallowed it: the
    fix was dead code that logged a warning and left the invented title
    standing. Every test still passed, because passing tests proved only
    that it failed SAFELY -- which is not the same as working.

    This asserts at the source that the tail binds the composer locally and
    calls it through that binding, so the failure cannot silently return.
    """
    writer_src = (REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
        encoding="utf-8")
    start = writer_src.index("--- J.7. The announcer's WORK phrase")
    end = writer_src.index("keeping the composed announcer opening", start)
    block = writer_src[start:end]

    assert "import _otr_line_composer as _OTRLC_TAIL" in block, (
        "the tail must import the composer itself -- run()'s _OTRLC is not "
        "in this method's scope")
    assert block.count("_OTRLC_TAIL.build_work_frame") == 1
    assert block.count("_OTRLC_TAIL.splice_work_frame") == 1
    # No call may go through the run()-scoped name.
    assert block.replace("_OTRLC_TAIL.", "").count("_OTRLC.") == 0, (
        "a call still routes through run()'s _OTRLC and will NameError")


def test_the_frame_is_exported():
    """The writer's tail imports these by name."""
    for symbol in ("build_work_frame", "splice_work_frame",
                   "WORK_FRAME_SENTENCE"):
        assert symbol in LC.__all__
        assert hasattr(LC, symbol)

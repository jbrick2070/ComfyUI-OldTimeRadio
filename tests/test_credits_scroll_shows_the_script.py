"""The credits roll must have the transcript ON SCREEN, first frame to last.

OPERATOR-CAUGHT 2026-08-12, watching a 45-word leg: "the script scroll is not
appearing at all". It was appearing -- for about 13 seconds of a 20.8 second
roll. Measured on the published artifact by counting lit pixels in the
transcript column:

    t=1s   blank        t=9s   full        t=16s  fading
    t=3s   blank        t=12s  full        t=18s  blank
    t=6s   full         t=14s  full        t=20s  blank

Both ends were dead. The scroll ran between CANVAS edges, and the canvas
overstates the content at both: ``_build_right`` starts drawing a quarter
screen down (breathing room) and pads below the last line, on a canvas sized to
at least three screens. So the window opened on blank panel above the first
line and closed on blank panel below the last one -- and the fade-out landed on
the empty end, which is why it read as "no script at all".

It hid on full-length episodes because a 1,500-word transcript rolls for a
minute and a few seconds of margin at each end vanish into it. A 45-word
episode is ~113 words, so the margins were most of the roll. The sweep that
renders one short episode per visual engine is exactly the thing that exposed
it.

The fix scrolls the CONTENT SPAN: open at the first line, rest the last line on
the bottom edge, and when the content is shorter than the screen, hold it fully
visible rather than drifting through emptiness.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from nodes.video_engine import _TelemetryHUDRenderer  # noqa: E402


def _panel(n_lines):
    """A HUD post-roll over a transcript of roughly ``n_lines`` spoken lines."""
    return _TelemetryHUDRenderer(1920, 1080, 25, {
        "title": "The Probe",
        "scenes": [{"scene_num": "1", "env": "Studio", "items": [
            # `type: dialogue` is what the renderer keys on -- an item without
            # it draws NOTHING, which silently makes a 60-line fixture the same
            # height as a 3-line one. Asserted below via the span, so this
            # fixture cannot rot back into testing an empty panel.
            {"type": "dialogue", "char": "ASTRA", "preset": "vz_a",
             "text": "Line number %d of the classified transcript." % i}
            for i in range(n_lines)]}],
        "dossier": [{"header": "BRIEF", "lines": ["A tense debate."]}],
        "telemetry": {"peak": "1.6", "speed": "42"},
    })


def _ink(img, x_from):
    grey = img.convert("L")
    col = grey.crop((x_from, 0, grey.width, grey.height))
    return sum(1 for px in col.getdata() if px > 70)


def _ink_at(panel, frac_of_roll, total):
    fi = int(total * frac_of_roll)
    return _ink(panel.render(fi, total), panel.RIGHT_X)


@pytest.mark.parametrize("n_lines", [4, 40])
def test_the_transcript_is_visible_at_the_START_of_the_roll(n_lines):
    """The old scroll opened on the breathing room above the first line."""
    panel = _panel(n_lines)
    total = panel.hud_frames()

    # Past the fade-in (6%), before any meaningful travel.
    assert _ink_at(panel, 0.10, total) > 500, "roll opens on an empty panel"


@pytest.mark.parametrize("n_lines", [4, 40])
def test_the_transcript_is_visible_at_the_END_of_the_roll(n_lines):
    """The old scroll ran past the last line into the trailing pad, so the
    credits faded out on nothing. This is the operator's actual complaint."""
    panel = _panel(n_lines)
    total = panel.hud_frames()

    # At 92% the scroll has reached frac=1.0 and holds; before the fade-out.
    assert _ink_at(panel, 0.93, total) > 500, "roll ends on an empty panel"


def test_a_SHORT_transcript_never_scrolls_out_of_frame():
    """Content shorter than the screen has nowhere to go: it must simply hold,
    fully visible, for the whole roll."""
    panel = _panel(3)
    total = panel.hud_frames()

    samples = [_ink_at(panel, f, total) for f in (0.10, 0.30, 0.50, 0.70, 0.93)]
    assert all(s > 500 for s in samples), (
        "short transcript left the frame: %s" % samples)


def test_the_scroll_measures_CONTENT_not_canvas():
    """The canvas is at least three screens tall regardless of how little is on
    it (`est_h = max(est_h, self.h * 3)`), so a canvas-based span would report
    travel for a transcript that needs none."""
    panel = _panel(3)

    assert panel._right.height >= panel.h * 3, "fixture no longer over-sized"
    assert panel._right_span() < panel._right.height, (
        "span is measuring the canvas, not the ink")


def test_the_fixture_actually_draws_its_lines():
    """Guards every other test here. The first version of this fixture used
    `speaker`/`voice` keys the renderer ignores, so a 60-line transcript built
    the same 711px panel as a 3-line one and 'it did not move' looked like a
    code defect. A longer script must produce a taller span."""
    assert _panel(60)._right_span() > _panel(3)._right_span() * 3


def test_a_LONG_transcript_still_actually_travels():
    """The fix must not pin a long roll in place -- it still has to scroll."""
    panel = _panel(60)
    total = panel.hud_frames()
    assert panel._right_span() > panel.h, "fixture is too short to scroll"

    first = panel.render(int(total * 0.10), total)
    last = panel.render(int(total * 0.93), total)
    assert list(first.getdata()) != list(last.getdata()), (
        "a long transcript did not move")

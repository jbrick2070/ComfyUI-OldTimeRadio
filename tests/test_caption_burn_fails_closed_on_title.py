"""A planned hero title must never be dropped by a silent passthrough.

``OTR_CaptionBurn`` has FIVE no-error exits that return the input video
untouched, and ``burn_captions`` DEFAULTS TO FALSE:

  1. captions OFF                       -- the default
  2. no timed ledger resolves
  3. the caption style is unknown       -- (None, reason) -> ValueError -> caught
  4. the ledger has no speech lines     -- same route
  5. any other ValueError from the burn -- ffmpeg missing, input missing, the
     ffmpeg call itself failing

All five were harmless while the title was baked into procgen pixels. The moment
the title lives ONLY in the ASS, each one publishes an episode with NO TITLE and
a cheerful log line. Three independent reviews of the build spec landed on this
same hazard, and it is why the procgen suppression is the FOURTH build step
rather than the first.

The contract these tests pin:
  * NO plan supplied  -> every historical passthrough is unchanged. Captions are
    genuinely best-effort and must stay that way.
  * A plan supplied   -> the burn runs regardless of ``burn_captions``, a missing
    or speechless ledger still yields a TITLE-ONLY .ass, and anything that
    prevents the title REFUSES.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from nodes._otr_captions import (TitlePlanError, build_ass_from_ledger,  # noqa: E402
                                 title_events_from_plan)
from nodes.otr_caption_burn import OTRCaptionBurn  # noqa: E402
from nodes.video_engine import _CRTRenderer  # noqa: E402

W, H, FPS, TOTAL = 1920, 1080, 25, 200


@pytest.fixture()
def plan():
    vol = np.linspace(0.0, 1.0, TOTAL).astype("float32")
    freqs = [np.zeros(32, dtype="float32") for _ in range(TOTAL)]
    waves = [np.zeros(64, dtype="float32") for _ in range(TOTAL)]
    r = _CRTRenderer(W, H, "The Probe", vol, freqs, waves, FPS, timing={
        "music_open_start_f": 10, "music_open_end_f": 60, "first_dialogue_f": 80,
    })
    return r.title_card_plan(main_frames=TOTAL)


@pytest.fixture()
def plan_json(plan):
    return json.dumps(plan)


# ---------------------------------------------------------------------------
# EXIT 1 -- captions OFF
# ---------------------------------------------------------------------------

def test_no_plan_still_passes_through_when_captions_are_off():
    """Unchanged behaviour, asserted so the fail-closed work cannot creep into
    the caption feature and start blocking clean masters."""
    out, report = OTRCaptionBurn().burn("C:/nowhere/ep_silent.mp4",
                                        burn_captions=False)
    assert out == "C:/nowhere/ep_silent.mp4"
    assert "passthrough" in report.lower()


def test_a_plan_burns_even_though_captions_are_off(plan_json, tmp_path):
    """The title is NOT a caption. It must not inherit the caption on/off widget.

    ffmpeg is absent here, so the burn fails -- the point is WHICH failure: a
    refusal, never the silent clean-master passthrough that exit 1 used to give.
    """
    vid = tmp_path / "ep_silent.mp4"
    vid.write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="hero title card was planned"):
        OTRCaptionBurn().burn(str(vid), burn_captions=False,
                              ffmpeg="definitely-not-a-real-ffmpeg-binary",
                              title_card_plan_json=plan_json)


# ---------------------------------------------------------------------------
# EXIT 2 + 4 -- no ledger / no speech lines still yield a TITLE-ONLY .ass
# ---------------------------------------------------------------------------

def test_an_unreadable_ledger_still_produces_the_title(plan, tmp_path):
    out = tmp_path / "title_only.ass"
    path, report = build_ass_from_ledger(str(tmp_path / "missing_ledger.json"),
                                         title_plan=plan, out_path=str(out))
    assert path, report
    text = out.read_text(encoding="utf-8")
    assert "Style: TITLE," in text
    assert "Dialogue: 1," in text


def test_a_speechless_ledger_still_produces_the_title(plan, tmp_path):
    led = tmp_path / "ep_ledger.json"
    led.write_text(json.dumps({"episode_id": "ep", "cast": [], "lines": []}),
                   encoding="utf-8")
    out = tmp_path / "title_only.ass"
    path, _ = build_ass_from_ledger(str(led), title_plan=plan, out_path=str(out))
    assert path, "a speechless ledger dropped the title -- exit 4 is still open"
    assert "Dialogue: 1," in out.read_text(encoding="utf-8")


def test_without_a_plan_a_speechless_ledger_still_declines(tmp_path):
    """The reorder must not turn "nothing to caption" into a spurious .ass."""
    led = tmp_path / "ep_ledger.json"
    led.write_text(json.dumps({"episode_id": "ep", "cast": [], "lines": []}),
                   encoding="utf-8")
    path, report = build_ass_from_ledger(str(led), out_path=str(tmp_path / "x.ass"))
    assert path is None and "no speech lines" in report


def test_a_title_only_burn_writes_its_ass_beside_the_episode_not_into_the_cwd(
        plan_json, tmp_path, monkeypatch):
    """A title-only burn has NO ledger, so the .ass path cannot be derived from
    one -- and the fallback wrote a RELATIVE name into the process CWD, which on
    a live server is the repo root. Rendered assets belong in the episode
    folder; nothing is ever parked elsewhere to be moved later.

    ``ffmpeg`` is pointed at THIS interpreter, deliberately: a missing binary
    refuses before the .ass is ever built, so it would not reach the window in
    which the stray file appeared. A real executable that exits nonzero does --
    the .ass gets written, then the burn fails. Asserting from a DIFFERENT CWD
    is the whole point.
    """
    import sys

    ep = tmp_path / "episodes" / "ep01"
    ep.mkdir(parents=True)
    vid = ep / "ep01_procgen_blended.mp4"
    vid.write_bytes(b"\x00\x00")
    elsewhere = tmp_path / "server_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(RuntimeError):
        OTRCaptionBurn().burn(str(vid), burn_captions=False,
                              ffmpeg=sys.executable,
                              title_card_plan_json=plan_json)

    assert list(elsewhere.iterdir()) == [], \
        f"stray files in the CWD: {[p.name for p in elsewhere.iterdir()]}"
    assert list(ep.glob("*.ass")), \
        "the title .ass was not written beside the episode"


# ---------------------------------------------------------------------------
# EXIT 3 -- the unknown style, the fifth exit a third reviewer found
# ---------------------------------------------------------------------------

def test_an_unknown_style_refuses_instead_of_dropping_the_title(plan_json, tmp_path):
    vid = tmp_path / "ep_silent.mp4"
    vid.write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="hero title card was planned"):
        OTRCaptionBurn().burn(str(vid), burn_captions=True,
                              caption_style="no_such_style",
                              title_card_plan_json=plan_json)


# ---------------------------------------------------------------------------
# The plan on the wire -- a broken plan is not "no title wanted"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,why", [
    ("{not json", "malformed JSON"),
    ("[1, 2, 3]", "not an object"),
    ('{"version": 1, "frames": []}', "no frames"),
])
def test_a_broken_plan_refuses_rather_than_reading_as_no_title(raw, why):
    with pytest.raises(RuntimeError):
        OTRCaptionBurn().burn("C:/nowhere/ep.mp4", burn_captions=False,
                              title_card_plan_json=raw)


def test_an_empty_plan_string_is_the_no_title_signal():
    """An episode with no card sends "" -- and only "". That is what keeps a
    card-less episode from being mistaken for a broken one."""
    out, report = OTRCaptionBurn().burn("C:/nowhere/ep.mp4", burn_captions=False,
                                        title_card_plan_json="   ")
    assert out == "C:/nowhere/ep.mp4" and "passthrough" in report.lower()


def test_an_unknown_plan_version_refuses(plan):
    plan["version"] = plan["version"] + 99
    with pytest.raises(TitlePlanError, match="refusing to guess"):
        title_events_from_plan(plan)


# ---------------------------------------------------------------------------
# The ASS the emitter writes
# ---------------------------------------------------------------------------

def test_events_are_frame_bounded_and_contiguous(plan):
    """One event per line per frame, and this frame's end IS next frame's start.

    A 1-centisecond gap between consecutive frames would flicker the title, and
    at 24 fps the naive rounding produces exactly that.
    """
    import re
    events = title_events_from_plan(plan)
    spans = {}
    for e in events:
        s, en = re.match(r"Dialogue: 1,([^,]+),([^,]+),", e).groups()
        spans.setdefault(s, en)
    starts = sorted(spans)
    for i in range(len(starts) - 1):
        assert spans[starts[i]] == starts[i + 1], (
            f"gap or overlap between {starts[i]} and {starts[i + 1]}")


def test_the_sdh_style_line_is_untouched(plan, tmp_path):
    """TITLE is an ADDITIVE second style. The accessibility master must not move."""
    out = tmp_path / "c.ass"
    build_ass_from_ledger(str(tmp_path / "none.json"), title_plan=plan,
                          out_path=str(out))
    text = out.read_text(encoding="utf-8")
    sdh = [ln for ln in text.splitlines() if ln.startswith("Style: SDH,")]
    assert sdh == [
        "Style: SDH,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H70000000,"
        "0,0,0,0,100,100,0,0,3,5,0,2,40,40,90,1"
    ], sdh


def test_the_title_style_carries_a_real_outline(plan, tmp_path):
    """The whole point: BorderStyle=1 with a non-zero Outline width, which is
    exactly what could not exist in the lighten-only procgen layer."""
    out = tmp_path / "c.ass"
    build_ass_from_ledger(str(tmp_path / "none.json"), title_plan=plan,
                          out_path=str(out))
    line = next(ln for ln in out.read_text(encoding="utf-8").splitlines()
                if ln.startswith("Style: TITLE,"))
    # Field 0 is the "Style: TITLE" token itself, so every V4+ field sits one
    # index later than the Format: header suggests. Index by NAME to make that
    # explicit -- getting it wrong is how this test first went red against a
    # style line that was correct all along.
    header = ("Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
              "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
              "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
              "Alignment,MarginL,MarginR,MarginV,Encoding").split(",")
    f = dict(zip(header, line.split(",")))
    assert f["BorderStyle"] == "1", \
        f"BorderStyle is {f['BorderStyle']}, expected 1 (outline + shadow)"
    assert int(f["Outline"]) >= 3, f"Outline width {f['Outline']} is too thin"
    assert f["OutlineColour"] == "&H00000000", "OutlineColour is not opaque black"
    assert f["Alignment"] == "7", \
        "Alignment must be top-left to match the planner's PIL origins"


@pytest.mark.parametrize("title", [
    "{braces}", "back\\slash", "line\nbreak", "comma, title",
    "Ünïcødé", "", "supercalifragilisticexpialidociousandthensome",
])
def test_hostile_titles_do_not_corrupt_the_ass(title):
    """Override tags go on AFTER escaping. An unescaped brace would otherwise
    open an override block and swallow the rest of the line."""
    vol = np.linspace(0.0, 1.0, TOTAL).astype("float32")
    freqs = [np.zeros(32, dtype="float32") for _ in range(TOTAL)]
    waves = [np.zeros(64, dtype="float32") for _ in range(TOTAL)]
    r = _CRTRenderer(W, H, title, vol, freqs, waves, FPS, timing={
        "music_open_start_f": 10, "music_open_end_f": 40, "first_dialogue_f": 60,
    })
    plan = r.title_card_plan(main_frames=TOTAL)
    events = title_events_from_plan(plan)
    for e in events:
        body = e.split(",,", 1)[1]
        # Exactly the override blocks the emitter opened -- no stray braces
        # smuggled in from the title text.
        assert body.count("{") == body.count("}")
        assert "\n" not in e

"""SDH caption builder regression: scripts/otr_captions.py.

Covers timecode formatting, word wrap, multi-line cue chunking, time
distribution, SDH line rules (CPS lint), and ASS structure -- including the
accessibility contract: dialogue + speaker name are WHITE with the name BOLD
(weight is the speaker cue, never color) on a single opaque box.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAP = _REPO_ROOT / "nodes" / "_otr_captions.py"

# `_otr_captions` is loaded FLAT (no parent package), so its internal
# `from . import _otr_ledger_consumers` falls back to a bare import that needs
# nodes/ on the path. Without this, build_ass_from_ledger raised
# ModuleNotFoundError whenever this file ran on its own, and only passed when a
# sibling test module happened to put nodes/ on sys.path first -- a test that
# depends on another test's import side effects is a test that does not run.
if str(_REPO_ROOT / "nodes") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "nodes"))

spec = importlib.util.spec_from_file_location("otr_captions", _CAP)
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


def test_timecode_format():
    assert cap.ass_timecode(0) == "0:00:00.00"
    assert cap.ass_timecode(9.5) == "0:00:09.50"
    assert cap.ass_timecode(3661.234) == "1:01:01.23"
    assert cap.ass_timecode(-5) == "0:00:00.00"


def test_wrap_respects_max_chars():
    lines = cap.wrap_words("the quick brown fox jumps over the lazy dog again", 20)
    assert all(len(ln) <= 20 for ln in lines)
    assert " ".join(lines).split() == "the quick brown fox jumps over the lazy dog again".split()


def test_chunk_into_two_line_cues():
    text = " ".join(["word"] * 40)  # forces several cues
    cues = cap.chunk_into_cues(text, max_chars=cap.MAX_CHARS_PER_LINE, max_lines=2)
    for c in cues:
        assert c.count("\\N") <= 1  # at most 2 physical lines per cue


def test_distribute_time_no_overlap_and_bounded():
    spans = cap.distribute_time(3, 10.0, 19.0)
    assert spans[0][0] == 10.0
    assert spans[-1][1] == 19.0
    for (s, e), (s2, _e2) in zip(spans, spans[1:]):
        assert e <= s2 + 1e-9  # non-overlapping, contiguous
        assert e >= s


def _write_ledger(tmp_path) -> Path:
    led = {
        "episode_id": "test_ep",
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "ANTON BEATTY"},
        ],
        "lines": [
            {"char_id": "c01", "speaker_role": "announcer", "start_s": 9.5,
             "dur_s": 9.19, "text": "Ladies and gentlemen, gather round your radios "
             "for an tale of science and urgency tonight."},
            {"char_id": "c02", "speaker_role": "character", "start_s": 18.7,
             "dur_s": 15.0, "text": "My, how time does fly when you're chasing the impossible."},
            {"char_id": "c02", "speaker_role": "character", "start_s": 33.7,
             "dur_s": 11.0, "text": "Backup generator, stat!"},
        ],
    }
    p = tmp_path / "test_ep_ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")
    return p


def test_build_ass_structure_and_accessibility(tmp_path):
    led = _write_ledger(tmp_path)
    out, report = cap.build_ass_from_ledger(led, style="sdh_standard")
    assert out is not None, report
    text = Path(out).read_text(encoding="utf-8")

    # Opaque box always drawn: BorderStyle=3 with Outline > 0.
    style_line = next(l for l in text.splitlines() if l.startswith("Style: SDH,"))
    fields = style_line.split(",")
    # fields[0]="Style: SDH"; BorderStyle=15, Outline=16, Shadow=17, Align=18.
    assert fields[2].strip() == "36", "standard captions must stay ~40% larger than the old 26"
    assert fields[15].strip() == "3", "BorderStyle must be 3 (opaque box)"
    assert int(fields[16]) > 0, "Outline must be >0 or libass draws no box"
    assert fields[19].strip() == "40"
    assert fields[20].strip() == "40"
    # PrimaryColour (fill) is opaque white.
    assert fields[3].strip().upper() == "&H00FFFFFF"

    # Speaker label present, BOLD WHITE, sharing the SINGLE caption box --
    # no per-name box/border/color (removed 2026-06-06: it drew an ugly
    # highlighted block behind the name). Weight is the only speaker cue;
    # \r resets the rest of the cue to the box style.
    assert "ANNOUNCER:" in text
    assert "ANTON BEATTY:" in text
    assert "\\b1" in text          # bold name (weight cue)
    assert "\\3c" not in text      # no per-name outline color/border
    assert "\\c&H" not in text     # no fill recolor anywhere
    assert "\\r" in text           # name resets to the box style

    # Long announcer line (>74 visible chars) splits into multiple cues.
    dlg = [l for l in text.splitlines() if l.startswith("Dialogue:")]
    assert len(dlg) >= 4

    # Consecutive same-speaker lines: label only on the first.
    anton_labeled = [l for l in dlg if "ANTON BEATTY:" in l]
    assert len(anton_labeled) == 1


def test_unknown_style_returns_error(tmp_path):
    led = _write_ledger(tmp_path)
    out, report = cap.build_ass_from_ledger(led, style="does_not_exist")
    assert out is None
    assert "unknown style" in report


def test_caption_wrap_is_wider_for_large_sdh_style():
    assert cap.MAX_CHARS_PER_LINE == 44
    cues = cap.chunk_into_cues(" ".join(["archive"] * 12))
    assert all(len(physical) <= 44 for cue in cues for physical in cue.split("\\N"))


def _dialogue_bodies(ass_text):
    """The visible text of each Dialogue: event, style overrides removed."""
    import re
    out = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        out.append(re.sub(r"\{\\[^}]*\}", "", parts[9]).replace("\\N", " "))
    return out


def test_caption_shows_only_what_the_voice_speaks(tmp_path):
    """A caption must never show words TTS strips before it speaks.

    The ledger keeps each line as WRITTEN -- parenthetical performance
    direction included -- because that direction is what a later motion /
    shot-direction pass reads. TTS has always voiced clean_spoken_text(text);
    the caption builder used to burn the raw string, so the viewer read stage
    direction nobody said (measured 2026-08-05 over the shipped .ass files:
    255 cues across 95 episodes).
    """
    led = {
        "episode_id": "spoken_surface_ep",
        "cast": [{"char_id": "c02", "name": "NORA DRAKE"}],
        "lines": [
            {"char_id": "c02", "speaker_role": "character", "start_s": 1.0,
             "dur_s": 8.0,
             "text": "(a slow drag of something heavy across the boards) "
                     "You feel that? The floor is giving."},
        ],
    }
    p = tmp_path / "spoken_surface_ep_ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")

    out, report = cap.build_ass_from_ledger(p, style="sdh_standard")
    assert out is not None, report
    bodies = " ".join(_dialogue_bodies(Path(out).read_text(encoding="utf-8")))

    assert "You feel that?" in bodies, "the spoken dialogue must survive"
    assert "NORA DRAKE:" in bodies, "the speaker label must survive"
    assert "slow drag" not in bodies, (
        "stage direction reached the caption -- TTS strips it, so the viewer "
        "would read words the voice never speaks")
    assert "(" not in bodies and ")" not in bodies


def test_caption_drops_a_line_that_is_entirely_direction(tmp_path):
    """A line with nothing sayable must produce NO cue at all.

    Ten shipped cues were entirely stage direction -- a caption on screen with
    no spoken words under it. One of them put a Project Gutenberg source URL
    in front of the viewer, which is why this is a publish-facing defect and
    not a matter of taste.
    """
    led = {
        "episode_id": "silent_cue_ep",
        "cast": [
            {"char_id": "c01", "name": "GULLIVER REEVES"},
            {"char_id": "c02", "name": "ALICE PALMER"},
        ],
        "lines": [
            {"char_id": "c01", "speaker_role": "character", "start_s": 1.0,
             "dur_s": 4.0,
             "text": "(pocketing his lighter, then heads for the door)"},
            {"char_id": "c02", "speaker_role": "character", "start_s": 6.0,
             "dur_s": 4.0,
             "text": "(https://www.gutenberg.org/cache/epub/1342/pg1342.txt)"},
            {"char_id": "c02", "speaker_role": "character", "start_s": 11.0,
             "dur_s": 4.0, "text": "Rufus, look at the plate."},
        ],
    }
    p = tmp_path / "silent_cue_ep_ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")

    out, report = cap.build_ass_from_ledger(p, style="sdh_standard")
    assert out is not None, report
    bodies = _dialogue_bodies(Path(out).read_text(encoding="utf-8"))

    assert len(bodies) == 1, (
        "only the one line with spoken words may become a cue, got %r" % bodies)
    assert "Rufus, look at the plate." in bodies[0]
    joined = " ".join(bodies)
    assert "gutenberg" not in joined.lower(), "a source URL reached the screen"
    assert "pocketing" not in joined


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

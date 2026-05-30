"""SDH caption builder regression: scripts/otr_captions.py.

Covers timecode formatting, word wrap, multi-line cue chunking, time
distribution, SDH line rules (CPS lint), and ASS structure -- including the
accessibility contract: dialogue + speaker-name fill are WHITE (color is a
subtle outline only, never the speaker cue) and an opaque box is always drawn.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAP = _REPO_ROOT / "scripts" / "otr_captions.py"

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
    cues = cap.chunk_into_cues(text, max_chars=37, max_lines=2)
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
    assert fields[15].strip() == "3", "BorderStyle must be 3 (opaque box)"
    assert int(fields[16]) > 0, "Outline must be >0 or libass draws no box"
    # PrimaryColour (fill) is opaque white.
    assert fields[3].strip().upper() == "&H00FFFFFF"

    # Speaker label present and color applied only as an OUTLINE (\3c), with
    # the fill reset to white (\r) -- never a \c fill-color recolor.
    assert "ANNOUNCER:" in text
    assert "ANTON BEATTY:" in text
    assert "\\3c" in text          # outline-only color
    assert "\\b1" in text          # bold name (weight cue)
    assert "\\c&H" not in text     # no fill recolor anywhere

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

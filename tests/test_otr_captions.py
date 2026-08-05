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


def test_caption_keeps_performance_direction_on_purpose(tmp_path):
    """Captions burn the RAW line, so stage direction is VISIBLE by design.

    Operator ruling 2026-08-05: "it's a nice easter egg as long as it's built
    and we know and it's documented." Caption and audio diverge deliberately --
    TTS strips the parenthetical via clean_spoken_text, the caption does not.
    Measured before the ruling: 255 such cues across 95 of 915 shipped episodes.

    This test exists so the divergence cannot be "corrected" by accident by a
    later reader who assumes a caption must equal the spoken surface. If you
    are here because you just made captions match TTS, that is a behavior
    change the operator has to approve, not a bug fix.
    """
    led = {
        "episode_id": "direction_ep",
        "cast": [{"char_id": "c02", "name": "NORA DRAKE"}],
        "lines": [
            {"char_id": "c02", "speaker_role": "character", "start_s": 1.0,
             "dur_s": 8.0,
             "text": "(a slow drag of something heavy across the boards) "
                     "You feel that? The floor is giving."},
        ],
    }
    p = tmp_path / "direction_ep_ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")

    out, report = cap.build_ass_from_ledger(p, style="sdh_standard")
    assert out is not None, report
    bodies = " ".join(_dialogue_bodies(Path(out).read_text(encoding="utf-8")))

    assert "You feel that?" in bodies, "the spoken dialogue must survive"
    assert "NORA DRAKE:" in bodies, "the speaker label must survive"
    assert "slow drag" in bodies, (
        "performance direction must stay VISIBLE -- operator ruling 2026-08-05")


def test_tts_and_caption_surfaces_differ_by_design(tmp_path):
    """Pin the divergence itself: what is SPOKEN is a strict subset of what is READ.

    The ledger stores one field, `text`, holding the line as written. TTS voices
    clean_spoken_text(text); the caption burns text. The same raw string also
    feeds the still-image prompt (otr_meta_brief_image_prompt.py:1313), which is
    why the direction stays in the ledger rather than being stripped upstream.
    """
    from _otr_script_prep import clean_spoken_text

    raw = "(defiant) The train doesn't wait for anyone, Sully."
    spoken = clean_spoken_text(raw)

    assert spoken == "The train doesn't wait for anyone, Sully."
    assert "defiant" not in spoken, "TTS must not speak the direction"

    led = {
        "episode_id": "divergence_ep",
        "cast": [{"char_id": "c02", "name": "PHYLLIS TERWILLIGER"}],
        "lines": [
            {"char_id": "c02", "speaker_role": "character", "start_s": 1.0,
             "dur_s": 6.0, "text": raw},
        ],
    }
    p = tmp_path / "divergence_ep_ledger.json"
    p.write_text(json.dumps(led), encoding="utf-8")

    out, report = cap.build_ass_from_ledger(p, style="sdh_standard")
    assert out is not None, report
    bodies = " ".join(_dialogue_bodies(Path(out).read_text(encoding="utf-8")))

    assert "defiant" in bodies, "the caption keeps what TTS drops -- by design"
    for word in spoken.split():
        assert word in bodies, "every spoken word must also be readable"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""tests/test_news_interpreter_wiring.py -- unit tests for the writer-
side news-wiring helpers (commit 4 of the news_interpreter sprint).

Targets ``nodes/_otr_news_wiring.py``:
  - post_assembly_keyterm_check

(``override_announcer_close`` was retired 2026-05-22 -- BUG-LOCAL-255;
its tests were removed with it. The announcer closing line is now
written by ``_otr_line_composer.compose_announcer_outro`` and covered
by ``tests/test_announcer_passes.py``.)

Hermetic: no GPU, no I/O, no LLM. Just exercises the pure helpers
on in-memory line_rows fixtures. The helpers themselves live in
their own small module so this test file doesn't have to import
the heavy OTR_LedgerScriptWriter (which pulls comfy.utils + the
LLM loader).

ADR docs/news_interpreter_adr.md section 5 commit 4 + section 4.4.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Same sys.path setup as the sibling tests so ``nodes`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_news_wiring as _NW  # noqa: E402


def _line(line_id: str, role: str, text: str) -> dict:
    """Build an in-flight line_row matching the writer's shape."""
    return {
        "line_id":       line_id,
        "shot_id":       None,
        "beat_id":       line_id,
        "char_id":       role if role != "character" else "c02",
        "text":          text,
        "traits":        "neutral",
        "boundary":      None,
        "bark_wav_path": None,
        "start_s":       None,
        "dur_s":         None,
        "_speaker_role": role,
    }


# ---------------------------------------------------------------------------
# post_assembly_keyterm_check
# ---------------------------------------------------------------------------


def test_post_assembly_keyterm_check_all_landed():
    rows = [
        _line("b001", "announcer", "Tonight on Signal Lost..."),
        _line("b002", "character", "I built the AI model myself."),
        _line("b003", "character", "We pointed the telescope at the source."),
    ]
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("AI", "telescope"),
    )
    assert sorted(landed) == ["AI", "telescope"]
    assert missing == []


def test_post_assembly_keyterm_check_some_missing():
    rows = [
        _line("b001", "character", "The telescope is offline."),
        _line("b002", "character", "It happens."),
    ]
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("telescope", "Mars", "Voyager"),
    )
    assert landed == ["telescope"]
    assert sorted(missing) == ["Mars", "Voyager"]


def test_post_assembly_keyterm_check_word_boundary():
    """Word-boundary regex must not match substrings. "AI" inside
    "afraid" / "paid" / "available" is NOT a match.
    """
    rows = [
        _line("b001", "character", "The system is paid. We are afraid."),
        _line("b002", "character", "It is available."),
    ]
    landed, missing = _NW.post_assembly_keyterm_check(rows, ("AI",))
    assert landed == []
    assert missing == ["AI"]


def test_post_assembly_keyterm_check_ignores_non_voiced():
    rows = [
        _line("b001", "music_open", "[music: bright fanfare]"),
        _line("b002", "sfx", "[sfx: door slam]"),
        _line("b003", "character", "Only this line counts."),
    ]
    # Term appears in a music_open beat but should NOT count there;
    # it does NOT appear in the character beat -> missing.
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("fanfare",),
    )
    assert landed == []
    assert missing == ["fanfare"]


def test_post_assembly_keyterm_check_case_insensitive():
    rows = [_line("b001", "character", "The CRISPR breakthrough is real.")]
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("crispr", "Real"),
    )
    assert sorted(landed) == ["Real", "crispr"]
    assert missing == []


def test_post_assembly_keyterm_check_accepts_public_speaker_role():
    """The helper tolerates lines that already went through
    set_lines + the speaker_role post-patch (so the field is
    'speaker_role' instead of the in-flight '_speaker_role').
    """
    rows = [
        {"line_id": "b001", "speaker_role": "character",
         "text": "The telescope is online."},
        {"line_id": "b002", "speaker_role": "announcer",
         "text": "More on the signal in a moment."},
    ]
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("telescope", "signal"),
    )
    assert sorted(landed) == ["signal", "telescope"]
    assert missing == []


def test_post_assembly_keyterm_check_empty_term_skipped():
    """Defensive: an empty string in key_terms must not match
    anything (re.search on empty pattern would otherwise match).
    """
    rows = [_line("b001", "character", "Some dialogue.")]
    landed, missing = _NW.post_assembly_keyterm_check(
        rows, ("", "Mars"),
    )
    assert landed == []
    assert missing == ["Mars"]


def test_post_assembly_keyterm_check_empty_terms_list():
    rows = [_line("b001", "character", "Some dialogue.")]
    landed, missing = _NW.post_assembly_keyterm_check(rows, ())
    assert landed == []
    assert missing == []

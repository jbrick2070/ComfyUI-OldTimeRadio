"""Sprint 4 -- scrub repair handoff regression (story-spine go-forward).

In the go-forward flow the beat-local micro-repair runs UPSTREAM, so the
deterministic Ledger Scrub is always called with repair_available=False
and runs LAST. This guards the Sprint 4 contract:

  * a clean ledger still PASSES with repair_available=False;
  * fail-closed: an un-normalizable mechanical defect (an empty spoken
    line) with repair_available=False is a FAIL -- never a silent pass --
    and consumes no repair;
  * the scrub normalizes in place but NEVER rewrites the story (the spoken
    words survive);
  * the spine calls the scrub AFTER micro_repair, with repair_available=False.

Pure-Python: the scrub is deterministic and takes no LLM/GPU/network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_ledger_scrub import scrub_ledger  # noqa: E402

_SPINE_SRC = (
    Path(__file__).resolve().parents[1] / "nodes" / "_otr_story_spine.py"
).read_text(encoding="utf-8")


def _clean_ledger() -> dict:
    return {
        "schema_version": "l3",
        "episode_id": "sprint4",
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "gender": "male",
             "tts_model": "kokoro", "voice_preset": "bm_george"},
            {"char_id": "c02", "name": "EDNA REEVES", "gender": "female",
             "tts_model": "bark", "voice_preset": "v2/en_speaker_4"},
        ],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer",
             "text": "Tonight, a signal from the dark side of the moon."},
            {"line_id": "b002", "beat_id": "b002", "char_id": "c02",
             "speaker_role": "character",
             "text": "The relay is dead. We have minutes, not hours."},
        ],
    }


def test_clean_ledger_passes_with_repair_unavailable():
    r = scrub_ledger(_clean_ledger(), repair_available=False)
    assert r.status == "PASS"
    assert not r.repair_consumed


def test_unnormalizable_defect_fails_closed_with_repair_unavailable():
    """An empty spoken line cannot be normalized; with no repair budget the
    scrub fails closed (FAIL), never a silent PASS, and consumes no repair.
    This is the go-forward Sprint 4 'aborts/fails on an un-normalizable
    mechanical defect' contract under the new repair_available=False call."""
    led = _clean_ledger()
    led["lines"][1]["text"] = "   "  # whitespace-only -> empty after normalize
    r = scrub_ledger(led, repair_available=False)
    assert r.status == "FAIL"
    assert not r.repair_consumed


def test_scrub_preserves_story_words_never_rewrites():
    """The scrub normalizes (whitespace / quotes / leaked formatting) but
    must never rewrite the story -- every spoken word survives."""
    led = _clean_ledger()
    original_words = [w.strip(".,") for w in led["lines"][1]["text"].split()]
    r = scrub_ledger(led, repair_available=False)
    cleaned = led["lines"][1]["text"]
    assert r.status == "PASS"
    for word in original_words:
        assert word in cleaned, f"scrub dropped the story word {word!r}"


def test_spine_runs_scrub_after_micro_repair_with_repair_false():
    """Source-order guard: the spine calls micro_repair BEFORE scrub_ledger
    (the scrub is LAST), and the scrub is told repair_available=False."""
    micro_idx = _SPINE_SRC.find(".micro_repair(")
    scrub_idx = _SPINE_SRC.find(".scrub_ledger(")
    assert micro_idx != -1, "spine must call the editor's micro_repair"
    assert scrub_idx != -1, "spine must call scrub_ledger"
    assert scrub_idx > micro_idx, (
        "the scrub must run AFTER micro_repair (mechanical + last)"
    )
    assert "repair_available=False" in _SPINE_SRC, (
        "the scrub must be called with repair_available=False"
    )

"""Chunk 4 -- the bare stage-direction FREEZE floor in _otr_ledger_scrub.

_strip_stage_directions calls the shared scrub AFTER the delimited strips
(BARE_STAGE_FLOOR_ACTIVE), preserving the Tuple[str,bool] contract; scrub_ledger
emits CODE_STAGE_DIRECTION + restamps word_count. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_ledger_scrub import (  # noqa: E402
    CODE_STAGE_DIRECTION,
    _strip_stage_directions,
    scrub_ledger,
)


class TestStripStageDirections:
    def test_bare_leading_stripped(self):
        out, changed = _strip_stage_directions("clenches jaw You mean it?")
        assert out == "You mean it?"
        assert changed is True

    def test_clean_line_noop(self):
        out, changed = _strip_stage_directions("We need a plan, Pinky.")
        assert out == "We need a plan, Pinky."
        assert changed is False

    def test_delimited_plus_bare(self):
        # an existing bracket cue AND a bare leading direction both go
        out, changed = _strip_stage_directions("[beat] twirls his pen Look, Pinky.")
        assert "[beat]" not in out
        assert out == "Look, Pinky."
        assert changed is True

    def test_returns_tuple(self):
        res = _strip_stage_directions("twirls his pen You mean it?")
        assert isinstance(res, tuple) and len(res) == 2
        assert isinstance(res[0], str) and isinstance(res[1], bool)


def _ledger(text, word_count=99):
    return {
        "cast": [{"char_id": "c01", "name": "CHRIS"}],
        "lines": [{
            "line_id": "b004", "beat_id": "b004", "char_id": "c01",
            "speaker_role": "character", "text": text,
            "word_count": word_count, "char_count": len(text),
        }],
        "meta": {},
    }


class TestScrubLedgerIntegration:
    def test_freeze_strips_bare_and_restamps(self):
        led = _ledger("twirls his pen nervously Look, Pinky, we are done.")
        res = scrub_ledger(led, repair_available=True)
        row = led["lines"][0]
        assert row["text"] == "Look, Pinky, we are done."
        # word_count restamped to the cleaned text, not the stale 99
        assert row["word_count"] == len("Look, Pinky, we are done.".split())
        codes = [f.code for f in res.findings]
        assert CODE_STAGE_DIRECTION in codes

    def test_clean_line_untouched(self):
        led = _ledger("We are done here, Pinky.")
        scrub_ledger(led, repair_available=True)
        assert led["lines"][0]["text"] == "We are done here, Pinky."

    def test_music_row_not_touched_by_floor(self):
        led = {
            "cast": [{"char_id": "c01", "name": "CHRIS"}],
            "lines": [{
                "line_id": "b005", "beat_id": "b005",
                "char_id": "music_inter", "speaker_role": "music_inter",
                "text": "twirls his pen Look here", "word_count": 4,
            }],
            "meta": {},
        }
        scrub_ledger(led, repair_available=True)
        # non-spoken rows are render contracts -> never scrubbed
        assert led["lines"][0]["text"] == "twirls his pen Look here"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

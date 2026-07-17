"""v4 campaign P1(iv): beat_bounds structural contract + G11 floor terminal.

Operator decision (2026-07-17): WORDS_PER_BEAT=40 is a SOFT target only --
episode length is recorded-not-gated, never pass/fail. beat_bounds gates ONLY a
STRUCTURAL FLOOR (min voiced beats below which an episode is not assemblable).
The per-family MAX + the word->beat derivation change are RECORDED and validated
LIVE in Phase 2 per bank; Phase 1 ships the contract + the deterministic floor
terminal (G11), which every current lane clears (codex beat_ids=max(3,..); the
legacy adapter pads to >=3) so it never false-fails a shipping lane.

Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_episode_budget as BB  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Pure contract
# ---------------------------------------------------------------------------

class TestBeatBoundsContract:
    def test_floor_is_structural_min(self):
        assert BB.STRUCTURAL_MIN_BEATS == 3
        assert BB.WORDS_PER_BEAT == 40

    def test_family_bands(self):
        assert BB.beat_bounds(350, "codex") == (3, 12)
        assert BB.beat_bounds(350, "inline") == (3, 40)
        assert BB.beat_bounds(350, "default") == (3, 40)
        assert BB.beat_bounds(350, "nonsense") == (3, 40)   # -> default
        assert BB.beat_bounds(350, None) == (3, 40)

    def test_target_beat_count_round_half_up_and_clamp(self):
        assert BB.target_beat_count(720, "inline") == 18     # 740//40
        assert BB.target_beat_count(200, "inline") == 5      # 220//40
        assert BB.target_beat_count(40, "inline") == 3       # 60//40=1 -> floor 3
        assert BB.target_beat_count(2000, "inline") == 40    # clamp to inline max
        assert BB.target_beat_count(2000, "codex") == 12     # clamp to codex max
        assert BB.target_beat_count(0, "inline") == 3        # floor

    def test_target_beat_count_bad_input_safe(self):
        assert BB.target_beat_count(None, "inline") == 3
        assert BB.target_beat_count("x", "inline") == 3

    def test_classify(self):
        assert BB.classify_beat_count(3, "codex") == "ok"
        assert BB.classify_beat_count(12, "inline") == "ok"
        assert BB.classify_beat_count(2, "inline") == "below_min"
        assert BB.classify_beat_count(0, "inline") == "below_min"
        assert BB.classify_beat_count("x", "inline") == "below_min"


# ---------------------------------------------------------------------------
# 2. G11 structural-floor terminal
# ---------------------------------------------------------------------------

def _led(bounds, lines):
    meta = {"source_bank": "x"}
    if bounds is not None:
        meta["beat_bounds"] = bounds
    return {"schema_version": "l3-2026-05-14", "meta": meta, "lines": lines}


_INLINE = {"family": "inline", "min": 3, "max": 40}


def _g11_errors(ledger_data):
    errors: list = []
    LF._check_g11_beat_floor(ledger_data, errors, [])
    return errors


def _spoken(beat_id, lid="l"):
    return {"line_id": f"{lid}{beat_id}", "beat_id": beat_id,
            "speaker_role": "character", "text": "a line"}


class TestG11BeatFloor:
    def test_fires_below_floor(self):
        led = _led(_INLINE, [_spoken("b1"), _spoken("b2")])  # 2 distinct beats
        errs = _g11_errors(led)
        assert errs and "G11" in errs[0]

    def test_ok_at_floor(self):
        led = _led(_INLINE, [_spoken("b1"), _spoken("b2"), _spoken("b3")])
        assert _g11_errors(led) == []

    def test_inert_without_opt_in(self):
        led = _led(None, [_spoken("b1")])  # 1 beat but no beat_bounds meta
        assert _g11_errors(led) == []

    def test_music_rows_excluded(self):
        led = _led(_INLINE, [
            _spoken("b1"), _spoken("b2"),
            {"line_id": "m1", "beat_id": "b3", "speaker_role": "music_inter",
             "text": "theme"},
        ])
        # only 2 SPOKEN beats -> below floor (music beat b3 does not count)
        assert _g11_errors(led) != []

    def test_fallback_to_spoken_line_count_without_beat_id(self):
        led = _led(_INLINE, [
            {"line_id": "l1", "speaker_role": "character", "text": "a"},
            {"line_id": "l2", "speaker_role": "character", "text": "b"},
            {"line_id": "l3", "speaker_role": "character", "text": "c"},
        ])
        assert _g11_errors(led) == []  # 3 spoken lines, no beat_id -> floor met

    def test_bad_min_safe(self):
        led = _led({"family": "inline", "min": "oops", "max": 40}, [_spoken("b1")])
        assert _g11_errors(led) == []  # unparseable floor -> inert, never crashes

    def test_run_gap_audit_includes_g11(self):
        led = _led(_INLINE, [_spoken("b1")])
        report = LF.run_gap_audit(led, label="pre")
        assert any("G11" in e for e in report.errors)

    def test_phase_10_raises_with_g11(self):
        led = _led(_INLINE, [_spoken("b1")])
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert any("G11" in e for e in ei.value.errors)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

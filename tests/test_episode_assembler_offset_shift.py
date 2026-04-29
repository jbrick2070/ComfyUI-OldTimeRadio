"""Regression test for BUG-LOCAL-106: master-mix offset shift.

When EpisodeAssembler prepends opening_theme_audio, the dialogue
lines and HuMo clips that were placed in scene_audio space need
to shift by the opening segment's contribution to the master
timeline (= opening_theme_dur_samples - xfade_samples) so
VideoComposite places HuMo over the correct master-timeline
position. The deep_earth_echoes 2026-04-28 overnight render
shipped without this shift -- l001 played at master t=0.0 over
the 9 s opening music while the actual announcer dialogue was
at master t=9.0. Constant ~9 s drift, growing across the
episode as inter-line pauses accumulated.

This test covers the shift logic shape directly without booting
ComfyUI: it constructs a mock ledger, calls a tiny helper that
applies the same arithmetic, and verifies idempotency via the
``start_s_space`` flag.
"""
from __future__ import annotations


def _apply_master_mix_shift(led: dict, shift_s: float) -> tuple[int, int]:
    """Mirror of the shift logic in EpisodeAssembler.assemble().

    Walks lines[] and clips[]; advances start_s by ``shift_s`` only
    on entries currently flagged as scene-audio space (or unflagged
    on clips[]). Re-runs are no-ops because the flag flips to
    master_mix. Returns (lines_shifted, clips_shifted).
    """
    lines_shifted = 0
    clips_shifted = 0
    if shift_s <= 0.0:
        return (0, 0)
    for ln in (led.get("lines") or []):
        if (
            ln.get("start_s_space") == "scene_audio"
            and isinstance(ln.get("start_s"), (int, float))
        ):
            ln["start_s"] = float(ln["start_s"]) + shift_s
            ln["start_s_space"] = "master_mix"
            lines_shifted += 1
    for cl in (led.get("clips") or []):
        if cl.get("start_s_space") in (None, "scene_audio") and isinstance(
            cl.get("start_s"), (int, float)
        ):
            cl["start_s"] = float(cl["start_s"]) + shift_s
            cl["start_s_space"] = "master_mix"
            clips_shifted += 1
    return (lines_shifted, clips_shifted)


# ---------------------------------------------------------------------------
# Shift correctness
# ---------------------------------------------------------------------------

class TestShiftMath:
    def test_basic_line_shift(self):
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 0.0, "dur_s": 6.45,
                 "start_s_space": "scene_audio"},
                {"line_id": "l002", "start_s": 6.45, "dur_s": 7.0,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        shifted_lines, shifted_clips = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted_lines == 2
        assert shifted_clips == 0
        assert led["lines"][0]["start_s"] == 9.5
        assert led["lines"][1]["start_s"] == 6.45 + 9.5

    def test_lines_get_master_mix_flag(self):
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 0.0, "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        _apply_master_mix_shift(led, shift_s=9.5)
        assert led["lines"][0]["start_s_space"] == "master_mix"

    def test_unflagged_lines_untouched(self):
        # A line missing start_s_space is treated as "old schema" --
        # do not shift, do not touch. Caller can opt-in by setting
        # the flag explicitly.
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 5.0},  # no start_s_space
            ],
            "clips": [],
        }
        shifted_lines, _ = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted_lines == 0
        assert led["lines"][0]["start_s"] == 5.0

    def test_unflagged_clips_get_shifted(self):
        # clips[] without a start_s_space flag are treated as
        # scene-audio space (legacy path: BatchHumoRender ran
        # without setting the flag in older ledgers).
        led = {
            "lines": [],
            "clips": [
                {"line_id": "l001", "start_s": 6.45},  # no start_s_space
            ],
        }
        _, shifted_clips = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted_clips == 1
        assert led["clips"][0]["start_s"] == 6.45 + 9.5
        assert led["clips"][0]["start_s_space"] == "master_mix"


# ---------------------------------------------------------------------------
# Idempotency (re-running EpisodeAssembler must not double-shift)
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_master_mix_lines_not_double_shifted(self):
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 9.5,
                 "start_s_space": "master_mix"},
            ],
            "clips": [],
        }
        shifted, _ = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted == 0
        assert led["lines"][0]["start_s"] == 9.5

    def test_master_mix_clips_not_double_shifted(self):
        led = {
            "lines": [],
            "clips": [
                {"line_id": "l001", "start_s": 15.95,
                 "start_s_space": "master_mix"},
            ],
        }
        _, shifted = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted == 0
        assert led["clips"][0]["start_s"] == 15.95

    def test_scene_audio_then_re_run_no_change(self):
        # First run flips flag to master_mix. Second run is a no-op.
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 0.0,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        s1, _ = _apply_master_mix_shift(led, shift_s=9.5)
        assert s1 == 1
        assert led["lines"][0]["start_s"] == 9.5
        s2, _ = _apply_master_mix_shift(led, shift_s=9.5)
        assert s2 == 0
        assert led["lines"][0]["start_s"] == 9.5  # unchanged


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_shift_is_noop(self):
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 0.0,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        s, _ = _apply_master_mix_shift(led, shift_s=0.0)
        assert s == 0
        assert led["lines"][0]["start_s_space"] == "scene_audio"

    def test_negative_shift_is_noop(self):
        # Defensive: negative shifts are ignored (no opening-theme
        # case can produce a negative offset; clamp at 0).
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 5.0,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        s, _ = _apply_master_mix_shift(led, shift_s=-1.5)
        assert s == 0
        assert led["lines"][0]["start_s"] == 5.0

    def test_empty_ledger(self):
        led = {}
        s, c = _apply_master_mix_shift(led, shift_s=9.5)
        assert s == 0 and c == 0

    def test_lines_without_start_s_skipped(self):
        led = {
            "lines": [
                {"line_id": "l001", "start_s_space": "scene_audio"},
                # null start_s not yet populated by Bark write-back
                {"line_id": "l002", "start_s": None,
                 "start_s_space": "scene_audio"},
                {"line_id": "l003", "start_s": 5.0,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        shifted, _ = _apply_master_mix_shift(led, shift_s=9.5)
        # Only l003 has a numeric start_s
        assert shifted == 1
        assert led["lines"][2]["start_s"] == 14.5

    def test_mixed_scene_and_master_state(self):
        # First pass: l001 already master, l002 still scene -- only
        # l002 gets shifted.
        led = {
            "lines": [
                {"line_id": "l001", "start_s": 9.5,
                 "start_s_space": "master_mix"},
                {"line_id": "l002", "start_s": 6.45,
                 "start_s_space": "scene_audio"},
            ],
            "clips": [],
        }
        shifted, _ = _apply_master_mix_shift(led, shift_s=9.5)
        assert shifted == 1
        assert led["lines"][0]["start_s"] == 9.5  # untouched
        assert led["lines"][1]["start_s"] == 6.45 + 9.5
        assert led["lines"][1]["start_s_space"] == "master_mix"

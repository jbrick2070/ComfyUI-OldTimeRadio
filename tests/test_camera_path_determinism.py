"""
test_camera_path_determinism.py  --  Phase C camera trajectory regression
=========================================================================

Locks in the determinism and frame-count contracts for
``otr_v2/hyworld/camera_path.py``.  Pure Python, no ffmpeg required:
all tests run in well under a second.

Contracts verified here:
  1. Hash equality  -- same (adjective, duration, fps, seed) => same
     SHA-256 across repeated calls.
  2. Frame-count exactness -- ``frame_count == round(duration*fps)``
     over a wide sweep; no off-by-one drift at long durations.
  3. Start/end pose values -- every catalog adjective starts and ends
     at the declared catalog poses.
  4. Adjective fallback -- unknown adjective resolves to the default
     ("slow drift") label without raising.
  5. Substring match -- "clean push in" resolves to the "clean push"
     catalog entry (preserves pre-Phase-C behavior).
  6. Input validation -- non-positive duration or fps raises
     ValueError.
  7. ffmpeg emit stability -- ``to_zoompan_filter`` output is stable
     across repeated calls for identical trajectories.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from otr_v2.hyworld.camera_path import (
    Trajectory,
    from_adjective,
    hash_trajectory,
    known_adjectives,
    sample,
    to_zoompan_filter,
    zoompan_for_shot,
)


# ---------------------------------------------------------------------------
# 1. Hash equality
# ---------------------------------------------------------------------------

class TestHashEquality:
    def test_same_inputs_same_hash_across_calls(self):
        h1 = hash_trajectory(from_adjective("clean push", 9.0, fps=24, seed=0))
        h2 = hash_trajectory(from_adjective("clean push", 9.0, fps=24, seed=0))
        assert h1 == h2

    def test_different_duration_different_hash(self):
        h1 = hash_trajectory(from_adjective("clean push", 9.0, fps=24))
        h2 = hash_trajectory(from_adjective("clean push", 10.0, fps=24))
        assert h1 != h2

    def test_different_adjective_different_hash(self):
        h1 = hash_trajectory(from_adjective("clean push", 9.0))
        h2 = hash_trajectory(from_adjective("slow drift", 9.0))
        assert h1 != h2

    def test_different_fps_different_hash(self):
        h1 = hash_trajectory(from_adjective("clean push", 9.0, fps=24))
        h2 = hash_trajectory(from_adjective("clean push", 9.0, fps=30))
        assert h1 != h2

    def test_hash_is_hex_sha256(self):
        h = hash_trajectory(from_adjective("locked off", 3.0))
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_every_catalog_adjective_hash_stable(self):
        """All catalog adjectives must produce a stable hash across
        two calls.  If this fails on any entry, the whole engine is
        non-deterministic."""
        for adj in known_adjectives():
            h1 = hash_trajectory(from_adjective(adj, 5.0, fps=24, seed=0))
            h2 = hash_trajectory(from_adjective(adj, 5.0, fps=24, seed=0))
            assert h1 == h2, f"hash drift on adjective: {adj}"


# ---------------------------------------------------------------------------
# 2. Frame-count exactness
# ---------------------------------------------------------------------------

class TestFrameCount:
    @pytest.mark.parametrize("duration,fps,expected", [
        (1.0, 24, 24),
        (9.0, 24, 216),
        (10.0, 24, 240),
        (3.5, 24, 84),
        (3.5, 30, 105),
        (0.5, 24, 12),
        (180.0, 24, 4320),     # 3-min shot, no overflow
    ])
    def test_exact_frame_count(self, duration, fps, expected):
        traj = from_adjective("locked off", duration, fps=fps)
        assert traj.frame_count == expected
        assert len(sample(traj)) == expected

    def test_minimum_one_frame(self):
        # Even a pathologically tiny duration must give at least one frame.
        traj = from_adjective("locked off", 0.001, fps=24)
        assert traj.frame_count >= 1
        assert len(sample(traj)) == traj.frame_count

    def test_no_off_by_one_over_wide_sweep(self):
        """Sweep 1..600s at 24fps.  Each frame count must match the
        simple formula."""
        for seconds in range(1, 601):
            traj = from_adjective("clean push", float(seconds), fps=24)
            assert traj.frame_count == seconds * 24


# ---------------------------------------------------------------------------
# 3. Start/end pose values
# ---------------------------------------------------------------------------

class TestStartEndPoses:
    def test_clean_push_ends_zoomed_in(self):
        traj = from_adjective("clean push", 9.0, fps=24)
        poses = sample(traj)
        assert poses[0] == pytest.approx((1.0, 0.5, 0.5))
        # Last pose should be the end zoom 1.40
        final_z, final_cx, final_cy = poses[-1]
        assert final_z == pytest.approx(1.40, abs=1e-6)
        assert final_cx == pytest.approx(0.5, abs=1e-6)
        assert final_cy == pytest.approx(0.5, abs=1e-6)

    def test_whip_pan_sweeps_horizontally(self):
        traj = from_adjective("whip-pan", 4.0, fps=24)
        poses = sample(traj)
        assert poses[0][1] == pytest.approx(0.0, abs=1e-6)
        assert poses[-1][1] == pytest.approx(1.0, abs=1e-6)

    def test_low_angle_moves_y_bottom_to_top(self):
        traj = from_adjective("low angle", 5.0, fps=24)
        poses = sample(traj)
        # Starts at bottom (cy=1.0), ends at top (cy=0.0).
        assert poses[0][2] == pytest.approx(1.0, abs=1e-6)
        assert poses[-1][2] == pytest.approx(0.0, abs=1e-6)

    def test_locked_off_is_constant(self):
        traj = from_adjective("locked off", 9.0, fps=24)
        poses = sample(traj)
        first = poses[0]
        for p in poses:
            assert p == pytest.approx(first, abs=1e-12)

    def test_macro_detail_end_zoom(self):
        traj = from_adjective("macro detail", 6.0, fps=24)
        poses = sample(traj)
        assert poses[-1][0] == pytest.approx(1.15, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Adjective fallback + substring match
# ---------------------------------------------------------------------------

class TestAdjectiveResolution:
    def test_unknown_adjective_falls_back_to_default(self):
        traj = from_adjective("zoom-crash-attack-mode-9000", 5.0, fps=24)
        assert traj.label == "slow drift"

    def test_empty_adjective_falls_back(self):
        traj = from_adjective("", 5.0, fps=24)
        assert traj.label == "slow drift"

    def test_none_adjective_falls_back(self):
        traj = from_adjective(None, 5.0, fps=24)  # type: ignore[arg-type]
        assert traj.label == "slow drift"

    def test_substring_match_clean_push_in(self):
        """Pre-Phase-C strings like "clean push in" must still resolve
        to the "clean push" catalog entry."""
        traj = from_adjective("clean push in", 5.0, fps=24)
        assert traj.label == "clean push"

    def test_case_insensitive_match(self):
        traj = from_adjective("LOCKED OFF on tripod", 5.0, fps=24)
        assert traj.label == "locked off"

    def test_first_match_wins(self):
        """Ordering contract: if two catalog keys appear in the input,
        the first-in-catalog wins.  "locked off" precedes "clean push"
        so this phrase resolves to locked off."""
        traj = from_adjective("locked off, then clean push", 5.0, fps=24)
        assert traj.label == "locked off"


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="duration_sec"):
            from_adjective("clean push", 0.0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration_sec"):
            from_adjective("clean push", -1.0)

    def test_zero_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            from_adjective("clean push", 5.0, fps=0)

    def test_negative_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            from_adjective("clean push", 5.0, fps=-24)


# ---------------------------------------------------------------------------
# 6. ffmpeg emit stability
# ---------------------------------------------------------------------------

class TestZoompanEmit:
    def test_emit_stable_across_calls(self):
        traj = from_adjective("clean push", 9.0, fps=24)
        f1 = to_zoompan_filter(traj, 1280, 720)
        f2 = to_zoompan_filter(traj, 1280, 720)
        assert f1 == f2

    def test_emit_contains_frame_count(self):
        traj = from_adjective("clean push", 9.0, fps=24)
        f = to_zoompan_filter(traj, 1280, 720)
        assert "d=216" in f
        assert "s=1280x720" in f
        assert "fps=24" in f

    def test_emit_locked_off_is_constant(self):
        """Locked-off shot emits a constant zoom (no expression)."""
        traj = from_adjective("locked off", 5.0, fps=24)
        f = to_zoompan_filter(traj, 1280, 720)
        # Zoom expression should be a bare "1.0", not a formula.
        assert "z='1.0'" in f

    def test_emit_ease_in_out_uses_cosine(self):
        traj = from_adjective("clean push", 5.0, fps=24)
        f = to_zoompan_filter(traj, 1280, 720)
        assert "cos(PI*" in f

    def test_emit_single_frame_no_divide_by_zero(self):
        """frame_count==1 means u_expr should be "0", not "(on/0)"."""
        # Artificially construct a 1-frame trajectory.
        traj = Trajectory(
            label="test_single",
            duration_sec=1 / 24,
            fps=24,
            zoom_start=1.0, zoom_end=1.3,
            cx_start=0.5, cx_end=0.5,
            cy_start=0.5, cy_end=0.5,
            easing="linear",
        )
        assert traj.frame_count == 1
        f = to_zoompan_filter(traj, 1280, 720)
        assert "on/0" not in f
        assert "d=1" in f


# ---------------------------------------------------------------------------
# 7. Worker-facing wrapper
# ---------------------------------------------------------------------------

class TestWorkerWrapper:
    def test_zoompan_for_shot_returns_label_and_filter(self):
        label, vfilter = zoompan_for_shot("clean push in", 9.0, 1280, 720)
        assert label == "clean push"
        assert vfilter.startswith("zoompan=")
        assert "d=216" in vfilter

    def test_zoompan_for_shot_unknown_adjective_falls_back(self):
        label, vfilter = zoompan_for_shot("nonsense-camera", 5.0, 1280, 720)
        assert label == "slow drift"
        assert vfilter.startswith("zoompan=")

    def test_zoompan_for_shot_covers_all_catalog(self):
        """Every catalog entry must emit a non-empty filter chain with
        a matching duration."""
        for adj in known_adjectives():
            label, vfilter = zoompan_for_shot(adj, 5.0, 1280, 720)
            assert label == adj
            assert vfilter.startswith("zoompan=")
            assert "d=120" in vfilter  # 5s * 24fps
            assert "s=1280x720" in vfilter

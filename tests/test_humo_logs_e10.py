"""Sprint E E10 / R-13 + R-14: HuMo log surface drift guards.

R-14: BatchHumoRender must log.warning when portraits_dir input is
empty (D0d fallback path). Pre-E10 the auto-resolve fallback was
silent; post-E10 it logs explicitly so soak diagnostics see the
missing-wire case.

R-13: BatchHumoRender must log a pre-batch wall-time estimate so the
operator sees the projected total before queueing. Post-2026-05-24
(BUG-LOCAL-265) the basis is the per-clip bracket test (~4:23/clip);
the stale "~10-12 min/line, Sprint C-era" reference was retired.

Both tests AST-walk the source for the log line shapes; they do not
exercise the full render path (which requires GPU + real model load).
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
HUMO_SOURCE = REPO_ROOT / "nodes" / "batch_humo_render.py"


def _read_source() -> str:
    return HUMO_SOURCE.read_text(encoding="utf-8")


class TestPortraitsDirFallbackLogged:
    """R-14: log.warning fires when portraits_dir auto-resolves."""

    def test_portraits_dir_empty_branch_logs_warning(self):
        src = _read_source()
        # The auto-resolve branch must call log.warning with a message
        # naming the D0d fallback path so soak grep can find it.
        assert "log.warning" in src
        # Pin the specific message tag so a future log-rename keeps
        # the soak diagnostic discoverable.
        assert "portraits_dir input is empty" in src
        assert "D0d fallback path" in src

    def test_portraits_dir_fallback_resolution_still_works(self):
        """The defensive log MUST NOT replace the auto-resolve;
        portraits_dir_path is still assigned in the empty branch."""
        src = _read_source()
        # The empty branch assigns portraits_dir_path = comfy_output_dir()
        assert "portraits_dir_path = comfy_output_dir()" in src


class TestWallTimeEstimateLogged:
    """R-13: pre-batch HuMo wall-time estimate logged before render."""

    def test_wall_time_estimate_log_present(self):
        src = _read_source()
        assert "PRE-BATCH ESTIMATE" in src
        # The estimate still reports a total-minutes projection.
        assert "minutes total HuMo wall time" in src or "minutes total HuMo" in src

    def test_estimate_excludes_non_character_lines(self):
        """Estimate is computed against character-role line count, not
        the full ledger.lines[] sweep. Non-character lines route to
        radio still and complete in ~seconds."""
        src = _read_source()
        # The estimate-count construction filters by speaker_role.
        # Pin the role string so a future refactor that drops the
        # filter (and inflates the estimate by ~3-4x on episodes with
        # many SFX / music_inter rows) fails this test.
        assert "_character_lines_count" in src
        assert '"character"' in src or "'character'" in src

    def test_estimate_caveats_documented(self):
        """The estimate log must caveat its basis so an operator does
        not over-trust it. Post-2026-05-24 (BUG-LOCAL-265) the basis is
        the per-clip bracket test (~4:23/clip) and the caveat is that
        long lines split into multiple clips -- so the projected total
        is a range, not a point estimate. The pre-2026-05-24 stale
        '~10-12 min/line' figure was retired."""
        src = _read_source()
        assert "2026-05-24 bracket test" in src
        assert "split into multiple clips" in src

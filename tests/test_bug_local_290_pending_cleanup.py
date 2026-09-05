"""tests/test_bug_local_290_pending_cleanup.py

BUG-LOCAL-290: sweep stale `pending_*` episode dirs that the writer
abandoned before composing any lines. 17 such dirs accumulated on
2026-05-27 before the cleanup was added.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from nodes._otr_pending_cleanup import (
    DEFAULT_PENDING_MAX_AGE_SECONDS,
    PendingSweepReport,
    sweep_empty_pending_dirs,
)


def _make_pending_dir(
    root: Path,
    name: str,
    *,
    lines: int = 0,
    age_seconds: int = 0,
    with_ledger: bool = True,
) -> Path:
    """Build a fake pending_* dir for testing.

    lines: number of fake line rows to write into the ledger.
    age_seconds: how many seconds to age the dir's mtime by.
    with_ledger: when False, no *_ledger.json file is created.
    """
    d = root / name
    audio = d / "audio"
    audio.mkdir(parents=True)
    if with_ledger:
        led_p = audio / f"{name}_ledger.json"
        ledger = {
            "episode_id": name,
            "lines": [
                {"line_id": f"l{i:03d}", "text": "x"}
                for i in range(lines)
            ],
            "meta": {"episode_title": ""},
        }
        led_p.write_text(json.dumps(ledger), encoding="utf-8")
    if age_seconds > 0:
        _age_tree(d, age_seconds)
    return d


def _age_tree(d: Path, age_seconds: int) -> None:
    """Age the dir AND everything beneath it. The sweep measures age from the
    newest mtime under the dir (an NTFS directory mtime moves only on
    direct-child changes), so aging the top level alone leaves a fresh ledger
    inside and the dir reads as in-flight -- which is correct for the sweep
    and wrong for a fixture claiming to be a 2-hour-old abandoned run."""
    old_time = time.time() - age_seconds
    for p in [d] + list(d.rglob("*")):
        os.utime(p, (old_time, old_time))


class TestSweepEmptyPendingDirs:
    """Behavior pins."""

    def test_deletes_old_empty_pending(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260101_010101",
            lines=0,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        report = sweep_empty_pending_dirs(tmp_path)
        assert str(d) in report.deleted
        assert not d.exists()

    def test_keeps_young_empty_pending(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260527_152033",
            lines=0,
            age_seconds=0,  # just created -- in-flight
        )
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert str(d) in report.skipped_too_young

    def test_keeps_old_pending_with_lines(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260101_010101",
            lines=3,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert str(d) in report.skipped_has_lines

    def test_deletes_old_pending_with_no_ledger(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260101_010101",
            with_ledger=False,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        report = sweep_empty_pending_dirs(tmp_path)
        assert not d.exists()
        assert str(d) in report.deleted
        # It was deleted, so it is NOT also booked as skipped -- the old
        # assertion pinned exactly that double-booking (2026-09-05).
        assert str(d) not in report.skipped_no_ledger

    def test_keeps_signal_lost_dirs(self, tmp_path):
        """Named episodes (signal_lost_*) are NEVER touched."""
        d = (tmp_path / "signal_lost_black_holes_echo_20260527")
        (d / "audio").mkdir(parents=True)
        # Even though it's old and has no ledger -- doesn't match
        # the pending_* prefix.
        old_time = time.time() - 86400
        os.utime(d, (old_time, old_time))
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert report.scanned == 0  # signal_lost_* not even scanned

    def test_dry_run_does_not_delete(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260101_010101",
            lines=0,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        report = sweep_empty_pending_dirs(tmp_path, dry_run=True)
        assert d.exists()
        assert str(d) in report.deleted  # would have been deleted

    def test_custom_max_age(self, tmp_path):
        d = _make_pending_dir(
            tmp_path, "pending_20260527_122512",
            lines=0,
            age_seconds=600,  # 10 minutes
        )
        # Default age cap (2h) -- too young, skip.
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        # Custom 5-minute cap -- now eligible.
        report = sweep_empty_pending_dirs(tmp_path, max_age_seconds=300)
        assert not d.exists()

    def test_empty_root_returns_empty_report(self, tmp_path):
        # Empty episodes_root -- nothing to scan.
        report = sweep_empty_pending_dirs(tmp_path)
        assert report.scanned == 0
        assert report.deleted == []

    def test_nonexistent_root_returns_empty_report(self):
        # Defensive -- missing root should not raise.
        report = sweep_empty_pending_dirs(
            Path("/nonexistent/path/episodes")
        )
        assert report.scanned == 0
        assert report.deleted == []

    def test_mixed_state_report(self, tmp_path):
        """One sweep with all four dispositions present."""
        old_empty = _make_pending_dir(
            tmp_path, "pending_20260101_010101",
            lines=0,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        young_empty = _make_pending_dir(
            tmp_path, "pending_20260527_152033",
            lines=0, age_seconds=0,
        )
        old_with_lines = _make_pending_dir(
            tmp_path, "pending_20260101_020202",
            lines=5,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )
        old_no_ledger = _make_pending_dir(
            tmp_path, "pending_20260101_030303",
            with_ledger=False,
            age_seconds=DEFAULT_PENDING_MAX_AGE_SECONDS + 60,
        )

        report = sweep_empty_pending_dirs(tmp_path)

        assert report.scanned == 4
        assert str(old_empty) in report.deleted
        assert not old_empty.exists()
        assert str(young_empty) in report.skipped_too_young
        assert young_empty.exists()
        assert str(old_with_lines) in report.skipped_has_lines
        assert old_with_lines.exists()
        assert str(old_no_ledger) in report.deleted
        assert str(old_no_ledger) not in report.skipped_no_ledger
        assert not old_no_ledger.exists()


class TestPendingSweepReportShape:
    """Report serialization pin."""

    def test_as_dict_round_trip(self, tmp_path):
        report = sweep_empty_pending_dirs(tmp_path)
        d = report.as_dict()
        assert "scanned" in d
        assert "deleted_count" in d
        assert "errors" in d
        # JSON-friendly
        import json as _j
        _j.dumps(d)  # raises if not serializable


class TestWriterWiring:
    """Source pin -- the writer calls the sweep helper at start-up."""

    def test_writer_imports_sweep(self):
        from pathlib import Path
        src = Path(
            __import__(
                "nodes.OTR_LedgerScriptWriter",
                fromlist=["OTR_LedgerScriptWriter"],
            ).__file__
        ).read_text(encoding="utf-8")
        assert "from ._otr_pending_cleanup import sweep_empty_pending_dirs" in src
        assert "BUG-LOCAL-290" in src


class TestUnreadableIsNotEmpty:
    """The fix (2026-09-05, Fable gate): 'cannot read the ledger' was being
    treated as 'may delete the directory'. These pin the four-state
    disposition so it cannot regress to a bare None."""

    def _old(self):
        return DEFAULT_PENDING_MAX_AGE_SECONDS + 60

    def test_a_corrupt_ledger_and_its_asset_survive(self, tmp_path):
        """A truncated ledger beside a rendered WAV. The whole directory used
        to be rmtree'd. Both must still be there afterwards."""
        d = _make_pending_dir(tmp_path, "pending_20260101_020202",
                              lines=3, age_seconds=self._old())
        led = next((d / "audio").glob("*_ledger.json"))
        led.write_bytes(led.read_bytes()[: len(led.read_bytes()) // 2])  # truncate
        asset = d / "audio" / "line_001.wav"
        asset.write_bytes(b"RIFF" + b"\x00" * 64)
        _age_tree(d, self._old())          # truncation + asset are "old" too
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists(), "an unreadable ledger is never permission to delete"
        assert asset.exists()
        assert str(d) in report.skipped_unreadable
        assert str(d) not in report.deleted

    def test_a_bom_prefixed_ledger_is_unreadable_not_empty(self, tmp_path):
        """json.load raises on a UTF-8 BOM -- a hand edit in the wrong editor.
        That is the exact case that used to erase a run."""
        d = _make_pending_dir(tmp_path, "pending_20260101_030303",
                              lines=2, age_seconds=self._old())
        led = next((d / "audio").glob("*_ledger.json"))
        led.write_bytes(b"\xef\xbb\xbf" + led.read_bytes())
        _age_tree(d, self._old())
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert str(d) in report.skipped_unreadable

    def test_no_ledger_but_files_present_is_preserved(self, tmp_path):
        """Absent ledger deletes only on POSITIVE emptiness. A directory with
        no ledger but a real file under it is not the failed-skeleton case."""
        d = _make_pending_dir(tmp_path, "pending_20260101_040404",
                              with_ledger=False, age_seconds=self._old())
        (d / "audio" / "opening_theme.wav").write_bytes(b"RIFF" + b"\x00" * 64)
        _age_tree(d, self._old())
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert str(d) in report.skipped_no_ledger
        assert str(d) not in report.deleted

    def test_age_is_measured_beneath_the_dir(self, tmp_path):
        """A dir whose TOP-LEVEL mtime is old but whose audio/ holds a
        fresh file is in flight, not stale. The old top-level stat missed it
        because an NTFS dir mtime moves only on direct-child changes."""
        d = _make_pending_dir(tmp_path, "pending_20260101_050505",
                              lines=0, age_seconds=self._old())
        fresh = d / "audio" / "just_written.wav"
        fresh.write_bytes(b"RIFF" + b"\x00" * 64)     # now -> young
        report = sweep_empty_pending_dirs(tmp_path)
        assert d.exists()
        assert str(d) in report.skipped_too_young

    def test_report_round_trip_carries_the_new_bucket(self, tmp_path):
        report = sweep_empty_pending_dirs(tmp_path)
        assert "skipped_unreadable_count" in report.as_dict()

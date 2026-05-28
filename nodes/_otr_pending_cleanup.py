"""nodes/_otr_pending_cleanup.py -- BUG-LOCAL-290 (2026-05-27).

Sweep stale `pending_*` episode dirs that the writer node abandoned
at init time.

Background. The writer stamps a `pending_<TIMESTAMP>_ledger.json`
skeleton up-front (see OTR_LedgerScriptWriter.run §6.A) so the
in-flight ledger has a disk anchor from minute 1. When the writer
node errors out before composing any lines (cancelled queue, RSS
fetch failure, LLM load failure), the skeleton ledger stays on disk
with 0 lines + 2 meta keys. Successful runs rename the dir to
`signal_lost_<title>_<TIMESTAMP>`; aborted runs leave the
`pending_*` dir behind forever.

Live evidence: 17 pending_20260527_* dirs accumulated on
2026-05-27 between 11:28 and 15:09 (all with 0 lines, all with
just `episode_id` + `style` in meta) before the operator restarted
ComfyUI and the cleanup was added.

This module is PURE: no torch, no Comfy. Pure stdlib + the
`_otr_paths.otr_episodes_root` helper. Called by
OTR_LedgerScriptWriter.run on its happy-path entry (so it runs
exactly once per writer queue, never blocks the writer's own work).

UTF-8 no BOM. No em-dashes (Windows cp1252 decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Tuple


log = logging.getLogger("OTR")


__all__ = [
    "sweep_empty_pending_dirs",
    "PendingSweepReport",
    "DEFAULT_PENDING_MAX_AGE_SECONDS",
]


# 2 hours. A run that hasn't progressed past skeleton-stamp in 2
# hours is definitely abandoned (a clean writer run takes minutes;
# a clean full audio+video pipeline takes ~35 minutes max). The
# threshold is conservative: NEVER touch a dir that might still be
# in flight.
DEFAULT_PENDING_MAX_AGE_SECONDS: int = 2 * 60 * 60


class PendingSweepReport:
    """Audit log for one sweep pass.

    Plain class with __init__ + as_dict (not a dataclass) so a
    consumer doesn't need to import the dataclass type.
    """

    def __init__(self) -> None:
        self.scanned: int = 0
        self.deleted: List[str] = []
        self.skipped_too_young: List[str] = []
        self.skipped_has_lines: List[str] = []
        self.skipped_no_ledger: List[str] = []
        self.errors: List[Tuple[str, str]] = []

    def as_dict(self) -> dict:
        return {
            "scanned": int(self.scanned),
            "deleted_count": len(self.deleted),
            "deleted": list(self.deleted),
            "skipped_too_young_count": len(self.skipped_too_young),
            "skipped_has_lines_count": len(self.skipped_has_lines),
            "skipped_no_ledger_count": len(self.skipped_no_ledger),
            "errors_count": len(self.errors),
            "errors": [
                {"path": p, "reason": r} for (p, r) in self.errors
            ],
        }


def _ledger_has_lines(audio_dir: Path) -> bool | None:
    """Inspect the dir's *_ledger.json. Returns True if it has any
    `lines` rows, False if the ledger is shaped but lines=[],
    None if no ledger file exists (in-flight rename race or empty
    dir).
    """
    if not audio_dir.is_dir():
        return None
    ledgers = list(audio_dir.glob("*_ledger.json"))
    if not ledgers:
        return None
    # Pick the most-recently-modified ledger if multiple exist
    # (shouldn't happen, but defensive).
    led_p = max(ledgers, key=lambda p: p.stat().st_mtime)
    try:
        with open(led_p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    lines = data.get("lines") or []
    return bool(lines)


def sweep_empty_pending_dirs(
    episodes_root: Path,
    *,
    max_age_seconds: int = DEFAULT_PENDING_MAX_AGE_SECONDS,
    dry_run: bool = False,
) -> PendingSweepReport:
    """Delete `pending_*` dirs whose ledger has 0 lines AND whose
    mtime is older than `max_age_seconds`.

    Args:
        episodes_root: Path to `output/otr/episodes/`. Returned by
            `_otr_paths.otr_episodes_root()`.
        max_age_seconds: dirs younger than this are SKIPPED no
            matter what (an in-flight run that paused). Default
            2 hours.
        dry_run: when True, report what WOULD be deleted without
            calling shutil.rmtree. The test suite uses this.

    Returns:
        PendingSweepReport with per-dir disposition.

    PD1: never raises. Filesystem errors are caught per-dir and
    appended to report.errors.
    """
    report = PendingSweepReport()

    if not episodes_root.is_dir():
        return report

    now = time.time()
    age_cutoff_mtime = now - float(max_age_seconds)

    for child in episodes_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("pending_"):
            continue
        report.scanned += 1

        try:
            mtime = child.stat().st_mtime
        except OSError as exc:
            report.errors.append((str(child), f"stat failed: {exc}"))
            continue

        if mtime > age_cutoff_mtime:
            report.skipped_too_young.append(str(child))
            continue

        audio_dir = child / "audio"
        ledger_state = _ledger_has_lines(audio_dir)
        if ledger_state is None:
            # No ledger -- treat as empty IF age cutoff was met.
            # Mainly catches abandoned dirs where even the
            # skeleton stamp failed.
            report.skipped_no_ledger.append(str(child))
            if dry_run:
                continue
            try:
                shutil.rmtree(child)
                report.deleted.append(str(child))
            except OSError as exc:
                report.errors.append(
                    (str(child), f"rmtree failed: {exc}"),
                )
            continue

        if ledger_state is True:
            # Real ledger with composed lines -- leave it alone.
            # The dir is forensic evidence of a run that got
            # somewhere; the operator may want to inspect it.
            report.skipped_has_lines.append(str(child))
            continue

        # ledger_state is False -- ledger exists but lines=[].
        # The writer abandoned before composing anything. Eligible
        # for cleanup.
        if dry_run:
            report.deleted.append(str(child))
            continue
        try:
            shutil.rmtree(child)
            report.deleted.append(str(child))
        except OSError as exc:
            report.errors.append(
                (str(child), f"rmtree failed: {exc}"),
            )

    if report.deleted and not dry_run:
        log.info(
            "[PendingSweep] BUG-LOCAL-290: deleted %d empty pending dir(s) "
            "(scanned=%d skipped_too_young=%d skipped_has_lines=%d "
            "skipped_no_ledger=%d errors=%d).",
            len(report.deleted),
            report.scanned,
            len(report.skipped_too_young),
            len(report.skipped_has_lines),
            len(report.skipped_no_ledger),
            len(report.errors),
        )
    elif dry_run:
        log.debug(
            "[PendingSweep] dry-run: would have deleted %d dir(s).",
            len(report.deleted),
        )
    return report

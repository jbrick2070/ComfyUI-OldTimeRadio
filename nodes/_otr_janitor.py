"""OH-3 janitor -- the ONE sanctioned auto-delete (output-tree contract,
operator law 2026-06-11).

Sweeps STALE entries under ``episodes/_shared/tmp`` ONLY (the scratch tier
the launchers point TEMP/TMP/OTR_GPU_LEASE_DIR at). Everything else in the
output tree is operator-gated: episode assets are never auto-deleted, the
OH-4 migration prints a dry-run table and STOPS for an explicit yes, and
the pending-dir sweep (BUG-LOCAL-290) keeps its own narrow contract.

Rules (the OUTPUT_TREE_CONTRACT OH-3 ticket, verbatim):

* age threshold (default 24h; ``OTR_TMP_SWEEP_MAX_AGE_S`` overrides);
* ONLY under ``otr_shared_tmp_dir()`` -- the sweep refuses any other root;
* skip-locked: an entry that cannot be deleted (open handle, permission)
  is SKIPPED with a log line, never an error;
* every unlink is LOGGED;
* never raises (PD1) -- a janitor failure must not block a render.

Runs at server boot (``__init__.py`` registration tail) and post-publish
(``OTR_MasterAudioMux`` after the obs copy). Also drops the ``_shared``
README.txt (OH-1) so an operator browsing the tree knows what the tier is.

UTF-8, no BOM, ASCII-only source. Dependency-free (stdlib only).
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("OTR.janitor")

#: Default stale threshold: 24 hours.
DEFAULT_TMP_MAX_AGE_S = 24 * 3600

_SHARED_README = """OTR reserved system tier (episodes/_shared) -- NOT an episode.

cache/  content-addressed cross-episode copies (e.g. the AS-5 portrait
        pool, {sha256}.png). Never the only copy: the episode dir holds
        the asset of record.
tmp/    scratch (TEMP/TMP/OTR_GPU_LEASE_DIR). Janitor-swept: stale
        entries are auto-deleted after an age threshold. Do not store
        anything you want to keep here.
state/  per-machine runtime state (news_history.json, node run reports).

Contract: docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md
(the otr/ top level is EXACTLY episodes/ + obs/).
"""


@dataclass
class TmpSweepReport:
    """Disposition of one sweep run (PD1: errors recorded, never raised)."""
    scanned: int = 0
    deleted: list = field(default_factory=list)
    skipped_young: list = field(default_factory=list)
    skipped_locked: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def ensure_shared_readme() -> None:
    """Drop README.txt into ``episodes/_shared/`` (OH-1; idempotent,
    never raises)."""
    try:
        from ._otr_paths import otr_shared_root
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_paths import otr_shared_root  # type: ignore
    try:
        root = otr_shared_root()
        root.mkdir(parents=True, exist_ok=True)
        readme = root / "README.txt"
        if not readme.exists():
            readme.write_text(_SHARED_README, encoding="utf-8")
            log.info("[OTR.janitor] wrote %s", readme)
    except Exception as exc:  # noqa: BLE001 -- PD1
        log.warning("[OTR.janitor] shared README write skipped: %s", exc)


def _entry_mtime(p: Path) -> float:
    """Newest mtime within an entry (a dir's recent children keep it alive)."""
    try:
        newest = p.stat().st_mtime
    except OSError:
        return time.time()          # unstat-able -> treat as fresh (skip)
    if p.is_dir():
        try:
            for child in p.rglob("*"):
                try:
                    m = child.stat().st_mtime
                    if m > newest:
                        newest = m
                except OSError:
                    continue
        except OSError:
            pass
    return newest


def sweep_shared_tmp(max_age_seconds: float | None = None,
                     dry_run: bool = False) -> TmpSweepReport:
    """Delete stale top-level entries under ``episodes/_shared/tmp``.

    The ONE sanctioned auto-delete. Age = newest mtime inside the entry;
    younger-than-threshold entries are skipped (an in-flight render's
    scratch stays). Locked/undeletable entries are SKIPPED loud. Returns
    a :class:`TmpSweepReport`; never raises."""
    report = TmpSweepReport()
    try:
        try:
            from ._otr_paths import otr_shared_tmp_dir
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_paths import otr_shared_tmp_dir  # type: ignore
        tmp_root = otr_shared_tmp_dir()
        if max_age_seconds is None:
            raw = os.environ.get("OTR_TMP_SWEEP_MAX_AGE_S", "")
            try:
                max_age_seconds = float(raw) if raw.strip() else \
                    DEFAULT_TMP_MAX_AGE_S
            except (TypeError, ValueError):
                max_age_seconds = DEFAULT_TMP_MAX_AGE_S
        if not tmp_root.is_dir():
            return report
        cutoff = time.time() - float(max_age_seconds)
        for entry in tmp_root.iterdir():
            report.scanned += 1
            if _entry_mtime(entry) > cutoff:
                report.skipped_young.append(str(entry))
                continue
            if dry_run:
                report.deleted.append(str(entry))
                continue
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                report.deleted.append(str(entry))
                log.info("[OTR.janitor] swept stale tmp entry: %s", entry)
            except OSError as exc:
                # skip-locked: an open handle / permission error is a SKIP.
                report.skipped_locked.append(str(entry))
                log.info("[OTR.janitor] tmp entry locked, skipped: %s (%s)",
                         entry, exc)
    except Exception as exc:  # noqa: BLE001 -- PD1: never raises
        report.errors.append(str(exc))
        log.warning("[OTR.janitor] sweep aborted (non-fatal): %s", exc)
    return report


def run_boot_sweep() -> None:
    """Server-boot hook: README + sweep, fully fail-soft (PD1)."""
    ensure_shared_readme()
    rep = sweep_shared_tmp()
    if rep.deleted or rep.skipped_locked or rep.errors:
        log.warning("[OTR.janitor] boot sweep: %d scanned, %d deleted, "
                    "%d locked-skip, %d young-skip, %d errors",
                    rep.scanned, len(rep.deleted), len(rep.skipped_locked),
                    len(rep.skipped_young), len(rep.errors))


__all__ = ["sweep_shared_tmp", "run_boot_sweep", "ensure_shared_readme",
           "TmpSweepReport", "DEFAULT_TMP_MAX_AGE_S"]

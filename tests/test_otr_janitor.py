"""OH-3 janitor tests -- the ONE sanctioned auto-delete.

Pins: sweep is scoped to episodes/_shared/tmp only; the age threshold
holds (young entries survive); deletions are reported; dry_run deletes
nothing; locked entries are SKIPPED not errored; the sweep never raises
(PD1); the _shared README drop is idempotent. Pure CPU. UTF-8, no BOM,
ASCII-only source.
"""
from __future__ import annotations

import os
import time

import pytest

from nodes import _otr_janitor as J
from nodes import _otr_paths as P


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("OTR_TMP_SWEEP_MAX_AGE_S", raising=False)
    return tmp_path


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))


def test_sweep_deletes_stale_keeps_young(out_root):
    tmp = P.otr_shared_tmp_dir()
    tmp.mkdir(parents=True)
    stale = tmp / "otr_old_scratch.bin"
    stale.write_bytes(b"x")
    _age(stale, J.DEFAULT_TMP_MAX_AGE_S + 600)
    young = tmp / "otr_inflight.bin"
    young.write_bytes(b"y")
    rep = J.sweep_shared_tmp()
    assert str(stale) in rep.deleted and not stale.exists()
    assert str(young) in rep.skipped_young and young.exists()
    assert rep.errors == []


def test_sweep_stale_dir_with_fresh_child_survives(out_root):
    """A dir's NEWEST inner mtime keeps it alive (in-flight scratch)."""
    tmp = P.otr_shared_tmp_dir()
    d = tmp / "otr_ffmpeg_job"
    d.mkdir(parents=True)
    inner = d / "frames.bin"
    inner.write_bytes(b"z")
    _age(d, J.DEFAULT_TMP_MAX_AGE_S + 600)       # dir LOOKS old...
    rep = J.sweep_shared_tmp()                   # ...child is fresh
    assert str(d) in rep.skipped_young and d.exists()


def test_sweep_dry_run_deletes_nothing(out_root):
    tmp = P.otr_shared_tmp_dir()
    tmp.mkdir(parents=True)
    stale = tmp / "otr_old.bin"
    stale.write_bytes(b"x")
    _age(stale, J.DEFAULT_TMP_MAX_AGE_S + 600)
    rep = J.sweep_shared_tmp(dry_run=True)
    assert str(stale) in rep.deleted and stale.exists()


def test_sweep_threshold_env_override(out_root, monkeypatch):
    tmp = P.otr_shared_tmp_dir()
    tmp.mkdir(parents=True)
    f = tmp / "otr_recent.bin"
    f.write_bytes(b"x")
    _age(f, 120)
    monkeypatch.setenv("OTR_TMP_SWEEP_MAX_AGE_S", "60")
    rep = J.sweep_shared_tmp()
    assert str(f) in rep.deleted and not f.exists()


def test_sweep_locked_entry_skipped_not_error(out_root):
    """An open handle (Windows lock) -> SKIP with a log, never an error."""
    tmp = P.otr_shared_tmp_dir()
    tmp.mkdir(parents=True)
    locked = tmp / "otr_locked.bin"
    locked.write_bytes(b"x")
    _age(locked, J.DEFAULT_TMP_MAX_AGE_S + 600)
    with open(locked, "rb"):
        rep = J.sweep_shared_tmp()
    assert rep.errors == []
    # On Windows the open handle blocks unlink -> skipped_locked; on a
    # POSIX runner the unlink would succeed -> deleted. Both are valid.
    assert (str(locked) in rep.skipped_locked
            or str(locked) in rep.deleted)


def test_sweep_missing_tmp_dir_is_noop(out_root):
    rep = J.sweep_shared_tmp()
    assert rep.scanned == 0 and rep.deleted == [] and rep.errors == []


def test_sweep_never_touches_outside_tmp(out_root):
    """Scope guard: episode assets + cache/state are NEVER swept."""
    ep = P.otr_stills_dir("ep_keep")
    ep.mkdir(parents=True)
    keep = ep / "still.png"
    keep.write_bytes(b"p")
    _age(keep, J.DEFAULT_TMP_MAX_AGE_S * 10)
    cache = P.otr_shared_cache_dir()
    cache.mkdir(parents=True)
    pooled = cache / ("ab" * 32 + ".png")
    pooled.write_bytes(b"p")
    _age(pooled, J.DEFAULT_TMP_MAX_AGE_S * 10)
    J.sweep_shared_tmp()
    assert keep.exists() and pooled.exists()


def test_shared_readme_idempotent(out_root):
    J.ensure_shared_readme()
    readme = P.otr_shared_root() / "README.txt"
    assert readme.exists()
    first = readme.read_text(encoding="utf-8")
    J.ensure_shared_readme()
    assert readme.read_text(encoding="utf-8") == first
    assert "janitor" in first.lower()      # the tier docs name the sweep

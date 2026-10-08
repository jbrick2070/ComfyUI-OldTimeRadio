"""_otr_janitor._entry_is_fresh gives the verdict the full walk gave.

sweep_shared_tmp used to stat EVERY file under an entry to find its newest
mtime, then compare that against the cutoff. The predicate stops at the first
child newer than the cutoff. These pin that the verdict is unchanged --
strict ``>``, nested activity, a root stat failure comparing "now" -- and that
the walk really does stop.
"""
import os
import time
from pathlib import Path

import pytest

from nodes import _otr_janitor as J


def _touch(path, t):
    os.utime(path, (t, t))


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "entry"
    (root / "a" / "b").mkdir(parents=True)
    old = root / "a" / "old.txt"
    new = root / "a" / "b" / "new.txt"
    old.write_text("x", encoding="utf-8")
    new.write_text("y", encoding="utf-8")
    base = time.time() - 10_000
    for p in (old, root / "a" / "b", root / "a", root):
        _touch(p, base)
    _touch(new, base + 5_000)
    return root, base


@pytest.mark.parametrize("offset", [-1, 0, 4_999, 5_000, 5_001, 9_000])
def test_same_verdict_as_the_full_walk(tree, offset):
    root, base = tree
    cutoff = base + offset
    assert J._entry_is_fresh(root, cutoff) == (J._entry_mtime(root) > cutoff)


def test_a_child_exactly_at_the_cutoff_is_not_fresh(tree):
    root, base = tree
    assert J._entry_is_fresh(root, base + 5_000) is False


def test_a_plain_file_entry(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"z")
    _touch(f, 1_000.0)
    for cutoff in (999.0, 1_000.0, 1_001.0):
        assert J._entry_is_fresh(f, cutoff) == (J._entry_mtime(f) > cutoff)


@pytest.mark.parametrize("max_age", [60.0, 0.0, -5.0])
def test_root_stat_failure_compares_now_not_always_fresh(tmp_path, monkeypatch, max_age):
    monkeypatch.setattr(J.time, "time", lambda: 1_000.0)
    missing = tmp_path / "gone"
    cutoff = 1_000.0 - max_age
    expected = 1_000.0 > cutoff
    assert J._entry_is_fresh(missing, cutoff) is expected
    assert (J._entry_mtime(missing) > cutoff) is expected


def test_the_walk_stops_at_the_first_fresh_child(tmp_path, monkeypatch):
    root = tmp_path / "e"
    root.mkdir()
    child = root / "x"
    child.write_text("1", encoding="utf-8")
    cutoff = time.time() - 3_600
    _touch(root, cutoff - 100)

    def fake_rglob(self, pattern):
        yield child
        raise AssertionError("walked past the first fresh child")

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    assert J._entry_is_fresh(root, cutoff) is True

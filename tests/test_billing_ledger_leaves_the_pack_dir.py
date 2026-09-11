"""The cloud-media billing ledger must not live inside the installed pack.

`billing_ledger.jsonl` is an append-only record of real money spent on partner APIs,
and it is the ONLY copy of that history anywhere on disk -- verified: nothing reads it
back, so nothing could rebuild it. It used to sit under `cache_root`, which resolves
to `<repo>/otr/cache/cloud_media` -- inside the pack. A registry update or a reinstall
deletes that tree.

It now lives under `otr_state_dir()`, the tier that already exists for durable
per-machine runtime state: under the user's output tree, and never swept (the janitor
is scoped to `_shared/tmp` alone). The CACHE tier would have been the wrong home for
the opposite reason -- its own contract says "a cache entry is NEVER the only copy",
and this is.

Existing ledgers are COPIED forward, not moved, so a half-finished migration cannot
lose spend history.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes._otr_shared import cloud_media_backend as cmb
from nodes import _otr_paths as P


@pytest.fixture()
def pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("OTR_CLOUD_MEDIA_CACHE_DIR", raising=False)
    return tmp_path


class _Session:
    """Minimal stand-in carrying only what ledger_path touches."""

    def __init__(self, cache_root):
        self.cache_root = Path(cache_root)
        self._ledger_path = None

    ledger_path = cmb.CloudMediaSession.ledger_path


def test_the_ledger_lands_under_state_not_inside_the_pack(pinned, tmp_path):
    """THE DEFECT: a registry update wiped the only copy of real spend history."""
    cache_root = tmp_path / "pack" / "otr" / "cache" / "cloud_media"
    path = _Session(cache_root).ledger_path()

    assert path == Path(P.otr_state_dir()) / "cloud_media" / "billing_ledger.jsonl"
    assert cache_root not in path.parents, (
        "the billing ledger is still inside the pack cache tier: %s" % path)


def test_the_ledger_is_not_in_the_janitor_swept_tier(pinned, tmp_path):
    """`otr_shared_tmp_dir()` is the ONE sanctioned auto-delete. An audit trail
    must never live there."""
    path = _Session(tmp_path / "cache").ledger_path()
    assert Path(P.otr_shared_tmp_dir()).resolve() not in path.resolve().parents


def test_an_existing_ledger_is_COPIED_forward_and_the_original_survives(
        pinned, tmp_path):
    """A copy, not a move: a half-finished migration must not lose history.

    This also exercises the branch that logs -- the module carried no logger
    before this change, so the copy path would have raised NameError on the very
    first real migration while every other test stayed green.
    """
    cache_root = tmp_path / "pack" / "cache"
    cache_root.mkdir(parents=True)
    legacy = cache_root / "billing_ledger.jsonl"
    legacy.write_text(json.dumps({"ts": 1, "usd": 0.42}) + "\n", encoding="utf-8")

    path = _Session(cache_root).ledger_path()

    assert path.is_file(), "the ledger was not copied forward"
    assert json.loads(path.read_text(encoding="utf-8").strip())["usd"] == 0.42
    assert legacy.is_file(), "the original must be left in place, not moved"


def test_an_existing_destination_is_never_overwritten(pinned, tmp_path):
    """If the new ledger already has entries, the legacy copy must not clobber it."""
    cache_root = tmp_path / "pack" / "cache"
    cache_root.mkdir(parents=True)
    (cache_root / "billing_ledger.jsonl").write_text(
        json.dumps({"ts": 1, "usd": 9.99}) + "\n", encoding="utf-8")

    dest_dir = Path(P.otr_state_dir()) / "cloud_media"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "billing_ledger.jsonl").write_text(
        json.dumps({"ts": 2, "usd": 0.01}) + "\n", encoding="utf-8")

    path = _Session(cache_root).ledger_path()
    assert json.loads(path.read_text(encoding="utf-8").strip())["usd"] == 0.01


def test_an_explicit_cache_override_still_owns_the_ledger(tmp_path, monkeypatch):
    """A box that sets OTR_CLOUD_MEDIA_CACHE_DIR named the whole tier on purpose
    and must not be silently relocated."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
    override = tmp_path / "chosen"
    monkeypatch.setenv("OTR_CLOUD_MEDIA_CACHE_DIR", str(override))

    path = _Session(override).ledger_path()
    assert path == override / "billing_ledger.jsonl"

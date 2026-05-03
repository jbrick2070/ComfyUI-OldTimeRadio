"""
tests/test_ledger_rename.py -- Phase B (BUG-LOCAL-015) regression suite

Covers Ledger.rename_episode invariant: either complete with canonical
episode dir + canonical ledger + canonical treatment, OR raise BEFORE
mutating in-memory episode state. No silent split state.

State matrix (one test per row):
  old exists, new missing  -> retry os.replace 3x; hard-fail after
  old exists, new exists   -> hard-fail conflict (NEVER merge)
  old missing, new exists  -> accept as already moved (idempotent)
  old missing, new missing -> hard-fail (nothing to rename)
  same path (case-insensitive normcase) -> no-op

Plus: dir-move retry semantics, treatment + sidecar rename, re-rename
to same id idempotency.

Consult source: docs/2026-05-02-phase-b-rename-consult__*
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from nodes import production_ledger
from nodes.production_ledger import Ledger, _slugify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pending_workspace(tmp_path):
    """Build a pending_<ts>/audio/ workspace with a ledger + treatment.

    Returns a dict with old_id, new_id, paths, and a fresh Ledger
    pointed at the pending audio dir.
    """
    old_id = "pending_20260502_120000"
    new_id = "the_test_episode"
    old_ep_dir = tmp_path / "episodes" / old_id
    old_audio_dir = old_ep_dir / "audio"
    old_audio_dir.mkdir(parents=True)
    # Pre-create the ledger + treatment + a notes sidecar to verify
    # the glob catches all <old>_*.txt files.
    (old_audio_dir / f"{_slugify(old_id, limit=120)}_ledger.json").write_text(
        '{"episode_id": "%s"}' % old_id, encoding="utf-8"
    )
    (old_audio_dir / f"{_slugify(old_id, limit=120)}_treatment.txt").write_text(
        "treatment body", encoding="utf-8"
    )
    (old_audio_dir / f"{_slugify(old_id, limit=120)}_notes.txt").write_text(
        "notes body", encoding="utf-8"
    )
    new_ep_dir = tmp_path / "episodes" / new_id
    new_audio_dir = new_ep_dir / "audio"
    led = Ledger(old_id, str(old_audio_dir))
    return {
        "old_id": old_id,
        "new_id": new_id,
        "old_ep_dir": old_ep_dir,
        "old_audio_dir": old_audio_dir,
        "new_ep_dir": new_ep_dir,
        "new_audio_dir": new_audio_dir,
        "led": led,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_renames_dir_ledger_and_sidecars(pending_workspace):
    w = pending_workspace
    led = w["led"]

    led.rename_episode(w["new_id"])

    # In-memory state advanced
    assert led.episode_id == w["new_id"]
    assert led.data["episode_id"] == w["new_id"]
    assert Path(led.out_dir) == w["new_audio_dir"]

    # On-disk: old gone, new present with canonical names
    assert not w["old_ep_dir"].exists()
    assert w["new_ep_dir"].exists()
    new_slug = _slugify(w["new_id"], limit=120)
    assert (w["new_audio_dir"] / f"{new_slug}_ledger.json").exists()
    assert (w["new_audio_dir"] / f"{new_slug}_treatment.txt").exists()
    assert (w["new_audio_dir"] / f"{new_slug}_notes.txt").exists()
    # No pending_* leftovers
    old_slug = _slugify(w["old_id"], limit=120)
    assert not (w["new_audio_dir"] / f"{old_slug}_ledger.json").exists()
    assert not (w["new_audio_dir"] / f"{old_slug}_treatment.txt").exists()


def test_same_id_is_noop(pending_workspace):
    w = pending_workspace
    led = w["led"]
    # Same id -> early return
    led.rename_episode(w["old_id"])
    assert led.episode_id == w["old_id"]
    assert w["old_ep_dir"].exists()


# ---------------------------------------------------------------------------
# State matrix
# ---------------------------------------------------------------------------

def test_conflict_old_and_new_both_exist_raises(pending_workspace):
    w = pending_workspace
    led = w["led"]
    # Pre-create new_ep_dir to simulate partial-state from prior crash
    (w["new_ep_dir"] / "audio").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="both source and destination"):
        led.rename_episode(w["new_id"])

    # In-memory state UNCHANGED
    assert led.episode_id == w["old_id"]
    assert led.data["episode_id"] == w["old_id"]
    assert Path(led.out_dir) == w["old_audio_dir"]
    # On-disk state UNCHANGED
    assert w["old_ep_dir"].exists()
    assert w["new_ep_dir"].exists()


def test_both_missing_raises(pending_workspace, tmp_path):
    """Source dir vanished AND destination doesn't exist -> RuntimeError."""
    w = pending_workspace
    led = w["led"]
    # Wipe source after Ledger init
    import shutil
    shutil.rmtree(w["old_ep_dir"])

    with pytest.raises(RuntimeError, match="neither source nor destination"):
        led.rename_episode(w["new_id"])

    assert led.episode_id == w["old_id"]


def test_idempotent_recovery_old_missing_new_exists(pending_workspace):
    """Previous run already moved the dir, then crashed before in-memory
    state advanced. Next call to rename_episode should accept the
    already-moved state."""
    w = pending_workspace
    led = w["led"]
    # Simulate prior successful os.replace, then crash:
    os.replace(str(w["old_ep_dir"]), str(w["new_ep_dir"]))
    assert not w["old_ep_dir"].exists()
    assert w["new_ep_dir"].exists()

    # rename_episode should NOT raise, should advance state
    led.rename_episode(w["new_id"])

    assert led.episode_id == w["new_id"]
    assert Path(led.out_dir) == w["new_audio_dir"]


# ---------------------------------------------------------------------------
# Retry semantics
# ---------------------------------------------------------------------------

def test_dir_move_retries_then_succeeds(pending_workspace, monkeypatch):
    w = pending_workspace
    led = w["led"]

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(src, dst):
        # Only flake on the dir move (src == old_ep_dir). Let ledger and
        # treatment renames pass through.
        if str(src) == str(w["old_ep_dir"]):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise OSError("simulated Defender lock")
        return real_replace(src, dst)

    monkeypatch.setattr(production_ledger.os, "replace", flaky_replace)
    # Skip the real sleep
    monkeypatch.setattr(
        "time.sleep", lambda _x: None
    )

    led.rename_episode(w["new_id"])

    assert call_count["n"] == 3
    assert led.episode_id == w["new_id"]
    assert w["new_ep_dir"].exists()


def test_dir_move_fails_all_attempts_raises_state_unchanged(
    pending_workspace, monkeypatch
):
    w = pending_workspace
    led = w["led"]

    def always_fail(src, dst):
        if str(src) == str(w["old_ep_dir"]):
            raise OSError("simulated unfixable lock")
        # Should never reach here -- if dir move always fails, downstream
        # renames never happen.
        raise AssertionError("downstream rename should not run")

    monkeypatch.setattr(production_ledger.os, "replace", always_fail)
    monkeypatch.setattr("time.sleep", lambda _x: None)

    with pytest.raises(RuntimeError, match="3 attempts"):
        led.rename_episode(w["new_id"])

    # In-memory state UNCHANGED
    assert led.episode_id == w["old_id"]
    assert led.data["episode_id"] == w["old_id"]
    assert Path(led.out_dir) == w["old_audio_dir"]
    # On-disk state UNCHANGED
    assert w["old_ep_dir"].exists()
    assert not w["new_ep_dir"].exists()


def test_dir_move_error_message_mentions_human_locks(
    pending_workspace, monkeypatch
):
    w = pending_workspace
    led = w["led"]

    def always_fail(src, dst):
        raise OSError("simulated lock")

    monkeypatch.setattr(production_ledger.os, "replace", always_fail)
    monkeypatch.setattr("time.sleep", lambda _x: None)

    with pytest.raises(RuntimeError) as excinfo:
        led.rename_episode(w["new_id"])

    # Must guide the user to check for human-held locks
    msg = str(excinfo.value)
    assert "Notepad" in msg or "VLC" in msg or "open" in msg.lower()


# ---------------------------------------------------------------------------
# Best-effort sidecar handling
# ---------------------------------------------------------------------------

def test_treatment_rename_failure_does_not_raise(
    pending_workspace, monkeypatch
):
    """If treatment rename fails after dir + ledger move succeed, the
    function should warn but not raise -- the dir invariant is already
    satisfied and downstream nodes can find the canonical ledger."""
    w = pending_workspace
    led = w["led"]

    real_replace = os.replace

    def fail_only_treatment(src, dst):
        if "_treatment.txt" in str(dst):
            raise OSError("simulated lock on treatment")
        return real_replace(src, dst)

    monkeypatch.setattr(production_ledger.os, "replace", fail_only_treatment)

    # Should NOT raise
    led.rename_episode(w["new_id"])

    # State + dir + ledger all advanced
    assert led.episode_id == w["new_id"]
    assert w["new_ep_dir"].exists()
    new_slug = _slugify(w["new_id"], limit=120)
    assert (w["new_audio_dir"] / f"{new_slug}_ledger.json").exists()
    # Treatment stuck at old name (best-effort tolerance)
    old_slug = _slugify(w["old_id"], limit=120)
    assert (w["new_audio_dir"] / f"{old_slug}_treatment.txt").exists()


def test_sidecar_glob_uses_old_id_prefix_not_pending_wildcard(
    pending_workspace
):
    """Drop a file with an unrelated pending_*_treatment.txt name into
    the destination after dir move. The rename should NOT touch it."""
    w = pending_workspace
    led = w["led"]
    # Make sure the destination dir exists so we can pre-seed unrelated file
    w["new_ep_dir"].mkdir(parents=True, exist_ok=False)
    (w["new_ep_dir"] / "audio").mkdir()
    # ...but conflict-state would now raise. Instead, run rename normally
    # then verify post-rename that an unrelated pending_*_treatment.txt
    # file pre-existing in the new dir would not be swept.
    # Reset: use a fresh ledger setup that doesn't trip the conflict guard.
    import shutil
    shutil.rmtree(w["new_ep_dir"])

    led.rename_episode(w["new_id"])
    # Now drop an unrelated file
    unrelated = w["new_audio_dir"] / "pending_20260101_999999_treatment.txt"
    unrelated.write_text("unrelated", encoding="utf-8")

    # Calling rename_episode again with a different id should leave the
    # unrelated file alone (its prefix doesn't match new_id)
    led.rename_episode("third_id")
    assert unrelated.exists() or (
        # If the dir got moved, the unrelated file moves with it
        Path(led.out_dir) / "pending_20260101_999999_treatment.txt"
    ).exists()

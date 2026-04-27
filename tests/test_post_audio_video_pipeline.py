"""Regression tests for OTR_PostAudioVideoPipeline node.

Covers:
  - _resolve_ledger_from_input flexibility (mp4 -> ledger swap, .json
    direct, directory scan, empty fallback).
  - dry-run mode returns a valid status string without spawning a
    child process (no Popen call hits in this code path).
  - INPUT_TYPES contract (required/optional shape, types).
  - Error path: bad input returns an error: status, not a crash.

The fire / wait modes are NOT exercised here (they spawn a real
PowerShell subprocess and are integration-tested by Jeffrey running
the FULL workflow). dry-run is enough to validate the node's
in-process logic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "nodes" / "post_audio_video_pipeline.py"


def _load_module():
    """Load the node module by file path (the package dir name has a
    hyphen which Python can't import as a normal package)."""
    spec = importlib.util.spec_from_file_location(
        "post_audio_video_pipeline_under_test", str(NODE_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_module()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_ledger_from_json_path_uses_as_is(tmp_path: Path, m) -> None:
    led = tmp_path / "test_episode_ledger.json"
    led.write_text(json.dumps({"episode_id": "test_episode"}), encoding="utf-8")
    found = m._resolve_ledger_from_input(str(led))
    assert found == led


def test_resolve_ledger_from_mp4_path_swaps_suffix(tmp_path: Path, m) -> None:
    """The audio mp4 path -> ledger pairing is `<stem>.mp4` ->
    `<stem>_ledger.json` per the EpisodeAssembler convention."""
    mp4 = tmp_path / "test_episode.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    led = tmp_path / "test_episode_ledger.json"
    led.write_text(json.dumps({"episode_id": "test_episode"}), encoding="utf-8")

    found = m._resolve_ledger_from_input(str(mp4))
    assert found == led


def test_resolve_ledger_from_mp4_path_returns_none_if_ledger_missing(
    tmp_path: Path, m
) -> None:
    """If the .mp4 is on disk but its sibling ledger isn't, return None
    rather than fabricating a path."""
    mp4 = tmp_path / "orphan_episode.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    found = m._resolve_ledger_from_input(str(mp4))
    assert found is None


def test_resolve_ledger_from_directory_picks_newest_non_pending(
    tmp_path: Path, m
) -> None:
    import os
    import time

    older = tmp_path / "older_episode_ledger.json"
    older.write_text("{}", encoding="utf-8")
    pending = tmp_path / "pending_in_progress_ledger.json"
    pending.write_text("{}", encoding="utf-8")
    newer = tmp_path / "newer_episode_ledger.json"
    newer.write_text("{}", encoding="utf-8")

    # Force mtime ordering so older is older, newer is newest
    now = time.time()
    os.utime(older, (now - 1000, now - 1000))
    os.utime(pending, (now, now))  # pending is newest by mtime
    os.utime(newer, (now - 100, now - 100))

    found = m._resolve_ledger_from_input(str(tmp_path))
    # pending_* must be excluded even if newest
    assert found == newer


def test_resolve_ledger_handles_empty_input_without_crashing(m) -> None:
    """Empty string falls back to scanning canonical audio dirs.
    Test environment may or may not have those dirs; the function must
    return None or a Path, never raise."""
    found = m._resolve_ledger_from_input("")
    assert found is None or isinstance(found, Path)


# ---------------------------------------------------------------------------
# Node contract
# ---------------------------------------------------------------------------

def test_node_class_attributes(m) -> None:
    cls = m.PostAudioVideoPipeline
    assert cls.CATEGORY == "OTR/v2/Pipeline"
    assert cls.FUNCTION == "execute"
    assert cls.RETURN_TYPES == ("STRING", "STRING")
    assert cls.RETURN_NAMES == ("status", "log_path")


def test_input_types_shape(m) -> None:
    inp = m.PostAudioVideoPipeline.INPUT_TYPES()
    assert "required" in inp
    assert "audio_or_ledger_path" in inp["required"]
    assert "optional" in inp
    for k in ("clip_length", "max_clips", "portrait_map", "skip_pass1", "mode"):
        assert k in inp["optional"], f"optional input {k!r} missing"


def test_clip_length_default_is_seven_seconds(m) -> None:
    """The FULL-workflow spec says 7s clips. Pin the default so a
    silent revert to 3.88 is caught immediately."""
    inp = m.PostAudioVideoPipeline.INPUT_TYPES()
    assert inp["optional"]["clip_length"][1]["default"] == 7.0


def test_clip_length_max_respects_humo_ceiling(m) -> None:
    """HuMo's empirical max on RTX 5080 16 GB is 7.08s (177 frames).
    The widget's max should not exceed this without code changes to
    HUMO_MAX_FRAMES in render_humo_batch.py."""
    inp = m.PostAudioVideoPipeline.INPUT_TYPES()
    assert inp["optional"]["clip_length"][1]["max"] == pytest.approx(7.08, abs=0.01)


# ---------------------------------------------------------------------------
# Dry-run mode (no subprocess)
# ---------------------------------------------------------------------------

def test_dry_run_mode_returns_valid_status_no_subprocess(
    tmp_path: Path, m
) -> None:
    """dry-run validates inputs and returns a status string without
    launching anything. No child process gets spawned (the in-process
    Popen-launch path is bypassed entirely by the mode dispatch)."""
    mp4 = tmp_path / "fake_episode.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    led = tmp_path / "fake_episode_ledger.json"
    led.write_text("{}", encoding="utf-8")

    inst = m.PostAudioVideoPipeline()
    status, log_path = inst.execute(
        audio_or_ledger_path=str(mp4),
        clip_length=7.0,
        max_clips=4,
        portrait_map="ANNOUNCER=17",
        skip_pass1=True,
        mode="dry-run",
    )
    assert status.startswith("dry-run:")
    assert "fake_episode_ledger.json" in status
    assert "clip_length=7.0" in status
    assert "max_clips=4" in status
    assert log_path.endswith(".log")


def test_unresolvable_input_returns_error_status_not_crash(m) -> None:
    """Bad input must not raise -- return an `error:` status string."""
    inst = m.PostAudioVideoPipeline()
    status, log_path = inst.execute(
        audio_or_ledger_path="/nonexistent/path/that/cannot/resolve.mp4",
        mode="dry-run",
    )
    assert status.startswith("error:")
    # log_path is "" on error
    assert log_path == ""


def test_dry_run_has_correct_command_shape(tmp_path: Path, m) -> None:
    """The dry-run status string includes the command that WOULD be
    launched. Pin the shape so PowerShell invocation can't drift."""
    led = tmp_path / "shape_test_ledger.json"
    led.write_text("{}", encoding="utf-8")

    inst = m.PostAudioVideoPipeline()
    status, _ = inst.execute(
        audio_or_ledger_path=str(led),
        clip_length=7.0,
        max_clips=8,
        portrait_map="ANNOUNCER=17,LEV=29",
        skip_pass1=True,
        mode="dry-run",
    )
    # Expected command tokens
    assert "powershell.exe" in status
    assert "-ExecutionPolicy" in status
    assert "Bypass" in status
    assert "test_humo_batch_concat.ps1" in status
    assert "-Ledger" in status
    assert "-AutoSlice" in status
    assert "7.0000" in status  # clip_length passed as AutoSlice
    assert "-MaxClips 8" in status
    assert "-PortraitMap ANNOUNCER=17,LEV=29" in status
    assert "-SkipPass1" in status


# ---------------------------------------------------------------------------
# Wrapper script presence (sanity)
# ---------------------------------------------------------------------------

def test_wrapper_script_exists_at_expected_location(m) -> None:
    """The node hardcodes the wrapper path; if someone moves the
    .ps1, this test catches it before a runtime fire-mode failure."""
    assert m._WRAPPER_PATH.exists(), \
        f"wrapper not at {m._WRAPPER_PATH}"
    assert m._WRAPPER_PATH.suffix == ".ps1"

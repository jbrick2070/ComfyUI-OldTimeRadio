"""Regression tests for scripts/render_humo_batch.py plan-builder.

BUG-LOCAL-074 (Bug A): the plan-builder used to read ledger lines via
``ln.get("speaker")``, but production_ledger.set_lines stores the
character reference as ``char_id`` and resolves the human-readable name
through ``cast[*].name``. The old code returned ``""`` for every speaker
which caused ``find_portrait_for_speaker`` to fall through to the first
PNG it found -- every clip used ``portrait_001.png`` and per-speaker
prompts collapsed to a generic fallback.

These tests pin three behaviours so the regression cannot return:
    1. ``char_id`` resolves to ``cast[*].name`` and lands in the plan.
    2. Legacy ledgers that populated ``ln.speaker`` directly still work.
    3. An unknown ``char_id`` empties out gracefully (does not crash).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


# Make scripts/ importable as a top-level module.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_humo_batch as rhb  # noqa: E402  (sys.path manipulation above)


_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _make_args(portraits_dir: Path, out_dir: Path) -> argparse.Namespace:
    """Minimal Namespace for build_clip_plan.

    Only the attributes build_clip_plan reads need to be set.
    ``auto_slice`` is on so we don't depend on ledger timing fields,
    and ``master_wav`` will be passed as a non-existent path so the
    function uses the ledger fallback duration instead of ffprobe.
    """
    return argparse.Namespace(
        scope="all",
        auto_slice=3.88,
        clip_length=3.88,
        max_clips=None,
        portraits_dir=portraits_dir,
        out_dir=out_dir,
        chain=True,
    )


def _make_pass1_portraits(dirpath: Path, n: int) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (dirpath / f"otr_humo_pass1_portrait_{i:03d}.png").write_bytes(_PNG_HEADER)


# ---------------------------------------------------------------------------
# Bug A: char_id -> name resolution
# ---------------------------------------------------------------------------

def test_build_clip_plan_resolves_char_id_to_speaker_name(tmp_path: Path) -> None:
    """Ledger lines carry char_id only -- plan must hydrate speaker
    from cast[*].name so portrait selection and prompt building work."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 3)

    ledger = {
        "episode_id": "test_episode",
        "total_episode_dur_s": 60.0,
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "LEV"},
            {"char_id": "c03", "name": "TODD"},
        ],
        "lines": [
            {"line_id": "l001", "char_id": "c01", "text": "Welcome.",       "boundary": "shot_start"},
            {"line_id": "l002", "char_id": "c02", "text": "What was that?", "boundary": "beat_start"},
            {"line_id": "l003", "char_id": "c03", "text": "Static.",        "boundary": "beat_start"},
            {"line_id": "l004", "char_id": "c01", "text": "Stay calm.",     "boundary": "shot_start"},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")

    assert len(plan) == 4
    assert [c["speaker"] for c in plan] == ["ANNOUNCER", "LEV", "TODD", "ANNOUNCER"]


def test_build_clip_plan_picks_distinct_portraits_per_speaker(tmp_path: Path) -> None:
    """find_portrait_for_speaker uses cast position to index into the
    PASS1 portrait list; each speaker should land on a different PNG."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 3)

    ledger = {
        "episode_id": "test_episode",
        "total_episode_dur_s": 60.0,
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "LEV"},
            {"char_id": "c03", "name": "TODD"},
        ],
        "lines": [
            {"line_id": "l001", "char_id": "c01", "text": "A."},
            {"line_id": "l002", "char_id": "c02", "text": "B."},
            {"line_id": "l003", "char_id": "c03", "text": "C."},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")

    portrait_names = [Path(c["portrait_src_path"]).name for c in plan]
    # The fix is doing its job when each speaker lands on a different
    # portrait (regression: pre-fix all three returned portrait_001).
    assert len(set(portrait_names)) == 3
    assert all(name.startswith("otr_humo_pass1_portrait_") for name in portrait_names)


# ---------------------------------------------------------------------------
# Back-compat: legacy ledgers with ln.speaker set directly
# ---------------------------------------------------------------------------

def test_build_clip_plan_legacy_speaker_field_still_works(tmp_path: Path) -> None:
    """A pre-char_id ledger that populates ln.speaker should still
    produce a real speaker. ln.speaker must take precedence over
    char_id lookup so an explicit override always wins."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 1)

    ledger = {
        "episode_id": "legacy_episode",
        "total_episode_dur_s": 30.0,
        "cast": [{"char_id": "c01", "name": "EDNA"}],
        "lines": [
            {"line_id": "l001", "speaker": "EDNA", "text": "Test.", "boundary": "shot_start"},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")
    assert len(plan) == 1
    assert plan[0]["speaker"] == "EDNA"


def test_build_clip_plan_speaker_field_overrides_char_id(tmp_path: Path) -> None:
    """If ln.speaker disagrees with cast[char_id].name, ln.speaker wins
    (explicit override). Useful for re-cast or shot-level overrides."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 1)

    ledger = {
        "episode_id": "override_episode",
        "total_episode_dur_s": 30.0,
        "cast": [{"char_id": "c01", "name": "EDNA"}],
        "lines": [
            {"line_id": "l001", "char_id": "c01", "speaker": "EDNA_DOUBLE", "text": "Test."},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")
    assert plan[0]["speaker"] == "EDNA_DOUBLE"


# ---------------------------------------------------------------------------
# Graceful empty: unknown char_id should not crash, just empty out
# ---------------------------------------------------------------------------

def test_build_clip_plan_unknown_char_id_falls_back_to_empty(tmp_path: Path) -> None:
    """If char_id doesn't match any cast row, speaker should be empty
    string (not crash) and the line should still appear in the plan
    with the first-portrait fallback so the run keeps moving."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 1)

    ledger = {
        "episode_id": "orphan_episode",
        "total_episode_dur_s": 30.0,
        "cast": [{"char_id": "c01", "name": "EDNA"}],
        "lines": [
            {"line_id": "l001", "char_id": "c99_orphan", "text": "Glitch."},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")
    assert len(plan) == 1
    assert plan[0]["speaker"] == ""


def test_build_clip_plan_no_cast_no_speaker_is_safe(tmp_path: Path) -> None:
    """Defensive: if cast is empty AND lines have neither speaker nor
    char_id, the plan should still build with empty speakers rather
    than throwing."""
    portraits = tmp_path / "portraits"
    _make_pass1_portraits(portraits, 1)

    ledger = {
        "episode_id": "anonymous_episode",
        "total_episode_dur_s": 30.0,
        "cast": [],
        "lines": [
            {"line_id": "l001", "text": "Drift."},
            {"line_id": "l002", "text": "Settle."},
        ],
    }

    args = _make_args(portraits, tmp_path / "out")
    plan = rhb.build_clip_plan(ledger, args, tmp_path / "missing.wav")
    assert len(plan) == 2
    assert all(c["speaker"] == "" for c in plan)

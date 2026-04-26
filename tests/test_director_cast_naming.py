"""Regression test for the LLMDirector cast description name-leak.

Bug: nodes/story_orchestrator.py:7815 used to prefix
voice_assignments[name].notes with profile['name'] (a procedural-flavor
last-name like 'PRIAM SMITHERS') even though the script uses the
original script-side name (e.g. 'MEREDITH'). This made cast.description
read 'PRIAM SMITHERS - Female, wry, 40s' for a character the dialogue
calls MEREDITH, and could even contradict the gender hint
('NORA SATO - Male, intense, 50s').

Fix: drop the procedural-name prefix; pass bare profile['notes'].

This test exercises LLMDirector._randomize_character_names directly with
a minimal plan dict and asserts:
  1. notes for regular characters do NOT contain the leaked
     procedural name (no leading 'WORD WORD - ' uppercase prefix).
  2. notes look like the trait string ('Female, wry, 40s' shape).
  3. The dict key is the ORIGINAL script-side name, not the procedural one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Make repo root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes.story_orchestrator import LLMDirector  # noqa: E402


# Pattern that flags the leak: starts with two ALLCAPS tokens followed by " - "
LEAK_PREFIX = re.compile(r"^[A-Z][A-Z]+\s+[A-Z][A-Z]+\s*-\s*")


def _minimal_plan(names: list[str]) -> dict:
    """Build a plan with the shape _randomize_character_names expects."""
    return {
        "voice_assignments": {
            n: {"voice_preset": None, "notes": ""} for n in names
        }
    }


def test_regular_character_notes_do_not_carry_procedural_name_prefix():
    director = LLMDirector()
    plan = _minimal_plan(["MEREDITH", "RYAN", "SEAN", "VANCE"])
    out = director._randomize_character_names(
        plan,
        episode_seed="testseed-2026-04-26",
        gender_map={"MEREDITH": "female", "RYAN": "male",
                    "SEAN": "male", "VANCE": "male"},
    )
    va = out["voice_assignments"]
    for name in ("MEREDITH", "RYAN", "SEAN", "VANCE"):
        assert name in va, f"script-side name {name!r} missing from voice_assignments"
        notes = va[name]["notes"]
        assert isinstance(notes, str) and notes, f"{name} has empty notes"
        assert not LEAK_PREFIX.match(notes), (
            f"{name} notes still carry a procedural-name prefix: {notes!r}"
        )


def test_regular_character_notes_look_like_trait_string():
    director = LLMDirector()
    plan = _minimal_plan(["MEREDITH"])
    out = director._randomize_character_names(
        plan,
        episode_seed="testseed-2026-04-26",
        gender_map={"MEREDITH": "female"},
    )
    notes = out["voice_assignments"]["MEREDITH"]["notes"]
    # Expect at least one comma-separated trait segment.
    assert "," in notes, f"notes look malformed: {notes!r}"
    # Expect 'female' or 'Female' tag matching the gender hint.
    assert re.search(r"\bfemale\b", notes, re.IGNORECASE), (
        f"gender hint did not propagate to notes: {notes!r}"
    )


def test_voice_assignments_keyed_on_original_script_name():
    director = LLMDirector()
    plan = _minimal_plan(["MEREDITH", "VANCE"])
    out = director._randomize_character_names(
        plan,
        episode_seed="testseed-2026-04-26",
        gender_map={"MEREDITH": "female", "VANCE": "male"},
    )
    va = out["voice_assignments"]
    # Original keys preserved, no procedural-name keys leaked.
    assert set(va.keys()) == {"MEREDITH", "VANCE"}


def test_lemmy_branch_unchanged():
    """Sanity: LEMMY branch still produces fixed profile with no leak."""
    director = LLMDirector()
    plan = _minimal_plan(["LEMMY"])
    out = director._randomize_character_names(plan, episode_seed="anything")
    notes = out["voice_assignments"]["LEMMY"]["notes"]
    assert notes
    assert not LEAK_PREFIX.match(notes), f"LEMMY notes leak: {notes!r}"

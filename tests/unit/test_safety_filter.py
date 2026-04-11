"""Unit tests for content safety filter (BUG-012 regression).

Tests the classify_prompt() and check_prompt() functions against
known-bad, known-flagged, and known-clean prompts.
"""

import os
import sys
import pytest

# Add project root to path so we can import without torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nodes.safety_filter import classify_prompt, check_prompt


class TestClassifyPromptBlock:
    """All known-bad prompts must return result='block'."""

    @pytest.fixture
    def bad_prompts(self):
        """Load bad prompts from the test file."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "safety", "bad_prompts.txt"
        )
        with open(path) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    def test_all_bad_prompts_blocked(self, bad_prompts):
        """Every line in bad_prompts.txt must be blocked."""
        for prompt in bad_prompts:
            result = classify_prompt(prompt)
            assert result["result"] == "block", (
                f"Expected BLOCK for: {prompt!r}, got {result['result']}"
            )

    def test_bad_prompts_file_has_10_entries(self, bad_prompts):
        """Acceptance criteria: exactly 10 known-bad prompts."""
        assert len(bad_prompts) == 10

    def test_gore_blocked(self):
        assert classify_prompt("a gory scene")["result"] == "block"

    def test_nude_blocked(self):
        assert classify_prompt("nude portrait")["result"] == "block"

    def test_nsfw_blocked(self):
        assert classify_prompt("nsfw content here")["result"] == "block"

    def test_torture_blocked(self):
        assert classify_prompt("medieval torture device")["result"] == "block"

    def test_bomb_making_blocked(self):
        assert classify_prompt("bomb making diagram")["result"] == "block"

    def test_child_abuse_blocked(self):
        assert classify_prompt("child abuse scene")["result"] == "block"

    def test_ethnic_cleansing_blocked(self):
        assert classify_prompt("ethnic cleansing poster")["result"] == "block"


class TestClassifyPromptFlag:
    """Borderline content returns result='flag' (logged but allowed)."""

    def test_blood_flagged(self):
        assert classify_prompt("blood on the floor")["result"] == "flag"

    def test_weapon_flagged(self):
        assert classify_prompt("a weapon on the table")["result"] == "flag"

    def test_death_flagged(self):
        assert classify_prompt("theme of death")["result"] == "flag"


class TestClassifyPromptAllow:
    """Clean prompts must return result='allow'."""

    def test_radio_studio(self):
        result = classify_prompt("A cozy 1950s radio studio with warm lighting")
        assert result["result"] == "allow"

    def test_sci_fi_scene(self):
        result = classify_prompt("A space station orbiting Jupiter at sunset")
        assert result["result"] == "allow"

    def test_portrait(self):
        result = classify_prompt("Professional portrait of a scientist in a lab coat")
        assert result["result"] == "allow"

    def test_empty_string(self):
        result = classify_prompt("")
        assert result["result"] == "allow"

    def test_none_safe(self):
        """None input should not crash."""
        result = classify_prompt(None)
        assert result["result"] == "allow"

    def test_whitespace_only(self):
        result = classify_prompt("   ")
        assert result["result"] == "allow"


class TestClassifyPromptReturnFormat:
    """Return dict must always have result, reason, matched keys."""

    def test_allow_has_all_keys(self):
        r = classify_prompt("clean prompt")
        assert set(r.keys()) == {"result", "reason", "matched"}

    def test_block_has_all_keys(self):
        r = classify_prompt("graphic gore scene")
        assert set(r.keys()) == {"result", "reason", "matched"}

    def test_flag_has_all_keys(self):
        r = classify_prompt("a knife in the drawer")
        assert set(r.keys()) == {"result", "reason", "matched"}

    def test_block_has_nonempty_reason(self):
        r = classify_prompt("nsfw artwork")
        assert r["reason"] != ""
        assert r["matched"] != ""


class TestCheckPromptConvenience:
    """check_prompt() returns True for allow/flag, False for block."""

    def test_clean_returns_true(self):
        assert check_prompt("A peaceful meadow") is True

    def test_flagged_returns_true(self):
        assert check_prompt("a gun on the desk") is True

    def test_blocked_returns_false(self):
        assert check_prompt("graphic gore and torture") is False

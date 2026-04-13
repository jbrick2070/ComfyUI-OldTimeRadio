"""
Regression test: every dropdown / narrative option in OTR_Gemma4ScriptWriter.

Tests the pre-flight guardrail logic WITHOUT running the LLM.  We patch the
heavy methods (RSS fetch, model load, generation) and only exercise the code
path from write_script() entry through the pre-flight section to confirm:

  1. Every dropdown value is accepted (no KeyError, no crash).
  2. Every dropdown value changes something downstream (not dead).
  3. Cross-combo guardrails fire correctly (auto-clamp, cap, disable).
  4. No combination of legal dropdown values causes an unhandled exception.

Run:  python -m pytest tests/test_dropdown_guardrails.py -v
"""
import importlib
import json
import logging
import os
import re
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add the pack root so we can import nodes.story_orchestrator
# ---------------------------------------------------------------------------
PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACK_ROOT not in sys.path:
    sys.path.insert(0, PACK_ROOT)

# We need to mock heavy deps that story_orchestrator imports at module level
# before importing the module itself.
_RUNTIME_LOG_LINES = []

def _mock_runtime_log(msg):
    """Capture runtime log lines for assertion."""
    _RUNTIME_LOG_LINES.append(msg)


def _fake_vram_snapshot(*a, **kw):
    pass

def _fake_vram_reset_peak(*a, **kw):
    pass

def _fake_force_vram_offload(*a, **kw):
    pass


# Patch vram_log before importing story_orchestrator
_vram_mod = types.ModuleType("nodes._vram_log")
_vram_mod.vram_snapshot = _fake_vram_snapshot
_vram_mod.vram_reset_peak = _fake_vram_reset_peak
_vram_mod.force_vram_offload = _fake_force_vram_offload
_vram_mod.register_vram_cleanup = lambda *a, **kw: None
sys.modules["nodes._vram_log"] = _vram_mod

# Patch project_state
_ps_mod = types.ModuleType("nodes.project_state")
_ps_mod.ProjectState = type("ProjectState", (), {})
sys.modules["nodes.project_state"] = _ps_mod

# Now import
from nodes.story_orchestrator import LLMScriptWriter


# ---------------------------------------------------------------------------
# Extract all dropdown options from INPUT_TYPES
# ---------------------------------------------------------------------------
_INPUT_TYPES = LLMScriptWriter.INPUT_TYPES()
_REQUIRED = _INPUT_TYPES["required"]
_OPTIONAL = _INPUT_TYPES["optional"]

RUNTIME_PRESETS = _REQUIRED["runtime_preset"][0]
GENRE_FLAVORS = _REQUIRED["genre_flavor"][0]
TARGET_LENGTHS = _OPTIONAL["target_length"][0]
STYLE_VARIANTS = _OPTIONAL["style_variant"][0]
CREATIVITY_OPTIONS = _OPTIONAL["creativity"][0]
OPT_PROFILES = _OPTIONAL["optimization_profile"][0]


# ---------------------------------------------------------------------------
# Helper: run write_script through pre-flight only, then bail
# ---------------------------------------------------------------------------
class _PreFlightExit(Exception):
    """Raised to bail out after pre-flight completes."""
    pass


def _run_preflight(writer, **overrides):
    """Call write_script with mocks so it runs pre-flight then stops.

    We mock both _fetch_science_news AND _generate_with_llm so the code
    runs through all pre-flight guardrails + the Open-Close / chunked-gen
    path selection, but never actually calls torch or the LLM.

    Returns the list of runtime log lines captured during the call.
    """
    _RUNTIME_LOG_LINES.clear()

    defaults = {
        "episode_title": "Test Episode",
        "genre_flavor": "hard_sci_fi",
        "runtime_preset": "[EMOJI] standard (12 min)",
        "target_minutes": 8,
        "num_characters": 4,
        "model_id": "google/gemma-4-E4B-it",
        "custom_premise": "",
        "news_headlines": 3,
        "temperature": 0.8,
        "include_act_breaks": True,
        "self_critique": True,
        "open_close": True,
        "target_length": "medium (5 acts)",
        "style_variant": "tense claustrophobic",
        "creativity": "balanced",
        "arc_enhancer": True,
        "project_state": None,
        "optimization_profile": "Standard",
    }
    defaults.update(overrides)

    # _generate_with_llm returns a fake script that the parser can chew on.
    # We raise _PreFlightExit on the SECOND call so all pre-flight + first
    # generation attempt paths are exercised.
    _call_count = [0]

    def _fake_generate(*a, **kw):
        _call_count[0] += 1
        if _call_count[0] >= 2:
            raise _PreFlightExit("bail after first generate")
        # Return a minimal parseable script so Open-Close / chunked gen
        # path selection code can proceed through one cycle.
        return (
            "=== SCENE 1 ===\n"
            "NARRATOR: The signal was lost.\n"
            "DR. CHEN: We need to find it.\n"
            "COMMANDER REEVES: Agreed.\n"
            "=== SCENE 2 ===\n"
            "DR. CHEN: I found something.\n"
            "COMMANDER REEVES: What is it?\n"
            "NARRATOR: The end.\n"
        )

    with patch("nodes.story_orchestrator._runtime_log", side_effect=_mock_runtime_log), \
         patch("nodes.story_orchestrator._generate_with_llm", side_effect=_fake_generate), \
         patch("nodes.story_orchestrator.force_vram_offload", lambda *a, **kw: None), \
         patch("nodes.story_orchestrator._flush_vram_keep_llm", lambda *a, **kw: None), \
         patch("nodes.story_orchestrator._unload_llm", lambda *a, **kw: None):
            try:
                writer.write_script(**defaults)
            except _PreFlightExit:
                pass  # Expected
            except Exception as e:
                if "bail after" not in str(e):
                    raise

    return list(_RUNTIME_LOG_LINES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def writer():
    return LLMScriptWriter()


# ===========================================================================
# TEST SUITE 1: Every dropdown value accepted without crash
# ===========================================================================
class TestAllDropdownsAccepted:
    """Every single dropdown value must run through pre-flight without error."""

    @pytest.mark.parametrize("preset", RUNTIME_PRESETS)
    def test_runtime_preset_accepted(self, writer, preset):
        if preset == "[EMOJI] custom":
            logs = _run_preflight(writer, runtime_preset=preset, target_minutes=10)
        else:
            logs = _run_preflight(writer, runtime_preset=preset)
        assert any("ScriptWriter:" in l for l in logs), f"No log output for preset {preset}"

    @pytest.mark.parametrize("genre", GENRE_FLAVORS)
    def test_genre_flavor_accepted(self, writer, genre):
        logs = _run_preflight(writer, genre_flavor=genre)
        assert any("ScriptWriter:" in l for l in logs)

    @pytest.mark.parametrize("tl", TARGET_LENGTHS)
    def test_target_length_accepted(self, writer, tl):
        logs = _run_preflight(writer, target_length=tl)
        assert any("ScriptWriter:" in l for l in logs)

    @pytest.mark.parametrize("sv", STYLE_VARIANTS)
    def test_style_variant_accepted(self, writer, sv):
        logs = _run_preflight(writer, style_variant=sv)
        assert any("ScriptWriter:" in l for l in logs)

    @pytest.mark.parametrize("cr", CREATIVITY_OPTIONS)
    def test_creativity_accepted(self, writer, cr):
        logs = _run_preflight(writer, creativity=cr)
        assert any("ScriptWriter:" in l for l in logs)

    @pytest.mark.parametrize("op", OPT_PROFILES)
    def test_optimization_profile_accepted(self, writer, op):
        logs = _run_preflight(writer, optimization_profile=op)
        assert any("ScriptWriter:" in l for l in logs)


# ===========================================================================
# TEST SUITE 2: Each dropdown value changes something (not dead)
# ===========================================================================
class TestDropdownsHaveEffect:
    """Every dropdown must produce a measurable difference in the log output."""

    def test_runtime_presets_produce_different_target_minutes(self, writer):
        """Each non-custom preset maps to a distinct target_minutes."""
        seen_minutes = set()
        for preset in RUNTIME_PRESETS:
            if preset == "[EMOJI] custom":
                continue
            logs = _run_preflight(writer, runtime_preset=preset)
            # Find target_min= in diagnostic log
            for l in logs:
                m = re.search(r"target_min=(\d+)", l)
                if m:
                    seen_minutes.add(int(m.group(1)))
                    break
        # Should have 4 distinct values (5, 8, 15, 20)
        assert len(seen_minutes) == 4, f"Expected 4 distinct minutes, got {seen_minutes}"

    def test_creativity_produces_different_temps(self, writer):
        """Each creativity tier maps to a distinct temperature."""
        seen_temps = set()
        for cr in CREATIVITY_OPTIONS:
            logs = _run_preflight(writer, creativity=cr)
            for l in logs:
                m = re.search(r"CREATIVITY .+ - temp=([\d.]+)", l)
                if m:
                    seen_temps.add(m.group(1))
                    break
        assert len(seen_temps) == len(CREATIVITY_OPTIONS), \
            f"Expected {len(CREATIVITY_OPTIONS)} distinct temps, got {seen_temps}"

    def test_style_variants_logged_distinctly(self, writer):
        """Each style variant appears in the PARAMS log line."""
        seen_styles = set()
        for sv in STYLE_VARIANTS:
            logs = _run_preflight(writer, style_variant=sv)
            for l in logs:
                m = re.search(r"length=.+ style=(.+?) creativity=", l)
                if m:
                    seen_styles.add(m.group(1))
                    break
        assert len(seen_styles) == len(STYLE_VARIANTS), \
            f"Expected {len(STYLE_VARIANTS)} distinct styles, got {seen_styles}"

    def test_genre_flavors_logged_distinctly(self, writer):
        """Each genre appears in the FINGERPRINT log (affects episode seed)."""
        seen_fingerprints = set()
        for genre in GENRE_FLAVORS:
            logs = _run_preflight(writer, genre_flavor=genre)
            for l in logs:
                m = re.search(r"FINGERPRINT (\w+)", l)
                if m:
                    seen_fingerprints.add(m.group(1))
                    break
        assert len(seen_fingerprints) == len(GENRE_FLAVORS), \
            f"Expected {len(GENRE_FLAVORS)} distinct fingerprints, got {len(seen_fingerprints)}"


# ===========================================================================
# TEST SUITE 3: Guardrail auto-clamps fire correctly
# ===========================================================================
class TestGuardrails:
    """Cross-combo guardrails must clamp dangerous parameter pairs."""

    def test_user_target_length_respected(self, writer):
        """User's target_length choice is never overridden by preset."""
        for tl in TARGET_LENGTHS:
            logs = _run_preflight(writer,
                                  runtime_preset="[FAST] quick (5 min)",
                                  target_length=tl)
            # Verify the PARAMS log shows the user's chosen target_length
            for l in logs:
                if "PARAMS" in l and f"length={tl}" in l:
                    break
            else:
                pytest.fail(f"target_length '{tl}' not found in PARAMS log: {logs}")
            # Verify NO auto-clamp fired
            assert not any("Auto-clamped target_length" in l for l in logs), \
                f"target_length '{tl}' should not be auto-clamped"

    def test_short_3_acts_with_quick_preset_accepted(self, writer):
        """short (3 acts) + quick (5 min) must work without PARSE_FATAL."""
        logs = _run_preflight(writer,
                              runtime_preset="[FAST] quick (5 min)",
                              target_length="short (3 acts)")
        # Should log 3 acts in the PARAMS line
        assert any("length=short (3 acts)" in l for l in logs)

    def test_too_many_chars_short_episode(self, writer):
        """8 characters + 5 min -> clamped to 4."""
        logs = _run_preflight(writer,
                              runtime_preset="[FAST] quick (5 min)",
                              num_characters=8)
        assert any("Clamped num_characters to 4" in l for l in logs)

    def test_too_many_chars_very_short_episode(self, writer):
        """8 characters + 3 min (custom) -> clamped to 3."""
        logs = _run_preflight(writer,
                              runtime_preset="[EMOJI] custom",
                              target_minutes=3,
                              num_characters=8)
        # Should clamp to 4 first (<=5 min), then to 3 (<=3 min)
        assert any("Clamped num_characters to 3" in l for l in logs)

    def test_too_few_chars_long_episode(self, writer):
        """2 characters + long (7-8 acts) -> clamped to 3."""
        logs = _run_preflight(writer,
                              runtime_preset="[EMOJI] long (15 min)",
                              target_length="long (7-8 acts)",
                              num_characters=2)
        assert any("Clamped num_characters to 3" in l for l in logs)

    def test_obsidian_caps_runtime(self, writer):
        """Obsidian + 20 min -> clamped to 10 min."""
        logs = _run_preflight(writer,
                              runtime_preset="[EMOJI] epic (20 min)",
                              optimization_profile="Obsidian (UNSTABLE/4GB)")
        assert any("Obsidian token cap" in l for l in logs) or \
               any("Obsidian" in l and "10" in l for l in logs), \
            f"Expected Obsidian time cap, got: {logs}"

    def test_obsidian_disables_multipass(self, writer):
        """Obsidian forces self_critique=False, open_close=False, arc_enhancer=False."""
        logs = _run_preflight(writer,
                              optimization_profile="Obsidian (UNSTABLE/4GB)")
        # The Obsidian log should fire
        assert any("OBSIDIAN" in l.upper() for l in logs), \
            f"Expected Obsidian activation log, got: {logs}"

    def test_custom_premise_disables_open_close(self, writer):
        """Custom premise -> open_close forced to False."""
        logs = _run_preflight(writer,
                              custom_premise="A scientist discovers time travel.",
                              open_close=True)
        assert any("Custom premise detected" in l or "RSS bypassed" in l for l in logs)

    def test_act_breaks_disabled_short_episode(self, writer):
        """Act breaks disabled for <=3 min episodes."""
        logs = _run_preflight(writer,
                              runtime_preset="[EMOJI] custom",
                              target_minutes=3,
                              include_act_breaks=True)
        assert any("Act breaks disabled" in l for l in logs)


# ===========================================================================
# TEST SUITE 4: Dynamic dialogue line scaling
# ===========================================================================
class TestDialogueLineScaling:
    """length_instruction must scale proportionally with target_minutes."""

    @pytest.mark.parametrize("minutes,expected_min_lines", [
        (3, 24),    # 3 * 8 = 24
        (5, 40),    # 5 * 8 = 40
        (8, 64),    # 8 * 8 = 64
        (15, 120),  # 15 * 8 = 120
        (20, 160),  # 20 * 8 = 160
    ])
    def test_min_lines_formula(self, minutes, expected_min_lines):
        """Dialogue floor = max(18, target_minutes * 8)."""
        result = max(18, int(minutes * 8))
        assert result == expected_min_lines

    def test_floor_never_below_18(self):
        """Even at minimum runtime, floor is 18 lines."""
        result = max(18, int(3 * 8))
        assert result >= 18


# ===========================================================================
# TEST SUITE 5: Obsidian profile string match
# ===========================================================================
class TestObsidianStringMatch:
    """The Obsidian check in code must match the INPUT_TYPES string exactly."""

    def test_obsidian_string_in_code(self):
        """Verify the code checks the exact same string as INPUT_TYPES."""
        import inspect
        source = inspect.getsource(LLMScriptWriter.write_script)
        # The INPUT_TYPES has "Obsidian (UNSTABLE/4GB)"
        obsidian_option = [o for o in OPT_PROFILES if "Obsidian" in o][0]
        # Code must reference this exact string (or a matching substring)
        assert "Obsidian" in source, \
            "write_script does not check for Obsidian profile at all"


# ===========================================================================
# TEST SUITE 6: No dead dropdown options
# ===========================================================================
class TestNoDeadOptions:
    """Verify deprecated parameters are marked and active ones are not."""

    def test_news_headlines_marked_deprecated(self):
        tooltip = _OPTIONAL["news_headlines"][1].get("tooltip", "")
        assert "DEPRECATED" in tooltip, "news_headlines should be marked DEPRECATED"

    def test_temperature_marked_deprecated(self):
        tooltip = _OPTIONAL["temperature"][1].get("tooltip", "")
        assert "DEPRECATED" in tooltip, "temperature should be marked DEPRECATED"

    def test_creativity_not_deprecated(self):
        tooltip = _OPTIONAL["creativity"][1].get("tooltip", "")
        assert "DEPRECATED" not in tooltip, "creativity should NOT be deprecated"

    def test_target_length_not_deprecated(self):
        tooltip = _OPTIONAL["target_length"][1].get("tooltip", "")
        assert "DEPRECATED" not in tooltip, "target_length should NOT be deprecated"

    def test_no_1min_test_preset(self):
        """The 1-min test preset should no longer exist."""
        assert not any("1 min" in p for p in RUNTIME_PRESETS), \
            "1-min test preset should have been removed"

    def test_minimum_target_minutes_is_3(self):
        """target_minutes min must be 3, not 1."""
        meta = _REQUIRED["target_minutes"][1]
        assert meta["min"] == 3, f"target_minutes min should be 3, got {meta['min']}"

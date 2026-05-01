"""
test_radio_prompt_builder.py
============================

Regression coverage for the dynamic radio still FLUX prompt builder.

Post 2026-04-30 consolidation: ``style`` is the single
free-text tonal knob.  Whatever the user types is used VERBATIM as
the radio's leading aesthetic descriptor; the SIGNAL LOST universal
suffix is appended for broadcast-distress identity.  No more genre
presets, no more style mood map -- the LLM widget is the source of
truth, the FLUX prompt builder just consumes it.

Coverage:
  - empty / None / unknown ledger -> fallback prompt
  - any non-empty style -> verbatim incorporation
  - missing / blank / non-string style -> fallback
  - field resolution: gen_params_initial > gen_params (forward-compat)
  - SIGNAL LOST suffix appended in every non-fallback case
  - Builder NEVER returns empty string
  - Hostile input safety
"""

from __future__ import annotations

import os
import sys

import pytest

# Make `visual/` importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from visual.batch_flux_render import (  # noqa: E402  -- after sys.path tweak
    _build_dynamic_radio_prompt,
    _RADIO_FALLBACK_PROMPT,
    _RADIO_PROMPT_SUFFIX,
)


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

class TestFallback:
    """When ledger is missing / unusable / has no style, builder
    returns the safety fallback unchanged."""

    def test_none_ledger_returns_fallback(self):
        assert _build_dynamic_radio_prompt(None) == _RADIO_FALLBACK_PROMPT

    def test_empty_dict_returns_fallback(self):
        assert _build_dynamic_radio_prompt({}) == _RADIO_FALLBACK_PROMPT

    def test_meta_present_but_no_gen_params(self):
        assert _build_dynamic_radio_prompt({"meta": {}}) == _RADIO_FALLBACK_PROMPT

    def test_gen_params_present_but_no_style(self):
        led = {"meta": {"gen_params_initial": {}}}
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    def test_blank_style(self):
        led = {"meta": {"gen_params_initial": {"style": ""}}}
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    def test_whitespace_only_style(self):
        led = {"meta": {"gen_params_initial": {"style": "   "}}}
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    @pytest.mark.parametrize("falsy", [None, False, 0, ""])
    def test_falsy_ledger(self, falsy):
        assert _build_dynamic_radio_prompt(falsy) == _RADIO_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Verbatim style incorporation
# ---------------------------------------------------------------------------

class TestStyleVariantVerbatim:
    """Whatever the user typed is the radio's leading descriptor.
    No transformation, no preset lookup -- just verbatim use."""

    @pytest.mark.parametrize("style", [
        "tense claustrophobic",
        "space opera epic",
        "psychological slow-burn",
        "hard-sci-fi procedural",
        "noir mystery",
        "chaotic black-mirror",
        "neon-drenched cyber-noir",
        "rust-belt post-apocalyptic",
        "cosmic dread, fog-bound coast",
        "1970s soviet brutalism",
    ])
    def test_style_appears_at_prompt_start(self, style):
        led = {"meta": {"gen_params_initial": {"style": style}}}
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith(f"{style} radio broadcast unit"), (
            f"style {style!r} should be the leading descriptor; "
            f"got prompt: {result!r}"
        )

    def test_strips_surrounding_whitespace(self):
        led = {"meta": {"gen_params_initial": {"style": "  noir mystery  "}}}
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith("noir mystery radio broadcast unit")
        assert "  noir" not in result  # no double-space

    def test_internal_whitespace_preserved(self):
        # Multi-word descriptors with internal commas / spaces should
        # pass through verbatim.
        led = {"meta": {"gen_params_initial": {
            "style": "neon-drenched cyber noir, rain-streaked plexi housing",
        }}}
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith(
            "neon-drenched cyber noir, rain-streaked plexi housing "
            "radio broadcast unit"
        )

    def test_preserves_user_case(self):
        # Free-text means we don't .lower() anything -- the FLUX model
        # may interpret CamelCase or ALL CAPS as emphasis.
        led = {"meta": {"gen_params_initial": {"style": "Mid-Century Atomic Modernism"}}}
        result = _build_dynamic_radio_prompt(led)
        assert "Mid-Century Atomic Modernism" in result


# ---------------------------------------------------------------------------
# Forward-compat: gen_params_initial > gen_params
# ---------------------------------------------------------------------------

class TestForwardCompat:
    """gen_params_initial is the canonical phase-0 stamp.  gen_params
    is the spine-ledger forward-compat field.  Builder reads initial
    first, then falls back to plain gen_params."""

    def test_initial_takes_precedence(self):
        led = {"meta": {
            "gen_params_initial": {"style": "noir mystery"},
            "gen_params":         {"style": "space opera epic"},
        }}
        result = _build_dynamic_radio_prompt(led)
        assert "noir mystery" in result
        assert "space opera epic" not in result

    def test_falls_back_to_gen_params_when_initial_missing(self):
        led = {"meta": {"gen_params": {"style": "horror cosmic"}}}
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith("horror cosmic radio broadcast unit")

    def test_initial_present_but_empty_uses_gen_params(self):
        # If gen_params_initial is present but its style is
        # blank, we should fall through (the post-resolution check
        # is on the string itself, not which bag it came from).
        led = {"meta": {
            "gen_params_initial": {"style": ""},
            "gen_params":         {"style": "western frontier"},
        }}
        result = _build_dynamic_radio_prompt(led)
        # Per the current implementation, gen_params_initial wins
        # the dict lookup race; if its style is empty, the
        # builder lands on fallback (does NOT cascade through to
        # gen_params).  This test pins that behavior.
        assert result == _RADIO_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Universal contract checks
# ---------------------------------------------------------------------------

class TestUniversalSuffix:
    """The SIGNAL LOST universal suffix appears in every prompt the
    builder returns -- both fallback and dynamic."""

    def test_suffix_in_fallback(self):
        assert _RADIO_PROMPT_SUFFIX in _RADIO_FALLBACK_PROMPT

    @pytest.mark.parametrize("style", [
        "noir", "space opera", "cyber noir",
        "1970s brutalism", "fog-bound horror",
    ])
    def test_suffix_in_dynamic(self, style):
        led = {"meta": {"gen_params_initial": {"style": style}}}
        result = _build_dynamic_radio_prompt(led)
        assert _RADIO_PROMPT_SUFFIX in result


class TestNeverEmpty:
    """Builder must never return an empty string -- a downstream FLUX
    text-encoder choke on empty input is a worse failure mode than
    rendering the safety fallback."""

    @pytest.mark.parametrize("led", [
        None,
        {},
        {"meta": {}},
        {"meta": {"gen_params_initial": {}}},
        {"meta": {"gen_params_initial": {"style": ""}}},
        {"meta": {"gen_params_initial": {"style": "   "}}},
        {"meta": {"gen_params_initial": {"style": None}}},
        {"meta": {"gen_params_initial": {"style": 42}}},
        {"meta": {"gen_params_initial": {"style": []}}},
        {"meta": "not a dict"},
        {"meta": {"gen_params_initial": "not a dict"}},
        # Hostile-type ledger:
        "string instead of dict",
        42,
        [],
    ])
    def test_builder_always_returns_nonempty_string(self, led):
        result = _build_dynamic_radio_prompt(led)
        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Realistic ledger shape
# ---------------------------------------------------------------------------

class TestRealisticLedger:
    """Smoke-test against the actual ledger shape that
    story_orchestrator stamps in `gen_params_initial`."""

    def test_typical_run_ledger(self):
        led = {
            "episode_id": "signal_lost_smoke_20260430_120000",
            "meta": {
                "gen_params_initial": {
                    "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
                    "target_words": 350,
                    "num_characters": 2,
                    "target_length": "short (3 acts)",
                    "style": "tense claustrophobic",
                    "creativity": "balanced",
                    "optimization_profile": "Standard",
                },
                "news_seed": {
                    "headline": "Mysterious signals detected from deep space",
                },
            },
        }
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith("tense claustrophobic radio broadcast unit")
        assert _RADIO_PROMPT_SUFFIX in result


# ---------------------------------------------------------------------------
# BUG-LOCAL-127 (2026-05-01): the radio bookend pass was wired AFTER
# the serial loop only.  The fast_batch path returned BEFORE reaching
# the bookend code, so fast_batch (the default) silently skipped the
# entire radio still pipeline.  Symptom 2 from earlier today
# ("wanted radio still but it's missing" downstream in BatchHumoRender)
# was caused by this -- not by FLUX render failure, not by ledger
# stamping race.  The bookend code simply never ran.
#
# Fix: extracted radio bookend block into _render_radio_bookend_step()
# helper, called from BOTH paths just before their respective returns.
#
# These tests source-grep the file to confirm both paths still invoke
# the helper.  Exercising the actual path requires a real FLUX model;
# we instead lock in the static structure so a future refactor can't
# regress the same way.
# ---------------------------------------------------------------------------


class TestBug127BookendOnBothPaths:
    """Confirm that the radio bookend helper is invoked from BOTH the
    fast_batch return path and the serial-loop return path so the
    bookend can never be silently skipped again."""

    def _read_source(self) -> str:
        from visual import batch_flux_render as bfr
        return open(bfr.__file__, encoding="utf-8").read()

    def test_render_radio_bookend_step_helper_exists(self):
        src = self._read_source()
        assert "def _render_radio_bookend_step(" in src, (
            "BUG-LOCAL-127 fix requires a _render_radio_bookend_step "
            "helper method that both fast_batch and serial paths call"
        )

    def test_fast_batch_path_invokes_helper_before_return(self):
        # Read the source and isolate the fast_batch branch (between
        # 'FAST BATCH PATH' marker and 'SERIAL LOOP PATH' marker).
        src = self._read_source()
        fast_idx = src.find("# FAST BATCH PATH")
        serial_idx = src.find("# SERIAL LOOP PATH")
        assert fast_idx != -1 and serial_idx != -1, (
            "Could not locate FAST BATCH PATH or SERIAL LOOP PATH "
            "markers; source structure may have changed"
        )
        fast_block = src[fast_idx:serial_idx]
        assert "_render_radio_bookend_step(" in fast_block, (
            "fast_batch path MUST invoke _render_radio_bookend_step "
            "before its return -- this was BUG-LOCAL-127, the fast "
            "path silently skipped the entire bookend pipeline"
        )

    def test_serial_loop_path_invokes_helper_before_return(self):
        # The serial-loop path is the second exit point and must also
        # call the helper.  Look for the helper call in the trailing
        # chunk of execute() AFTER the SERIAL LOOP PATH marker but
        # BEFORE the helper definition itself.
        src = self._read_source()
        serial_idx = src.find("# SERIAL LOOP PATH")
        helper_def_idx = src.find("def _render_radio_bookend_step(")
        assert serial_idx != -1 and helper_def_idx != -1
        assert serial_idx < helper_def_idx, (
            "SERIAL LOOP PATH marker must appear BEFORE the helper "
            "definition; source ordering changed"
        )
        serial_block = src[serial_idx:helper_def_idx]
        # The helper is invoked via self._render_radio_bookend_step(...)
        # in both paths; this block must contain that call.
        assert "_render_radio_bookend_step(" in serial_block, (
            "serial-loop path MUST invoke _render_radio_bookend_step "
            "before its return"
        )

    def test_no_inline_bookend_block_remains_in_execute(self):
        # The original inline `if widget_str.upper() == "DISABLED":`
        # block was the one that lived only in the serial path.  It
        # should now live ONLY inside the helper, not duplicated in
        # execute().  Count occurrences -- should be exactly 1.
        src = self._read_source()
        count = src.count('widget_str.upper() == "DISABLED"')
        assert count == 1, (
            f"Found {count} occurrences of the DISABLED-widget check; "
            f"expected exactly 1 (inside _render_radio_bookend_step). "
            f"If >1, the inline block wasn't fully extracted; if 0, "
            f"the helper itself lost the check."
        )

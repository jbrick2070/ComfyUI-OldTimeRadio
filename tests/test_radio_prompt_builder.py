"""
test_radio_prompt_builder.py
============================

Regression coverage for the dynamic radio still FLUX prompt builder.

BUG-LOCAL-249 (2026-05-20): the radio prompt is DOWNSTREAM of story
creation, so it is driven by ``meta.story_brief`` -- the post-script
reflection. The widget style preset is an UPSTREAM input (it shapes
the story-writing LLM only) and is NOT read by the radio prompt
builder. Every downstream visual derives from the brief, not the
preset.

Coverage:
  - empty / None / hostile ledger -> fallback prompt
  - story_brief (status ok) -> brief is the descriptor, the
    "radio broadcast unit" anchor + SIGNAL LOST suffix present
  - style preset present but no brief -> style is IGNORED; the chain
    falls to the episode_id slug or the hardcoded fallback
  - episode_id slug used as last resort before the hardcoded fallback
  - SIGNAL LOST suffix appended in every non-fallback case
  - builder NEVER returns an empty string
  - hostile-input safety
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
# Ledger fixture builder
# ---------------------------------------------------------------------------

def _ledger(*, brief=None, status=None, episode_id=None, style=None):
    """Construct a ledger dict for the radio prompt builder.

    brief=None   -> no story_brief key at all (legacy ledger)
    brief=""     -> story_brief present but empty
    status=None  -> derived: "ok" if brief truthy else "absent"
    style=...    -> stamped into gen_params_initial.style; the builder
                    must IGNORE it (upstream-only input).
    """
    meta: dict = {}
    if style is not None:
        meta["gen_params_initial"] = {"style": style}
    if brief is not None:
        meta["story_brief"] = brief
        meta["story_brief_status"] = (
            status if status is not None
            else ("ok" if brief else "absent")
        )
        meta["story_brief_terms"] = {
            "setting": [], "lighting": [], "atmosphere": [],
        }
    led: dict = {"meta": meta}
    if episode_id is not None:
        led["episode_id"] = episode_id
    return led


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

class TestFallback:
    """When the ledger has no usable brief and no usable episode_id,
    the builder returns the safety fallback unchanged."""

    def test_none_ledger_returns_fallback(self):
        assert _build_dynamic_radio_prompt(None) == _RADIO_FALLBACK_PROMPT

    def test_empty_dict_returns_fallback(self):
        assert _build_dynamic_radio_prompt({}) == _RADIO_FALLBACK_PROMPT

    def test_meta_present_but_no_brief(self):
        assert (
            _build_dynamic_radio_prompt({"meta": {}})
            == _RADIO_FALLBACK_PROMPT
        )

    def test_failed_brief_no_episode_id_returns_fallback(self):
        led = _ledger(brief="", status="failed")
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    def test_absent_brief_no_episode_id_returns_fallback(self):
        led = _ledger(brief=None)
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    @pytest.mark.parametrize("falsy", [None, False, 0, ""])
    def test_falsy_ledger(self, falsy):
        assert _build_dynamic_radio_prompt(falsy) == _RADIO_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Brief-driven path -- the primary contract
# ---------------------------------------------------------------------------

class TestBriefDriven:
    """meta.story_brief (status ok) is the radio prompt's descriptor."""

    @pytest.mark.parametrize("brief", [
        "A silent world reveals ancient life beneath a vast, lonely sky.",
        "A salvage crew hears their own distress call from the future.",
        "A lighthouse keeper logs a tide that never goes back out.",
        "Two researchers trade places with their reflections.",
    ])
    def test_brief_appears_in_prompt(self, brief):
        led = _ledger(brief=brief)
        result = _build_dynamic_radio_prompt(led)
        # Trailing sentence punctuation is stripped; the body of the
        # brief must survive verbatim.
        assert brief.rstrip(" .,;:") in result

    def test_radio_anchor_present(self):
        led = _ledger(brief="a quiet town under a copper sky")
        result = _build_dynamic_radio_prompt(led)
        assert result.startswith("radio broadcast unit, ")

    def test_trailing_period_stripped(self):
        led = _ledger(brief="a quiet town under a copper sky.")
        result = _build_dynamic_radio_prompt(led)
        assert "copper sky." not in result
        assert "copper sky," in result

    def test_suffix_present_with_brief(self):
        led = _ledger(brief="a derelict station drifting silent")
        result = _build_dynamic_radio_prompt(led)
        assert _RADIO_PROMPT_SUFFIX in result

    def test_long_brief_capped(self):
        # A pathologically long brief is word-boundary trimmed; the
        # prompt stays bounded and still carries the suffix.
        long_brief = ("word " * 200).strip()
        led = _ledger(brief=long_brief)
        result = _build_dynamic_radio_prompt(led)
        assert _RADIO_PROMPT_SUFFIX in result
        assert len(result) < 400
        assert " wor " not in result  # no mid-word cut

    def test_failed_status_brief_not_used(self):
        # story_brief text present but status=failed -> get_story_brief
        # _full returns "", so the brief text is NOT used; the chain
        # falls through to the episode_id slug.
        led = _ledger(
            brief="this brief text must not appear",
            status="failed",
            episode_id="signal_lost_back_channel_20260519_120000",
        )
        result = _build_dynamic_radio_prompt(led)
        assert "this brief text must not appear" not in result
        assert "back channel" in result


# ---------------------------------------------------------------------------
# Style preset is ignored -- the BUG-LOCAL-249 contract
# ---------------------------------------------------------------------------

class TestStylePresetIgnored:
    """The upstream style preset must never reach the radio prompt."""

    @pytest.mark.parametrize("style", [
        "noir mystery", "space opera epic", "noir_interrogation",
        "deep_space_distress_call", "1970s soviet brutalism",
    ])
    def test_style_preset_never_in_prompt(self, style):
        led = _ledger(
            brief="a tense room with rain on the tin roof",
            style=style,
        )
        result = _build_dynamic_radio_prompt(led)
        assert style not in result, (
            f"upstream style preset {style!r} leaked into the radio "
            f"prompt: {result!r}"
        )
        assert "a tense room with rain on the tin roof" in result

    def test_style_alone_does_not_drive_prompt(self):
        # Style set, but NO brief and NO episode_id -> the style is
        # ignored and the builder lands on the hardcoded fallback.
        led = _ledger(brief=None, style="noir mystery")
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    def test_style_alone_falls_to_episode_slug(self):
        # Style set, no brief, but an episode_id exists -> the slug is
        # used; the style is still ignored.
        led = _ledger(
            brief=None,
            style="noir mystery",
            episode_id="signal_lost_amber_signal_20260519_120000",
        )
        result = _build_dynamic_radio_prompt(led)
        assert "noir mystery" not in result
        assert "amber signal" in result


# ---------------------------------------------------------------------------
# episode_id slug fallback
# ---------------------------------------------------------------------------

class TestEpisodeSlugFallback:
    """When the brief is unusable, the episode_id slug is the last
    resort before the hardcoded fallback."""

    def test_slug_used_when_no_brief(self):
        led = _ledger(
            brief=None,
            episode_id="signal_lost_haunted_signal_20260519_120000",
        )
        result = _build_dynamic_radio_prompt(led)
        assert "haunted signal" in result
        assert result.startswith("radio broadcast unit, ")
        assert _RADIO_PROMPT_SUFFIX in result

    def test_signal_lost_prefix_and_timestamp_stripped(self):
        led = _ledger(
            brief=None,
            episode_id="signal_lost_orbital_decay_20260519_153000",
        )
        result = _build_dynamic_radio_prompt(led)
        assert "signal_lost" not in result
        assert "20260519" not in result
        assert "orbital decay" in result

    def test_pending_episode_id_skipped(self):
        # A pre-rename pending_* id is not a meaningful descriptor.
        led = _ledger(brief=None, episode_id="pending_20260519_120000")
        assert _build_dynamic_radio_prompt(led) == _RADIO_FALLBACK_PROMPT

    def test_brief_wins_over_episode_slug(self):
        led = _ledger(
            brief="a flooded archive lit by failing bulbs",
            episode_id="signal_lost_other_thing_20260519_120000",
        )
        result = _build_dynamic_radio_prompt(led)
        assert "a flooded archive lit by failing bulbs" in result
        assert "other thing" not in result


# ---------------------------------------------------------------------------
# Universal contract checks
# ---------------------------------------------------------------------------

class TestUniversalSuffix:
    """The SIGNAL LOST universal suffix appears in every prompt the
    builder returns -- both fallback and dynamic."""

    def test_suffix_in_fallback(self):
        assert _RADIO_PROMPT_SUFFIX in _RADIO_FALLBACK_PROMPT

    @pytest.mark.parametrize("brief", [
        "a cold mission control room",
        "a fog-bound coastal station",
        "a cramped cargo bay vibrating",
    ])
    def test_suffix_in_dynamic(self, brief):
        led = _ledger(brief=brief)
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
        {"meta": {"story_brief": "", "story_brief_status": "absent"}},
        {"meta": {"story_brief": "", "story_brief_status": "failed"}},
        {"meta": {"story_brief": None, "story_brief_status": "ok"}},
        {"meta": {"story_brief": 42, "story_brief_status": "ok"}},
        {"meta": {"gen_params_initial": {"style": "noir"}}},
        {"meta": "not a dict"},
        {"meta": {"story_brief": "ok brief", "story_brief_status": "ok"}},
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
    """Smoke-test against the first-stable-run ledger shape
    (docs/2026-05-19-first-stable-run-ledger.json)."""

    def test_first_stable_run_ledger(self):
        led = {
            "episode_id": (
                "signal_lost_orbital_panorama_reveal_20260519_120000"
            ),
            "meta": {
                "gen_params_initial": {
                    "model_id": "google/gemma-4-E4B-it",
                    "target_words": 350,
                    "num_characters": 2,
                    "style": "orbital_panorama_reveal",
                },
                "style": "orbital_panorama_reveal",
                "story_brief": (
                    "A silent world reveals ancient life beneath a "
                    "vast, lonely sky."
                ),
                "story_brief_status": "ok",
                "story_brief_terms": {
                    "setting": ["crater"],
                    "lighting": [],
                    "atmosphere": [],
                },
            },
        }
        result = _build_dynamic_radio_prompt(led)
        assert "A silent world reveals ancient life" in result
        assert "orbital_panorama_reveal" not in result, (
            "the style preset must not appear in the radio prompt"
        )
        assert result.startswith("radio broadcast unit, ")
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

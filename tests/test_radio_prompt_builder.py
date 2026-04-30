"""
test_radio_prompt_builder.py
============================

Regression coverage for the dynamic radio still FLUX prompt builder
introduced 2026-04-30 alongside the ROADMAP P0 architecture lock
(every audio second is a HuMo clip; non-dialogue lines use the radio
still as I2V reference; per-episode aesthetic comes from
ledger.meta.gen_params).

Coverage:
  - empty / None / unknown ledger -> fallback prompt
  - known genre -> genre-keyed aesthetic + universal SIGNAL LOST suffix
  - unknown genre -> sci-fi default with universal suffix
  - style_variant -> mood layer appended before suffix
  - Suffix is universal across all branches
  - Builder NEVER returns empty string
"""

from __future__ import annotations

import os
import sys

import pytest

# Make `visual/` importable.  Tests run from repo root in CI; this
# block keeps them runnable from any working directory too.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from visual.batch_flux_render import (  # noqa: E402  -- after sys.path tweak
    _build_dynamic_radio_prompt,
    _GENRE_RADIO_AESTHETIC,
    _RADIO_FALLBACK_PROMPT,
    _RADIO_PROMPT_SUFFIX,
    _STYLE_MOOD_LAYER,
)


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

class TestFallback:
    """When ledger is missing/unusable, builder returns the safety
    fallback unchanged."""

    def test_none_ledger_returns_fallback_verbatim(self):
        assert _build_dynamic_radio_prompt(None) == _RADIO_FALLBACK_PROMPT

    def test_empty_dict_ledger_returns_sci_fi_dynamic(self):
        # Empty dict has no meta/gen_params; falls through to sci-fi
        # default INSIDE the dynamic builder (not the fallback const).
        result = _build_dynamic_radio_prompt({})
        assert _GENRE_RADIO_AESTHETIC["sci-fi"] in result
        assert result.endswith(_RADIO_PROMPT_SUFFIX)

    def test_falsy_ledger_returns_fallback(self):
        # Defensive: any falsy value should land on fallback.
        for falsy in (None, False, 0, ""):
            assert _build_dynamic_radio_prompt(falsy) == _RADIO_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Genre branching
# ---------------------------------------------------------------------------

class TestGenreSelection:
    """Each known genre routes to its keyed aesthetic phrase."""

    @pytest.mark.parametrize("genre", sorted(_GENRE_RADIO_AESTHETIC.keys()))
    def test_each_known_genre_uses_its_aesthetic(self, genre):
        led = {"meta": {"gen_params_initial": {"genre_flavor": genre}}}
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC[genre] in result, (
            f"Genre {genre} did not surface its aesthetic phrase in "
            f"the result"
        )
        assert result.endswith(_RADIO_PROMPT_SUFFIX)

    def test_unknown_genre_falls_through_to_sci_fi(self):
        led = {"meta": {"gen_params_initial": {"genre_flavor": "musical-theatre"}}}
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC["sci-fi"] in result
        assert result.endswith(_RADIO_PROMPT_SUFFIX)

    def test_blank_genre_falls_through_to_sci_fi(self):
        led = {"meta": {"gen_params_initial": {"genre_flavor": ""}}}
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC["sci-fi"] in result

    def test_genre_is_case_insensitive(self):
        for variant in ("NOIR", "Noir", "  noir  ", "nOiR"):
            led = {"meta": {"gen_params_initial": {"genre_flavor": variant}}}
            result = _build_dynamic_radio_prompt(led)
            assert _GENRE_RADIO_AESTHETIC["noir"] in result, (
                f"Genre case variant {variant!r} did not normalize to noir"
            )


# ---------------------------------------------------------------------------
# Style mood layering
# ---------------------------------------------------------------------------

class TestStyleVariantLayer:
    """style_variant adds a mood phrase between the genre aesthetic
    and the universal suffix."""

    @pytest.mark.parametrize("style", sorted(_STYLE_MOOD_LAYER.keys()))
    def test_known_style_adds_its_mood_phrase(self, style):
        led = {"meta": {"gen_params_initial": {
            "genre_flavor": "sci-fi",
            "style_variant": style,
        }}}
        result = _build_dynamic_radio_prompt(led)
        assert _STYLE_MOOD_LAYER[style] in result, (
            f"Style {style} mood phrase missing from result"
        )

    def test_unknown_style_silently_ignored(self):
        led = {"meta": {"gen_params_initial": {
            "genre_flavor": "sci-fi",
            "style_variant": "made-up-style-name",
        }}}
        result = _build_dynamic_radio_prompt(led)
        # Should still render, just without a mood layer.
        assert _GENRE_RADIO_AESTHETIC["sci-fi"] in result
        assert result.endswith(_RADIO_PROMPT_SUFFIX)
        # Verify no mood phrase leaked in.
        for mood_phrase in _STYLE_MOOD_LAYER.values():
            assert mood_phrase not in result

    def test_no_style_variant_yields_no_mood_layer(self):
        led = {"meta": {"gen_params_initial": {"genre_flavor": "noir"}}}
        result = _build_dynamic_radio_prompt(led)
        for mood_phrase in _STYLE_MOOD_LAYER.values():
            assert mood_phrase not in result, (
                "Mood phrase leaked in despite style_variant being "
                "unset"
            )


# ---------------------------------------------------------------------------
# Universal contract checks
# ---------------------------------------------------------------------------

class TestUniversalSuffix:
    """The SIGNAL LOST universal suffix appears in every branch."""

    def test_suffix_in_fallback(self):
        assert _RADIO_PROMPT_SUFFIX in _RADIO_FALLBACK_PROMPT

    @pytest.mark.parametrize("genre", sorted(_GENRE_RADIO_AESTHETIC.keys()))
    def test_suffix_in_every_genre(self, genre):
        led = {"meta": {"gen_params_initial": {"genre_flavor": genre}}}
        result = _build_dynamic_radio_prompt(led)
        assert _RADIO_PROMPT_SUFFIX in result


class TestNeverEmpty:
    """Builder must never return an empty string -- a downstream
    FLUX text encoder choke on empty prompt is a worse failure mode
    than producing the fallback."""

    @pytest.mark.parametrize("led", [
        None,
        {},
        {"meta": {}},
        {"meta": {"gen_params_initial": {}}},
        {"meta": {"gen_params_initial": {"genre_flavor": ""}}},
        {"meta": {"gen_params_initial": {"genre_flavor": "garbled"}}},
        {"meta": {"gen_params_initial": {"genre_flavor": "noir", "style_variant": ""}}},
        # Hostile-input safety:
        {"meta": "not a dict"},
        {"meta": {"gen_params_initial": "not a dict"}},
    ])
    def test_builder_always_nonempty(self, led):
        result = _build_dynamic_radio_prompt(led)
        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Forward-compat: old-shape ledger uses `gen_params` not
# `gen_params_initial`.  Builder must read both, preferring the new
# field name.
# ---------------------------------------------------------------------------

class TestForwardCompat:
    """When `gen_params_initial` is absent, fall back to `gen_params`
    (forward-compat with the spine-ledger ticket's bundled schema)."""

    def test_old_shape_gen_params_used_when_initial_missing(self):
        led = {"meta": {"gen_params": {"genre_flavor": "horror"}}}
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC["horror"] in result

    def test_initial_takes_precedence_over_legacy(self):
        # When both are present, `gen_params_initial` wins -- it's
        # the canonical phase-0 stamp.
        led = {"meta": {
            "gen_params_initial": {"genre_flavor": "noir"},
            "gen_params":         {"genre_flavor": "horror"},
        }}
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC["noir"] in result
        assert _GENRE_RADIO_AESTHETIC["horror"] not in result


# ---------------------------------------------------------------------------
# Realistic ledger shape sanity
# ---------------------------------------------------------------------------

class TestRealisticLedger:
    """Smoke-test against a ledger shape mirroring what
    story_orchestrator stamps in `gen_params_initial`.  Exercises
    the integration boundary, not just the unit signature."""

    def test_typical_sci_fi_thriller_ledger(self):
        led = {
            "episode_id": "signal_lost_smoke_20260430_120000",
            "meta": {
                "gen_params_initial": {
                    "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
                    "target_words": 350,
                    "num_characters": 2,
                    "target_length": "short (3 acts)",
                    "style_variant": "atmospheric",
                    "creativity": "balanced",
                    "genre_flavor": "thriller",
                    "optimization_profile": "Standard",
                },
                "news_seed": {
                    "headline": "Mysterious signals detected from deep space",
                },
            },
        }
        result = _build_dynamic_radio_prompt(led)
        assert _GENRE_RADIO_AESTHETIC["thriller"] in result
        assert _STYLE_MOOD_LAYER["atmospheric"] in result
        assert result.endswith(_RADIO_PROMPT_SUFFIX)

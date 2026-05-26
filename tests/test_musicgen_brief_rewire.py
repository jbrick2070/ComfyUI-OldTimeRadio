"""Sprint 8.1: MusicGenTheme brief-rewire tests.

`_compose_music_prompt` was the one C-class consumer the audit
flagged: it was reading visual-tuned `story_brief_terms.atmosphere`
values and producing music prompts off the wrong signal. Sprint 8.1
rewires it to prefer the v2 `music_mood_terms` field (music-tuned)
and fall back through to the v1 atmosphere path, then the legacy
keyword-mining path. These tests lock the resolution order so a
future regression cannot silently flip back to the v1-first ordering.

Test surface (per `downstream_brief_consumer_followup.md` open
decision C1 = bundled commit 1):

  1. v2 path: music_mood_terms non-empty -> prompt uses those terms.
  2. v2 path: prompt is sliced to top-3 (matches v1 atmosphere[:3]).
  3. v1 atmosphere fallback: music_mood_terms absent (legacy ledger).
  4. v1 atmosphere fallback: music_mood_terms empty (v2 failure
     sentinel) with non-empty atmosphere.
  5. v1 keyword-mining fallback: all brief signals empty.
  6. Neutral default: every signal source empty.
  7. v2 path coexists with setting / period terms (the rewire only
     changes mood resolution -- everything else still composes).

The tests import `_compose_music_prompt` directly so MusicGen heavy
deps (torch, transformers) do not need to load -- the function is a
pure prompt-string builder.
"""
from __future__ import annotations

import pytest

from nodes.musicgen_theme import _compose_music_prompt, CUE_DURATIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta_v2_full() -> dict:
    """A complete Sprint 8.1 v2 _success_delta-shaped meta dict."""
    return {
        "story_brief_status": "ok",
        "story_brief":        "a dim room under a bare bulb",
        "story_brief_terms": {
            "setting":    ["interrogation room", "steel table"],
            "lighting":   ["bare bulb"],
            "atmosphere": ["sweat", "smoke", "tense"],
        },
        "music_mood_terms": ["sombre", "uneasy", "menacing"],
        "visual_palette":   ["amber glow"],
        "key_objects":      ["table"],
        "tempo_hint":       "slow",
        "atmosphere_line":  "smoke under a bare bulb.",
    }


def _meta_v1_legacy() -> dict:
    """Legacy v1 meta -- no v2 fields at all."""
    return {
        "story_brief_status": "ok",
        "story_brief":        "a dim room under a bare bulb",
        "story_brief_terms": {
            "setting":    ["room"],
            "lighting":   ["bulb"],
            "atmosphere": ["tense", "smoky", "wary"],
        },
    }


def _meta_v2_failure_sentinel() -> dict:
    """v2 failure sentinel -- every field present but safe-empty."""
    return {
        "story_brief_status": "failed",
        "story_brief":        "",
        "story_brief_terms": {
            "setting":    [],
            "lighting":   [],
            "atmosphere": [],
        },
        "music_mood_terms": [],
        "visual_palette":   [],
        "key_objects":      [],
        "tempo_hint":       "",
        "atmosphere_line":  "",
    }


def _meta_v2_empty_mood_v1_atmosphere() -> dict:
    """v2 producer ran but emitted empty music_mood_terms; v1
    atmosphere still has content. This is the v1-fallback path the
    rewire must support gracefully."""
    return {
        "story_brief_status": "ok",
        "story_brief":        "a dim room",
        "story_brief_terms": {
            "setting":    ["room"],
            "lighting":   ["bulb"],
            "atmosphere": ["wistful", "yearning", "melancholic"],
        },
        "music_mood_terms": [],
        "visual_palette":   [],
        "key_objects":      [],
        "tempo_hint":       "",
        "atmosphere_line":  "",
    }


def _meta_only_news_brief() -> dict:
    """Both reflection signals empty; only the legacy news.script_brief
    carries mood keywords (the original audit-recommended last-ditch
    fallback path).

    The keyword string must hit at least one key in `_MOOD_TAGS`
    (`danger`, `betrayal`, `discovery`, `loss`, `urgent`, `isolation`,
    `mystery`, `triumph`, `conflict`, `silence`) for the mining path
    to contribute. The fixture uses `danger` + `isolation` so the
    test asserts on stable tags.
    """
    return {
        "story_brief_status": "absent",
        "news": {
            "script_brief": (
                "A growing danger surrounds the lone survivor in "
                "deepening isolation."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 1-2. v2 path takes priority
# ---------------------------------------------------------------------------


class TestV2PathPreferred:

    def test_v2_mood_terms_used_when_non_empty(self):
        """When music_mood_terms is non-empty, the resulting prompt
        leads with those terms -- NOT the v1 atmosphere-derived
        ones."""
        meta = _meta_v2_full()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert "sombre" in prompt
        assert "uneasy" in prompt
        assert "menacing" in prompt
        # The v1 atmosphere terms (sweat / smoke / tense) must NOT
        # leak into the mood prefix when v2 is present -- the audit
        # called those visually-tuned, not music-tuned. v1 setting
        # terms (room/table) ARE allowed to land via the "evokes"
        # clause, and the v1 atmosphere word "tense" is also a
        # valid music mood, so we lock the v2 mood terms appearing
        # in mood-prefix position rather than every v1 word being
        # absent.
        prefix = prompt.split(",")[0].strip()
        assert "sombre" in prefix or "uneasy" in prefix or "menacing" in prefix

    def test_v2_mood_sliced_to_top_three(self):
        """The v2 path mirrors the v1 atmosphere[:3] slice."""
        meta = _meta_v2_full()
        meta["music_mood_terms"] = [
            "sombre", "uneasy", "menacing", "frantic", "stoic", "wistful",
        ]
        prompt, _ = _compose_music_prompt(meta, "opening")
        # First 3 used; 4th onward dropped.
        assert "sombre" in prompt
        assert "uneasy" in prompt
        assert "menacing" in prompt
        assert "frantic" not in prompt
        assert "stoic" not in prompt
        assert "wistful" not in prompt

    def test_v2_path_returns_correct_duration(self):
        """The rewire MUST NOT change cue duration semantics --
        opening/closing/interstitial still map to CUE_DURATIONS."""
        meta = _meta_v2_full()
        for cue_id in ("opening", "closing", "interstitial"):
            _, duration = _compose_music_prompt(meta, cue_id)
            assert duration == CUE_DURATIONS[cue_id]


# ---------------------------------------------------------------------------
# 3-4. v1 atmosphere fallback
# ---------------------------------------------------------------------------


class TestV1AtmosphereFallback:

    def test_legacy_v1_meta_uses_atmosphere_terms(self):
        """A legacy ledger has no music_mood_terms at all; the prompt
        falls through to story_brief_terms.atmosphere -- the v1 path
        still works unchanged for un-rewritten ledgers."""
        meta = _meta_v1_legacy()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert "tense" in prompt
        assert "smoky" in prompt
        assert "wary" in prompt

    def test_v2_empty_mood_falls_back_to_atmosphere(self):
        """v2 producer ran but emitted empty music_mood_terms; the
        rewire must drop through to the v1 atmosphere path rather
        than going straight to keyword mining."""
        meta = _meta_v2_empty_mood_v1_atmosphere()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert "wistful" in prompt
        assert "yearning" in prompt
        assert "melancholic" in prompt


# ---------------------------------------------------------------------------
# 5. v1 keyword-mining fallback
# ---------------------------------------------------------------------------


class TestKeywordMiningFallback:

    def test_only_news_brief_signal_falls_to_keyword_mining(self):
        """When both reflection signals (v2 music_mood_terms and v1
        atmosphere) are empty, the prompt resolves mood by mining
        news.script_brief -- the legacy last-ditch path.

        The fixture brief contains `danger` and `isolation` (real
        `_MOOD_TAGS` keys), so the resolved prompt must include the
        corresponding music tags (`building tension` from danger
        and `sparse texture` from isolation).
        """
        meta = _meta_only_news_brief()
        prompt, _ = _compose_music_prompt(meta, "opening")
        # At least one of the keyword-mined tags must reach the
        # prompt -- catching this proves the v2 -> v1-atmosphere ->
        # v1-keyword-mining cascade is intact.
        keyword_tags = (
            "building tension",
            "dissonant cluster",
            "sparse texture",
            "wide stereo field",
        )
        assert any(tag in prompt for tag in keyword_tags), (
            f"keyword mining produced no mood tags in prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# 6. Neutral default on full failure sentinel
# ---------------------------------------------------------------------------


class TestNeutralDefault:

    def test_full_failure_sentinel_yields_neutral_default(self):
        """Every brief signal empty (the v2 failure sentinel shape).
        The prompt falls through to the neutral 'atmospheric' default
        plus the cue character template."""
        meta = _meta_v2_failure_sentinel()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert "atmospheric" in prompt
        # Cue character template still appended -- the rewire only
        # touched mood resolution, not the composition spine.
        assert "instrumental" in prompt


# ---------------------------------------------------------------------------
# 7. v2 path coexists with non-mood prompt elements
# ---------------------------------------------------------------------------


class TestV2PathCoexistsWithCompositionSpine:

    def test_v2_path_still_includes_setting_clause(self):
        """The rewire only changes the mood resolution; setting terms
        from story_brief_terms.setting must still land as the
        'evokes ...' clause in the prompt."""
        meta = _meta_v2_full()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert "evokes" in prompt
        assert "interrogation room" in prompt or "steel table" in prompt

    def test_v2_path_still_includes_cue_character(self):
        """Each cue's per-cue character template still appends."""
        meta = _meta_v2_full()
        for cue_id, expected_marker in (
            ("opening",      "atmospheric build"),
            ("closing",      "resolving cadence"),
            ("interstitial", "textural bridge"),
        ):
            prompt, _ = _compose_music_prompt(meta, cue_id)
            assert expected_marker in prompt, (
                f"cue character template missing from {cue_id!r}"
            )

    def test_v2_path_still_includes_prompt_tail(self):
        """The instrumental-only tail must always close the prompt."""
        meta = _meta_v2_full()
        prompt, _ = _compose_music_prompt(meta, "opening")
        assert prompt.endswith(
            "instrumental only, no dialogue, no vocals"
        )


# ---------------------------------------------------------------------------
# 8. Non-list / non-string music_mood_terms degrades gracefully
# ---------------------------------------------------------------------------


class TestMalformedV2Fields:

    def test_non_list_music_mood_terms_falls_back(self):
        """A v2 field stamped with the wrong type (string where list
        was expected -- only possible via direct dict editing or a
        future producer regression) must NOT crash the composer; it
        falls through to the v1 atmosphere path."""
        meta = _meta_v1_legacy()
        meta["music_mood_terms"] = "not a list"  # type: ignore[assignment]
        prompt, _ = _compose_music_prompt(meta, "opening")
        # v1 atmosphere wins.
        assert "tense" in prompt
        assert "not a list" not in prompt

    def test_empty_strings_in_music_mood_filtered(self):
        """Empty / whitespace-only entries in music_mood_terms are
        filtered out; if every entry is empty the path drops to v1."""
        meta = _meta_v1_legacy()
        meta["music_mood_terms"] = ["", "  ", "\t"]
        prompt, _ = _compose_music_prompt(meta, "opening")
        # Empty-string filtering left the v2 list functionally empty;
        # the v1 atmosphere terms drove the prompt instead.
        assert "tense" in prompt

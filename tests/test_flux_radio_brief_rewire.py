"""Sprint 8.4: FLUX radio bookend brief-rewire tests.

`_build_dynamic_radio_prompt` now reads `meta.visual_palette` via
`_read_brief_field` and inserts the top-3 palette terms between the
brief-driven context and the always-on broadcast-distress suffix.
v1-era ledgers and the v2 failure sentinel both drop through to the
legacy `body, _RADIO_PROMPT_SUFFIX` shape unchanged.

The broadcast-distress identity stays locked: the palette ENRICHES
the suffix rather than replacing it, so 35mm grain, sharp focus,
centered composition, 1080p continue to ride every radio render.
"""
from __future__ import annotations

import inspect

import pytest

from visual import batch_flux_render as bfr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(
    *,
    brief: str | None = "a silent world reveals ancient life",
    brief_status: str | None = "ok",
    palette: list | None = None,
    episode_id: str | None = "signal_lost_haunted_signal_20260525_120000",
) -> dict:
    meta: dict = {}
    if brief is not None:
        meta["story_brief"] = brief
    if brief_status is not None:
        meta["story_brief_status"] = brief_status
        meta["story_brief_terms"] = {
            "setting":    [],
            "lighting":   [],
            "atmosphere": [],
        }
    if palette is not None:
        meta["visual_palette"] = palette
    led: dict = {"meta": meta}
    if episode_id is not None:
        led["episode_id"] = episode_id
    return led


# ---------------------------------------------------------------------------
# 1. v2 palette terms appear in the prompt when present
# ---------------------------------------------------------------------------


class TestPaletteLandsInPrompt:

    def test_palette_terms_appear(self):
        led = _ledger(palette=["amber glow", "smoke-grey", "wet asphalt"])
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "amber glow" in prompt
        assert "smoke-grey" in prompt
        assert "wet asphalt" in prompt

    def test_palette_sliced_to_top_three(self):
        led = _ledger(
            palette=[
                "amber", "smoke", "wet", "neon", "rust", "moss",
            ],
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "amber" in prompt
        assert "smoke" in prompt
        assert "wet" in prompt
        assert "neon" not in prompt
        assert "rust" not in prompt
        assert "moss" not in prompt

    def test_palette_lands_between_context_and_suffix(self):
        led = _ledger(
            brief="CONTEXT_CANARY",
            palette=["PALETTE_CANARY_A", "PALETTE_CANARY_B"],
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        i_ctx = prompt.index("CONTEXT_CANARY")
        i_pal_a = prompt.index("PALETTE_CANARY_A")
        i_pal_b = prompt.index("PALETTE_CANARY_B")
        i_suffix = prompt.index("35mm film grain")
        assert i_ctx < i_pal_a, "context must precede palette"
        assert i_pal_b < i_suffix, "palette must precede broadcast suffix"


# ---------------------------------------------------------------------------
# 2. The broadcast-distress identity always rides
# ---------------------------------------------------------------------------


class TestBroadcastSuffixAlwaysPresent:

    def test_suffix_rides_with_palette_present(self):
        led = _ledger(palette=["amber", "smoke"])
        prompt = bfr._build_dynamic_radio_prompt(led)
        for token in (
            "35mm film grain", "broadcast-distressed",
            "sharp focus", "centered composition", "1080p",
        ):
            assert token in prompt, (
                f"baseline broadcast-distress token {token!r} dropped "
                f"by Sprint 8.4 palette enrichment"
            )

    def test_suffix_rides_without_palette(self):
        led = _ledger(palette=None)
        prompt = bfr._build_dynamic_radio_prompt(led)
        for token in (
            "35mm film grain", "broadcast-distressed",
            "sharp focus", "centered composition", "1080p",
        ):
            assert token in prompt


# ---------------------------------------------------------------------------
# 3. Fallback paths -- v1-era, v2 failure sentinel, malformed
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_v1_era_ledger_byte_identical_to_pre_8_4(self):
        """A ledger with no visual_palette key at all must render the
        exact pre-Sprint-8.4 prompt: body + _RADIO_PROMPT_SUFFIX."""
        led = _ledger(brief="silent world", palette=None)
        prompt = bfr._build_dynamic_radio_prompt(led)
        expected = (
            f"radio broadcast unit, silent world, {bfr._RADIO_PROMPT_SUFFIX}"
        )
        assert prompt == expected

    def test_v2_failure_sentinel_byte_identical_to_legacy(self):
        """Empty visual_palette list (the v2 failure-sentinel shape)
        renders the same prompt as the v1-era ledger."""
        led_failure = _ledger(brief="silent world", palette=[])
        led_v1 = _ledger(brief="silent world", palette=None)
        out_failure = bfr._build_dynamic_radio_prompt(led_failure)
        out_v1 = bfr._build_dynamic_radio_prompt(led_v1)
        assert out_failure == out_v1

    def test_non_list_palette_falls_through(self):
        led = _ledger(brief="silent world", palette="not a list")  # type: ignore[arg-type]
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "not a list" not in prompt
        # Suffix still rides.
        assert "35mm film grain" in prompt

    def test_empty_string_entries_filtered(self):
        led = _ledger(
            brief="silent world",
            palette=["", "  ", "amber"],
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "amber" in prompt
        # No double-comma indicating empty segment.
        assert ", ," not in prompt


# ---------------------------------------------------------------------------
# 4. Tier-1 / tier-2 / tier-3 resolution -- palette layers on every tier
# ---------------------------------------------------------------------------


class TestPaletteAcrossTiers:

    def test_palette_layers_on_brief_tier(self):
        led = _ledger(
            brief="silent world",
            palette=["amber", "smoke"],
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "silent world" in prompt
        assert "amber" in prompt

    def test_palette_layers_on_slug_tier(self):
        """When the brief is empty but the episode_id slug fires, the
        palette still enriches the tail."""
        led = _ledger(
            brief="",
            brief_status="failed",
            palette=["amber", "smoke"],
            episode_id="signal_lost_haunted_signal_20260525_120000",
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        assert "haunted signal" in prompt
        assert "amber" in prompt

    def test_palette_does_not_save_hardcoded_fallback(self):
        """When BOTH brief and slug are absent, the function returns
        _RADIO_FALLBACK_PROMPT verbatim -- palette enrichment cannot
        revive a fundamentally empty prompt path."""
        led = _ledger(
            brief="",
            brief_status="failed",
            palette=["amber", "smoke"],
            episode_id=None,
        )
        prompt = bfr._build_dynamic_radio_prompt(led)
        # Tier 3: the hardcoded fallback shape. Palette is NOT layered
        # onto this tier -- the fallback is a frozen string by design.
        assert prompt == bfr._RADIO_FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# 5. Source-level wiring lock
# ---------------------------------------------------------------------------


class TestReaderHelperUsed:

    def test_function_source_uses_brief_reader(self):
        src = inspect.getsource(bfr._build_dynamic_radio_prompt)
        assert "read_brief_field" in src, (
            "_build_dynamic_radio_prompt no longer uses "
            "_read_brief_field -- Sprint 8.4 wiring regressed"
        )

    def test_function_reads_visual_palette_key(self):
        src = inspect.getsource(bfr._build_dynamic_radio_prompt)
        assert '"visual_palette"' in src, (
            "_build_dynamic_radio_prompt no longer reads "
            "meta.visual_palette -- Sprint 8.4 wiring regressed"
        )


# ---------------------------------------------------------------------------
# 6. Log observability
# ---------------------------------------------------------------------------


class TestLogObservability:

    def test_log_surfaces_palette(self, caplog):
        caplog.set_level("INFO")
        led = _ledger(palette=["amber", "smoke"])
        bfr._build_dynamic_radio_prompt(led)
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "palette=" in log_text
        assert "amber" in log_text

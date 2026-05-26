"""Sprint 8.7: OTR_VideoPlan era-tail brief-rewire tests.

`_resolve_era_tail` now reads `meta.visual_palette` + `meta.atmosphere_line`
via `_read_brief_field` and composes the era tail in order:

  atmosphere_line -> visual_palette (top 3) -> v1 lighting+atmosphere

The v1 lighting + atmosphere path is preserved as the trailing
baseline. All-empty inputs still fall through to `_DEFAULT_ERA_TAIL`.

Tests cover: v2 enrichment, ordering, top-3 slice, fallback paths
(v1-era ledger, v2 failure sentinel, malformed v2 fields), and
source-level wiring locks.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import otr_video_plan as ovp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta(
    *,
    atmosphere_line: str | None = None,
    visual_palette: list | None = None,
    v1_lighting: list | None = None,
    v1_atmosphere: list | None = None,
    brief_status: str = "ok",
) -> dict:
    m: dict = {
        "story_brief":        "a dim room",
        "story_brief_status": brief_status,
        "story_brief_terms": {
            "setting":    [],
            "lighting":   v1_lighting or [],
            "atmosphere": v1_atmosphere or [],
        },
    }
    if atmosphere_line is not None:
        m["atmosphere_line"] = atmosphere_line
    if visual_palette is not None:
        m["visual_palette"] = visual_palette
    return m


# ---------------------------------------------------------------------------
# 1. v2 fields appear in the tail
# ---------------------------------------------------------------------------


class TestV2FieldsLand:

    def test_atmosphere_line_appears(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="smoke and sweat under a bare bulb.",
        ))
        assert "smoke and sweat under a bare bulb." in tail

    def test_palette_appears(self):
        tail = ovp._resolve_era_tail(_meta(
            visual_palette=["amber glow", "smoke-grey", "wet asphalt"],
        ))
        assert "amber glow" in tail
        assert "smoke-grey" in tail
        assert "wet asphalt" in tail

    def test_palette_sliced_to_top_three(self):
        tail = ovp._resolve_era_tail(_meta(
            visual_palette=[
                "amber", "smoke", "wet", "neon", "rust", "moss",
            ],
        ))
        assert "amber" in tail
        assert "smoke" in tail
        assert "wet" in tail
        assert "neon" not in tail
        assert "rust" not in tail
        assert "moss" not in tail


# ---------------------------------------------------------------------------
# 2. Ordering: atmosphere_line -> palette -> v1 tail
# ---------------------------------------------------------------------------


class TestOrdering:

    def test_atmosphere_line_precedes_palette_and_v1(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="ATMO_CANARY",
            visual_palette=["PALETTE_CANARY"],
            v1_lighting=["V1_LIGHT_CANARY"],
        ))
        i_atmo = tail.index("ATMO_CANARY")
        i_pal = tail.index("PALETTE_CANARY")
        i_v1 = tail.index("V1_LIGHT_CANARY")
        assert i_atmo < i_pal < i_v1, (
            f"era-tail order wrong: atmo@{i_atmo} "
            f"palette@{i_pal} v1@{i_v1}"
        )

    def test_palette_precedes_v1_when_atmosphere_absent(self):
        tail = ovp._resolve_era_tail(_meta(
            visual_palette=["PALETTE_CANARY"],
            v1_lighting=["V1_LIGHT_CANARY"],
        ))
        i_pal = tail.index("PALETTE_CANARY")
        i_v1 = tail.index("V1_LIGHT_CANARY")
        assert i_pal < i_v1


# ---------------------------------------------------------------------------
# 3. Fallback paths
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_v1_era_ledger_byte_identical_to_pre_8_7(self):
        """No v2 fields at all -- output must equal the pre-Sprint-8.7
        legacy v1 tail."""
        meta_pre = _meta(
            v1_lighting=["bare bulb"],
            v1_atmosphere=["tense"],
        )
        # The v1 tail is exactly what get_story_brief_lighting returns:
        # ", ".join(lighting + atmosphere).
        legacy_expected = "bare bulb, tense"
        tail = ovp._resolve_era_tail(meta_pre)
        assert tail == legacy_expected

    def test_v2_failure_sentinel_falls_to_default(self):
        """v2 sentinel: all fields empty, brief failed -> default."""
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="",
            visual_palette=[],
            brief_status="failed",
        ))
        assert tail == ovp._DEFAULT_ERA_TAIL

    def test_all_signals_absent_returns_default(self):
        tail = ovp._resolve_era_tail({})
        assert tail == ovp._DEFAULT_ERA_TAIL

    def test_non_list_palette_falls_through(self):
        tail = ovp._resolve_era_tail(_meta(
            visual_palette="not a list",  # type: ignore[arg-type]
            v1_lighting=["bare bulb"],
        ))
        assert "not a list" not in tail
        assert "bare bulb" in tail

    def test_non_string_atmosphere_line_falls_through(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line=42,  # type: ignore[arg-type]
            v1_lighting=["bare bulb"],
        ))
        assert "42" not in tail
        assert "bare bulb" in tail

    def test_whitespace_atmosphere_line_treated_as_empty(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="   \t  ",
            v1_lighting=["bare bulb"],
        ))
        # Whitespace-only treated as empty; same as no atmosphere_line.
        baseline = ovp._resolve_era_tail(_meta(
            v1_lighting=["bare bulb"],
        ))
        assert tail == baseline


# ---------------------------------------------------------------------------
# 4. v2-only paths (no v1 lighting/atmosphere)
# ---------------------------------------------------------------------------


class TestV2OnlyPath:

    def test_atmosphere_line_alone(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="lone v2 sentence."
        ))
        assert tail == "lone v2 sentence."

    def test_palette_alone(self):
        tail = ovp._resolve_era_tail(_meta(
            visual_palette=["amber", "smoke"]
        ))
        assert tail == "amber, smoke"

    def test_atmosphere_plus_palette_no_v1(self):
        tail = ovp._resolve_era_tail(_meta(
            atmosphere_line="lone v2 sentence.",
            visual_palette=["amber"],
        ))
        assert tail == "lone v2 sentence., amber"


# ---------------------------------------------------------------------------
# 5. Source-level wiring lock
# ---------------------------------------------------------------------------


class TestReaderHelperUsed:

    def test_function_source_uses_brief_reader(self):
        src = inspect.getsource(ovp._resolve_era_tail)
        assert "_read_brief_field" in src, (
            "_resolve_era_tail no longer uses _read_brief_field -- "
            "Sprint 8.7 wiring regressed"
        )

    def test_function_reads_both_v2_keys(self):
        src = inspect.getsource(ovp._resolve_era_tail)
        assert '"visual_palette"' in src, (
            "_resolve_era_tail no longer reads meta.visual_palette"
        )
        assert '"atmosphere_line"' in src, (
            "_resolve_era_tail no longer reads meta.atmosphere_line"
        )

    def test_function_still_uses_v1_lighting_helper(self):
        src = inspect.getsource(ovp._resolve_era_tail)
        assert "get_story_brief_lighting" in src, (
            "_resolve_era_tail dropped the v1 helper -- additive "
            "rewire turned into replacement"
        )


# ---------------------------------------------------------------------------
# 6. Log observability
# ---------------------------------------------------------------------------


class TestLogObservability:

    def test_log_surfaces_all_three_signal_counts(self, caplog):
        caplog.set_level("INFO")
        ovp._resolve_era_tail(_meta(
            atmosphere_line="brief atmo line",
            visual_palette=["amber"],
            v1_lighting=["bulb"],
        ))
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "atmosphere_line_chars" in log_text
        assert "palette_terms" in log_text
        assert "v1_chars" in log_text

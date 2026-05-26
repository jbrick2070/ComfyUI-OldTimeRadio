"""Sprint 8.2: FLUX env brief-rewire tests.

Locks the contract for the FLUX env consumer wired to the v2 brief:

  * `visual_palette` (top-level meta key, list[str]) and `key_objects`
    (top-level, list[str]) land between the env description and the
    cinematic style_suffix tail.
  * The v1 `meta.story_brief` prose still LEADS (BUG-LOCAL-250 rule).
  * Top-3 slice on each list (composition-dilution guard).
  * v1-era ledgers (no v2 fields) and v2 failure sentinel (empty v2
    lists) both drop through to the legacy brief-leads composition.
  * Malformed v2 values (non-list, list with empty strings) degrade
    gracefully -- no crash, the legacy path runs.

The test imports `_parse_env_prompts` directly and feeds hand-crafted
script_json strings. The function is a pure prompt-string builder
once the helpers are imported, so no FLUX / torch heavy deps load.
"""
from __future__ import annotations

import json

import pytest

from visual import batch_flux_render as bfr


_FALLBACK = "a dim warehouse"
_STYLE_SUFFIX = "35mm film grain, cinematic"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _env_token(description: str = "a foggy dock at midnight") -> dict:
    return {"type": "environment", "description": description}


def _payload(
    *,
    brief: str | None = "a dim room under a bare bulb",
    brief_status: str | None = "ok",
    palette: list | None = None,
    key_objects: list | None = None,
    extra_meta: dict | None = None,
    env_description: str = "a foggy dock at midnight",
) -> str:
    """Build a script_json string with controllable brief + v2 fields."""
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
    if key_objects is not None:
        meta["key_objects"] = key_objects
    if extra_meta:
        meta.update(extra_meta)
    payload = {
        "tokens": [
            _env_token(env_description),
            {"type": "dialogue", "text": "noise to exercise filtering"},
        ],
        "meta": meta,
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# 1. v2 palette + key_objects land in the prompt
# ---------------------------------------------------------------------------


class TestV2FieldsLand:

    def test_palette_terms_appear_in_prompt(self):
        sj = _payload(
            palette=["amber glow", "smoke-grey", "wet asphalt"],
            key_objects=None,
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert len(prompts) == 1
        assert "amber glow" in prompts[0]
        assert "smoke-grey" in prompts[0]
        assert "wet asphalt" in prompts[0]

    def test_key_objects_appear_in_prompt(self):
        sj = _payload(
            palette=None,
            key_objects=["steel table", "bare bulb", "ashtray"],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert "steel table" in prompts[0]
        assert "bare bulb" in prompts[0]
        assert "ashtray" in prompts[0]

    def test_both_v2_fields_appear_together(self):
        sj = _payload(
            palette=["amber glow"],
            key_objects=["steel table"],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert "amber glow" in prompts[0]
        assert "steel table" in prompts[0]


# ---------------------------------------------------------------------------
# 2. Ordering: brief LEADS; v2 fields between desc and style_suffix
# ---------------------------------------------------------------------------


class TestOrdering:

    def test_brief_still_leads(self):
        """BUG-LOCAL-250 invariant: meta.story_brief is the first segment."""
        sj = _payload(
            brief="LEADING_BRIEF_CANARY",
            palette=["amber"],
            key_objects=["table"],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        first_segment = prompts[0].split(",")[0].strip()
        assert first_segment == "LEADING_BRIEF_CANARY"

    def test_v2_fields_after_desc_before_style_suffix(self):
        sj = _payload(
            brief="brief_text",
            palette=["palette_token_PT"],
            key_objects=["object_token_OT"],
            env_description="env_token_ET",
        )
        prompts = bfr._parse_env_prompts(
            sj, 1, _FALLBACK, "style_token_ST",
        )
        segments = [s.strip() for s in prompts[0].split(",")]
        i_brief = segments.index("brief_text")
        i_desc = segments.index("env_token_ET")
        i_palette = segments.index("palette_token_PT")
        i_object = segments.index("object_token_OT")
        i_style = segments.index("style_token_ST")
        assert i_brief < i_desc, "brief must lead"
        assert i_desc < i_palette, "env desc must precede v2 palette"
        assert i_palette < i_style, "palette must precede style_suffix"
        assert i_desc < i_object, "env desc must precede v2 key_objects"
        assert i_object < i_style, "key_objects must precede style_suffix"


# ---------------------------------------------------------------------------
# 3. Top-3 slice on each list
# ---------------------------------------------------------------------------


class TestTopThreeSlice:

    def test_palette_sliced_to_top_three(self):
        sj = _payload(
            palette=[
                "amber", "smoke", "wet", "neon", "rust", "moss",
            ],
            key_objects=None,
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert "amber" in prompts[0]
        assert "smoke" in prompts[0]
        assert "wet" in prompts[0]
        assert "neon" not in prompts[0]
        assert "rust" not in prompts[0]
        assert "moss" not in prompts[0]

    def test_key_objects_sliced_to_top_three(self):
        sj = _payload(
            palette=None,
            key_objects=[
                "table", "bulb", "ashtray", "ledger", "phone", "coat",
            ],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert "table" in prompts[0]
        assert "bulb" in prompts[0]
        assert "ashtray" in prompts[0]
        assert "ledger" not in prompts[0]
        assert "phone" not in prompts[0]
        assert "coat" not in prompts[0]


# ---------------------------------------------------------------------------
# 4. Fallback paths: v1-era ledger and v2 failure sentinel
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_v1_era_ledger_falls_through(self):
        """No v2 fields at all -- only v1 brief present. Function must
        compose the legacy brief-leads-desc-style_suffix string."""
        sj = _payload(
            brief="LEGACY_BRIEF_LEAD_canary",
            palette=None,
            key_objects=None,
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        # Legacy shape: brief LEADS, then desc, then style_suffix.
        segments = [s.strip() for s in prompts[0].split(",")]
        assert "LEGACY_BRIEF_LEAD_canary" in segments
        assert "a foggy dock at midnight" in segments
        # No v2 tokens leaked. Pick canaries that cannot collide with
        # the brief / desc / style_suffix fixture strings.
        for token in (
            "palette_canary_amber", "object_canary_table",
        ):
            assert token not in prompts[0]

    def test_v2_failure_sentinel_falls_through(self):
        """V2 fields present but empty (the failure-sentinel shape).
        Same composition as v1-era ledger -- empty lists contribute
        no segments. Identity check against the legacy path: the
        rendered prompt must equal the v1-era render with the same
        brief / desc / style_suffix."""
        sj_failure = _payload(
            brief="SENTINEL_PROBE", palette=[], key_objects=[],
        )
        sj_legacy = _payload(
            brief="SENTINEL_PROBE", palette=None, key_objects=None,
        )
        out_failure = bfr._parse_env_prompts(
            sj_failure, 1, _FALLBACK, _STYLE_SUFFIX,
        )
        out_legacy = bfr._parse_env_prompts(
            sj_legacy, 1, _FALLBACK, _STYLE_SUFFIX,
        )
        assert out_failure == out_legacy, (
            "v2 failure-sentinel-shaped meta produced a different "
            "prompt than the v1-era ledger; empty v2 lists must "
            "drop through cleanly"
        )


# ---------------------------------------------------------------------------
# 5. Malformed v2 values degrade gracefully
# ---------------------------------------------------------------------------


class TestMalformedV2Values:

    def test_non_list_palette_is_ignored(self):
        sj = _payload(
            palette="not a list",  # type: ignore[arg-type]
            key_objects=["table"],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        # The non-list value must NOT reach the prompt.
        assert "not a list" not in prompts[0]
        # key_objects still works.
        assert "table" in prompts[0]

    def test_empty_string_entries_are_filtered(self):
        sj = _payload(
            palette=["", "  ", "\t", "amber"],
            key_objects=["", "table"],
        )
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert "amber" in prompts[0]
        assert "table" in prompts[0]
        # Two consecutive commas would indicate an empty segment landed.
        assert ", ," not in prompts[0]

    def test_none_palette_falls_through(self):
        """A meta with story_brief but no v2 keys at all (the v1-era
        ledger case) still composes without crashing."""
        sj = _payload(palette=None, key_objects=None)
        prompts = bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        assert len(prompts) == 1


# ---------------------------------------------------------------------------
# 6. Log line surfaces v2 fields for operator observability
# ---------------------------------------------------------------------------


class TestLogObservability:

    def test_log_includes_palette_and_key_objects(self, caplog):
        caplog.set_level("INFO")
        sj = _payload(
            palette=["amber"],
            key_objects=["table"],
        )
        bfr._parse_env_prompts(sj, 1, _FALLBACK, _STYLE_SUFFIX)
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "palette=" in log_text
        assert "key_objects=" in log_text
        assert "amber" in log_text
        assert "table" in log_text


# ---------------------------------------------------------------------------
# 7. _read_brief_field is the canonical reader (source-level lock)
# ---------------------------------------------------------------------------


class TestReaderHelperUsed:

    def test_function_source_uses_brief_reader(self):
        """Mechanical grep: a future refactor that swaps to direct
        meta[...] access loses the v2 default-on-missing safety net.
        Pin the helper as the canonical access path."""
        import inspect
        src = inspect.getsource(bfr._parse_env_prompts)
        assert "read_brief_field" in src, (
            "_parse_env_prompts no longer uses _read_brief_field -- "
            "Sprint 8.2 wiring regressed"
        )

    def test_brief_reader_helper_resolves(self):
        """The `_brief_reader` sys.path-prepend helper must return the
        actual `_read_brief_field` callable -- mirrors the
        `_story_brief_helpers` pattern that BUG-LOCAL-249 locked."""
        fn = bfr._brief_reader()
        assert callable(fn)
        # Smoke: round-trip a value through it.
        out = fn({"story_brief_status": "ok", "visual_palette": ["x"]},
                 "visual_palette", default=[])
        assert out == ["x"]

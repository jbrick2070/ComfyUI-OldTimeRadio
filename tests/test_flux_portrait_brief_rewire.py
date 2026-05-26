"""Sprint 8.3: FLUX portrait brief-rewire tests.

`_build_portrait_prompt` grew two new optional kwargs in Sprint 8.3:

  * `palette` -- comma-joined top-3 of `meta.visual_palette`,
    resolved at the caller via `_read_brief_field`.
  * `atmosphere_line` -- one prose sentence from
    `meta.atmosphere_line`, resolved the same way.

Both land in the brief-leads region of the portrait prompt, right
after the v1 lighting signal and before the style anchor + speaker
description. Order is lighting -> atmosphere_line -> palette.
Empty inputs contribute nothing.

These tests pin the function-level contract. The caller-side wiring
in the BatchFluxPortraitRender.execute() method is covered by source-
level inspection (the caller's `_read_brief_field` import must stay
present so future refactors do not silently drop the v2 reads).
"""
from __future__ import annotations

import inspect

import pytest

from visual import batch_flux_portrait_render as bfp


# ---------------------------------------------------------------------------
# 1. v2 fields appear in the prompt when supplied
# ---------------------------------------------------------------------------


class TestV2FieldsLand:

    def test_atmosphere_line_appears_in_prompt(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="",
            palette="",
            atmosphere_line="smoke and sweat under a bare bulb.",
        )
        assert "smoke and sweat under a bare bulb." in prompt

    def test_palette_appears_in_prompt(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="",
            palette="amber glow, smoke-grey, wet asphalt",
            atmosphere_line="",
        )
        assert "amber glow" in prompt
        assert "smoke-grey" in prompt
        assert "wet asphalt" in prompt

    def test_both_v2_fields_appear_together(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="harsh top-down shadow, tense",
            palette="amber glow, smoke-grey",
            atmosphere_line="smoke and sweat under a bare bulb.",
        )
        assert "harsh top-down shadow" in prompt
        assert "smoke and sweat under a bare bulb." in prompt
        assert "amber glow" in prompt


# ---------------------------------------------------------------------------
# 2. Ordering: lighting -> atmosphere_line -> palette -> style_anchor
# ---------------------------------------------------------------------------


class TestOrdering:

    def test_v2_segments_after_lighting_before_style_anchor(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="APPEARANCE_CANARY",
            style_anchor="STYLE_ANCHOR_CANARY",
            lighting="LIGHTING_CANARY",
            palette="PALETTE_CANARY",
            atmosphere_line="ATMOSPHERE_CANARY",
        )
        segments = [s.strip() for s in prompt.split(",")]
        i_light = segments.index("LIGHTING_CANARY")
        i_atmo = segments.index("ATMOSPHERE_CANARY")
        i_pal = segments.index("PALETTE_CANARY")
        i_style = segments.index("STYLE_ANCHOR_CANARY")
        i_app = segments.index("APPEARANCE_CANARY")
        # Brief-leads region must lead.
        assert i_light < i_atmo
        assert i_atmo < i_pal
        # Style anchor and appearance follow.
        assert i_pal < i_style
        assert i_style < i_app


# ---------------------------------------------------------------------------
# 3. Fallback paths -- empty inputs degrade structurally to legacy
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_empty_palette_and_atmosphere_line_match_legacy(self):
        """With v2 inputs empty, the rendered prompt must equal the
        pre-Sprint-8.3 legacy shape with just lighting."""
        new_prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="harsh top-down shadow",
            palette="",
            atmosphere_line="",
        )
        legacy_prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="harsh top-down shadow",
        )
        assert new_prompt == legacy_prompt, (
            "empty v2 fields drifted the rendered prompt away from "
            "the legacy lighting-only path"
        )

    def test_all_brief_signals_empty_falls_to_legacy_composition(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="",
            palette="",
            atmosphere_line="",
        )
        # No empty leading comma; style anchor is the first segment.
        first = prompt.split(",")[0].strip()
        assert first == "noir studio portrait"

    def test_only_v2_atmosphere_line_present(self):
        """v2 producer ran but v1 helper returned empty -- atmosphere_line
        still leads, no spurious empty-string segment."""
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="",
            palette="",
            atmosphere_line="smoke and sweat",
        )
        first = prompt.split(",")[0].strip()
        assert first == "smoke and sweat"

    def test_whitespace_inputs_treated_as_empty(self):
        prompt = bfp._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
            lighting="   ",
            palette="\t",
            atmosphere_line="  \n",
        )
        first = prompt.split(",")[0].strip()
        assert first == "noir studio portrait"


# ---------------------------------------------------------------------------
# 4. Signature contract -- new kwargs default to empty + optional
# ---------------------------------------------------------------------------


class TestSignatureContract:

    def test_v2_kwargs_default_to_empty(self):
        """A pre-8.3 caller that does NOT supply palette / atmosphere_line
        must keep working -- the v2 kwargs default to empty strings."""
        sig = inspect.signature(bfp._build_portrait_prompt)
        assert sig.parameters["palette"].default == ""
        assert sig.parameters["atmosphere_line"].default == ""

    def test_legacy_call_still_works(self):
        """The pre-8.3 4-arg call form must still produce a valid prompt
        without raising. Identity test against the equivalent 6-arg
        call with v2 fields empty."""
        prompt_legacy = bfp._build_portrait_prompt(
            "Jones", "weary detective", "noir portrait",
        )
        prompt_explicit = bfp._build_portrait_prompt(
            "Jones", "weary detective", "noir portrait",
            lighting="", palette="", atmosphere_line="",
        )
        assert prompt_legacy == prompt_explicit


# ---------------------------------------------------------------------------
# 5. Caller-side wiring -- `_read_brief_field` is the canonical reader
# ---------------------------------------------------------------------------


class TestCallerWiringSourceLevel:

    def test_execute_imports_brief_reader(self):
        """Mechanical grep: the execute() method must import
        `_read_brief_field` from `_otr_brief_reader`. A future refactor
        that swaps to direct meta[...] access loses the v2 default-on-
        missing safety net."""
        src = inspect.getsource(bfp)
        assert "_otr_brief_reader" in src, (
            "BatchFluxPortraitRender source no longer references "
            "_otr_brief_reader -- Sprint 8.3 caller wiring regressed"
        )
        assert "_read_brief_field" in src, (
            "BatchFluxPortraitRender source no longer references "
            "_read_brief_field -- Sprint 8.3 caller wiring regressed"
        )

    def test_execute_passes_v2_kwargs_to_builder(self):
        """The execute() method must pass `palette` and `atmosphere_line`
        kwargs into `_build_portrait_prompt`. Locks the wiring end-to-
        end (caller resolves the brief fields, builder consumes them)."""
        src = inspect.getsource(bfp)
        # The caller's _build_portrait_prompt call site carries the
        # two new kwargs.
        assert "palette=_brief_palette" in src, (
            "BatchFluxPortraitRender no longer passes "
            "palette=_brief_palette to _build_portrait_prompt"
        )
        assert "atmosphere_line=_brief_atmosphere_line" in src, (
            "BatchFluxPortraitRender no longer passes "
            "atmosphere_line=_brief_atmosphere_line to "
            "_build_portrait_prompt"
        )

    def test_execute_reads_v2_keys_from_meta(self):
        """The caller resolves both v2 fields by their exact meta key
        names. Catches a typo / rename in the dotted-path argument."""
        src = inspect.getsource(bfp)
        assert '"visual_palette"' in src, (
            "BatchFluxPortraitRender no longer reads "
            "meta.visual_palette via _read_brief_field"
        )
        assert '"atmosphere_line"' in src, (
            "BatchFluxPortraitRender no longer reads "
            "meta.atmosphere_line via _read_brief_field"
        )

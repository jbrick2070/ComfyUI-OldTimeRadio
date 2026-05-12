"""Regression test for the two-LLM split feature (task #56 / Block 4).

Feature: OTR_LedgerScriptWriter has a `cleanup_model_id` widget that lets
users pair a creative LLM (rich draft prose, e.g. Captain-Eris-Violet)
with a structured LLM (format-compliant rescue paths, e.g. base
Mistral-Nemo-Instruct-2407). When `cleanup_model_id = "auto (use story
model)"` (the default), the run uses the same model for both roles --
backward-compatible with every saved workflow.

Structured paths that should use cleanup_model_id when configured:
    - WORD_EXTEND (rescue dialogue extension)
    - ANNOUNCER bookend generation
    - FORMAT_NORM (canonical format conversion)
    - Grammarian (light copy-edit)
    - LLM_RESCUE on parser-exception path
    - LLM_RESCUE on BUG-066 floor-trigger path

Creative paths that should ALWAYS use the story model (never cleanup):
    - Cast names, news summary, spine outlines, draft, critique,
      revision, autotitle, arc-enhancer, per-act gen, story-editor,
      act summaries (these are intentionally on the creative model so
      it has continuity over the whole story-writing flow).

This test exercises the resolver helper without requiring transformers /
torch / comfy by re-implementing the resolution as a standalone function.
"""

from __future__ import annotations

import pytest


def _resolve_cleanup_model_id(model_id: str, cleanup_model_id: str) -> str:
    """Mirror of the resolver embedded in write_script() at line ~3070.

    Returns the effective model id to use for structured cleanup phases.
    """
    if (not cleanup_model_id
            or str(cleanup_model_id).strip().lower().startswith("auto")):
        return model_id
    return str(cleanup_model_id).strip()


# ---------------------------------------------------------------------------
# Backward-compat: "auto (use story model)" must keep everything on one model
# ---------------------------------------------------------------------------


class TestAutoModeBackwardCompat:

    def test_auto_with_full_label_returns_story_model(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "auto (use story model)",
        )
        assert out == "mistralai/Mistral-Nemo-Instruct-2407"

    def test_auto_with_short_label_returns_story_model(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "auto",
        )
        assert out == "mistralai/Mistral-Nemo-Instruct-2407"

    def test_auto_case_insensitive(self):
        out = _resolve_cleanup_model_id(
            "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
            "Auto (use story model)",
        )
        assert out == "Nitral-AI/Captain-Eris_Violet-V0.420-12B"

    def test_empty_cleanup_id_returns_story_model(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "",
        )
        assert out == "mistralai/Mistral-Nemo-Instruct-2407"

    def test_none_cleanup_id_returns_story_model(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            None,
        )
        assert out == "mistralai/Mistral-Nemo-Instruct-2407"


# ---------------------------------------------------------------------------
# Recommended preset: creative=Captain-Eris-Violet, cleanup=Mistral-Nemo
# ---------------------------------------------------------------------------


class TestCreativeCaptainCleanupMistralPreset:
    """The pairing the user actually wants tonight: rich RP voice for the
    draft + base instruct for every structured rescue."""

    CREATIVE = "Nitral-AI/Captain-Eris_Violet-V0.420-12B"
    CLEANUP = "mistralai/Mistral-Nemo-Instruct-2407"

    def test_resolver_returns_cleanup_when_explicit(self):
        out = _resolve_cleanup_model_id(self.CREATIVE, self.CLEANUP)
        assert out == self.CLEANUP

    def test_resolver_does_not_fall_back_to_auto(self):
        out = _resolve_cleanup_model_id(self.CREATIVE, self.CLEANUP)
        assert out != self.CREATIVE


# ---------------------------------------------------------------------------
# Inverse pairing: creative=Mistral-Nemo, cleanup=Gemma-2-9b (smaller)
# ---------------------------------------------------------------------------


class TestSmallCleanupModelPairing:
    """Power-user pairing: use a small, fast cleanup model on a slow
    creative draft."""

    def test_gemma_cleanup_small_creative_big(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "google/gemma-2-9b-it",
        )
        assert out == "google/gemma-2-9b-it"


# ---------------------------------------------------------------------------
# Same model both sides should behave identically to "auto"
# ---------------------------------------------------------------------------


class TestSameModelBothSides:

    def test_same_creative_and_cleanup_resolves_clean(self):
        m = "mistralai/Mistral-Nemo-Instruct-2407"
        assert _resolve_cleanup_model_id(m, m) == m


# ---------------------------------------------------------------------------
# Whitespace + edge cases
# ---------------------------------------------------------------------------


class TestResolverEdgeCases:

    def test_leading_trailing_whitespace_stripped(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "  google/gemma-2-9b-it  ",
        )
        assert out == "google/gemma-2-9b-it"

    def test_qwen_alpha_label_passes_through(self):
        out = _resolve_cleanup_model_id(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
        )
        # The [ALPHA] suffix is part of the dropdown label; the production
        # _load_llm strips bracket suffixes before lookup. The resolver
        # itself just passes the string through as-is.
        assert out == "Qwen/Qwen2.5-14B-Instruct [ALPHA]"


# ---------------------------------------------------------------------------
# Active-state derivation (matches the SINGLE_LLM vs TWO_LLM_SPLIT log)
# ---------------------------------------------------------------------------


def _two_llm_active(model_id: str, cleanup_model_id: str) -> bool:
    effective = _resolve_cleanup_model_id(model_id, cleanup_model_id)
    return effective != model_id


class TestTwoLLMActiveDerivation:

    def test_auto_means_single_llm_mode(self):
        assert not _two_llm_active(
            "mistralai/Mistral-Nemo-Instruct-2407",
            "auto (use story model)",
        )

    def test_explicit_same_model_is_single_llm(self):
        m = "mistralai/Mistral-Nemo-Instruct-2407"
        assert not _two_llm_active(m, m)

    def test_explicit_different_model_is_two_llm(self):
        assert _two_llm_active(
            "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
            "mistralai/Mistral-Nemo-Instruct-2407",
        )

    def test_empty_cleanup_is_single_llm(self):
        assert not _two_llm_active(
            "mistralai/Mistral-Nemo-Instruct-2407", "",
        )

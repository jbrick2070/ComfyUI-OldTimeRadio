"""Creative prompt router resolver.

Core assertions:

  test_router_returns_modern_for_default_mistral_nemo
      The default writer slot (Mistral-Nemo) resolves to the modern
      phase prompt for every phase.

  test_router_production_caller_count_pinned
      `resolve_creative_system_prompt` has the expected production
      call sites for outline, line composition, and closing layers.

  test_router_raises_on_unknown_phase
      Typo at a call site fails loud with ValueError.

  test_render_few_shot_block_has_zero_production_callers
      Encodes the D2c few-shot OMIT decision as a zero-callers test.
      Re-introducing render_few_shot_block fires the test and
      forces a deliberate scope decision.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_creative_prompt_router as router  # noqa: E402
from nodes import _otr_compose_exchange  # noqa: E402
from nodes import _otr_line_composer  # noqa: E402
from nodes import _otr_model_catalog  # noqa: E402
from nodes import _otr_outline  # noqa: E402
from nodes import _otr_period_prompts  # noqa: E402
from nodes._otr_story_pack import get_pack_prompt  # noqa: E402
from nodes._otr_story_routing import resolve_story_pack  # noqa: E402

MISTRAL_NEMO = "mistralai/Mistral-Nemo-Instruct-2407"

PHASES: tuple[str, ...] = (
    "outline",
    "line_composer_system",
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "exchange_system",
    "coda_system",
    "announcer_intro_system",
    "announcer_intro_safe_system",
    "announcer_outro_system",
)

# The router's default source_bank_id for pack-routed seams. Every phase
# except the object-identity "outline" sentinel now resolves through this
# lane's story pack, so the modern-value expectation is the pack seam, not a
# Python constant. (The old science byte-identity baseline was retired with
# the science_news roster trim.)
_DEFAULT_SEAM_BANK = "media_archive"


def _expected_modern_by_phase() -> dict[str, str]:
    """Value a modern-profile / non-curated slot must resolve to per phase:
    the default bank's pack seam for pack-routed phases, and the modern
    outline constant (object-identity) for the plain ``outline`` phase."""
    pack = resolve_story_pack(_DEFAULT_SEAM_BANK)
    return {
        "outline":                       _otr_outline._SYSTEM_PROMPT,
        "line_composer_system":          get_pack_prompt(pack, "line_composer_system"),
        "outline_macro_system":          get_pack_prompt(pack, "outline_macro_system"),
        "outline_phase_system":          get_pack_prompt(pack, "outline_phase_system"),
        "outline_beat_system":           get_pack_prompt(pack, "outline_beat_system"),
        "exchange_system":               get_pack_prompt(pack, "exchange_system"),
        "coda_system":                   get_pack_prompt(pack, "coda_system"),
        "announcer_intro_system":        get_pack_prompt(pack, "announcer_intro_system"),
        "announcer_intro_safe_system":   get_pack_prompt(pack, "announcer_intro_safe_system"),
        "announcer_outro_system":        get_pack_prompt(pack, "announcer_outro_system"),
    }


# ---------------------------------------------------------------------------
# Resolver behavior
# ---------------------------------------------------------------------------


def test_router_returns_modern_for_default_mistral_nemo() -> None:
    """Every phase under Mistral-Nemo returns the corresponding modern
    phase prompt VALUE. The pack-routed seams resolve from the router's
    default lane's story pack (media_archive) -- value-equality is the
    contract, not object identity. outline stays object-identical (untouched).
    """
    expected_modern_by_phase = _expected_modern_by_phase()
    for phase in PHASES:
        out = router.resolve_creative_system_prompt(MISTRAL_NEMO, phase)
        assert out == expected_modern_by_phase[phase], (
            f"router({MISTRAL_NEMO!r}, {phase!r}) did NOT return the "
            f"modern phase prompt value; phase prompt may have been "
            f"reassigned or the pack drifted from the constant"
        )


def test_router_returns_modern_for_every_curated_row() -> None:
    """With no curated otr_1940s_v1 period row at present, the router
    resolves EVERY curated repo_id to a modern phase prompt. Pins the
    parked state of the period-routing branch -- adding a period model
    later should be a deliberate change that updates this test.
    """
    for m in _otr_model_catalog.CURATED_LLM_MODELS:
        assert m.prompt_profile == "modern", (
            f"curated row {m.repo_id!r} has prompt_profile "
            f"{m.prompt_profile!r}; no period row is expected at present"
        )
        for phase in PHASES:
            out = router.resolve_creative_system_prompt(m.repo_id, phase)
            assert out is not _otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT, (
                f"router({m.repo_id!r}, {phase!r}) returned the period "
                f"system prompt for a modern-profile row"
            )


def test_router_returns_modern_for_remote_slot_handles() -> None:
    """A remote creative writer (openrouter:slot-a/b, comfy:slot-a/b) is NOT a
    curated local row, so it has no per-model prompt_profile -> the router must
    return the MODERN phase prompt, NOT crash. Regression for the KeyError that
    aborted a whole episode when the creative slot was an OpenRouter model
    (2026-06-18).
    """
    expected_modern_by_phase = _expected_modern_by_phase()
    for handle in ("openrouter:slot-a", "openrouter:slot-b",
                   "comfy:slot-a", "comfy:slot-b", "some/uncurated-model"):
        for phase in PHASES:
            out = router.resolve_creative_system_prompt(handle, phase)
            assert out == expected_modern_by_phase[phase], (
                f"router({handle!r}, {phase!r}) must return the modern prompt "
                f"for a non-curated/remote slot, never KeyError or the period "
                f"prompt"
            )


def test_router_raises_on_unknown_phase() -> None:
    """Typo at a call site fails loud with ValueError naming the
    bad phase identifier.
    """
    with pytest.raises(ValueError, match="unknown creative phase"):
        router.resolve_creative_system_prompt(MISTRAL_NEMO, "outlineX")


# ---------------------------------------------------------------------------
# Caller-count invariants
# ---------------------------------------------------------------------------


# Production code search roots (excludes tests/ + docs/ + scripts/ scratch).
PROD_ROOTS = (
    REPO_ROOT / "nodes",
    REPO_ROOT / "visual",
)


def _count_callers(symbol_name: str, exclude_path: Path) -> tuple[int, list[str]]:
    """Count `ast.Call` nodes where the called function is named
    `symbol_name`, across production .py files. Excludes
    `exclude_path` (the symbol's own definition file).

    Matches both bare-name calls `fn(...)` and attribute-access
    calls `mod.fn(...)`. AST-based so docstrings, comments,
    string-literal mentions, and `__all__` entries do not count.
    A future commit that re-introduces the symbol by adding any
    of those still passes this test -- only a real CALL site fires
    the counter.

    Returns (count, list of "path:line" hits).
    """
    hits: list[str] = []
    for root in PROD_ROOTS:
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            if py.resolve() == exclude_path.resolve():
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called_name: str | None = None
                if isinstance(func, ast.Name):
                    called_name = func.id
                elif isinstance(func, ast.Attribute):
                    called_name = func.attr
                if called_name == symbol_name:
                    lineno = getattr(node, "lineno", -1)
                    hits.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    return len(hits), hits


def test_router_production_caller_count_pinned() -> None:
    """The router has three production dispatch points.

    Outline and character-line composition each dispatch directly. All closing
    layers share ``_resolved_closing_prompt`` so their runtime-selected phase
    stays centralized; closing-phase behavior is covered by the bank-routing
    tests rather than duplicated static call sites.
    """
    router_src = REPO_ROOT / "nodes" / "_otr_creative_prompt_router.py"
    count, hits = _count_callers("resolve_creative_system_prompt", router_src)
    assert count == 3, (
        f"resolve_creative_system_prompt has {count} production "
        f"caller(s); expected outline, compose_line, and the shared "
        f"closing resolver. Hits:\n  " + "\n  ".join(hits)
    )


def test_render_few_shot_block_has_zero_production_callers() -> None:
    """Encodes the D2c few-shot OMIT decision. The helper exists in
    `_otr_period_prompts` but is intentionally not wired into any
    production-side prompt assembly at D2a. Re-introducing it would
    cost ~600 tokens of context budget per call and should be a
    deliberate D-future scope decision, not a silent add.
    """
    period_src = REPO_ROOT / "nodes" / "_otr_period_prompts.py"
    count, hits = _count_callers("render_few_shot_block", period_src)
    assert count == 0, (
        f"render_few_shot_block has {count} production caller(s) at "
        f"the D2a boundary; expected 0 per the v3 plan D2c few-shot "
        f"OMIT decision. If a deliberate re-introduction is intended, "
        f"update this test to assert the new expected count and "
        f"document the few-shot context-budget cost in the commit "
        f"body. Hits:\n  " + "\n  ".join(hits)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

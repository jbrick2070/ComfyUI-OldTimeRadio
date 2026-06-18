"""Sprint E E4 / R-2: creative prompt router exact-match drift guard.

The cold-read review surfaced a HIGH finding (M1 in the Sprint E fix
plan) that the router selection "relies on substring matching" and
could mis-dispatch the period profile against a non-curated forked
HF id. Reading nodes/_otr_creative_prompt_router.py shows the router
actually does exact-match against CURATED_LLM_MODELS by repo_id
(L101-102) and dispatches on row.prompt_profile == "otr_1940s_v1"
(L103). The finding was a misread; this test locks the
already-correct behavior so a future refactor cannot introduce
substring matching.
"""
from __future__ import annotations

import pytest

from nodes import _otr_creative_prompt_router as router
from nodes import _otr_model_catalog


def _curated_repo_ids() -> list[str]:
    return [m.repo_id for m in _otr_model_catalog.CURATED_LLM_MODELS]


def _curated_period_repo_ids() -> list[str]:
    return [
        m.repo_id
        for m in _otr_model_catalog.CURATED_LLM_MODELS
        if getattr(m, "prompt_profile", None) == "otr_1940s_v1"
    ]


def _curated_modern_repo_ids() -> list[str]:
    return [
        m.repo_id
        for m in _otr_model_catalog.CURATED_LLM_MODELS
        if getattr(m, "prompt_profile", None) != "otr_1940s_v1"
    ]


class TestExactMatchEnforcement:
    """Drift guard: router must reject non-curated repo_ids loud, not
    fall back to substring match against curated entries."""

    def test_curated_modern_dispatches_modern(self):
        for repo_id in _curated_modern_repo_ids():
            result = router.resolve_creative_system_prompt(repo_id, "outline")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_curated_period_dispatches_period(self):
        period_ids = _curated_period_repo_ids()
        if not period_ids:
            pytest.skip("no curated period rows in catalog")
        from nodes._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT
        for repo_id in period_ids:
            for phase in ("outline", "line_composer_system",
                          "polish_character", "polish_announcer"):
                result = router.resolve_creative_system_prompt(repo_id, phase)
                assert result is OTR_PERIOD_SYSTEM_PROMPT, (
                    f"period repo_id {repo_id!r} phase {phase!r} must "
                    f"return OTR_PERIOD_SYSTEM_PROMPT object identity, "
                    f"got distinct string of len {len(result)}"
                )

    def test_forked_substring_does_not_match(self):
        """The cold-read finding M1 hypothetical: a non-curated HF id with the
        talkie substring must NOT trigger the period profile. Exact-match dict
        lookup means a forked id MISSES the catalog -> it resolves to the MODERN
        default (it used to KeyError, which crashed the whole episode when the
        creative slot was a non-curated/remote model -- 2026-06-18). The M1
        invariant that matters is unchanged: a forked id never gets PERIOD."""
        period_ids = _curated_period_repo_ids()
        if not period_ids:
            pytest.skip("no curated period rows to fork")
        from nodes._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT
        forked = period_ids[0] + "-Forked"
        result = router.resolve_creative_system_prompt(forked, "outline")
        assert result is not OTR_PERIOD_SYSTEM_PROMPT

    def test_arbitrary_uncurated_id_returns_modern(self):
        """A non-curated repo_id -- a forged HF id, an empty string, or a remote
        slot handle (openrouter:slot-a/b) -- resolves to the MODERN prompt, never
        the period prompt and never a crash (the episode-aborting KeyError fix)."""
        from nodes._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT
        for forged in (
            "someone/random-finetune",
            "test/talkie-imposter",
            "mradermacher/Mistral-Nemo-Talkie-Forked",
            "openrouter:slot-a",
            "comfy:slot-b",
            "",
        ):
            result = router.resolve_creative_system_prompt(forged, "outline")
            assert isinstance(result, str) and result, (
                f"uncurated id {forged!r} must return a non-empty modern prompt"
            )
            assert result is not OTR_PERIOD_SYSTEM_PROMPT

    def test_unknown_phase_raises_valueerror(self):
        curated = _curated_repo_ids()
        if not curated:
            pytest.skip("no curated rows in catalog")
        with pytest.raises(ValueError):
            router.resolve_creative_system_prompt(curated[0], "nonexistent_phase")


class TestSourceShapeGuard:
    """AST-level guard: router source must NOT contain substring-match
    constructs (`in repo_id`, `repo_id.startswith`, `repo_id.endswith`,
    fnmatch, regex match against repo_id). The exact-match dict lookup
    is the only allowed dispatch surface."""

    def test_router_source_no_substring_dispatch(self):
        src = open(router.__file__, encoding="utf-8").read()
        # Each pattern is a substring construct that would dispatch
        # based on partial repo_id match. None should appear in the
        # router source.
        forbidden_patterns = (
            "if 'talkie' in",
            "if \"talkie\" in",
            "repo_id.startswith",
            "repo_id.endswith",
            "fnmatch",
            ".match(repo_id)",
            "re.search(",
        )
        for pat in forbidden_patterns:
            assert pat not in src, (
                f"router source contains substring-dispatch pattern "
                f"{pat!r}; only exact-match against CURATED_LLM_MODELS "
                f"by repo_id is allowed (Sprint E E4 M1 drift guard)"
            )

    def test_router_uses_curated_dict_lookup(self):
        """Positive assertion: router source must construct a
        {repo_id: row} dict from CURATED_LLM_MODELS and look up via
        plain dict access. Pins the implementation pattern so a future
        refactor doesn't accidentally bypass the catalog."""
        src = open(router.__file__, encoding="utf-8").read()
        assert "CURATED_LLM_MODELS" in src
        # The exact-match idiom: build {repo_id: row} from the curated catalog
        # and look up by repo_id. The lookup is `.get` (returns None -> modern
        # default) rather than a subscript, so a non-curated/remote slug never
        # KeyError-crashes an episode -- but it is STILL an exact dict lookup
        # keyed on the full repo_id (no substring dispatch).
        assert "m.repo_id" in src
        assert "rows[repo_id]" in src or "rows.get(repo_id)" in src

"""Best-of-N structural story-refine selector (2026-06-23).

Local-only (default), opt-in remote, deterministic best-of-N OUTLINE selector.
This file grows one chunk at a time:

  Chunk 1 -- OutlineRequest.diversity_hint + _build_user_prompt render
             (flag-off / empty-hint => byte-identical prompt).
  Chunk 2 -- score_outline pure scorer + StoryScore (raw-intent metrics).
  Chunk 3 -- select_best_outline selector + flag parse + provider gate.
  Chunk 4 -- optional remote best-of-N + fail-closed cost guard.

Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_outline as OUT  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a valid OutlineRequest (budget is required by the v2.0 contract).
# ---------------------------------------------------------------------------
def _req(diversity_hint=None):
    from nodes import _otr_episode_budget as EB
    budget = EB.compute_episode_budget(
        target_words=400,
        act_count=EB.default_act_count(400),
        include_act_breaks=True,
        num_characters=2,
    )
    kwargs = dict(
        news_seed="A deep-space signal is detected near a dying star.",
        style="hard sci-fi procedural",
        target_words=400,
        character_cast=("MALI", "MANFRED"),
        budget=budget,
    )
    if diversity_hint is not None:
        kwargs["diversity_hint"] = diversity_hint
    return OUT.OutlineRequest(**kwargs)


# ---------------------------------------------------------------------------
# Chunk 1 -- diversity_hint field + prompt render
# ---------------------------------------------------------------------------
class TestDiversityHint:
    def test_field_defaults_to_empty(self):
        assert _req().diversity_hint == ""

    def test_empty_hint_prompt_is_byte_identical_to_default(self):
        # Field defaulted vs explicitly "" must produce the SAME prompt, and
        # neither may carry the structural-variation overlay. This is the
        # byte-identical guarantee for candidate 0 / every non-selector call.
        defaulted = OUT._build_user_prompt(_req())
        explicit_empty = OUT._build_user_prompt(_req(""))
        assert defaulted == explicit_empty
        assert "Structural variation" not in defaulted

    def test_whitespace_only_hint_is_treated_as_empty(self):
        # The render strips; a whitespace-only hint must not perturb the prompt.
        assert OUT._build_user_prompt(_req("   ")) == OUT._build_user_prompt(_req(""))

    def test_nonempty_hint_is_rendered_verbatim(self):
        hint = "open on the personal stake, not the institutional threat"
        prompt = OUT._build_user_prompt(_req(hint))
        assert "Structural variation" in prompt
        assert hint in prompt

    def test_nonempty_hint_only_appends(self):
        # The hinted prompt is the empty prompt plus the appended overlay block:
        # every line of the empty prompt is still present, in order.
        empty = OUT._build_user_prompt(_req(""))
        hinted = OUT._build_user_prompt(_req("vary which stake opens the story"))
        assert empty != hinted
        assert hinted.startswith(empty.split("\nBuild a dramatic outline")[0])
        assert "vary which stake opens the story" in hinted

    def test_different_hints_produce_different_prompts(self):
        a = OUT._build_user_prompt(_req("open on the turn"))
        b = OUT._build_user_prompt(_req("open on the consequence"))
        assert a != b

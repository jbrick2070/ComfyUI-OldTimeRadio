"""A GGUF row's badge must price the context a user actually gets.

PBUG-20260829-17. A GGUF row costs weights + KV, and KV scales with n_ctx, so
one number only means something once you know the context it assumes. The badge
assumed the row's MAXIMUM: `unsloth/Qwen3-4B-Instruct-2507-GGUF` showed
"(7.9 GB)" -- 2.3 weights + 5.6 KV at n_ctx 8192 -- while the same model needs
5.2 GB at the default 4096, measured on a real 8 GB card.

Only 6 of 94 shipped profiles request 8192; 69 request 4096. The badge was
pricing the rarest case as the norm, in the one place a user chooses -- and an
8 GB owner reading "7.9" against an 8 GB card walks away from the smallest,
cheapest, Apache-2.0 writer in the list.

Same shape as PBUG-08 (pricing the ROW's maximum instead of the REQUEST),
fixed in the gate and left surviving in the label.
"""
from __future__ import annotations

import re

from nodes._otr_model_catalog import vram_badge_for
from nodes._otr_shared.llm_policy import LLMRuntimePolicy

QWEN_2507 = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


def test_the_2507_badge_is_not_the_max_context_number():
    badge = vram_badge_for(QWEN_2507)
    gb = float(re.search(r"\(([\d.]+) GB", badge).group(1))
    assert gb < 7.0, (
        "badge reads %r -- that is still the n_ctx=8192 price (2.3 weights + "
        "5.6 KV), not what a default run costs" % badge)
    # measured on an 8 GB card at n_ctx 4096: 5.23 GB from the backend preflight
    assert 4.5 <= gb <= 5.6, "badge %r is not near the measured 5.2 GB" % badge


def test_the_badge_names_the_context_it_assumed():
    """A context-dependent number that hides its context is a trap."""
    badge = vram_badge_for(QWEN_2507)
    assert "ctx" in badge, (
        "badge %r gives a KV-dependent number without saying which context it "
        "assumes; the reader cannot tell whether it applies to them" % badge)


def test_it_prices_the_policy_default_not_a_hardcoded_guess():
    default_ctx = LLMRuntimePolicy.__dataclass_fields__["gguf_n_ctx"].default
    badge = vram_badge_for(QWEN_2507)
    assert "@%dk" % (default_ctx // 1024) in badge, (
        "badge %r does not reflect the policy default gguf_n_ctx=%d -- if the "
        "default moves, the badge must move with it"
        % (badge, default_ctx))


def test_transformers_rows_are_unchanged():
    """Only GGUF rows carry a KV term; the rest must not grow a suffix."""
    for repo in ("google/gemma-4-E2B-it", "google/gemma-2-2b-it"):
        badge = vram_badge_for(repo)
        assert badge and "ctx" not in badge, (
            "%s badge %r gained a context suffix it has no use for"
            % (repo, badge))


def test_a_badge_never_raises_on_an_unknown_row():
    """A picker must render even for a row nothing can estimate."""
    assert vram_badge_for("definitely/not-a-real-model-xyz") == ""

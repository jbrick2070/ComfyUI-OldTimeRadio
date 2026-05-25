"""S32 B1 -- helper signatures accept paired `creative_fn` +
`technical_fn` kwargs.

The four helpers (`pick_style`, `lock_cast`, `compose_line`,
`build_news_briefs`) post-S32 B1 refactor accept the paired
generator-callable contract from the writer. B1 routes all
sub-passes through `creative_fn` (or `technical_fn` for
`build_news_briefs`); B2/B3/B4 land per-sub-pass dispatches.

Tests:
  1-4. Signature acceptance: each helper exposes `creative_fn` +
       `technical_fn` as keyword-only parameters.
  5-8. Internal routing: when called with distinct mock generators,
       each helper routes its calls to the expected slot.
       - `pick_style` (test 5) is per-sub-pass post-S32 B2:
         pass 1 (inventor) routes to `creative_fn`, pass 2
         (chooser) routes to `technical_fn`.
       - `lock_cast` and `compose_line` (tests 6-7) route their
         generation to `creative_fn` by default.
       - `build_news_briefs` (test 8) routes to `technical_fn`.
"""

from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 1-4: Signature acceptance
# ---------------------------------------------------------------------------


def test_pick_style_accepts_paired_generators():
    from nodes._otr_style_picker import pick_style

    sig = inspect.signature(pick_style)
    assert "creative_fn" in sig.parameters
    assert "technical_fn" in sig.parameters
    # Both must be keyword-only.
    assert sig.parameters["creative_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["technical_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    # Legacy positional `generate_fn` must be gone.
    assert "generate_fn" not in sig.parameters


def test_lock_cast_accepts_paired_generators():
    from nodes._otr_casting import lock_cast

    sig = inspect.signature(lock_cast)
    assert "creative_fn" in sig.parameters
    assert "technical_fn" in sig.parameters
    assert sig.parameters["creative_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["technical_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "generate_fn" not in sig.parameters


def test_compose_line_accepts_paired_generators():
    from nodes._otr_line_composer import compose_line

    sig = inspect.signature(compose_line)
    assert "creative_fn" in sig.parameters
    assert "technical_fn" in sig.parameters
    assert sig.parameters["creative_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["technical_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    # S32 B4 (no-widget drift, plan D1 architectural rejection): the
    # originally-projected `use_technical_critic` opt-in parameter was
    # dropped. Critic always routes to creative; the parameter must
    # NOT exist on the signature.
    assert "use_technical_critic" not in sig.parameters, (
        "`use_technical_critic` parameter was rejected at S32 B4 "
        "(no-widget rule). Re-introducing it brings back the dead "
        "opt-in surface the rule guards against."
    )
    assert "generate_fn" not in sig.parameters


def test_build_news_briefs_accepts_paired_generators():
    from nodes.news_interpreter import build_news_briefs

    sig = inspect.signature(build_news_briefs)
    assert "creative_fn" in sig.parameters
    assert "technical_fn" in sig.parameters
    assert sig.parameters["creative_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["technical_fn"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "generate_fn" not in sig.parameters


# ---------------------------------------------------------------------------
# 5-8: Internal routing
# ---------------------------------------------------------------------------


# Five distinct real snake_case style slugs that pass the inventor
# grammar (DESCRIPTOR_RE) AND the distinctness / mode-collapse rule
# (no two share more than one root word). Same proven fixture as
# `tests/test_pick_style_routing.py::_VALID_INVENTOR_RESPONSE`. Using
# valid data makes the inventor pass genuinely succeed so the chooser
# (pass 2) pass is actually exercised.
_VALID_INVENTOR_RESPONSE = (
    "1. closed_room_suspense\n"
    "2. noir_interrogation\n"
    "3. arctic_research_horror\n"
    "4. desert_outpost_thriller\n"
    "5. jungle_expedition_mystery\n"
)


def test_pick_style_routes_inventor_creative_and_chooser_technical():
    """S32 B2 per-sub-pass routing: pick_style dispatches each pass
    to its own slot. Pass 1 (inventor -- a narrative style-invention
    pass) routes to `creative_fn`; pass 2 (chooser -- a structured
    short-output index pick) routes to `technical_fn`.

    The mock `creative_fn` returns five distinct valid snake_case
    descriptors so the inventor pass genuinely succeeds -- only then
    does control reach the chooser, letting us confirm pass 2 lands
    on `technical_fn`. With both fns call-tracked independently, the
    assertions below pin that the B2 flip is live.
    """
    from nodes import _otr_style_picker as sp

    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append({"temperature": temperature, "max_new_tokens": max_new_tokens})
        # Pass 1 inventor: five distinct valid descriptors.
        return _VALID_INVENTOR_RESPONSE

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append({"temperature": temperature, "max_new_tokens": max_new_tokens})
        # Pass 2 chooser: returns one candidate STRING (not an index).
        return "closed_room_suspense"

    rng = random.Random(42)
    try:
        sp.pick_style(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            article_text="A test article about radio drama production.",
            seed_pool=["s1", "s2", "s3", "s4", "s5", "s6"],
            rng=rng,
            model_id="test/model",
        )
    except Exception:
        # The picker may fail validation downstream of our mocks;
        # that's OK -- the routing assertions below are what we test.
        pass

    assert len(creative_calls) >= 1, (
        "Pass 1 (inventor) must call creative_fn at least once "
        f"(S32 B2 per-sub-pass routing); got {len(creative_calls)} "
        f"creative call(s)."
    )
    assert len(technical_calls) >= 1, (
        "Pass 2 (chooser) must call technical_fn at least once "
        f"after the S32 B2 flip; got {len(technical_calls)} "
        f"technical call(s). The inventor pass must succeed first "
        f"so the chooser is actually reached."
    )


def test_lock_cast_internally_uses_creative_fn_default():
    """lock_cast generation (fresh attempts) routes through
    creative_fn. After S32 B3, the repair attempt flips to
    technical_fn (D2 schema-validation slot, single-attempt
    fail-fast). This test asserts ONLY that creative_fn carries
    the generation -- the B3-specific routing assertions live in
    `tests/test_lock_cast_routing.py`.
    """
    from nodes import _otr_casting as cast

    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append({"temperature": temperature})
        return "not-valid-json"

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append({"temperature": temperature})
        return "not-valid-json"

    try:
        cast.lock_cast(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            num_characters=1,
            news_seed="seed",
            style="noir",
            rng=random.Random(7),
            # force_lemmy=False: the LEMMY ~11% cameo rolls on OS
            # entropy, not the seeded rng (BUG-LOCAL-260), so an unset
            # force_lemmy makes this routing test flaky -- a LEMMY hit
            # changes cast assembly and the generation call pattern.
            # Pinning it OFF keeps the creative/technical-slot assertion
            # below deterministic.
            force_lemmy=False,
            # Single attempt only -- skips the B3 repair branch so
            # this test stays focused on creative-slot generation.
            max_attempts_per_call=1,
        )
    except Exception:
        # Casting will fail validation; we only test routing.
        pass

    assert len(creative_calls) >= 1, (
        "creative_fn must be called at least once for generation"
    )
    assert len(technical_calls) == 0, (
        f"With max_attempts_per_call=1 the B3 repair branch is "
        f"never reached, so technical_fn should not fire; got "
        f"{len(technical_calls)} call(s)."
    )


def test_compose_line_internally_uses_creative_fn_default():
    """At B1 compose_line routes all sub-passes through creative_fn.
    With `use_technical_critic=False` (default), critic stays on
    creative. B4 lands the opt-in dispatch."""
    from nodes import _otr_line_composer as lc

    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens, **kw):
        creative_calls.append({"temperature": temperature})
        return "test dialogue"

    def technical_fn(messages, *, temperature, max_new_tokens, **kw):
        technical_calls.append({"temperature": temperature})
        return "should_not_be_called"

    # Build a minimal LineRequest. Inspect the dataclass for required
    # fields and pass placeholders.
    req_cls = lc.LineRequest
    req_fields = {f.name for f in (
        req_cls.__dataclass_fields__.values() if hasattr(req_cls, "__dataclass_fields__")
        else []
    )}
    # Build with the known-required surface; let other fields default.
    req_kwargs = {
        "speaker": "ALICE",
        "scene_header": "INT. STUDIO -- NIGHT",
        "canon_header": "INT. STUDIO -- NIGHT",
        "target_words": 8,
        "beat_traits": "calm",
        "allowed_roster": ["ALICE", "BOB"],
        "is_announcer": False,
    }
    # Filter to only fields the dataclass actually accepts.
    req_kwargs = {k: v for k, v in req_kwargs.items() if not req_fields or k in req_fields}
    try:
        req = req_cls(**req_kwargs)
    except TypeError:
        # If the LineRequest needs different fields, skip the runtime
        # part and assert only via signature inspection.
        pytest.skip("LineRequest signature changed; this test needs update")

    try:
        lc.compose_line(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            req=req,
        )
    except Exception:
        pass

    assert len(creative_calls) >= 1, (
        "creative_fn must be called at least once for the composer pass"
    )
    assert len(technical_calls) == 0, (
        f"technical_fn must NEVER be called by compose_line "
        f"(S32 B4 no-widget decision; critic always routes to "
        f"creative). Got {len(technical_calls)} call(s)."
    )


def test_build_news_briefs_internally_uses_technical_fn():
    """build_news_briefs is the ONE helper of the 4 that routes to
    technical_fn by default. All V0-V3 sub-passes are structured-
    output GBNF emits; the canonical slot for that is technical."""
    from nodes import news_interpreter as ni

    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens, **kw):
        creative_calls.append({"temperature": temperature})
        return "should_not_be_called"

    def technical_fn(messages, *, temperature, max_new_tokens, **kw):
        technical_calls.append({"temperature": temperature})
        return "not-valid-json"

    try:
        ni.build_news_briefs(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            full_text="A short news article about radio.",
            headline="Radio Returns",
            summary="A summary.",
            outlet="The Outlet",
            pub_date="2026-05-14",
            style="noir",
            seed=42,
        )
    except Exception:
        pass

    assert len(technical_calls) >= 1, (
        "technical_fn must be called at least once for "
        "build_news_briefs at B1 (all sub-passes route here)"
    )
    assert len(creative_calls) == 0, (
        f"creative_fn must NOT be called at B1 for "
        f"build_news_briefs (accepted for contract uniformity but "
        f"unused); got {len(creative_calls)} call(s)."
    )


# ---------------------------------------------------------------------------
# S32 B5: build_news_briefs all retries (V0-V3) on technical_fn
# ---------------------------------------------------------------------------


def test_build_news_briefs_all_retries_use_technical():
    """Plan B5 verification: V0-V3 sub-passes (V0 emit + V1/V2/V3
    retries on validation failure) ALL route to technical_fn.
    Drive `build_news_briefs` with always-malformed responses so
    every retry tier fires; assert zero creative-side calls
    across the full retry stack.
    """
    from nodes import news_interpreter as ni

    creative_calls: list[float] = []
    technical_calls: list[float] = []

    def creative_fn(messages, *, temperature, max_new_tokens, **kw):
        creative_calls.append(temperature)
        return "should_not_be_called"

    def technical_fn(messages, *, temperature, max_new_tokens, **kw):
        technical_calls.append(temperature)
        # Always-malformed so every retry tier fires.
        return "not-valid-json"

    try:
        ni.build_news_briefs(
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            full_text="A short news article about radio.",
            headline="Radio Returns",
            summary="A summary.",
            outlet="The Outlet",
            pub_date="2026-05-14",
            style="noir",
            seed=42,
            max_attempts=3,  # V0 + V1 retry + V2 repair attempt
        )
    except Exception:
        pass

    assert len(technical_calls) >= 1, (
        "All V0-V3 sub-passes must route to technical_fn. Got 0."
    )
    assert len(creative_calls) == 0, (
        f"creative_fn must NOT be called by build_news_briefs at "
        f"any retry tier; got {len(creative_calls)} call(s). "
        f"This would break the structured-output slot routing."
    )


# ---------------------------------------------------------------------------
# S32 B5: outline retry stays creative (D3)
# ---------------------------------------------------------------------------


def test_outline_retry_uses_creative():
    """Plan B5 + D3: `_otr_outline.generate_outline` retry stays
    creative. Schema validation is pure pydantic (no LLM); the retry
    is CONTENT regeneration (the outline got rejected for structural
    reasons -- length, missing field -- and we want a fresh creative
    take, not a technical-slot re-check). No paired-fn refactor
    needed for the outline.

    Structural assertion: the outline module exposes a
    `generate_outline` function that takes a single `generate_fn`
    positional arg (not paired). This pins the D3 decision -- if
    a future change introduces paired generators to the outline,
    THIS test breaks first.
    """
    from nodes import _otr_outline as outline
    import inspect

    sig = inspect.signature(outline.generate_outline)
    params = list(sig.parameters.values())
    assert len(params) >= 1, "generate_outline must have at least one arg"
    first = params[0]
    assert first.name in ("generate_fn", "fn"), (
        f"generate_outline's first arg should be a single "
        f"generate_fn (D3: outline stays single-fn creative). "
        f"Got {first.name!r}."
    )
    # Paired-contract surface must NOT have been added.
    assert "creative_fn" not in sig.parameters, (
        "D3: outline retry stays creative. The outline must NOT "
        "have been refactored to paired (creative_fn, technical_fn) "
        "kwargs -- that would imply per-sub-pass dispatch, which "
        "the architectural decision rejects."
    )
    assert "technical_fn" not in sig.parameters

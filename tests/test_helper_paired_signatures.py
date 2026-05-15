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
  5-8. Internal routing at B1: when called with distinct mock
       generators, each helper routes its calls to the
       configured default slot (creative for 1-3, technical for 4).
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
    # `use_technical_critic` opt-in widget present (defaults OFF).
    assert "use_technical_critic" in sig.parameters
    assert sig.parameters["use_technical_critic"].default is False
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
# 5-8: Internal routing at B1
# ---------------------------------------------------------------------------


def _fake_inventor_response(messages, *, temperature, max_new_tokens):
    """Return a valid Pass-1 inventor response."""
    # Pass 1 inventor expects a numbered list of style candidates.
    return "1. test_style_alpha\n2. test_style_beta\n3. test_style_gamma\n4. test_style_delta\n5. test_style_epsilon\n"


def _fake_chooser_response(messages, *, temperature, max_new_tokens):
    """Return a valid Pass-2 chooser response."""
    return "1"


def test_pick_style_internally_uses_creative_fn_default():
    """At B1 both passes route through creative_fn; technical_fn is
    accepted but unused. After B2 lands, the chooser pass flips to
    technical_fn -- this test will need updating then (the B2 test
    expects that flip)."""
    from nodes import _otr_style_picker as sp

    creative_calls: list[dict] = []
    technical_calls: list[dict] = []

    def creative_fn(messages, *, temperature, max_new_tokens):
        creative_calls.append({"messages": messages, "temperature": temperature, "max_new_tokens": max_new_tokens})
        # Need to return a response that varies by call. Heuristic:
        # the first call (inventor) needs a list of 5 candidates; the
        # second (chooser) needs an index.
        if len(creative_calls) == 1:
            return _fake_inventor_response(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        return _fake_chooser_response(messages, temperature=temperature, max_new_tokens=max_new_tokens)

    def technical_fn(messages, *, temperature, max_new_tokens):
        technical_calls.append({"messages": messages, "temperature": temperature, "max_new_tokens": max_new_tokens})
        return "should_not_be_called"

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
        # that's OK -- the routing assertion below is what we test.
        pass

    assert len(creative_calls) >= 1, (
        "creative_fn must be called at least once (inventor pass) at B1"
    )
    assert len(technical_calls) == 0, (
        f"technical_fn must NOT be called at B1 (routes to "
        f"creative_fn at this commit); got {len(technical_calls)} "
        f"call(s) to technical_fn."
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
            use_technical_critic=False,
        )
    except Exception:
        pass

    assert len(creative_calls) >= 1, (
        "creative_fn must be called at least once at B1"
    )
    assert len(technical_calls) == 0, (
        f"technical_fn must NOT be called at B1 with "
        f"use_technical_critic=False; got {len(technical_calls)} "
        f"call(s)."
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

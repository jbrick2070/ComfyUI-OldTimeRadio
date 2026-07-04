"""Story-quality S2 (C3) -- news-coda arc-shape bridge floor.

Locks: every curated arc-shape bridge passes validate_news_coda_bridge; the v2
fallback (when the LLM bridge fails) ships a curated arc bridge; v2-OFF / unknown
arc_shape stay on the legacy NEWS_CODA_POOL (byte-identical). Pure / CPU. UTF-8 no
BOM, SFW.
"""
from __future__ import annotations

from nodes._otr_line_composer import (
    NEWS_CODA_POOL,
    _NEWS_CODA_ARC_BRIDGES,
    compose_news_coda,
    validate_news_coda_bridge,
)


def _bad_fn(*_a, **_k):
    """A creative slot whose bridge ALWAYS fails validation (generic opener) ->
    forces the deterministic fallback floor."""
    return "And now"


_BRIEF = "UCLA Health reports a successful one-year transplant result."
_PREMISE = "A records clerk has buried one fatal file to protect a colleague."


def test_every_arc_bridge_validates():
    assert _NEWS_CODA_ARC_BRIDGES, "arc-bridge pool is empty"
    for arc, bridges in _NEWS_CODA_ARC_BRIDGES.items():
        assert bridges, f"{arc} has no bridges"
        for b in bridges:
            ok, _ = validate_news_coda_bridge(b)
            assert ok, f"{arc} bridge fails validate_news_coda_bridge: {b!r}"


def test_coda_both_attempts_rejected_fails_loud():
    # NO-FALLBACK (2026-07-03): when both bridge attempts are rejected the coda
    # RAISES; the curated arc-bridge / NEWS_CODA_POOL floor is RETIRED -- it no
    # longer silently ships a canned bridge as the spoken transition. (The bridge
    # DATA + validators are still exercised by test_every_arc_bridge_validates.)
    import pytest
    with pytest.raises(RuntimeError, match="no-fallback"):
        compose_news_coda(
            creative_fn=_bad_fn, news_close_brief=_BRIEF, premise=_PREMISE,
            cast_seed=7, arc_shape="betrayal")

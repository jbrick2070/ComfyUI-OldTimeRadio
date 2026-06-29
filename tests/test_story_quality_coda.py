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


def test_v2_fallback_ships_curated_arc_bridge():
    res = compose_news_coda(
        creative_fn=_bad_fn, news_close_brief=_BRIEF, premise=_PREMISE,
        cast_seed=7, story_quality_v2_enabled=True, arc_shape="betrayal")
    assert "news_coda_arc_bridge" in res.compose_flags
    # the shipped text opens with one of the curated betrayal bridges (+ ":")
    starts = [b for b in _NEWS_CODA_ARC_BRIDGES["betrayal"]
              if res.text.startswith(b)]
    assert starts, f"coda did not open with a betrayal arc bridge: {res.text!r}"
    assert _BRIEF.split()[0] in res.text  # the real fact is appended


def test_v2_off_uses_legacy_pool_byte_identical():
    res = compose_news_coda(
        creative_fn=_bad_fn, news_close_brief=_BRIEF, premise=_PREMISE,
        cast_seed=7, story_quality_v2_enabled=False, arc_shape="betrayal")
    assert "news_coda_arc_bridge" not in res.compose_flags
    assert any(res.text.startswith(p) for p in NEWS_CODA_POOL)


def test_unknown_arc_shape_falls_back_to_legacy():
    res = compose_news_coda(
        creative_fn=_bad_fn, news_close_brief=_BRIEF, premise=_PREMISE,
        cast_seed=7, story_quality_v2_enabled=True, arc_shape="not_a_real_arc")
    assert "news_coda_arc_bridge" not in res.compose_flags
    assert any(res.text.startswith(p) for p in NEWS_CODA_POOL)


def test_arc_selection_deterministic_by_cast_seed():
    kw = dict(creative_fn=_bad_fn, news_close_brief=_BRIEF, premise=_PREMISE,
              story_quality_v2_enabled=True, arc_shape="heist")
    a = compose_news_coda(cast_seed=3, **kw).text
    b = compose_news_coda(cast_seed=3, **kw).text
    assert a == b  # same seed -> same curated bridge

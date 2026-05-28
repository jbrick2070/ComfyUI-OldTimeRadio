"""tests/test_writers_room_resolver.py

Auto-resolve cast_names + news_seed from the in-flight ledger when
the operator left the Director / Story Room / Extract widgets
empty. Lets the writers' room run pure from scratch.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from nodes._otr_writers_room_resolver import (
    resolve_cast_names,
    resolve_news_seed,
)


def _ledger_with_cast(names=None, news_seed=None):
    cast = [{"char_id": "announcer", "name": "ANNOUNCER"}]
    for i, name in enumerate(names or [], start=1):
        cast.append({"char_id": f"c0{i}", "name": name})
    meta = {}
    if news_seed is not None:
        meta["news_seed"] = news_seed
    return {"cast": cast, "meta": meta, "lines": []}


class TestResolveCastNames:
    def test_widget_value_wins(self):
        ledger = _ledger_with_cast(["FROM_LEDGER"])
        out = resolve_cast_names("TYPED, NAMES", fallback_ledger=ledger)
        assert out == ["TYPED", "NAMES"]

    def test_widget_empty_falls_back_to_ledger(self):
        ledger = _ledger_with_cast(["REN BLACK", "DR MAEVE COLE"])
        out = resolve_cast_names("", fallback_ledger=ledger)
        assert out == ["REN BLACK", "DR MAEVE COLE"]

    def test_announcer_excluded(self):
        ledger = _ledger_with_cast(["PAM PIERCE"])
        # ANNOUNCER is always in the cast list but should be filtered.
        out = resolve_cast_names("", fallback_ledger=ledger)
        assert "ANNOUNCER" not in out
        assert out == ["PAM PIERCE"]

    def test_typed_announcer_also_excluded(self):
        out = resolve_cast_names(
            "ANNOUNCER, REN BLACK", fallback_ledger=None,
        )
        assert out == ["REN BLACK"]

    def test_empty_widget_and_no_ledger_returns_empty(self):
        out = resolve_cast_names("", fallback_ledger=None)
        assert out == []

    def test_dedup_in_ledger(self):
        ledger = {
            "cast": [
                {"char_id": "c01", "name": "REN BLACK"},
                {"char_id": "c02", "name": "REN BLACK"},  # dup
            ],
            "meta": {},
        }
        out = resolve_cast_names("", fallback_ledger=ledger)
        assert out == ["REN BLACK"]

    def test_falsy_ledger_returns_empty(self):
        assert resolve_cast_names("", fallback_ledger={}) == []
        assert resolve_cast_names("", fallback_ledger=None) == []


class TestResolveNewsSeed:
    def test_widget_value_wins(self):
        led = _ledger_with_cast(
            news_seed={"headline": "Ledger headline"},
        )
        out = resolve_news_seed("typed seed", fallback_ledger=led)
        assert out == "typed seed"

    def test_widget_empty_falls_back_to_dict_headline(self):
        led = _ledger_with_cast(
            news_seed={"headline": "NASA Develops Sensor",
                       "source": "NASA News",
                       "body": "With peak wildfire season..."},
        )
        out = resolve_news_seed("", fallback_ledger=led)
        assert "NASA Develops Sensor" in out
        assert "wildfire" in out

    def test_widget_empty_falls_back_to_string_seed(self):
        # Pre-BUG-277 shape: meta.news_seed is a plain string.
        led = _ledger_with_cast(news_seed="Plain string headline")
        out = resolve_news_seed("", fallback_ledger=led)
        assert out == "Plain string headline"

    def test_widget_empty_and_no_ledger_returns_empty(self):
        out = resolve_news_seed("", fallback_ledger=None)
        assert out == ""

    def test_dict_with_only_headline(self):
        led = _ledger_with_cast(news_seed={"headline": "Only headline"})
        out = resolve_news_seed("", fallback_ledger=led)
        assert out == "Only headline"

    def test_widget_whitespace_treated_as_empty(self):
        led = _ledger_with_cast(news_seed={"headline": "From ledger"})
        out = resolve_news_seed("   ", fallback_ledger=led)
        assert out == "From ledger"


class TestNodeWiring:
    """Pin: the three nodes import the resolver."""

    def test_director_imports_resolver(self):
        from pathlib import Path
        src = Path(
            __import__(
                "nodes.OTR_DirectorBrief",
                fromlist=["OTR_DirectorBrief"],
            ).__file__
        ).read_text(encoding="utf-8")
        assert "_otr_writers_room_resolver" in src
        assert "resolve_cast_names" in src
        assert "resolve_news_seed" in src

    def test_story_room_imports_resolver(self):
        from pathlib import Path
        src = Path(
            __import__(
                "nodes.OTR_StoryRoom",
                fromlist=["OTR_StoryRoom"],
            ).__file__
        ).read_text(encoding="utf-8")
        assert "_otr_writers_room_resolver" in src

    def test_extract_imports_resolver(self):
        from pathlib import Path
        src = Path(
            __import__(
                "nodes.OTR_StoryRoomExtract",
                fromlist=["OTR_StoryRoomExtract"],
            ).__file__
        ).read_text(encoding="utf-8")
        assert "_otr_writers_room_resolver" in src

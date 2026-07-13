"""Regression coverage for strict Sci-Fi RSS source admission."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from nodes import _otr_scifi_codex as codex
from nodes import _otr_scifi_gemini as gemini
from nodes import _otr_scifi_sonnet as sonnet
from nodes import production_ledger
from nodes import story_orchestrator as orchestrator
from nodes._otr_rss_source_contract import meets_v4_rss_source_floor


_TOKENS = [f"w{index:03d}" for index in range(12)]


def _body(word_count: int, *, terminal: str = "") -> str:
    """Build a body with four-character words and all twelve unique tokens."""
    words = [_TOKENS[index % len(_TOKENS)] for index in range(word_count)]
    return " ".join(words) + terminal


def _source_payload(body: str) -> dict[str, str]:
    return {
        "headline": "Test science report",
        "summary": "Test source summary.",
        "full_text": body,
        "source": "Test Wire",
        "date": "2026-07-12",
        "link": "https://example.invalid/source",
        "seed_text": body,
    }


def _hyphenated_body() -> str:
    return " ".join(
        f"{_TOKENS[(index * 2) % 12]}-{_TOKENS[(index * 2 + 1) % 12]}"
        for index in range(40)
    ) + "!"


class _InlineFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _InlineExecutor:
    """Synchronous executor that records the actual body-fetch attempt list."""

    map_inputs: list[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *args: Any) -> bool:
        del args
        return False

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> _InlineFuture:
        return _InlineFuture(fn(*args, **kwargs))

    def map(self, fn: Callable[[dict[str, Any]], Any], iterable: Any) -> list[Any]:
        items = list(iterable)
        type(self).map_inputs.append(items)
        return [fn(item) for item in items]


class _Ledger:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def save(self) -> None:
        pass


def _entry(headline: str, rss_full: str, *, link: str = "") -> dict[str, Any]:
    return {
        "title": headline,
        "content": [{"value": rss_full}],
        "summary": "Brief summary.",
        "link": link,
        "published": "2026-07-12",
    }


def _install_fake_feed_run(
    monkeypatch: pytest.MonkeyPatch,
    feeds: dict[str, list[dict[str, Any]]],
) -> None:
    """Patch the real fetch function's network and persistence seams only."""
    _InlineExecutor.map_inputs = []
    feedparser = SimpleNamespace(
        parse=lambda url: SimpleNamespace(
            entries=feeds[url], feed={"title": f"Feed {url}"}
        )
    )
    monkeypatch.setitem(sys.modules, "feedparser", feedparser)
    monkeypatch.setattr(orchestrator, "SCIENCE_NEWS_FEEDS", list(feeds))
    monkeypatch.setattr(orchestrator, "ThreadPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(orchestrator, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(orchestrator.random, "shuffle", lambda sequence: None)
    monkeypatch.setattr(orchestrator, "_load_news_history", lambda: set())
    monkeypatch.setattr(orchestrator, "_record_news_usage", lambda **kwargs: None)
    monkeypatch.setattr(production_ledger, "get_ledger", _Ledger)


def test_strict_fetch_prioritizes_later_qualified_inline_rss_without_widening_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thin = _body(80)
    eligible = _body(80, terminal="!")
    feeds = {
        "feed://one": [_entry(f"thin-{index}", thin) for index in range(6)],
        "feed://two": [
            *[_entry(f"thin-{index}", thin) for index in range(6, 10)],
            _entry("eligible-first", eligible),
            _entry("eligible-second", eligible),
        ],
    }
    _install_fake_feed_run(monkeypatch, feeds)

    chosen = orchestrator._fetch_science_news(
        model_id=None, require_science_floor=True
    )[0]

    attempted_headlines = [
        candidate["headline"] for candidate in _InlineExecutor.map_inputs[-1]
    ]
    assert attempted_headlines == [
        "eligible-first",
        "eligible-second",
        *[f"thin-{index}" for index in range(8)],
    ]
    assert chosen["headline"] == "eligible-first"
    assert chosen["_body_source"] == "rss_full"


def test_strict_thin_inline_rss_scrapes_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thin = _body(80)
    scraped = _body(80, terminal="!")
    link = "https://example.invalid/strict-thin"
    _install_fake_feed_run(
        monkeypatch,
        {"feed://one": [_entry("thin-inline", thin, link=link)]},
    )
    calls: list[tuple[str, int]] = []

    def _scrape(url: str, timeout: int) -> str:
        calls.append((url, timeout))
        return scraped

    monkeypatch.setattr(orchestrator, "_fetch_full_article", _scrape)

    chosen = orchestrator._fetch_science_news(
        model_id=None, require_science_floor=True
    )[0]

    assert calls == [(link, 5)]
    assert chosen["full_text"] == scraped
    assert chosen["_body_source"] == "url_scrape"


def test_legacy_science_news_keeps_399_character_inline_rss_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thin = _body(80)
    link = "https://example.invalid/legacy-thin"
    _install_fake_feed_run(
        monkeypatch,
        {"feed://one": [_entry("legacy-thin", thin, link=link)]},
    )

    def _unexpected_scrape(url: str, timeout: int) -> str:
        raise AssertionError(f"legacy inline RSS should not scrape {url} ({timeout})")

    monkeypatch.setattr(orchestrator, "_fetch_full_article", _unexpected_scrape)

    chosen = orchestrator._fetch_science_news(
        model_id=None, require_science_floor=False
    )[0]

    assert chosen["full_text"] == thin
    assert chosen["_body_source"] == "rss_full"


@pytest.mark.parametrize("value", [None, 400])
def test_shared_rss_floor_rejects_non_strings(value: object) -> None:
    assert not meets_v4_rss_source_floor(value)


def test_shared_rss_floor_preserves_selector_regex_semantics() -> None:
    assert len(_body(80, terminal="!")) == 400
    assert meets_v4_rss_source_floor(_body(80, terminal="!"))
    assert not meets_v4_rss_source_floor(_body(79, terminal="!!!!!!"))
    assert not meets_v4_rss_source_floor(" ".join(["same"] * 80) + "!")

    punctuation_only_token = _body(79) + " ...!!"
    assert len(punctuation_only_token) == 400
    assert len(punctuation_only_token.split()) == 80
    assert not meets_v4_rss_source_floor(punctuation_only_token)

    hyphenated = _hyphenated_body()
    assert len(hyphenated) == 400
    assert len(hyphenated.split()) == 40
    assert meets_v4_rss_source_floor(hyphenated)


def test_legacy_floor_wrapper_forwards_threshold_kwargs() -> None:
    eligible = _body(80, terminal="!")
    assert orchestrator._science_body_meets_v4_floor(eligible)
    assert not orchestrator._science_body_meets_v4_floor(eligible, min_chars=401)
    assert not orchestrator._science_body_meets_v4_floor(eligible, min_words=81)


@pytest.mark.parametrize(
    ("validator", "thin_error"),
    [
        (codex.validate_payload_envelope, codex.CodexPayloadThinError),
        (gemini.validate_gemini_payload, gemini.SciFiGeminiPayloadThinError),
        (sonnet.validate_sonnet_payload, sonnet.SonnetThinPayloadError),
    ],
)
def test_v4_envelopes_share_the_rss_floor(
    validator: Callable[[dict[str, str], dict[str, Any]], Any],
    thin_error: type[Exception],
) -> None:
    resolved = {"seed_source": "rss_fetch", "target_words": 30}
    punctuation_only_token = _body(79) + " ...!!"

    for rejected in (
        _body(80),
        _body(79, terminal="!!!!!!"),
        " ".join(["same"] * 80) + "!",
        punctuation_only_token,
    ):
        with pytest.raises(thin_error):
            validator(_source_payload(rejected), resolved)

    for accepted in (_body(80, terminal="!"), _hyphenated_body()):
        validator(_source_payload(accepted), resolved)

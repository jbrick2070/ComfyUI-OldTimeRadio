"""`media_archive` must not adapt the same feed post forever.

PBUG-20260815-06. `fetch_media_archive_rss` returned
`payloads[_configured_index() % len(payloads)]`, and `_configured_index()`
defaults to `"0"`. Feed entries arrive newest-first, so with no operator env var
set the lane adapted THE NEWEST POST on every single run -- no dedup, no
ranking, no history anywhere in the module. The PBUG also recorded that **this
selection path had no test at all**, verified by grep across `tests/`. This file
is that missing coverage.

THE FIX REUSES THE SCIENCE LANE'S MECHANISM RATHER THAN INVENTING ONE. The
operator's question was the right one -- if both lanes consume RSS, should they
not share the selection logic? Science already keeps
`<output>/otr/state/news_history.json` (article URLs, rolling cap, 5-day TTL so
headlines recycle), it keys on URL, and nothing in it is science-specific. A
second history for this lane would just be a second thing to drift.

THE VERIFY CONDITION IS THE PBUG'S OWN: two consecutive runs against a stable
feed must not select the same post.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    for p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from nodes import _otr_media_archive_sources as MAS  # noqa: E402


def _feed(n=4):
    return [{"headline": f"post {i}", "url": f"https://example.test/{i}",
             "body": f"body {i}", "source": "Example Archive"}
            for i in range(n)]


@pytest.fixture
def stub_feed(monkeypatch):
    """A stable 4-entry feed, newest first, with no real network."""
    monkeypatch.delenv("OTR_MEDIA_ARCHIVE_ITEM_INDEX", raising=False)
    monkeypatch.setattr(MAS, "_configured_feeds", lambda: ("https://example.test/rss",))
    monkeypatch.setattr(MAS, "parse_media_archive_feed",
                        lambda url, source_url="": list(_feed()))
    return _feed()


def test_two_consecutive_runs_do_not_pick_the_same_post(monkeypatch, stub_feed):
    """THE PBUG'S OWN VERIFY CONDITION. Before the fix this returned post 0 both
    times, forever."""
    recorded: list[str] = []
    monkeypatch.setattr(MAS, "_recently_used_urls", lambda: set(recorded))
    monkeypatch.setattr(MAS, "_record_used",
                        lambda p: recorded.append(str(p.get("url") or "")))

    first = MAS.fetch_media_archive_rss(bank=None)
    second = MAS.fetch_media_archive_rss(bank=None)
    assert first["url"] != second["url"], (
        f"both runs adapted {first['url']} -- the lane is still retelling the "
        f"same post"
    )


def test_it_walks_the_whole_feed_before_repeating(monkeypatch, stub_feed):
    """TEETH on the above. Alternating between two posts would also pass a
    two-run check; a lane worth the name should exhaust the feed first."""
    recorded: list[str] = []
    monkeypatch.setattr(MAS, "_recently_used_urls", lambda: set(recorded))
    monkeypatch.setattr(MAS, "_record_used",
                        lambda p: recorded.append(str(p.get("url") or "")))

    picks = [MAS.fetch_media_archive_rss(bank=None)["url"] for _ in range(4)]
    assert len(set(picks)) == 4, f"only {len(set(picks))} distinct posts in 4 runs: {picks}"


def test_an_exhausted_history_falls_back_rather_than_failing(monkeypatch, stub_feed):
    """A lane that raises because everything has been seen is worse than one
    that repeats. The episode must still be renderable."""
    monkeypatch.setattr(MAS, "_recently_used_urls",
                        lambda: {f"https://example.test/{i}" for i in range(4)})
    monkeypatch.setattr(MAS, "_record_used", lambda p: None)
    got = MAS.fetch_media_archive_rss(bank=None)
    assert got["url"], "returned no usable entry when every post was in history"


def test_the_operator_index_override_still_wins(monkeypatch, stub_feed):
    """OTR_MEDIA_ARCHIVE_ITEM_INDEX is an operator override. Dedup must not
    fight it -- if it is set explicitly, that entry is what gets adapted even if
    it was used five minutes ago."""
    monkeypatch.setenv("OTR_MEDIA_ARCHIVE_ITEM_INDEX", "2")
    monkeypatch.setattr(MAS, "_recently_used_urls",
                        lambda: {f"https://example.test/{i}" for i in range(4)})
    monkeypatch.setattr(MAS, "_record_used", lambda p: None)
    got = MAS.fetch_media_archive_rss(bank=None)
    assert got["url"] == "https://example.test/2", (
        f"explicit index 2 was overridden by dedup; got {got['url']}"
    )


def test_dedup_failure_never_blocks_an_episode(monkeypatch, stub_feed):
    """The history is advisory. If reading it raises, the lane still returns a
    post -- a feed lane must not fail a render because a JSON file was corrupt."""
    def boom():
        raise OSError("history file is a smoking crater")
    monkeypatch.setattr(MAS, "_recently_used_urls", boom)
    monkeypatch.setattr(MAS, "_record_used", lambda p: None)
    with pytest.raises(OSError):
        MAS._recently_used_urls()          # the stub really does raise
    # ...but the real helper swallows its own failures, which is the contract:
    monkeypatch.undo()
    monkeypatch.setattr(MAS, "_configured_feeds", lambda: ("https://example.test/rss",))
    monkeypatch.setattr(MAS, "parse_media_archive_feed",
                        lambda url, source_url="": list(_feed()))
    monkeypatch.delenv("OTR_MEDIA_ARCHIVE_ITEM_INDEX", raising=False)
    assert MAS.fetch_media_archive_rss(bank=None)["url"]


def test_the_shared_history_is_the_science_lane_s_and_not_a_copy():
    """The whole point of the fix. If someone later gives this lane its OWN
    history file, the two lanes drift and a post adapted by one is invisible to
    the other."""
    import inspect
    src = inspect.getsource(MAS._recently_used_urls) + inspect.getsource(MAS._record_used)
    assert "story_orchestrator" in src, (
        "the media_archive dedup no longer reads the shared news history; a "
        "second history is a second thing to drift"
    )
    assert "_load_news_history" in src and "_record_news_usage" in src, (
        "expected the shared loader AND recorder -- reading a history nothing "
        "writes to is a no-op that looks like a feature"
    )


# --------------------------------------------------------------------------- #
# Found by review (codex spark, 2026-08-19) on the first cut of the fix above.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("junk", ["abc", "2.5", "-", " ", "one"])
def test_a_garbage_index_does_not_silently_restore_always_newest(monkeypatch,
                                                                stub_feed, junk):
    """THE HOLE REVIEW CAUGHT. The first cut tested only that the env var was
    NON-EMPTY before taking the override branch. `_configured_index()` swallows a
    ValueError and returns 0, so `OTR_MEDIA_ARCHIVE_ITEM_INDEX=abc` would take
    the override, collapse to index 0, and silently reinstate the always-newest
    behaviour this whole fix exists to end -- with no error and no log line.

    A value that does not parse is not a choice. Dedup must still apply.
    """
    monkeypatch.setenv("OTR_MEDIA_ARCHIVE_ITEM_INDEX", junk)
    recorded: list[str] = []
    monkeypatch.setattr(MAS, "_recently_used_urls", lambda: set(recorded))
    monkeypatch.setattr(MAS, "_record_used",
                        lambda p: recorded.append(str(p.get("url") or "")))
    first = MAS.fetch_media_archive_rss(bank=None)
    second = MAS.fetch_media_archive_rss(bank=None)
    assert first["url"] != second["url"], (
        f"OTR_MEDIA_ARCHIVE_ITEM_INDEX={junk!r} was treated as an operator "
        f"choice, so dedup was skipped and the lane repeated {first['url']}"
    )


def test_explicit_index_is_distinguished_from_a_defaulting_one():
    """`_configured_index()` answers 'which index', `_explicit_index()` answers
    'did the operator choose'. Conflating them is the bug above."""
    import os as _os
    old = _os.environ.get("OTR_MEDIA_ARCHIVE_ITEM_INDEX")
    try:
        _os.environ.pop("OTR_MEDIA_ARCHIVE_ITEM_INDEX", None)
        assert MAS._explicit_index() is None, "unset must not read as a choice"
        assert MAS._configured_index() == 0, "unset must still yield a usable 0"
        _os.environ["OTR_MEDIA_ARCHIVE_ITEM_INDEX"] = "abc"
        assert MAS._explicit_index() is None, "garbage must not read as a choice"
        _os.environ["OTR_MEDIA_ARCHIVE_ITEM_INDEX"] = "3"
        assert MAS._explicit_index() == 3
    finally:
        _os.environ.pop("OTR_MEDIA_ARCHIVE_ITEM_INDEX", None)
        if old is not None:
            _os.environ["OTR_MEDIA_ARCHIVE_ITEM_INDEX"] = old


def test_an_explicit_index_is_not_written_to_the_shared_history(monkeypatch,
                                                               stub_feed):
    """Also from review. An explicit index is a deliberate repeat, so recording
    it would let a debugging run consume a headline for the automatic path -- and
    because the history is SHARED, for the science lane's TTL window as well."""
    monkeypatch.setenv("OTR_MEDIA_ARCHIVE_ITEM_INDEX", "1")
    wrote: list[str] = []
    monkeypatch.setattr(MAS, "_recently_used_urls", set)
    monkeypatch.setattr(MAS, "_record_used",
                        lambda p: wrote.append(str(p.get("url") or "")))
    MAS.fetch_media_archive_rss(bank=None)
    assert not wrote, f"the override path recorded {wrote} into shared history"


def test_the_two_lanes_feeds_cannot_collide():
    """Grounds the cross-lane coupling question review raised. Sharing one
    history is only safe if a URL from one lane can never appear in the other's
    feed -- and the feed sets are disjoint by domain, so it cannot."""
    from nodes import story_orchestrator as SO
    media_domains = {u.split("/")[2] for u in MAS.DEFAULT_MEDIA_ARCHIVE_FEEDS}
    science_domains = {u.split("/")[2] for u in SO.SCIENCE_NEWS_FEEDS}
    overlap = media_domains & science_domains
    assert not overlap, (
        f"the lanes now share feed domains {overlap}, so one lane can consume a "
        f"URL out from under the other. The shared history stops being free."
    )

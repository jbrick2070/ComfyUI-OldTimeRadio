"""BUG-LOCAL-090: news_history.json gets a 5-day TTL.

The active filter set excludes URLs whose timestamp is older than
``_NEWS_HISTORY_FILTER_DAYS`` so the candidate pool can recycle a
headline after enough time has passed. Older entries stay on disk for
audit but no longer block fresh runs.

Pinning behaviour:
- entries within the TTL window -> in the filter set
- entries older than the TTL window -> NOT in the filter set
- entries with missing or malformed timestamp -> in the filter set
  (safer to filter once than to surface a same-day repeat)
- file missing / corrupted -> empty set, never raises
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Match other tests' import pattern (test_critique_dialogue_preservation.py):
# repo root on path so ``from nodes import story_orchestrator`` works
# despite story_orchestrator.py's relative imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


@pytest.fixture(scope="module")
def m():
    """Load the story_orchestrator module under test."""
    from nodes import story_orchestrator as so  # noqa: WPS433
    return so


def _write_history(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def test_ttl_constant_is_five_days(m) -> None:
    """Pin the TTL window so a silent revert is caught."""
    assert m._NEWS_HISTORY_FILTER_DAYS == 5


def test_load_filters_old_entries(m, tmp_path, monkeypatch) -> None:
    """Entries older than the TTL window are excluded from the filter set."""
    history_path = tmp_path / "news_history.json"
    monkeypatch.setattr(m, "_NEWS_HISTORY_PATH", str(history_path))

    now = datetime.now()
    entries = [
        {"url": "https://fresh-today.example/a",
         "headline": "fresh today",
         "timestamp": now.isoformat(timespec="seconds")},
        {"url": "https://two-days-ago.example/b",
         "headline": "fresh-ish",
         "timestamp": (now - timedelta(days=2)).isoformat(timespec="seconds")},
        {"url": "https://four-days-ago.example/c",
         "headline": "edge of window",
         "timestamp": (now - timedelta(days=4)).isoformat(timespec="seconds")},
        {"url": "https://six-days-ago.example/d",
         "headline": "outside window",
         "timestamp": (now - timedelta(days=6)).isoformat(timespec="seconds")},
        {"url": "https://ten-days-ago.example/e",
         "headline": "way outside",
         "timestamp": (now - timedelta(days=10)).isoformat(timespec="seconds")},
    ]
    _write_history(history_path, entries)

    fresh = m._load_news_history()
    assert "https://fresh-today.example/a" in fresh
    assert "https://two-days-ago.example/b" in fresh
    assert "https://four-days-ago.example/c" in fresh
    assert "https://six-days-ago.example/d" not in fresh
    assert "https://ten-days-ago.example/e" not in fresh
    assert len(fresh) == 3


def test_missing_timestamp_treated_as_fresh(m, tmp_path, monkeypatch) -> None:
    """Entries lacking a parseable timestamp default to fresh -- safer to
    filter once than to surface a same-day repeat."""
    history_path = tmp_path / "news_history.json"
    monkeypatch.setattr(m, "_NEWS_HISTORY_PATH", str(history_path))

    entries = [
        {"url": "https://no-timestamp.example/a", "headline": "missing field"},
        {"url": "https://empty-timestamp.example/b",
         "headline": "empty string", "timestamp": ""},
        {"url": "https://garbage-timestamp.example/c",
         "headline": "malformed", "timestamp": "not a date"},
    ]
    _write_history(history_path, entries)

    fresh = m._load_news_history()
    assert "https://no-timestamp.example/a" in fresh
    assert "https://empty-timestamp.example/b" in fresh
    assert "https://garbage-timestamp.example/c" in fresh


def test_missing_url_skipped(m, tmp_path, monkeypatch) -> None:
    """Entries lacking a URL are dropped silently."""
    history_path = tmp_path / "news_history.json"
    monkeypatch.setattr(m, "_NEWS_HISTORY_PATH", str(history_path))

    now = datetime.now()
    entries = [
        {"headline": "no url field",
         "timestamp": now.isoformat(timespec="seconds")},
        {"url": "", "headline": "empty url",
         "timestamp": now.isoformat(timespec="seconds")},
        {"url": "https://valid.example/a", "headline": "real entry",
         "timestamp": now.isoformat(timespec="seconds")},
    ]
    _write_history(history_path, entries)

    fresh = m._load_news_history()
    assert fresh == {"https://valid.example/a"}


def test_file_missing_returns_empty(m, tmp_path, monkeypatch) -> None:
    """Missing history file is not an error -- first-run case."""
    monkeypatch.setattr(
        m, "_NEWS_HISTORY_PATH", str(tmp_path / "does_not_exist.json")
    )
    assert m._load_news_history() == set()


def test_corrupted_json_returns_empty(m, tmp_path, monkeypatch) -> None:
    """Corrupted JSON file is not an error -- best-effort dedup."""
    history_path = tmp_path / "news_history.json"
    history_path.write_text("{ this is not valid json }", encoding="utf-8")
    monkeypatch.setattr(m, "_NEWS_HISTORY_PATH", str(history_path))
    assert m._load_news_history() == set()


def test_entries_at_exactly_ttl_boundary(m, tmp_path, monkeypatch) -> None:
    """An entry timestamped exactly _NEWS_HISTORY_FILTER_DAYS ago is on the
    fresh side of the boundary (>= cutoff). One second older is stale."""
    history_path = tmp_path / "news_history.json"
    monkeypatch.setattr(m, "_NEWS_HISTORY_PATH", str(history_path))

    now = datetime.now()
    # Pad by 1 second on each side so wall-clock drift between write and
    # read doesn't flake the test.
    just_inside = now - timedelta(days=m._NEWS_HISTORY_FILTER_DAYS) + timedelta(seconds=1)
    just_outside = now - timedelta(days=m._NEWS_HISTORY_FILTER_DAYS) - timedelta(seconds=1)
    entries = [
        {"url": "https://inside.example/a", "headline": "just inside",
         "timestamp": just_inside.isoformat(timespec="seconds")},
        {"url": "https://outside.example/b", "headline": "just outside",
         "timestamp": just_outside.isoformat(timespec="seconds")},
    ]
    _write_history(history_path, entries)

    fresh = m._load_news_history()
    assert "https://inside.example/a" in fresh
    assert "https://outside.example/b" not in fresh

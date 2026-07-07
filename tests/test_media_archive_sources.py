"""Media archive source fetcher fixture tests (QA must-fix test gap).

Pure/CPU. Proves:
(1) RSS fixture parses into exact SOURCE_PAYLOAD_KEYS and passes
    validate_source_payload.
(2) Atom fixture parses into exact keys and passes validation.
(3) Strict mode raises MediaArchiveSourceError on missing title/body/link.
(4) Source hash is deterministic.
(5) Module never imports writer or news_interpreter.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from nodes._otr_media_archive_sources import (
    MediaArchiveSourceError,
    _clean_text,
    _payload_from_entry,
    _source_hash,
    parse_media_archive_feed,
)
from nodes._otr_source_payload import SOURCE_PAYLOAD_KEYS, validate_source_payload

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "nodes" / "_otr_media_archive_sources.py"

# ---------------------------------------------------------------------------
# minimal synthetic feed fixtures
# ---------------------------------------------------------------------------

_RSS_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Library of Congress Now See Hear!</title>
    <link>https://blogs.loc.gov/now-see-hear/</link>
    <item>
      <title>Restoring a Lost 1948 Radio Broadcast</title>
      <link>https://blogs.loc.gov/now-see-hear/2026/01/lost-1948</link>
      <description>Archivists recently discovered a forgotten acetate disc
      containing what appears to be a 1948 live radio broadcast. The disc
      was found during a routine inventory of uncatalogued materials.</description>
      <pubDate>Mon, 06 Jan 2026 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

_ATOM_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Film Preservation Blog</title>
  <entry>
    <title>Nitrate Film Vault Cataloging Project</title>
    <id>https://www.filmpreservation.org/blog/2026/02/nitrate-vault</id>
    <link href="https://www.filmpreservation.org/blog/2026/02/nitrate-vault" />
    <summary>A team of volunteers is cataloging a vault of nitrate film reels
    discovered in a former studio warehouse. The project aims to identify
    and prioritize reels for digitization.</summary>
    <updated>2026-02-15T10:00:00Z</updated>
  </entry>
</feed>
"""


# ---------------------------------------------------------------------------
# (1) RSS fixture -> exact SOURCE_PAYLOAD_KEYS
# ---------------------------------------------------------------------------

class TestRSSFixture:
    def test_rss_parses_into_source_payload_keys(self):
        results = parse_media_archive_feed(_RSS_FIXTURE, source_url="test://rss")
        assert len(results) >= 1
        payload = results[0]
        assert set(payload.keys()) == SOURCE_PAYLOAD_KEYS

    def test_rss_passes_validate_source_payload(self):
        results = parse_media_archive_feed(_RSS_FIXTURE, source_url="test://rss")
        payload = results[0]
        out = validate_source_payload(payload, origin="test_rss")
        assert out == payload and out is not payload  # shallow copy

    def test_rss_headline_matches(self):
        results = parse_media_archive_feed(_RSS_FIXTURE, source_url="test://rss")
        assert "1948" in results[0]["headline"]

    def test_rss_all_values_are_str(self):
        results = parse_media_archive_feed(_RSS_FIXTURE, source_url="test://rss")
        for key, val in results[0].items():
            assert isinstance(val, str), f"key {key!r} is {type(val).__name__}"


# ---------------------------------------------------------------------------
# (2) Atom fixture -> exact SOURCE_PAYLOAD_KEYS
# ---------------------------------------------------------------------------

class TestAtomFixture:
    def test_atom_parses_into_source_payload_keys(self):
        results = parse_media_archive_feed(_ATOM_FIXTURE, source_url="test://atom")
        assert len(results) >= 1
        assert set(results[0].keys()) == SOURCE_PAYLOAD_KEYS

    def test_atom_passes_validate_source_payload(self):
        results = parse_media_archive_feed(_ATOM_FIXTURE, source_url="test://atom")
        out = validate_source_payload(results[0], origin="test_atom")
        assert "Nitrate" in out["headline"]


# ---------------------------------------------------------------------------
# (3) strict mode raises on missing title/body/link
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_missing_title_raises_in_strict(self):
        bad_rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><description>body here</description>
<link>https://example.com</link></item></channel></rss>
"""
        with pytest.raises(MediaArchiveSourceError, match="title"):
            parse_media_archive_feed(bad_rss, source_url="test://bad",
                                     strict=True)

    def test_missing_body_raises_in_strict(self):
        bad_rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Has Title</title>
<link>https://example.com</link></item></channel></rss>
"""
        with pytest.raises(MediaArchiveSourceError, match="title/body/link"):
            parse_media_archive_feed(bad_rss, source_url="test://bad",
                                     strict=True)

    def test_non_strict_skips_bad_entries(self):
        bad_rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Has Title</title>
<link>https://example.com</link></item></channel></rss>
"""
        with pytest.raises(MediaArchiveSourceError, match="no usable"):
            parse_media_archive_feed(bad_rss, source_url="test://bad",
                                     strict=False)


# ---------------------------------------------------------------------------
# (4) source hash determinism
# ---------------------------------------------------------------------------

class TestSourceHash:
    def test_deterministic(self):
        a = _source_hash("hello", "world")
        b = _source_hash("hello", "world")
        assert a == b
        assert len(a) == 16

    def test_different_inputs_differ(self):
        a = _source_hash("hello", "world")
        b = _source_hash("world", "hello")
        assert a != b


# ---------------------------------------------------------------------------
# (5) import posture: no writer/news_interpreter imports
# ---------------------------------------------------------------------------

class TestImportPosture:
    def test_no_writer_import(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [a.name for a in getattr(node, "names", [])]
                combined = module + " " + " ".join(names)
                assert "OTR_LedgerScriptWriter" not in combined
                assert "news_interpreter" not in combined


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_html(self):
        assert _clean_text("<b>hello</b> <i>world</i>") == "hello world"

    def test_truncates_long(self):
        result = _clean_text("a " * 5000, max_chars=20)
        assert len(result) <= 20

"""The fast pop-up page (``/direct``): served by the same server under the
same rules, and its script asks for nothing the markup does not have. It
posts the Stream Deck's four dial fields plus the menu workflow, so the
deck's contract pins what it may send."""
from __future__ import annotations

import pathlib
import re
import threading
from html.parser import HTMLParser

import pytest

from radio_play_studio import server as server_module
from radio_play_studio.runner import Studio
from radio_play_studio.server import make_server

from test_rps_runner import FakeClient, _good_job  # noqa: E402 - sibling test module
from test_rps_server import _request  # noqa: E402 - one request over real HTTP

PAGE = pathlib.Path(__file__).resolve().parents[1] / "radio_play_studio" / "static" / "direct.html"


@pytest.fixture
def app(settings, schemas, monkeypatch):
    """The real server on a free port, without its GPU watcher: that thread
    loops for good after shutdown and would ask the fixture's comfy.invalid
    to /free (Codex QA, 2026-10-30; the deck contract does the same)."""
    monkeypatch.setattr(server_module, "_watch_gpu", lambda runs: None)
    studio = Studio(settings, client_factory=lambda url: FakeClient(_good_job()), fetch=lambda url: schemas)
    httpd = make_server(studio, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], settings
    httpd.shutdown()
    httpd.server_close()
SETTINGS_TS = (pathlib.Path(__file__).resolve().parents[1] / "streamdeck" / "src" / "types"
               / "settings.ts").read_text(encoding="utf-8")
HTML = PAGE.read_text(encoding="utf-8")


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list = []

    def handle_starttag(self, tag, attrs) -> None:
        for name, value in attrs:
            if name == "id":
                self.ids.append(value)


def _static_ids() -> list:
    parser = _Ids()
    parser.feed(HTML.split("<script>", 1)[0])
    return parser.ids


def test_direct_is_served_as_html(app):
    port, _settings = app
    status, headers, data = _request(port, "GET", "/direct")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>Radio Play Studio - Direct</title>" in data


def test_direct_keeps_the_loopback_rule(app):
    port, _settings = app
    status, _headers, data = _request(port, "GET", "/direct", headers={"Host": "studio.example:8765"})
    assert status == 403
    assert b"localhost" in data


def test_every_static_id_is_unique():
    ids = _static_ids()
    assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)


def test_script_asks_only_for_ids_the_markup_has():
    script = HTML.split("<script>", 1)[1]
    wanted = set(re.findall(r'\$\("([a-z-]+)"\)', script))
    missing = wanted - set(_static_ids())
    assert not missing, sorted(missing)


def test_the_page_posts_the_decks_dial_fields():
    """What the page sends as choices is exactly what the deck's dials send
    (plus the menu workflow), so one contract covers both."""
    deck = re.findall(r'"([a-z_]+)"', re.search(r"DIAL_FIELDS = \[([^\]]+)\]", SETTINGS_TS).group(1))
    page = re.findall(r'"([a-z_]+)"', re.search(r'const DIAL_FIELDS = \[([^\]]+)\]', HTML).group(1))
    assert page == deck


def test_the_page_skips_my_story_like_the_deck():
    assert 'MY_STORY_BANK = "my_story"' in HTML
    assert "c === MY_STORY_BANK" in HTML

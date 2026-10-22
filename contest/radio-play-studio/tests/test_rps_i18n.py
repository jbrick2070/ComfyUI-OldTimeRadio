"""The page in eight languages: every language file says what English says,
English is the form's own words, and the app hands the files out."""
from __future__ import annotations

import http.client
import json
import re

import pytest

from radio_play_studio import server as rps_server
from radio_play_studio.form import FIELDS
from radio_play_studio.server import I18N_DIR, available_languages

from test_rps_server import _request, app  # noqa: F401 - the live-server fixture

PAGE = I18N_DIR.parent / "index.html"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
CODES = sorted(p.stem for p in I18N_DIR.glob("*.json"))


def _load(code):
    return json.loads((I18N_DIR / f"{code}.json").read_text(encoding="utf-8"))


def _slug(text):
    """The page's own slug (index.html): lower case, every run of other characters one _."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


EN = _load("en")


def test_the_page_speaks_the_languages_the_pack_writes_episodes_in():
    assert CODES == sorted(["en", "es", "pt", "it", "fr", "hi", "ja", "zh"])


@pytest.mark.parametrize("code", CODES)
def test_each_language_says_everything_english_says_and_nothing_else(code):
    strings = _load(code)
    assert sorted(set(EN) - set(strings)) == [], f"{code}.json is missing these"
    assert sorted(set(strings) - set(EN)) == [], f"{code}.json has keys English does not"
    assert all(isinstance(v, str) and v.strip() for v in strings.values())


@pytest.mark.parametrize("code", CODES)
def test_each_language_keeps_every_placeholder(code):
    """A translation that drops {error} hides the error; one that adds {x} prints it raw."""
    strings = _load(code)
    wrong = [k for k in EN if sorted(PLACEHOLDER.findall(strings[k])) != sorted(PLACEHOLDER.findall(EN[k]))]
    assert wrong == []


def test_english_is_the_forms_own_words():
    """form.py stays the source of what a field says; en.json must say the same,
    so changing a label there shows up here and in every translation."""
    for f in FIELDS:
        assert EN[f"field.{f.key}.label"] == f.label
        assert EN[f"field.{f.key}.help"] == f.help
        assert EN[f"group.{_slug(f.group)}"] == f.group


def test_every_key_the_page_names_is_in_english():
    page = PAGE.read_text(encoding="utf-8")
    used = set(re.findall(r'"([a-z_]+(?:\.[a-z0-9_]+)+)"', page))
    assert used, "the page names no keys at all"
    assert sorted(k for k in used if k not in EN) == []


def test_a_missing_string_never_shows_its_key():
    """With no strings loaded at all (en.json unreachable), an element keeps the
    English it was created with: t() returns null, and paint() leaves it alone."""
    page = PAGE.read_text(encoding="utf-8")
    assert "if (s == null) return null;" in page
    assert 'if (fallback == null) fallback = attr ? el.getAttribute(attr) : el.textContent;' in page


def test_a_failed_language_fetch_keeps_the_language_and_the_saved_choice():
    page = PAGE.read_text(encoding="utf-8")
    body = page[page.index("async function setLang"):page.index("async function initLang")]
    assert 'catch (err) { $("lang").value = lang; return; }' in body
    assert body.index("localStorage.setItem") > body.index("if (mine !== langRequest) return;")


def test_the_slug_matches_the_page():
    """The page builds choice keys with this exact expression; a drift would
    quietly show every translated choice under its raw name."""
    page = PAGE.read_text(encoding="utf-8")
    assert 'replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "")' in page
    assert _slug("roll (any eligible bank)") == "roll_any_eligible_bank"


def test_languages_are_listed_english_first_by_their_own_names(app):  # noqa: F811
    port, _ = app
    status, _, body = _request(port, "GET", "/api/languages")
    langs = json.loads(body)["languages"]
    assert status == 200 and langs[0] == {"code": "en", "name": "English"}
    assert {lang["code"]: lang["name"] for lang in langs}["ja"] == "日本語"
    assert sorted(lang["code"] for lang in langs) == CODES


def test_a_language_file_is_served_as_utf8_json(app):  # noqa: F811
    port, _ = app
    status, headers, body = _request(port, "GET", "/i18n/ja.json")
    assert status == 200 and "charset=utf-8" in headers["Content-Type"]
    assert json.loads(body.decode("utf-8")) == _load("ja")


@pytest.mark.parametrize("path", ["/i18n/xx.json", "/i18n/EN.json", "/i18n/../server.py",
                                  "/i18n/%2e%2e/server.py", "/i18n/en.json.bak", "/i18n/"])
def test_anything_else_under_i18n_is_a_404(app, path):  # noqa: F811
    port, _ = app
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    assert resp.status == 404 and b"_name" not in body


def test_a_file_without_a_name_is_not_offered(tmp_path, monkeypatch):
    (tmp_path / "en.json").write_text('{"_name": "English"}', encoding="utf-8")
    (tmp_path / "xx.json").write_text('{"a": "b"}', encoding="utf-8")
    (tmp_path / "yy.json").write_text("not json", encoding="utf-8")
    (tmp_path / "zz.json").write_text('["a list, not an object"]', encoding="utf-8")
    monkeypatch.setattr(rps_server, "I18N_DIR", tmp_path)
    assert available_languages() == [{"code": "en", "name": "English"}]

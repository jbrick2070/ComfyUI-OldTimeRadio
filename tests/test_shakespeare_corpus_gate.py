"""The acceptance gate's fetch, pin and verdict wiring. No network.

Spec 2026-09-18: *"nothing enters the pipeline on the strength of this
document"*. v1 scored 28 cells READY without opening the pages, so what this
file guards is the difference between a claim and a measurement.
UTF-8 no BOM, CPU only.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes import _otr_verbatim_corpus as C  # noqa: E402


def _gate():
    spec = importlib.util.spec_from_file_location(
        "otr_shakespeare_corpus_gate",
        REPO / "scripts" / "otr_shakespeare_corpus_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _gate()


# --------------------------------------------------------------------------- #
# pinning the bytes
# --------------------------------------------------------------------------- #


def test_a_mediawiki_revision_number_wins_outright():
    body = 'var x = {"wgCurRevisionId":12345678,"wgTitle":"Macbeth"};'
    assert GATE.revision_id("https://fr.wikisource.org/wiki/Macbeth", body,
                            {"ETag": "abc"}) == "rev:12345678"


def test_an_explicit_oldid_is_next():
    assert GATE.revision_id("https://x/wiki/Y?oldid=999", "no config here",
                            {"ETag": "abc"}) == "oldid:999"


def test_a_transport_validator_is_next():
    assert GATE.revision_id("https://x/y", "plain text",
                            {"ETag": '"W/abc123"'}) == "etag:W/abc123"
    assert GATE.revision_id("https://x/y", "plain text",
                            {"Last-Modified": "Wed, 01 Jan 2020 00:00:00 GMT"}) \
        .startswith("last-modified:")


def test_a_host_with_no_version_is_pinned_by_the_bytes_we_read():
    """Project Gutenberg and archive.org offer nothing; the digest of what we
    actually read is the edition we vendored."""
    pin = GATE.revision_id("https://www.gutenberg.org/ebooks/25667.txt.utf-8",
                           "HAMLET: Ser ou nao ser", {})
    assert pin.startswith("sha256:") and len(pin) == len("sha256:") + 16
    other = GATE.revision_id("https://x/y", "a different text", {})
    assert other != pin


def test_nothing_at_all_pins_nothing():
    assert GATE.revision_id("https://x/y", "", {}) == ""


# --------------------------------------------------------------------------- #
# what a lead is worth before anyone opens it
# --------------------------------------------------------------------------- #


def _lead(**over):
    lead = {
        "iso": "fr", "play": "macbeth", "scene": "1.3",
        "translator": "Hugo", "translator_death_date": 1873,
        "translation_first_published": 1865,
        "transcription_license": "CC BY-SA 4.0",
        "url": "https://fr.wikisource.org/wiki/Macbeth_(trad._Hugo)",
    }
    lead.update(over)
    return lead


def test_without_fetch_a_clean_lead_is_partial_and_says_why(tmp_path):
    r = GATE.assess_lead(_lead(), cache_dir=str(tmp_path), do_fetch=False)
    assert r.verdict == C.PARTIAL
    assert r.reasons == ["not fetched -- rights checked only"]


def test_without_fetch_rights_still_block(tmp_path):
    r = GATE.assess_lead(_lead(translator_death_date=1968,
                                translation_first_published=1955),
                         cache_dir=str(tmp_path), do_fetch=False)
    assert r.verdict == C.BLOCKED


def test_a_lead_with_no_url_is_empty_not_a_crash(tmp_path):
    """The leads file deliberately carries rows whose page is not located yet
    -- the gate reports that rather than dying on it."""
    r = GATE.assess_lead(_lead(url=""), cache_dir=str(tmp_path), do_fetch=True)
    assert r.verdict == C.EMPTY
    assert "locate the page first" in r.reasons[0]


def test_a_lead_with_no_url_but_bad_rights_is_blocked_not_empty(tmp_path):
    r = GATE.assess_lead(_lead(url="", translator_death_date=1968,
                                translation_first_published=1955),
                         cache_dir=str(tmp_path), do_fetch=True)
    assert r.verdict == C.BLOCKED


def test_tracking_parameters_are_stripped_from_the_lead_url(tmp_path):
    r = GATE.assess_lead(_lead(url="https://x/y?utm_source=gemini"),
                         cache_dir=str(tmp_path), do_fetch=False)
    assert r.url == "https://x/y"


# --------------------------------------------------------------------------- #
# a throttled host is not an empty page
# --------------------------------------------------------------------------- #


def test_a_transport_failure_reads_as_retry_not_as_an_empty_page(tmp_path, monkeypatch):
    """Measured 2026-09-18: two Gutenberg ids failed mid-run and both fetched
    cleanly seconds later. Reporting that as 'no bytes' would cut a good lead."""
    monkeypatch.setattr(GATE, "fetch", lambda *a, **k: {
        "status": 0, "final_url": "https://x/y", "text": "", "bytes": 0,
        "encoding": "", "headers": {}, "error": "URLError: timed out"})
    r = GATE.assess_lead(_lead(), cache_dir=str(tmp_path), do_fetch=True)
    assert r.verdict == C.EMPTY
    assert r.transport_error == "URLError: timed out"
    joined = " ".join(r.reasons)
    assert "re-run before judging" in joined
    assert "no bytes" not in joined


def test_a_real_empty_page_still_says_no_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(GATE, "fetch", lambda *a, **k: {
        "status": 200, "final_url": "https://x/y", "text": "", "bytes": 0,
        "encoding": "utf-8", "headers": {}})
    r = GATE.assess_lead(_lead(), cache_dir=str(tmp_path), do_fetch=True)
    assert "no bytes" in " ".join(r.reasons)
    assert not r.transport_error


def test_a_fetched_scene_that_measures_well_is_ready(tmp_path, monkeypatch):
    # Two speakers ALTERNATING: twelve unique names would be a cast list.
    page = ('var c = {"wgCurRevisionId":42};\nACTE I, SCENE 3.\n'
            + "".join("ANTIPHOLUS: une replique assez longue pour compter.\n"
                      "BALTHASAR: une reponse tout aussi longue ici.\n"
                      for _ in range(6)))
    monkeypatch.setattr(GATE, "fetch", lambda *a, **k: {
        "status": 200, "final_url": "https://fr.wikisource.org/wiki/Macbeth",
        "text": page, "bytes": len(page), "encoding": "utf-8", "headers": {}})
    r = GATE.assess_lead(_lead(), cache_dir=str(tmp_path), do_fetch=True)
    assert r.verdict == C.READY, r.reasons
    assert r.revision_id == "rev:42"
    assert r.speaker_labels >= C.MIN_SPEAKER_LABELS


def test_a_pending_transcription_is_never_ready(tmp_path, monkeypatch):
    page = ('var c = {"wgCurRevisionId":42};\nACTE I, SCENE 3.\nA transcribir\n'
            + "".join("PERSONNAGE %d: una replica.\n" % i for i in range(12)))
    monkeypatch.setattr(GATE, "fetch", lambda *a, **k: {
        "status": 200, "final_url": "https://x/y", "text": page,
        "bytes": len(page), "encoding": "utf-8", "headers": {}})
    r = GATE.assess_lead(_lead(), cache_dir=str(tmp_path), do_fetch=True)
    assert r.verdict != C.READY
    assert any("transcription pending" in x for x in r.reasons)


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #


def test_the_cache_is_keyed_by_url_and_survives_a_second_read(tmp_path):
    url = "https://example.invalid/scene"
    name = GATE._cache_name(url)
    (tmp_path / name).write_text("MACBETH: cached text\n", encoding="utf-8")
    got = GATE.fetch(url, str(tmp_path))
    assert got["cached"] is True and "cached text" in got["text"]
    assert GATE._cache_name(url) == name
    assert GATE._cache_name(url + "x") != name


# --------------------------------------------------------------------------- #
# the shipped leads file
# --------------------------------------------------------------------------- #


LEADS = REPO / "config" / "source_banks" / "shakespeare" / "translations" / "leads.json"


def test_the_shipped_leads_file_parses_and_carries_the_rights_fields():
    leads = GATE.load_leads(str(LEADS))
    assert leads
    for lead in leads:
        for key in ("iso", "play", "scene", "translator",
                    "translator_death_date", "translation_first_published",
                    "transcription_license"):
            assert key in lead, (lead.get("iso"), lead.get("play"), key)


def test_no_shipped_lead_carries_a_tracking_parameter():
    for lead in GATE.load_leads(str(LEADS)):
        url = lead.get("url") or ""
        assert "utm_" not in url, url
        assert url == C.strip_tracking(url)


def test_the_blocked_languages_are_blocked_for_the_reason_the_spec_gives():
    """zh has no source that clears both tests -- Zhu fails the US clock,
    Tian Han fails life+70. That is the finding, not a gap in the leads."""
    leads = {(l["iso"], l["translator"]): l for l in GATE.load_leads(str(LEADS))}
    zhu = next(v for (iso, t), v in leads.items() if iso == "zh" and "Zhu" in t)
    tian = next(v for (iso, t), v in leads.items() if iso == "zh" and "Tian" in t)
    assert not C.clears_publication_anywhere(
        zhu["translation_first_published"], zhu["translator_death_date"])
    assert not C.clears_publication_anywhere(
        tian["translation_first_published"], tian["translator_death_date"])

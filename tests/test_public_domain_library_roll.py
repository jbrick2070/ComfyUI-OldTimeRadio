"""The public_domain bank is a DECK, not a shelf with one book on it.

This bank shipped with a single vendored work and no selector: a blank
source_ref fell straight through to defaults.source_ref, so every episode it
ever produced was H.G. Wells' The Time Machine. With one source on disk a
broken picker and a working one are indistinguishable, which is why these tests
exist only now that the library is full.
"""
from __future__ import annotations

import collections
import random

import pytest

from nodes import _otr_public_domain_sources as PD
from nodes import _otr_story_routing as SR


def _bank():
    return SR.require_runnable_bank("public_domain")


def _manifest():
    defaults = _bank().defaults or {}
    path = PD._resolve_repo_relative_path(
        str(defaults.get("manifest_path", "")), key="manifest_path")
    return PD.load_public_domain_manifest(path)


def test_the_library_is_actually_stocked():
    """The defect was an empty shelf, so the shelf is what gets asserted."""
    sources = _manifest()["sources"]
    assert len(sources) >= 20, f"only {len(sources)} sources vendored"
    authors = {s["author"] for s in sources}
    assert len(authors) >= 20, f"only {len(authors)} distinct authors"
    # The point of the bank is variety; one author must not dominate it.
    counts = collections.Counter(s["author"] for s in sources)
    assert counts.most_common(1)[0][1] <= 3


def test_the_authentic_wells_survives_a_manifest_regeneration():
    """Slot zero was fetched and hash-verified after the bank was found
    shipping an INVENTED Wells fixture. Regenerating the manifest from the
    curation table must never silently drop it."""
    ids = {s["source_id"] for s in _manifest()["sources"]}
    assert "time_machine" in ids


def test_a_blank_ref_deals_from_the_whole_library():
    bank = _bank()
    drawn = {PD.fetch_public_domain_source(bank=bank).payload["headline"]
             for _ in range(200)}
    # 200 draws over ~28 units: seeing only a handful means it is not dealing.
    assert len(drawn) >= 15, f"only {len(drawn)} distinct units in 200 draws"


def test_the_draw_is_uniform_enough_to_be_a_real_roll():
    """A selector that always returns the first unit would still pass a
    distinctness check if it drifted; this pins the SHAPE of the draw."""
    manifest = _manifest()
    refs = [f"{s['source_id']}:{u['unit_id']}"
            for s in manifest["sources"] for u in s["units"]]
    rng = random.Random(4242)          # deterministic; this is a shape test
    seen = collections.Counter(
        PD.select_public_domain_unit_ref(manifest, rng=rng)
        for _ in range(len(refs) * 40))
    assert set(seen) == set(refs), "some units are unreachable by the roll"
    expected = 40
    assert max(seen.values()) < expected * 3


def test_an_explicit_ref_still_pins():
    bank = _bank()
    res = PD.fetch_public_domain_source(bank=bank, source_ref="monkeys_paw:main")
    assert res.payload["headline"].startswith("The Monkey's Paw")
    again = PD.fetch_public_domain_source(bank=bank, source_ref="monkeys_paw:main")
    assert again.payload["headline"] == res.payload["headline"]


def test_selection_mode_fixed_honours_the_pinned_default():
    class _Bank:
        source_bank_id = "public_domain"
        defaults = dict(_bank().defaults or {})

    _Bank.defaults["selection_mode"] = "fixed"
    _Bank.defaults["source_ref"] = "dracula:main"
    res = PD.fetch_public_domain_source(bank=_Bank())
    assert res.payload["headline"].startswith("Dracula")


def test_an_unsupported_selection_mode_fails_loud():
    class _Bank:
        source_bank_id = "public_domain"
        defaults = dict(_bank().defaults or {})

    _Bank.defaults["selection_mode"] = "whatever"
    with pytest.raises(PD.PublicDomainManifestError, match="selection_mode"):
        PD.fetch_public_domain_source(bank=_Bank())


def test_fixed_mode_without_a_ref_refuses_rather_than_guessing():
    class _Bank:
        source_bank_id = "public_domain"
        defaults = dict(_bank().defaults or {})

    _Bank.defaults["selection_mode"] = "fixed"
    _Bank.defaults["source_ref"] = ""
    with pytest.raises(PD.PublicDomainSourceRefError):
        PD.fetch_public_domain_source(bank=_Bank())


def test_no_vendored_text_carries_gutenberg_boilerplate():
    """Boilerplate is not the author's words, and it reaches the writer LLM and
    can be SPOKEN by TTS.

    Found by an independent QA pass: two of forty-five leaked, because the
    stripper only knew the modern asterisked markers. The Cask of Amontillado
    ends with a possessive "End of Project Gutenberg's <title>" and no
    asterisks; The Secret Sharer opens with a "Produced by ..." transcriber
    credit sitting AFTER the start marker, where header-trimming never looked.
    """
    import re

    bad = []
    for source in _manifest()["sources"]:
        for unit in source["units"]:
            path = (PD._resolve_repo_relative_path(
                str((_bank().defaults or {}).get("manifest_path", "")),
                key="manifest_path").parent / unit["text_path"])
            body = path.read_text(encoding="utf-8", errors="replace")
            for pat in (r"Project Gutenberg",
                        r"^\s*Produced by ",
                        r"^\s*Transcriber'?s? Note",
                        r"Distributed Proofread"):
                m = re.search(pat, body, re.IGNORECASE | re.MULTILINE)
                if m:
                    bad.append(f"{source['source_id']}: {m.group(0).strip()[:60]!r}")
    assert not bad, "boilerplate in vendored text:\n  " + "\n  ".join(bad)


def test_every_vendored_unit_actually_loads():
    """A manifest row whose text is missing is a broken draw waiting to happen,
    and the roll would hit it eventually rather than immediately."""
    bank = _bank()
    manifest = _manifest()
    for source in manifest["sources"]:
        for unit in source["units"]:
            ref = f"{source['source_id']}:{unit['unit_id']}"
            res = PD.fetch_public_domain_source(bank=bank, source_ref=ref)
            body = res.payload["full_text"]
            assert len(body.split()) > 200, f"{ref} body is a stub"
            assert res.source_rights["license_status"] == "public_domain_us"

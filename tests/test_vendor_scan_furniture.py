# -*- coding: utf-8 -*-
"""The page-furniture rules of `scripts/otr_vendor_scan.py`.

WHY THIS FILE EXISTS. Every rule tested here was written to fix a scene that
had already gone wrong, and SIX of them were then broken again by the fix for
the next one -- twice by a guard that outlived the hazard it was written for.
The shapes are cheap to state and impossible to hold in a reader's head, so
they are stated here instead: each test names the volume it came from and what
the reader loses when it regresses.

None of this touches the network. A scanned volume is a list of page strings,
which is exactly what `pdf_text` returns, so the real functions run unchanged
against a handful of synthetic pages.
"""
import importlib.util
import os

import pytest


def _scan():
    """The vendor module, loaded from `scripts/` the way the corpus tests do."""
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_scan_under_test",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_scan.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_running_head_that_is_also_a_character_loses_the_header_not_the_speaker():
    """The 1912 Macbeth heads all 240 of its pages with the word `MACBETH`.

    The first cut at this filtered furniture by TOKEN, so the play's title
    character was deleted wherever he spoke: four speeches survived out of
    seventeen. The header and the man are the same string, and only POSITION
    tells them apart.
    """
    scan = _scan()
    pages = ["MACBETH\n%d\nMACBETH\nFala %d.\nBanquo\nResposta %d." % (n, n, n)
             for n in range(1, 21)]
    stripped = scan.strip_running_titles(pages)
    for page in stripped:
        lines = [line for line in page.splitlines() if line.strip()]
        assert lines[0] == "MACBETH", (
            "the speech label was eaten with the header: %r" % (lines[:3],))
        assert "Banquo" in page


def test_the_same_page_read_from_both_ends_spends_one_title_budget():
    """`MACBETH / 40 / MACBETH` -- header, folio, then a real speech label.

    On a page this short the head and foot windows are the same physical band
    read from opposite ends. Held per call, the budget let the head walk SPARE
    the second label and the foot walk delete it, and every page of six lost
    its speaker. Sparing a line is a decision about the page, so the second
    reading has to inherit it.
    """
    scan = _scan()
    pages = ["MACBETH\n%d\nMACBETH" % n for n in range(40, 46)]
    stripped = scan.strip_running_titles(pages)
    survived = sum(1 for page in stripped if "MACBETH" in page)
    assert survived == len(pages), (
        "%d of %d pages lost their real speech label" % (
            len(pages) - survived, len(pages)))


def test_a_title_printed_at_the_foot_is_furniture_too():
    """Text, then title, then folio -- the footer page.

    A de-duplication that skipped lines the head walk had already seen broke
    this outright: the head walk reads dialogue on its first line, returns
    having blanked nothing, and the foot walk that would have caught the title
    never ran. The title survived on 25 pages of 25 and became a false speaker.
    """
    scan = _scan()
    pages = ["Continua o dialogo da pagina %d\nMACBETH\n%d" % (n, n)
             for n in range(1, 26)]
    stripped = scan.strip_running_titles(pages)
    assert not any("MACBETH" in page for page in stripped)
    assert all("Continua o dialogo" in page for page in stripped)


def test_a_title_the_text_layer_split_in_half_is_rejoined_before_it_is_judged():
    """The 1919 Rei Lear prints `Rei Lear`, which arrives as `REI` then `LEAR`.

    Neither half is the title on its own, and the caps rule downstream rejoins
    them into one token that resolves to LEAR -- so the header stood as a
    speaker on every verso and split his real speeches at each page boundary.
    The joined form has to be tested BEFORE the single line, or `REI` matches
    alone, counts as the page's one title, and leaves `LEAR` behind.
    """
    scan = _scan()
    pages = ["%d\nREI\nLEAR\nKent\nFala numero %d." % (n, n) for n in range(1, 21)]
    stripped = scan.strip_running_titles(pages)
    for page in stripped:
        assert "REI" not in page and "LEAR" not in page, repr(page)
        assert "Kent" in page


def test_an_act_heading_is_furniture_only_when_it_repeats_and_carries_a_folio():
    """Both tests, because either one alone deletes a real heading.

    Recurrence alone deletes the genuine heading of the 1919 Rei Lear, whose
    act line is the same string as its running head. The folio alone deletes an
    ordinary act-opening page that happens to print a page number -- which is
    how an order-based rule ("the first one is the real one") was replaced by a
    rule with the same failure wearing different clothes.
    """
    scan = _scan()
    real = "ACTO PRIMEIRO\nSCENA I\nUma camara no palacio.\nKent\nFala."
    pages = [real] + ["ACTO PRIMEIRO\n%d\ntexto da pagina %d" % (n, n)
                      for n in range(3, 16)]
    stripped = scan.strip_running_titles(pages)
    assert stripped[0].splitlines()[0] == "ACTO PRIMEIRO", (
        "the real act heading was blanked; the scene is now unfindable")
    for page in stripped[1:]:
        assert "ACTO PRIMEIRO" not in page, "an echo survived: %r" % (page,)

    # A heading that appears once, on a page that does carry a folio.
    once = scan.strip_running_titles(["ACTO PRIMEIRO\nTRAGEDIA\n7"])[0]
    assert "ACTO PRIMEIRO" in once


def test_one_printed_line_gets_one_vote_even_when_the_windows_overlap():
    """`recurring_headings` counted per EDGE, and a short page has two.

    So one printed heading voted twice, and a heading on two short pages
    cleared a floor of three that the same heading on two ordinary pages could
    not -- biased, exactly backwards, toward flagging the short title pages
    whose headings are most likely to be real.
    """
    scan = _scan()
    short = ["ACTO PRIMEIRO\n7\nTexto um", "ACTO PRIMEIRO\n9\nTexto dois"]
    assert scan.recurring_headings(short) == set(), (
        "two pages cleared a floor of %d" % scan._ECHO_FLOOR)


def test_a_heading_is_recognised_whatever_case_the_scanner_printed_it_in():
    """The refusal in `main` asks this set whether a label is a running head.

    Asked with a raw string match, a volume printing `Scena III` did not match
    the conventional `SCENA III` a caller types, so the guard stayed silent on
    exactly the volume it exists for.
    """
    scan = _scan()
    pages = ["Scena III\n%d\nfala %d" % (n, n) for n in range(1, 9)]
    assert "SCENA III" in scan.recurring_headings(pages)
    assert scan._normalise_heading("  scena   iii . ") == "SCENA III"


@pytest.mark.parametrize("page", ["", "\n\n\n", "MACBETH", "7"])
def test_a_degenerate_page_is_survivable(page):
    """Empty, all-blank, and one-line pages reach this code from real scans."""
    scan = _scan()
    assert isinstance(scan.strip_running_titles([page])[0], str)


# --- The two known limits. Both are recorded as tests rather than as prose so
# --- that a reader who changes this behaviour finds out from the suite, and so
# --- that neither can be rediscovered later and mistaken for a fresh bug.


def test_known_limit_a_speaker_parked_in_the_edge_band_reads_as_furniture():
    """Judging by POSITION costs this, and there is no cheap way to buy it back.

    A label that lands in an edge band on a fifth of the pages is taken for a
    running title and removed. What makes it unlikely in a real book is that
    pagination moves a speech around the page, so a character does not sit in
    the band that consistently; neither measured volume trips it. It cannot be
    fixed by exempting roster names, because the hard case this file exists for
    is a header that IS a roster name -- the 1912 Macbeth prints exactly that.
    """
    scan = _scan()
    pages = ["fala %d\nmeio %d\nfim %d\nKENT" % (n, n, n) for n in range(1, 21)]
    stripped = scan.strip_running_titles(pages)
    assert not any("KENT" in page for page in stripped), (
        "the limit no longer holds -- if this was fixed deliberately, delete "
        "this test and say so; if it changed by accident, find out why")


def test_known_limit_the_head_band_is_read_first_when_the_two_overlap():
    """One band, one title -- and on a short page the head edge claims it.

    Where the windows overlap they are one physical band read from both ends,
    so the budget is shared and the walk order decides which of TWO distinct
    captions is removed. Reading the foot first would remove the other one.
    Neither measured volume prints two captions on one short page, so the order
    is arbitrary rather than wrong, and it is pinned here so that it stays
    deliberate.
    """
    scan = _scan()
    pages = ["TITULO\nmeio %d\nfim %d\nRODAPE" % (n, n) for n in range(1, 21)]
    forms = scan.running_titles(pages)
    assert {"TITULO", "RODAPE"} <= forms, "fixture no longer sets up the case"
    stripped = scan.strip_running_titles(pages)
    assert not any("TITULO" in page for page in stripped)
    assert all("RODAPE" in page for page in stripped)

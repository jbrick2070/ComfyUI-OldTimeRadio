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


def test_a_broken_word_is_rejoined_only_onto_a_lower_case_continuation():
    """NOT A FURNITURE RULE -- a page-text rule of the same script.

    The 1912 Macbeth breaks a word on the last line before a speaker label:

        Nunca vi assim um dia tao bello e tao horri-
        BANQUO
        Que distancia fazem d'aqui a Forres ? ...

    The rejoin welded the two into `horriBANQUO`, which ate Banquo's label --
    filing his opening speech under MACBETH -- and left a nonsense word in the
    corpus for a voice engine to read aloud. It shipped that way, at
    `alignment_confidence: 1.0`, and no count could see it.

    A typesetter breaks a word in its middle, so a genuine continuation is
    always lower-case. That is the whole guard.
    """
    scan = _scan()
    join = lambda text: scan._LINE_BREAK_HYPHEN.sub(r"\1\2", text)

    # the case the rule exists for, and still does
    assert join("por-\ntos") == "portos"
    assert join("esta-\nrem sobre ella") == "estarem sobre ella"

    # the case that shipped a defect
    assert join("tao horri-\nBANQUO") == "tao horri-\nBANQUO"
    # a numbered label is refused for the same reason
    assert join("tao horri-\n1.a FEITICEIRA") == "tao horri-\n1.a FEITICEIRA"
    # and the author's own hyphen in a compound is left alone
    assert join("Anglo-\nSaxao") == "Anglo-\nSaxao"


def _one_page_pdf(placements):
    """A real one-page PDF with text at exact positions. Returns a page.

    Built rather than fixtured because the rule under test is geometric: it
    only means anything against real glyph origins.
    """
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for x, y, text in placements:
        page.insert_text((x, y), text, fontsize=10)
    # reopen from bytes so the page carries a parsed text layer
    reopened = pymupdf.open("pdf", doc.tobytes())
    doc.close()
    return reopened, reopened[0]


def test_a_hanging_cue_joins_the_row_it_labels_not_the_page_bottom():
    """The whole reason the coordinate reader exists.

    A speaker label is printed in the margin, to the LEFT of the first line of
    its speech and on the same baseline. `page.get_text()` does not
    necessarily emit it there -- on Clark page 59 it emits the marginal cues
    at the BOTTOM of the page, detached from their dialogue, so the previous
    speaker absorbs two speeches that are not his. Rebuilt from baselines, the
    cue leads its own row.
    """
    scan = _scan()
    doc, page = _one_page_pdf([
        (30, 100, "MIR."), (80, 100, "Se me antoja"),
        (30, 120, "FER."), (80, 120, "No, mi noble duena"),
    ])
    rows = scan.rows_from_coordinates(page)
    doc.close()
    assert rows[0] == "MIR. Se me antoja", rows
    assert rows[1] == "FER. No, mi noble duena", rows


def test_a_row_is_measured_from_its_first_baseline_not_its_last():
    """Chaining near-neighbours walks a row down the page one step at a time.

    Three words at 100, 104 and 108 are within the span of their PREDECESSOR
    at every step, so a chaining rule swallows all three into one row even
    though the total drop is 8. The row is anchored on its first baseline
    instead. (The span is adaptive -- `max(3.0, 0.45 * median word height)`,
    about 6.2 at this synthetic font's 13.7pt box -- so the fixture keeps
    each 4pt step inside it and the 8pt whole outside it.)
    """
    scan = _scan()
    doc, page = _one_page_pdf([
        (30, 100, "alpha"), (80, 104, "beta"), (130, 108, "gamma"),
    ])
    rows = scan.rows_from_coordinates(page)
    doc.close()
    assert len(rows) == 2, rows
    assert rows[0] == "alpha beta" and rows[1] == "gamma", rows


def test_the_coordinate_reader_is_opt_in_and_flat_is_the_default():
    """It is NOT certified across every volume, so it may not become default.

    Measured exceptions remain: a hanging cue 3.29 points off its dialogue on
    one Clark page, another at 3.12, six rotated pages in the Macpherson
    volume, and words whose own glyphs straddle two rows. Prove a volume and
    pin it; do not flip this globally.
    """
    import inspect
    scan = _scan()
    assert inspect.signature(scan.pdf_text).parameters[
        "reading_order"].default == "flat"
    source = inspect.getsource(scan.pdf_text)
    assert "rows_from_coordinates" in source, "the reader is not wired in"
    assert "fused_words" in source, (
        "a page whose words straddle rows must be named, not shipped quietly")


def test_a_page_window_is_the_only_stable_address_in_a_scanned_volume():
    """NOT A FURNITURE RULE -- the addressing rule of the same script.

    A line index moves whenever a furniture rule or a PyMuPDF version
    changes, and a heading label does not identify a scene on its own: the
    Spanish volumes print `ESCENA PRIMERA .` five, six and ten times each, so
    asking by name returns whichever copy the matcher reaches first. That is
    how a request for King Lear 1.1 returned Act 2 Scene 1, and how the
    Tempest's act 3 came back spanning two plays.

    A bad window fails loudly rather than storing a wrong scene quietly.
    """
    scan = _scan()
    pages = ["page %d" % n for n in range(20)]

    got, why = scan.slice_pages(pages, "5-8")
    assert got == ["page 5", "page 6", "page 7", "page 8"], got
    assert "5-8" in why or "5-8" in why.replace(" ", "")

    # a single page is a window of one
    assert scan.slice_pages(pages, "5")[0] == ["page 5"]

    # and every bad shape refuses rather than guessing
    for bad in ("8-5", "19-20", "-3", "abc", "5-x"):
        got, why = scan.slice_pages(pages, bad)
        assert got is None, (bad, got)
        assert why, bad


def test_a_split_heading_rejoins_through_the_printer_s_trailing_stop():
    """NOT A FURNITURE RULE -- the other page-text rule of the same script.

    A book sets its heading with a stop after the numeral. Demanding a bare
    numeral made the rejoin fire on NOTHING in either Spanish scanned volume,
    so every split heading in both was invisible to the scene finder and
    `es/twelfth_night 1.5` could not be located at all -- the volume prints
    `ESCENA` over `V .` and the trailing period defeated the match.

    The numeral alternation is what keeps this narrow: only a roman, a small
    integer or a spelled ordinal may follow, so a line of dialogue under a
    stray heading word is still never rejoined.
    """
    scan = _scan()
    join = lambda text: scan._SPLIT_HEADING.sub(r"\1 \2", text)

    # the shapes that were being rejected, one per real volume
    assert join("ESCENA\nV .") == "ESCENA V"
    assert join("ACTO\nPRIMERO .") == "ACTO PRIMERO"
    assert join("ACTO\nQUINTO ,") == "ACTO QUINTO"
    assert join("ESCENA\nII.") == "ESCENA II"

    # the shape that already worked keeps working
    assert join("SCENA\nI") == "SCENA I"

    # and dialogue under a heading word is still left alone
    assert join("ESCENA\nUma camara no palacio.") == "ESCENA\nUma camara no palacio."
    assert join("ACTO\nprimeiro que tudo .") == "ACTO\nprimeiro que tudo ."


def test_the_identity_of_an_edge_row_is_its_title():
    """The coordinate reader joins a running head into ONE row, and the
    1914 Tempestade prints a three-part head: `SCENA II A TEMPESTADE 9` on
    the recto, `10 A TEMPESTADE ACTO I` on the verso. A rule that peels only
    the folio leaves a string that starts with a heading word and is barred
    from the vote -- measured, 38 of them inside act 1 scene 2's dialogue.
    Folio and heading are decoration; the title is the identity. A leading
    article is peeled too, because the scanner drops the `A` on a third of
    the pages and two spellings of one title are two keys.
    """
    scan = _scan()
    parts = scan._edge_folio_parts
    assert parts("SCENA II A TEMPESTADE 9") == ("TEMPESTADE", True)
    assert parts("10 A TEMPESTADE ACTO I") == ("TEMPESTADE", True)
    assert parts("98 A TEMPESTADE ACTO 111") == ("TEMPESTADE", True)   # OCR III
    assert parts("A TEMPESTADE") == ("TEMPESTADE", False)     # plain line, same key
    assert parts("8 REI LEAR") == ("REI LEAR", True)
    # what must NOT change
    assert parts("ACTO 1") == ("ACTO 1", False)               # a heading's own numeral
    assert parts("MACBETH") == ("MACBETH", False)
    assert parts("Fala numero 12") == ("Fala numero 12", False)   # prose stays prose


def test_a_heading_is_peeled_only_when_a_title_is_left():
    """THE 1912 MACBETH'S OLDEST HAZARD, RE-ENTERED FROM A NEW SIDE.

    Its recto head is two headings and a folio on one row, `ACTO I- SCENA
    III 17`. Peeling the scene half left `ACTO I-`, which qualified for the
    act-line FREE branch of the walk -- the one that spends no budget -- so
    the walk carried on into the page and spent the budget on the first
    `MACBETH` it met, which was the speaker. It cost a label in a shipped
    scene before this guard existed. A peel that leaves another heading is
    not a peel.

    The proof that this holds on the book itself is not a fixture: both
    vendored Domingos Ramos scenes re-extract BYTE FOR BYTE under the
    coordinate reader after the rule landed, where the unguarded peel had
    cost Macbeth one label. A synthetic page cannot stand in for that --
    repeat the same dialogue on every page and the floor makes all of it
    furniture -- so this test pins the identity and the commit cites the
    measurement.
    """
    scan = _scan()
    for head in ("ACTO I- SCENA III 17", "ACTO II - SCENA III 9", "ACTO I, SCENA II."):
        identity, _ = scan._edge_folio_parts(head)
        assert scan._HEADING_SHAPED.match(identity), (head, identity)
        assert not identity.rstrip(" -,").endswith(("I-", "ACTO I", "ACTO II")), (head, identity)


def test_a_three_part_running_head_is_stripped_and_the_speaker_below_it_is_not():
    """The Tempestade band, end to end, on both page sides."""
    scan = _scan()
    pages = []
    for n in range(9, 45):
        head = ("SCENA II A TEMPESTADE %d" % n) if n % 2 else ("%d A TEMPESTADE ACTO I" % n)
        pages.append("%s\nMIRANDA\nFala %d.\nPRÓSPERO\nResposta." % (head, n))
    for n, page in enumerate(scan.strip_running_titles(pages), 9):
        kept = [l for l in page.splitlines() if l.strip()]
        assert "TEMPESTADE" not in page, (n, kept[:2])
        assert kept and kept[0] == "MIRANDA", (n, kept[:2])


def test_a_folio_may_carry_the_printers_stop_and_a_comma_may_join_the_headings():
    scan = _scan()
    assert scan._FOLIO.match("198 .")
    assert scan._FOLIO.match("17")
    assert not scan._FOLIO.match("198 . Cual nunca")
    for head in ("ACTO I, SCENA II.", "ACTO I,SCENA II .", "ACTO I - SCENA III 17"):
        assert scan._RUNNING_HEADER.match(head), head


def test_a_fold_binds_a_printed_form_only_when_its_target_is_in_the_scene():
    """NOT A FURNITURE RULE -- the one declared alias the operator allowed.

    Domingos Ramos prints Ferdinand as FERNANDO: eight letters, past the
    length floor, refused on the prefix test because neither name starts
    with the other. Ten of his speeches were discarded and merged into the
    previous speaker in a scene he opens. A fold names the binding for ONE
    scene, and the roster check is what keeps it there: the same fold
    declared for a scene without Ferdinand binds nothing.
    """
    import inspect
    scan = _scan()
    tempest = {"FERDINAND", "MIRANDA", "PROSPERO"}
    lear = {"LEAR", "KENT", "CORDELIA"}
    assert scan.resolve("FERNANDO", tempest) is None          # the defect
    scan.FOLDS.clear()
    scan.FOLDS[scan.fold("FERNANDO")] = "FERDINAND"
    try:
        assert scan.resolve("FERNANDO", tempest) == "FERDINAND"
        assert scan.resolve("Fernando (aparte)", tempest) == "FERDINAND"
        assert scan.resolve("FERNANDO", lear) is None         # not on this stage
        assert scan.resolve("MIRANDA", tempest) == "MIRANDA"  # nothing else moves
    finally:
        scan.FOLDS.clear()
    # populated from the command line, validated against the roster, recorded
    source = inspect.getsource(scan.main)
    assert "FOLDS.clear()" in source and "not in this scene" in source
    assert '"folds"' in source, "a fold that is not recorded is not provenance"


def test_a_swallowed_label_has_a_shape_and_the_write_refuses_it():
    """NOT A FURNITURE RULE -- the write gate of the same script.

    The corpus shipped this defect twice, invisible to every count:
    `horriBANQUO` (1912 Macbeth) and `AfasKent` (1919 Rei Lear), each a
    broken word welded onto the next speaker's name, each filing a whole
    speech under the wrong character. A lower-case letter never runs
    straight into capitals inside a token in these languages, so the weld
    is refusable by shape. It is checked on speech BODIES only: a label may
    carry capitals and punctuation of its own.
    """
    import inspect
    scan = _scan()
    weld = scan._INTERIOR_WELD.search

    # the two that shipped
    assert weld("Nunca vi assim um dia tao bello e tao horriBANQUO Que distancia")
    assert weld("a flecha ja partiu. AfasKent Nao ; deixai-a")

    # ordinary prose, a sentence-opening capital, an accented capital name,
    # and a title-case label spelling are all left alone
    for clean in ("tao bello e tao horrivel! Que distancia fazem",
                  "Nada, meu senhor. REI LEAR fala.",
                  "Quem vem la? Entram Ross e Angus.",
                  "MIRANDA fala; Prospero escuta."):
        assert not weld(clean), clean

    # and it is enforced at the write, not merely defined
    source = inspect.getsource(scan.main)
    assert "_INTERIOR_WELD" in source and "REFUSING to write" in source, (
        "the weld gate is defined but the write does not consult it")


def test_the_rejoin_is_applied_where_the_pages_are_read():
    """A regex nothing calls is not a rule. Assert the real call site.

    The test above proves the pattern; only this proves that `pdf_text` is
    what applies it, which is the difference between a fixed corpus and a
    fixed constant.
    """
    import inspect
    scan = _scan()
    source = inspect.getsource(scan.pdf_text)
    assert "_LINE_BREAK_HYPHEN.sub" in source, (
        "pdf_text no longer rejoins broken words; the corpus will carry the "
        "typesetter's hyphens into a voice engine")


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

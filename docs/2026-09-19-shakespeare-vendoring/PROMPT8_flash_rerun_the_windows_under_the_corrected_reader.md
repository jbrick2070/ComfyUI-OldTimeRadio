Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

YOUR WINDOWS BRIEF WENT STALE UNDER YOU. It asked for the eight Spanish
windows re-run under the coordinate reader, and the reader it named was
refuted and replaced while you held it: commit ae792152 gave a word its
neighbours' glyphs (a word's bounding box reaches into the row above), and
commit 0b35b419 fixed ownership by identity -- one TextPage, block/line/
word numbers -- and folded in an adaptive span, a fused-word sweep that
crosses span boundaries, and a furniture rule for `<folio> <title>` rows.
If you ran the old brief, its numbers are against the wrong reader. Run
this one.

YOUR JOB IS TO MEASURE, AND TO REFUTE THE CLAIM THAT THE WINDOWS SURVIVE.
Read-only on the repo; scratch in %TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.
The PDFs are cached under tmp\scan_cache, keyed by the first 24 hex of a
sha256 of the source URL; the URLs are on the `es` page_scan rows of
config\source_banks\shakespeare\translations\leads.json.

THE EIGHT CELLS AND THEIR MEASURED WINDOWS (zero-based, inclusive):

    es tempest 3.1         58-62     label  ESCENA PRIMERA .
    es twelfth_night 1.5   110-122   label  ESCENA V
    es twelfth_night 2.5   136-144   label  ESCENA II        (the BOOK misprints 2.5 as II)
    es much_ado 2.3        220-230   label  ESCENA III.
    es much_ado 3.1        230-234   label  ESCENA PRIMERA .
    es king_lear 1.1       244-255   label  ESCENA PRIMERA .
    es midsummer 3.1       414-421   label  ESCENA PRIMERA
    es midsummer 3.2       421-440   label  ESCENA II

RUN EACH TWICE, flat and coordinates, through the real script:

    python scripts\otr_vendor_scan.py --iso es --play <play> --scene <scene> ^
        --stem <play>__act<A>_scene<S> --pages <window> --scene-label "<label>"

(the flat path) and the same pipeline with the reader on, which the CLI
does not expose yet, so call it from a probe:

    pages = strip_running_titles(pdf_text(url, reading_order="coordinates"))
    pages, _ = slice_pages(pages, "<window>")
    flat  = _SPLIT_HEADING.sub(r"\1 \2", _RUNNING_HEADER.sub(" ", "\n".join(pages)))
    body, reason = extract(flat.splitlines(), None, None, "<label>", end_label=None)

REPORT PER CELL, flat beside coordinates:
  * span in characters, and the first and last three lines of each
  * whether `find_label` still finds the label inside the window -- the
    reader can join a heading to its folio, and a joined heading is a
    different string
  * the cast found, against the English sidecar under
    config\source_banks\shakespeare\sources\ -- read the roster with
    nodes\_otr_roster_gender.py::load_roster_characters, do not parse the
    file yourself
  * every running-header or folio row that sits INSIDE the span, quoted.
    The furniture rule in 0b35b419 was measured on the two Portuguese
    scenes (16 and 3 headers to 0 and 0); nobody has measured it on Spanish
  * the reader's own `[scan] page N has K word(s) welded` warnings for pages
    in the window, and whether any is inside the scene span

THEN THE TWO THINGS THAT DECIDE IT:
  1. Does the reader MOVE any window? A cell whose label is found under
     flat and not under coordinates, or whose span changes by more than the
     furniture it removed, is a finding about the reader. Quote the row.
  2. Macpherson is NOT certified under the adaptive span -- the fix's own
     commit message names four image-verified cue rows it still splits
     (pages 322, 346, 358, 431). Three of the eight windows are Macpherson
     (244-255, 414-421, 421-440). Do any of those four pages fall inside a
     window, and if so which cue is split and whose speech does it cost?

DO NOT propose changes to extract() or find_label() in
scripts\otr_vendor_shakespeare.py -- scoped out by the operator, shared
with 38 HTML-vendored scenes.

FINISH WITH THREE LISTS, headed exactly:
    WINDOWS UNDER THE READER:  per cell -- found / moved / cast / furniture count
    SPLIT CUES:                per Macpherson cell, any of the four pages inside
                               it and the speech it costs
    REFUTED:                   every claim above you could not confirm

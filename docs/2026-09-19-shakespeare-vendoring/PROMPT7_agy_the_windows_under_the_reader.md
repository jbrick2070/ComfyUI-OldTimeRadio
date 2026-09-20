Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

YOUR EIGHT WINDOWS SHIPPED. `--pages START-END` is on main (76f88d8c) with
your measurement in the commit message: all eight cells, complete casts.
Your PDF hashes are recorded. Your chrome-fix kill stands.

YOUR JOB IS TO REFUTE AND TO MEASURE. Read-only on the repo; scratch in
%TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THE QUESTION: DO YOUR WINDOWS SURVIVE THE COORDINATE READER?

Everything you measured was on FLATTENED text -- `page.get_text()`. A
second reader is now on main, opt-in, that rebuilds each page in its printed
order from glyph baselines (commit ae792152, `rows_from_coordinates`). It
fixes real misattributions -- Clark's marginal cues come back to their
lines, `horri-` and `vel!` rejoin -- but it changes the shape of every page
the furniture rules see. `strip_running_titles` was tuned on the flattened
shape where a folio and a title arrive on separate lines; reconstruction
joins them into one row like `8 REI LEAR`, and another reviewer measured
19 running headers leaking into speeches because of it.

So re-run all eight of your windows with the reader ON:

    pages = strip_running_titles(pdf_text(url, reading_order="coordinates"))[start:end+1]

and for each cell report, beside your flat numbers: span, whether the
scene label is still found, whether the span still starts and ends where
it did, the cast found, and -- the new column -- every running-header or
folio row that now sits INSIDE the span. Quote them.

THEN THE TWO THINGS THAT DECIDE IT:
  1. Does the reader move any window? A heading that was one line under
     flatten may be one row under reconstruction or may be joined to its
     folio. If `find_label` no longer finds the label inside the window, say
     which cell and quote the row it became.
  2. The furniture leak. Count per cell. Then read `_strip_one_edge` and
     say whether a `<folio> <title>` row at a page edge could be removed by
     the existing walk with one added shape, and name every test in
     tests/test_vendor_scan_furniture.py that shape would put at risk.

DO NOT propose changes to extract() or find_label() -- scoped out.

FINISH WITH:
    WINDOWS UNDER THE READER:  per cell, found / moved / cast
    FURNITURE INSIDE SPANS:    per cell, count and quoted rows
    REFUTED:                   every claim above you could not confirm

Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

THE READER YOU QUALIFIED IS ON MAIN. Commit ae792152 implements your rule --
median character baseline, 3.0 point span from the row's FIRST baseline, no
chaining, words from PyMuPDF's own `words` and only the baseline from the raw
layer, joined by geometry because the block numbers disagree. It reproduces
your image-verified row counts on Tempest 3.1 exactly (24, 38, 38, 39) and
your conservation result exactly (0 failures, 1,450 pages, five volumes). It
is OPT-IN, `reading_order="coordinates"`, and `flat` stays the default, for
the reasons you gave. Read `rows_from_coordinates`, `_baseline_index`,
`_word_baseline_from` and `fused_words` in scripts/otr_vendor_scan.py.

YOUR JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a number you
produced. Read-only on the repo; scratch in %TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THREE THINGS YOU LEFT OPEN, NOW YOURS TO CLOSE

1. THE FURNITURE REGRESSION IS THE BLOCKER, AND YOU FOUND IT. Reconstruction
   inserts 19 running headers into speeches -- 16 in Lear, 3 in Macbeth --
   because it correctly joins a folio and a title into one printed row like
   `8 REI LEAR`, and `strip_running_titles` was built for the flattened
   shape where they arrive on separate lines. Read `running_titles`,
   `_strip_one_edge` and `_page_edges`. Design the smallest change that
   removes a `<folio> <title>` or `<title> <folio>` row at a page edge
   WITHOUT breaking any of the thirteen rules in
   tests/test_vendor_scan_furniture.py -- six of which were each broken by
   the fix for the next one. For each rule you think is at risk, name the
   test. Then MEASURE: with your change, how many of the 19 remain?

2. THE TOLERANCE. Another reviewer clustered on the bounding-box MIDPOINT
   with an adaptive span, `max(3.0, 0.45 * median_word_height)` -- about 4.4
   on Clark, 3.5 on Macpherson -- and reports zero verse splits or merges on
   Tempest 3.1. Your fixed 3.0 misses the hanging cues at 3.29 (page 133)
   and 3.12 (page 182). The adaptive span would catch both and stays under
   your 5.445 minimum row gap. Test it on YOUR baseline, not the midpoint:
   does `max(3.0, 0.45 * median_word_height)` applied to median character
   baselines catch those two cues without merging any verse row on the four
   Tempest pages, and does it hold on Macpherson where the median height is
   smaller? Give the numbers. If it is better, say so in one sentence.

3. THE SHIPPED SCENES. You diffed both Portuguese scenes under
   reconstruction and found five real repairs and no new misattribution.
   Now that the reader is on main, re-run that diff through the actual
   script -- `pdf_text(url, reading_order="coordinates")` -- and report
   whether the script's output matches your earlier hand reconstruction.
   Any difference is a defect in the implementation of your own rule, and
   that is the most useful thing you could find.

DO NOT propose changes to extract() or find_label() -- scoped out, shared
with 38 shipping scenes.

FINISH WITH:
    SHIP:        the furniture change, precise enough to code from
    TOLERANCE:   fixed 3.0 or adaptive, with the numbers that decided it
    REFUTED:     every claim above you could not confirm

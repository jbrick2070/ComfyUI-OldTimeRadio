Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit efee731a.

YOUR LAST PASS KILLED THE DRIVER'S FIX AND WAS RIGHT TO. Occurrence counts
settled it: `ESCENA PRIMERA .` appears 5 times in Macpherson, 10 in the Clark
Tempest volume and 6 in the Clark Otelo volume, so recording the printed
chrome only works for king_lear 1.1 and only because King Lear happens to be
the first play in its book. Withdrawn. Page windows are now the only boundary
mechanism in the design.

**Your split-heading widening is SHIPPED** -- commit efee731a, with your
measurement in the message and both Portuguese volumes proven unmoved (0
rejoins before, 0 after; pt/macbeth 1.3 re-extracts byte for byte).

Your probe files are still in %TEMP%; the cleanup command failed but nothing
in the repo was touched, and they do not need removing.

YOUR JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a number you
produced. Where you cannot, say "refuted -- could not confirm". Read-only on
the repo: do not edit any tracked file, do not run any git command that
writes, do not touch the GPU, and keep scratch in %TEMP%.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THE CLAIM TO PROVE OR BREAK: THAT YOUR WINDOWS WORK END TO END

You measured eight windows and said act labels and `--end-label` can both be
deleted. Prove it by running it, not by reasoning about it. For each of the
eight cells:

    pages = strip_running_titles(pdf_text(url))[start : end + 1]
    flat  = _SPLIT_HEADING.sub(r"\1 \2", _RUNNING_HEADER.sub(" ", "\n".join(pages)))
    body, reason = extract(flat.splitlines(), None, None, "<scene label>", end_label=None)

using YOUR measured windows:

    king_lear 1.1     244-255      midsummer 3.1     414-421
    midsummer 3.2     421-440      tempest 3.1        58-62
    twelfth_night 1.5 110-122      twelfth_night 2.5 136-144
    much_ado 2.3      220-230      much_ado 3.1      230-234

1. REPORT, PER CELL: the span in characters, its first three and last three
   lines, and -- this is the part length cannot tell you -- WHICH CHARACTERS
   SPEAK in it, against the English scene sidecar under
   config/source_banks/shakespeare/sources/. A span of plausible length
   holding the wrong cast is the failure this corpus keeps shipping, and a
   character count cannot see it.

2. WHAT ACTUALLY ENDS THE SCENE NOW? With `act_label=None` and
   `end_label=None`, read the termination loop in `extract()`
   (scripts/otr_vendor_shakespeare.py, in the `for j in range(scene_at + 1,
   ...)` block). The act-heading branch is guarded by `if act_label` -- so
   with no act label, ONLY the scene stem can stop the scan. For a scene that
   is the LAST in its act, the next scene-shaped heading belongs to the NEXT
   ACT. Does the scene then run past the act boundary, and is it only the
   window edge that saves it? Say which cells are last-in-act and what
   terminates each one in fact.

3. THE SHARED PAGE IS YOUR OWN FINDING AND IT IS THE SHARP EDGE. Macpherson
   page 421 carries the end of Midsummer 3.1 AND the start of 3.2, and your
   windows use 421 as both the end of one and the start of the other. So:
   * does `midsummer 3.1` (414-421) pick up the head of 3.2 from page 421?
   * does `midsummer 3.2` (421-440) pick up the tail of 3.1 from page 421?
   Quote the boundary lines. If either leaks, say what fixes it -- and note
   that a page window cannot, because the unit is the page.

4. TWELFTH NIGHT 2.5 IS THE ONE YOU CLAIMED THE WINDOW RESCUES. Run it with
   the misprinted label `ESCENA II.` inside 136-144 and confirm the span is
   Folger 2.5 -- Toby, Andrew, Fabian, Maria, the letter -- and not 2.2.
   Name the speakers you actually find.

5. PIN THE EDITIONS. The registry will key on the PDF's own content hash
   rather than its URL, because a URL is not an identity. Give the full
   sha256 of each of the five cached volumes -- the three Spanish and the two
   Portuguese -- with the filename under tmp/scan_cache. The Clark
   Tempest/Twelfth Night volume is
   b79223db195e6b0c75bfed0b14c8d4120171ab0f99d8731e14f269db76f129d3; confirm
   it and supply the other four.

6. HOW WOULD THIS BE WRONG IN A YEAR? A window is recorded once and read
   forever. Name what could change underneath it -- a re-scanned file at the
   same URL, a PyMuPDF version that paginates differently, a Commons
   re-upload -- and say which of those the content hash catches and which it
   does not. Then say the cheapest thing a vendor run can assert every time
   to notice.

DO NOT propose changes to extract() or find_label(). The operator scoped
scripts/otr_vendor_shakespeare.py out and it is shared with 38 HTML-vendored
scenes already shipping; a fix needing it is a finding to report, not a
recommendation.

FINISH WITH THREE LISTS, headed exactly:

    WINDOWS THAT PRODUCE THE RIGHT SCENE:
      one line per cell, with the speakers found

    LEAKS:
      every cell whose span carries text from a neighbouring scene, with the
      quoted boundary line

    REFUTED:
      every claim in this brief you could not confirm against the real files

Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

THIS IS THE FINISHED-DIFF QA THE LAW OWES ON THE DAY'S BIGGEST CODE
COMMIT. Push-then-QA is the rule now (CLAUDE.md, 2026-09-19); this is the
QA. The commit is already on main, it replaced a reader that a challenger
refuted, and nobody outside the author has read the replacement as a
diff. Read it with `git show 0b35b419` -- 234 lines changed in
scripts\otr_vendor_scan.py, 8 in tests\test_vendor_scan_furniture.py.

YOUR ONLY JOB IS TO REFUTE. Find the reason this should not have shipped.
If you cannot ground an objection in a file and a line, do not raise it.
Read-only; no git writes; no render pipeline; no GPU; scratch in %TEMP%.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.
PDFs are cached under tmp\scan_cache.

WHAT THE COMMIT CLAIMS, so you know what to break:
  * `_word_baselines` builds ONE TextPage with `pymupdf.TEXTFLAGS_WORDS`
    and reads both `words` and `rawdict` from it, joining by
    (block, line, word_no). Claim: 302,837 words, zero mismatches.
  * The span is `max(3.0, 0.45 * median word height)`.
  * `fused_words` accumulates across spans and skips lines whose `dir`
    is not horizontal.
  * `_edge_folio_parts` gives a `<folio> <title>` row a comparison
    identity for the furniture vote and walk without touching the line.
  * Verified after the change: `que` at 100.3, `gusano,` in Prospero's
    row, Tempest rows 24/38/38/39, word conservation 0/1450, furniture
    inside both Portuguese spans 0/0, flat path byte-identical.

ATTACK THESE, EACH GROUNDED IN A FILE AND A LINE:

1. THE TEXTPAGE API. `page.get_textpage(flags=...)` and
   `page.get_text("words", textpage=tp)` -- confirm both exist with those
   signatures in the INSTALLED PyMuPDF (print `pymupdf.__version__` and
   read the method signatures). If `textpage=` is silently ignored on
   this version, the two views are built separately again and the join
   is back to the broken index. Prove it one way or the other by
   constructing a page where a separate-TextPage join WOULD mismatch and
   showing the shared one does not.

2. THE FALLBACK. `_word_baselines` falls back to `(y0 + y1) / 2.0` when
   `w_no >= len(runs)`. The commit says zero mismatches. Count the
   fallbacks across all five volumes yourself. If any fire, name the page
   and word -- a fallback is the bounding-box midpoint, the metric the
   whole change exists to avoid.

3. THE WHITESPACE SPLIT IN `_word_baselines`. It splits a raw line's
   characters on `ch["c"].isspace()` to produce runs in `word_no` order.
   PyMuPDF's `words` tokenizer splits on whitespace too -- but is it the
   SAME whitespace set? A non-breaking space (U+00A0), a thin space, a
   zero-width space in the raw layer: does `words` split on it, and does
   `isspace()`? A disagreement desynchronises `word_no` for every word
   after it on that line. Search all five volumes for any such character
   in a raw line and say what happens.

4. THE ADAPTIVE SPAN ON A DEGENERATE PAGE. A page with one word has a
   median height of that word; a page with two words of very different
   sizes (a drop cap, a folio) has a median that is one of them. Does the
   span ever collapse below 3.0 (no -- the floor), or balloon so that two
   real rows merge? Find the page in any volume with the LARGEST computed
   span and say whether its rows are right.

5. `fused_words` READS `line.get("dir", (1, 0))`. Is `dir` a key on a
   rawdict LINE in this PyMuPDF version, or only on `dict` lines, or
   only on spans? If it is absent, every line is treated as horizontal
   and the rotated-table skip never fires -- which is exactly the false
   positive the commit claims to have removed. Read the rawdict for a
   Macpherson rotated page (82-85, 87-88) and show the key.

6. `_edge_folio_parts` REGEXES. `([0-9]{1,3})\s+(.+)` then `(.+?)\s+([0-9]{1,3})`.
   Construct a line of real dialogue from any of the five volumes that
   ends in a one-to-three-digit number and starts with an uppercase word
   -- a year, a sum, a verse number -- and say whether the guard
   `title.isupper() or _HEADING_SHAPED.match(title)` keeps it out of the
   furniture vote. The challenger reported that an unrestricted first
   draft "turned numbered prose into repeated furniture"; verify the
   restriction actually holds on the real text.

7. THE DOUBLE CALL. In `strip_running_titles`, the `folios` list now
   calls `_edge_folio_parts(lines[i])` three times per edge line. Not a
   correctness issue -- but read `_strip_one_edge`: `identity` is
   computed from `text` BEFORE the `_FOLIO.match(text)` branch. If a
   line is a bare folio, `identity` is that folio string and
   `attached_folio` is False. Trace whether any later branch in the walk
   uses `identity` where it should have used `text`, or vice versa, for
   a bare folio line, a bare title line, and a `<folio> <title>` line.

8. THE TEST. `test_a_row_is_measured_from_its_first_baseline_not_its_last`
   was retuned from 100/103/106 to 100/104/108 because the synthetic
   font's box is 13.7pt and the span is 6.18. Is that test now pinning
   the RULE (anchor on first baseline, not chaining) or pinning a
   NUMBER (6.18)? Change the font size in the fixture to 8 and to 14 and
   say whether the assertion still tests what its docstring claims.

9. THE THING THE COMMIT MESSAGE DOES NOT SAY. Run the four suites --
   tests\test_vendor_scan_furniture.py tests\test_verbatim_corpus.py
   tests\test_shakespeare_corpus_gate.py tests\test_shakespeare_sources.py
   -- and report the real count against the claimed 191.

DO NOT propose changes to extract() or find_label() in
scripts\otr_vendor_shakespeare.py.

FINISH WITH EXACTLY ONE OF:
    VERDICT: HOLDS
    MUST-FIX: none
or
    VERDICT: DEFECTS FOUND
    MUST-FIX:
      <numbered, each with file:line and the concrete failure>
and in either case:
    SUITE:    the real pass count
    REFUTED:  every claim above you could not confirm

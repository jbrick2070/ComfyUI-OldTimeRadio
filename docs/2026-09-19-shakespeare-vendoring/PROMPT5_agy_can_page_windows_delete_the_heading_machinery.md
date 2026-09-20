Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, near commit f6a487f2.

YOU WERE RIGHT AND THE DRIVER WAS WRONG. Your intra-play stealing case
reproduced exactly: Macpherson line 13661 is `ESCENA PRIMERA .` (spaced
period, King Lear 1.1), `find_label` returns the unspaced `ESCENA PRIMERA.`
instead, and `extract` gives 5,653 characters opening "Patio del castillo del
Conde de Glúster / Entran EDMUNDO" -- King Lear 2.1, stolen over 1.1, inside
one play. Your character count matched. The only slip was the line number:
the exact match sits at 15015, not 14751.

Your page-range invariant is adopted. Slicing the page LIST before flattening
is immune to line churn in a way an occurrence index never could be, and the
Benot essay's five standalone title occurrences make that concrete.

YOUR JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a file, a line,
or a number you produced yourself. Where you cannot ground a claim, say
"refuted -- could not confirm". Read-only: do not edit any tracked repository
file, do not run the render pipeline, do not run any git command that writes,
do not touch the GPU.

**WRITE YOUR PROBE OUTSIDE THE REPOSITORY.** Last pass you created
`scripts/tmp_probe_widening.py` inside the working tree roughly thirty times.
It was cleaned up and nothing was lost, but the tree holds an uncommitted
change and a reviewer writing into it is how that gets destroyed. Use
%TEMP% or any path outside the repo.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with PYTHONUTF8=1. PDFs are cached under tmp/scan_cache, keyed by the first
24 hex of a sha256 of the source URL.

THE QUESTION: HOW MUCH HEADING MACHINERY SURVIVES A PAGE WINDOW?

If the registry records each SCENE's page range, the heading search stops
being a hunt through a 472-page volume and becomes a look inside two or three
pages. That may delete most of the fragile machinery -- or it may not, and
saying which is the job.

1. MEASURE THE WINDOWS. For each of the eight Spanish cells, give the PDF
   page index where the scene STARTS and where it ENDS, measured from the
   cached files. Say how you determined each one and how confident you are.

       es king_lear 1.1 · es midsummer 3.1 · es midsummer 3.2
       es tempest 3.1 · es twelfth_night 1.5 · es twelfth_night 2.5
       es much_ado 2.3 · es much_ado 3.1

   Two of these -- the much_ado pair, in "Otelo - Mucho ruido para nada" --
   have never been measured by anyone, so that volume needs mapping first.

2. ATTACK THE DRIVER'S OTHER FIX, WHICH MAY NOT WORK. The proposal for the
   intra-play theft you found is to record the label EXACTLY AS PRINTED,
   chrome included -- ask for `ESCENA PRIMERA .` with its space and period so
   the exact pass matches 13661 before 15015. But measured here, Macpherson
   prints that identical chromed string FIVE times:

       13661   16026   19543   20057   22223

   So chrome alone does not disambiguate; it only works if the ACT label
   reliably scopes the search first -- and Macpherson's act headings are OCR
   damaged (`AOTO PRIMERO .`, and `AOTO` / `TERCERO .` split across lines
   where the rejoin cannot help because `AOTO` is not a heading word).
   Does the chrome fix actually hold for all six Macpherson and Clark cells,
   or does it only appear to because the first occurrence happens to be the
   right one? Test each cell, and say plainly if the fix is luck.

3. WITH A PAGE WINDOW, WHAT CAN BE DELETED? Be concrete and name the code.
   Inside a two-or-three page window, does the caller still need an act
   label at all? A scene label? `--end-label`? The `_RUNNING_HEADER` strip?
   For each piece of machinery you would keep, say what breaks without it.
   **If a page window makes act labels unnecessary, say so in one sentence --
   that deletes the `AOTO` problem outright rather than working around it.**

4. WHAT A PAGE WINDOW DOES NOT FIX. Press on this hardest, because it is
   where the design will fail:
   * A scene STARTS MID-PAGE. Page granularity cannot bound it, so something
     still has to find the start inside the first page. What?
   * TWO scenes on one page -- does that occur in these volumes? Measure.
   * Clark misprints Twelfth Night 2.5's heading as `ESCENA II.` (you found
     this; it is a defect in the BOOK). With a page window, does that cell
     become vendorable, and what identifies the scene start if its printed
     label is simply wrong?
   * The LAST play running into the back-matter `ÍNDICE`. A page window
     should close this -- confirm it does.

5. HOW WOULD A WINDOW BE WRONG, AND WOULD ANYONE NOTICE? A mis-recorded page
   range is a silent defect of exactly the kind this corpus keeps shipping.
   Name the cheapest check that a recorded window actually contains the scene
   it claims -- something the vendor run can assert every time, not something
   a human verifies once.

DO NOT propose changes to extract() or find_label() in
scripts/otr_vendor_shakespeare.py. The operator scoped that file out and it
is shared with 38 HTML-vendored scenes already shipping; a fix needing it is
a finding to report, not a recommendation to make.

FINISH WITH THREE LISTS, headed exactly:

    WINDOWS:
      one line per cell -- start page, end page, how measured, confidence

    STILL NEEDED AFTER A WINDOW:
      the machinery that cannot be deleted, each with what breaks without it

    REFUTED:
      every claim in this brief you could not confirm against the real files

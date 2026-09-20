Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, near commit 8c5fdd85.

YOUR ONLY JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a file, a
line, or a number you produced yourself. Where you cannot ground a claim, say
"refuted -- could not confirm" rather than agreeing with it. Read-only: do not
edit any repository file, do not run the render pipeline, do not run any git
command that writes, do not touch the GPU. You MAY write and run a throwaway
read-only probe script; delete it when done.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with PYTHONUTF8=1. The PDFs are already cached under tmp/scan_cache (filename
is the first 24 hex of a sha256 of the source URL), so nothing downloads.
scripts/otr_vendor_scan.py gives you pdf_text(), strip_running_titles(),
_SPLIT_HEADING and _RUNNING_HEADER; scripts/otr_vendor_shakespeare.py gives
you find_label() and extract().

WHAT YOUR LAST PASS ESTABLISHED, AND WHAT IT DID NOT

Your front-matter finding was confirmed independently and it killed the
proposed play-slice as specified. Measured here afterwards: in the Macpherson
volume `EL REY LEAR` first matches at flat line 14 and
`SUENO EN NOCHE DE VERBENA` at flat line 15, both on the volume's title page,
so a slice between sibling anchors degenerates to a single line. That part of
your report stands.

Your headline claim -- that slicing merely shifts the defect from cross-play
to intra-play -- was NOT reproduced. Inside the Clark volume's LA TEMPESTAD
range there is no exact `ESCENA II.` for the first pass to steal, so the
prefix pass correctly returns line 1882. The mechanism you describe is real;
the instance was not found. Do not re-argue it from mechanism. If you want it
to stand, produce the measured case.

Your `_SPLIT_HEADING` recommendation was confirmed and undersold. Widening the
numeral line to tolerate trailing punctuation gives:

    Clark        rejoins now 0  ->  12    (6 spellings newly caught)
    Macpherson   rejoins now 0  ->  19    (8 spellings newly caught)
    pt Macbeth   rejoins now 0  ->   0
    pt Rei Lear  rejoins now 0  ->   0

The first column is the part worth noticing: the current rejoin fires ZERO
times in either Spanish volume, so every split heading in both is invisible
today. Both shipped Portuguese scenes are untouched.

THE QUESTION FOR THIS PASS, AND IT IS THE FIRST ONE ONLY IF IT SETTLES THINGS

Every boundary measurement taken so far was taken on text where NO split
heading had been rejoined. The widening makes 31 headings appear that the
boundary finder has never seen. So:

1. RE-MEASURE THE BOUNDARY BEHAVIOUR WITH THE WIDENING APPLIED, AND ASK
   WHETHER ANY SLICE IS STILL NEEDED AT ALL. Apply the widened rejoin, then
   run `extract()` with NO play anchor for each cell below and report the
   returned span length and its first and last three lines:

       es tempest 3.1        act `ACTO III.`      scene `ESCENA PRIMERA`
       es twelfth_night 1.5  act `ACTO PRIMERO`   scene `ESCENA V`
       es twelfth_night 2.5  act `ACTO SEGUNDO`   scene `ESCENA V`
       es king_lear 1.1      act -- read it off `--probe`; page 244 prints
                             `AOTO PRIMERO .`, with the C misread as an O
       es midsummer 3.1      act and scene off `--probe`
       es midsummer 3.2      act and scene off `--probe`

   For each: does the span stay inside its own play, and is its length
   plausible for that scene? A span of tens of thousands of characters is not.
   Choose sensible end labels yourself and SAY which you used -- that choice
   is part of the finding.

   **If the widening alone bounds every cell correctly, say so plainly. That
   result would delete an entire planned mechanism, and it is the most
   valuable sentence you could write.**

2. IF SOME CELLS STILL RUN AWAY, name them and name the exact heading that
   was missed. Then, and only then, attack the replacement design below.

THE REPLACEMENT DESIGN, TO BE ATTACKED ONLY IF (1) SHOWS IT IS STILL NEEDED

Rather than slicing between the first match of each sibling anchor, the
edition registry would record WHICH OCCURRENCE of the anchor begins the play
-- read off a probe by hand, the way `EDITION_LABELS` in
scripts/otr_vendor_shakespeare.py already treats a label as a measurement
rather than an inference. Read the comment above `EDITION_LABELS` before
judging it.

  3. What makes a recorded occurrence index right or wrong over time? Consider
     what happens to it when anything upstream changes the number of lines --
     the widened rejoin itself, a furniture rule, a different PyMuPDF version.
     If you think the index is fragile, say what invariant you would record
     instead, and check that your alternative is actually unique in BOTH
     Spanish volumes. An alternative that is unique in one and ambiguous in
     the other is not an answer.

  4. THE LAST PLAY IN A VOLUME still has no following sibling to bound
     against. You raised this and it was not answered. Say what bounds the
     last play, and whether that matters for es/twelfth_night 1.5 and 2.5,
     es/midsummer 3.1 and 3.2 -- name which of those four sit in the last play
     of their volume.

  5. Does the widened rejoin change WHICH line the play anchors themselves
     match at? The anchors are matched on the same flat text. Re-measure the
     Macpherson anchors after widening and say whether line 14 and line 15
     move.

DO NOT propose changes to `extract()` or `find_label()`. The operator has
ruled that scripts/otr_vendor_shakespeare.py is not open in this ticket, and
it is shared with 38 HTML-vendored scenes already shipping. A fix that
requires it is a finding to report, not a recommendation to make.

FINISH WITH TWO LISTS, headed exactly:

    STILL UNBOUNDED:
      one line per cell whose span escapes its scene or its play after the
      widening, with the missed heading named

    REFUTED:
      every claim in this brief you could not confirm against the real files

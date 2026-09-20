Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, near commit d9a5d82a.

YOUR FINDING WAS CONFIRMED AND IT CHANGED THE TICKET. Now attack what
replaced it, because the whole lane is about to be rebuilt on a second
assumption and nobody has tested that one either.

YOUR JOB IS TO REFUTE AND TO MEASURE. Ground every claim in a file, a line,
or a number you produced yourself. Where you cannot ground a claim, say
"refuted -- could not confirm" rather than agreeing with it. Read-only: do
not edit any repository file, do not run the render pipeline, do not run any
git command that writes, do not touch the GPU. You MAY write and run a
throwaway read-only probe; delete it when done.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with PYTHONUTF8=1. PDFs are cached under tmp/scan_cache; the filename is the
first 24 hex of a sha256 of the source URL. Clark "La tempestad - La noche
de Reyes" is 0fe883861a983e5a818ad82b.pdf, sha256
b79223db195e6b0c75bfed0b14c8d4120171ab0f99d8731e14f269db76f129d3.

WHAT WAS CONFIRMED FROM YOUR LAST PASS

Reproduced here independently, twice:

  * Clark page index 59. Rebuilt from word coordinates the page reads
    `MIR. Se me antoja / Que estás rendido.` and `FER. No, mi noble dueña:`.
    Flattened by page.get_text() those two cues are emitted at the BOTTOM of
    the page, detached from their dialogue, so Prospero absorbs both
    speeches. Your claim, exactly.
  * Domingos Ramos Macbeth page index 36 -- a scene ALREADY SHIPPING. Visual
    order is `MACBETH / ...tão horri- / vel! / BANQUO / Que distancia...`.
    Flattened, `vel!` is thrown to the TOP of the page and lands on the
    `TODAS TRES` speech. Same root cause, in vendored data.

The operator ruled the same day that a mangled native text beats a clean AI
translation, and that the gates therefore refuse MISATTRIBUTION and never
IMPERFECTION. That makes this the one class of defect still worth refusing,
so reading order is now the prerequisite and the label registry sits on top
of it.

THE REPLACEMENT, AND THE REASON TO DISTRUST IT

The plan is to stop using page.get_text() and rebuild each printed row from
word coordinates. The probe used to confirm your finding did this:

    words = sorted(page.get_text("words"), key=lambda w: (w[1], w[0]))
    # cluster consecutive words into a row while abs(y0 - row_y0) <= 3.0
    # then order each row by x0

IT IS ALREADY VISIBLY WRONG IN PLACES, which is why you are being asked. On
the same Clark page 59 it split single printed lines into fragments:

    row: "Estosmaderos que tusmiembros rinden"
    row: "."
    row: "Suéltalo ven descansa Cuando arda"
    row: ", y , ."
    row: "Mi , ."
    row: "padre en hondo estudio está sumido,"

Commas and periods are landing on their own baselines and breaking rows, and
once a row breaks, the x-ordering inside it is meaningless. A reader who
trusted this output would conclude the book is gibberish.

ANSWER THESE, EACH GROUNDED IN A MEASUREMENT

1. WHAT IS THE RIGHT ROW RULE? The y0 tolerance is a guess. Determine
   empirically what clusters a printed row correctly on these volumes.
   Consider using the span or line structures from page.get_text("dict")
   rather than "words", using the baseline or the vertical midpoint instead
   of y0, and clustering on character height. Give the rule you would ship
   and the evidence for it. State the tolerance as a number and say whether
   one number works for ALL FIVE volumes -- the three Spanish and the two
   Portuguese -- or whether it has to be per-edition, which would be a much
   worse answer and should be said plainly if true.

2. PROVE THE RULE WITHOUT ASSUMING IT. A reconstruction cannot be validated
   by reading its own output. Propose and RUN a check that is independent of
   the row rule. The obvious candidate: the multiset of words must be
   identical between flattened and reconstructed text, so only ORDER differs
   -- if a word is lost or duplicated, the rule is broken regardless of how
   plausible the rows look. Run that across every page of all five volumes
   and report the failures.

3. THE HANGING CUE. The whole point is that a marginal speaker label sits to
   the LEFT of its first dialogue line. Is x-order within a row sufficient to
   put the cue first, always? Look for a page where the cue's x0 is NOT the
   smallest on its row -- an indented verse line, a centred stage direction,
   a line beginning with a quotation dash. Name it if it exists.

4. VERSE IS THE HARD CASE, NOT PROSE. The Clark Tempest is set in verse:
   short, sometimes centred, sometimes indented lines. Does baseline
   clustering ever MERGE two verse lines into one row, or split one? Measure
   on the Tempest 3.1 pages specifically and say how often.

5. THE CUE THAT IS NOT THERE AT ALL. You reported 25 printed cues against 24
   in the text layer -- the final `FER.` before "¡Adiós! ¡adiós mil veces!" is
   absent from the text layer entirely. Coordinates cannot recover a glyph
   that was never extracted. So: what is the CHEAPEST RELIABLE DETECTOR that
   a cue is missing, one that does not require a human to read every page?
   Candidates to test rather than assert: a speech whose length is wildly out
   of family, a turn-order violation against the English scene, a vertical
   gap in the row sequence where a cue's x-band is empty, or rendering the
   page and testing the cue column for ink. Say which you would ship and what
   it costs per page.

6. DOES THE FIX ACTUALLY REPAIR THE SHIPPED SCENES? Two Portuguese scenes are
   already vendored from this extractor: pt/king_lear 1.1 and pt/macbeth 1.3.
   Reconstruct both scenes in coordinate order and DIFF against the stored
   files under config/source_banks/shakespeare/translations/pt/. Report how
   many speeches change speaker, how many lines move, and whether the result
   is better or merely different. The claim being tested is that
   reconstruction repairs `horriBANQUO` and the displaced `vel!` outright. If
   it introduces a NEW misattribution anywhere in either scene, that is the
   most important sentence in your report.

DO NOT propose changes to extract() or find_label() in
scripts/otr_vendor_shakespeare.py. The operator scoped that file out and it
is shared with 38 HTML-vendored scenes already shipping; a fix needing it is
a finding to report, not a recommendation to make.

FINISH WITH THREE LISTS, headed exactly:

    SHIP THIS RULE:
      the row-reconstruction rule you would implement, stated precisely
      enough to code from, with its tolerance and its validation

    STILL BROKEN AFTER IT:
      what coordinate ordering does NOT fix, the missing cue included

    REFUTED:
      every claim in this brief you could not confirm against the real files

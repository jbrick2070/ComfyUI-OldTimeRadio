Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

PRIORITY TWO. YOUR DETECTOR SPEC IS ACCEPTED AND IS BEING BUILT. Its
runtime side was confirmed by another lane: `_bind` returns an unbound
speech unchanged, `lock_cast` seats every passage speaker, `_unseated`
cannot raise, and `_is_upper_label` refuses `Mır` and `Fer` but takes
every upper-cased form -- so emission goes through `clean_label`, and
unbound labels are OMITTED from `speaker_map` because an empty roster
entry is malformed at nodes/_otr_verbatim_corpus.py:669.

What is NOT yet anywhere is the data your spec depends on: the FOLD
tables. You wrote "BUF + BUR is one Fool -- fold them in the edition table
or pay two voices." Nobody has measured that table. It is yours, because
your cue inventories are the only complete ones.

YOUR JOB IS TO MEASURE THE FOLD TABLE FOR EVERY CELL, THEN TRY TO BREAK IT.
Read-only on the repo; scratch in %TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

USE THE CORRECTED READER. Commit 0b35b419 fixed the reader's character
ownership -- your last inventories were on flattened text, and the reader
changes which cues sit on which row. Run every inventory through
`pdf_text(url, reading_order="coordinates")` and the `--pages` windows
already measured (58-62, 110-122, 136-144, 220-230, 230-234, 244-255,
414-421, 421-440). If a count moves against your flattened numbers, say
so; that is a finding about the reader, not about the table.

FOR EACH OF THE EIGHT CELLS, PRODUCE THE TABLE, precise enough to load:

    printed form  ->  canonical spoken name  ->  roster target or UNBOUND
                      (upper-cased)               (exact sidecar name)

with, per row, the COUNT of that printed form in the window and the
EVIDENCE that it is the same person as its siblings. "Same three letters"
is not evidence; the page image, the alternation, or the PERSONAJES list
is. Group siblings: `BUF`/`Buf`/`BUR` under one spoken name; `Tor`/`Toe`/
`Tok` under TOB; `Pep`/`Per`/`Pes`/`ÞED` under PED; `ELEN`/`Elen` under
ELEN with target HELENA. Mark every one-occurrence OCR form.

THEN ATTACK IT:
  1. AMBIGUITY INSIDE ONE EDITION. Is any printed form two people in the
     same BOOK? You measured `SEB` as Sebastian "later in the same Clark
     book" -- is that a different play, and does the fold therefore have
     to be scoped per SCENE rather than per volume? Find every form that
     appears in two windows of one volume and say whether it means the
     same person in both.
  2. THE FORBIDDEN TABLE. The operator forbade a global Spanish alias
     table. What structural property of YOUR table keeps it edition-scoped
     -- the sha256 key? the scene key? both? Say what would make it drift
     into the forbidden shape and what stops it.
  3. TRANSLATOR'S OWN PARTS ship UNBOUND by ruling (TISBE for Folger's
     FLUTE, the Spanish mechanicals). Separate those from OCR damage in
     every cell: a row is either a damaged spelling of a roster name (fold
     it, target the roster) or the translator's own name for a part (keep
     it, target UNBOUND). Getting that wrong either steals a voice or
     invents a binding. Name your rule for telling them apart.
  4. THE ASSERTION. A stale fold entry must fail loudly. Propose the check
     a write run makes -- "every form in the table occurs in the window",
     or stronger -- and say what it catches and what it cannot.
  5. THE VOICE COST. Under your table, how many DISTINCT VOICES does each
     cell need, against how many the English roster has? A cell needing
     nine voices for a six-name scene is shipping three OCR ghosts; say
     which cells do and whether folding closes the gap.

DO NOT propose changes to extract() or find_label().

FINISH WITH:
    TABLES:      the eight fold tables, loadable
    PER-SCENE:   yes / no -- must the fold be scoped per scene, with the
                 form that decides it
    VOICES:      per cell, distinct voices needed vs roster size
    REFUTED:     every claim above you could not confirm

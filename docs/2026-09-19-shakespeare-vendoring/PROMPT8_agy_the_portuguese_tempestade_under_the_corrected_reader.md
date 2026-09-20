Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

THE TWO CELLS THAT WERE CALLED READY ON THE MORNING OF 2026-09-19 AND
FELL TO A PROBE THAT AFTERNOON. `pt/tempest 1.2` and `pt/tempest 3.1`,
Domingos Ramos 1914, source
https://upload.wikimedia.org/wikipedia/commons/6/68/William_Shakespeare_-_A_Tempestade_%28trad._Domingos_Ramos%2C_1914%29.pdf
(216 pages, cached under tmp\scan_cache by the first 24 hex of the URL's
sha256). They were blocked on two measured defects:

  D1  Ferdinand is printed FERNANDO. Eight characters, so it clears the
      resolver's five-character floor and then fails the PREFIX test --
      neither FERNANDO nor FERDINAND starts with the other. His every
      speech was discarded and merged into the previous speaker. Act 3
      scene 1 came back as 14 speeches from two mouths in a scene he
      opens.
  D2  A page whose edge band OPENS with the scene heading kept its whole
      band, because a scene line is deliberately not free in the furniture
      walk and heading shapes never enter the running-title vote. Page 112
      survived as `SCENA I / 95 / A TEMPESTADE / FERNANDO`, and the header
      joined the caps run behind it: `TEMPESTADE PROSPERO x3`,
      `FERNANDO MIRANDA x1` -- each a real label eaten.

Since then the lane changed under both defects, and nobody has re-measured
them. That is your job -- this is the boundary layer you have owned all
day, and both defects are boundary-adjacent.

YOUR JOB IS TO RE-MEASURE, AND TO REFUTE THE CLAIM THAT EITHER DEFECT IS
STILL LIVE. Read-only on the repo; scratch in %TEMP%, never the working
tree; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

WHAT CHANGED SINCE THE DEFECTS WERE MEASURED (all on main):
  * `--pages START-END` addressing (76f88d8c). Act headings measured on
    this volume: ACTO PRIMEIRO p18, SEGUNDO p62, TERCEIRO p110, QUARTO
    p140, QUINTO p162. Scene 1.2 is `SCENA II` under ACTO PRIMEIRO and ends
    at ACTO SEGUNDO; scene 3.1 is `SCENA I` under ACTO TERCEIRO and ends
    at `SCENA II` (around p117). MEASURE THE WINDOWS YOURSELF off --probe;
    those page numbers are a starting point, not a result.
  * The coordinate reader, opt-in, `pdf_text(url, reading_order=
    "coordinates")`, fixed in 0b35b419 to own only a word's own glyphs.
  * A furniture rule for `<folio> <title>` rows joined by the reader
    (same commit), measured on the other two Ramos volumes only.
  * The standing ruling that an unresolvable cue is EMITTED as an unbound
    speaker, never merged (docs\OTR_STANDING_RULINGS.md, 2026-09-19). NOT
    YET BUILT -- today's `speeches_from_span` still discards. So D1 is
    still live in the code; what you are measuring is whether the
    proposed fix -- a scene-scoped fold `FERNANDO -> FERDINAND`, which the
    operator explicitly allowed and explicitly kept out of PLACE_NAMES --
    is the WHOLE of D1 or only part of it.

MEASURE, FOR EACH OF THE TWO CELLS, flat and coordinates side by side:
  1. The window, off `--probe`, and the span it yields with the label
     alone (no act label, no end label) inside that window. Does the
     scene finder still need `--end-label` on this volume once the window
     bounds it? This volume prints `ACTO I / SCENA II` as a running head
     on every page, which is exactly the case `--end-label` was built for.
  2. D2 under the reader plus the furniture rule: does page 112's band
     still survive? Quote the page's first six rows under coordinates,
     after `strip_running_titles`. Then count every `A TEMPESTADE`,
     `ACTO I`, `SCENA I/II` row that sits INSIDE the extracted span. Zero
     is the bar; name each survivor and the page it came from.
  3. D1 as it would stand with the fold: list every all-caps run the
     script reports as resolving to nobody, under both paths. Which of
     them is FERNANDO, which is furniture that D2 explains, and which is
     something ELSE nobody has named? That third set is the finding.
  4. The cast against the English sidecar (read it with
     nodes\_otr_roster_gender.py::load_roster_characters): 1.2 is ARIEL,
     CALIBAN, FERDINAND, MIRANDA, PROSPERO; 3.1 is FERDINAND, MIRANDA,
     PROSPERO. Report speech counts per character under flat and under
     coordinates, and -- the only number that matters -- any speech that
     lands in the WRONG mouth under either path, with the quoted line.
  5. The reader's `[scan] page N has K word(s) welded` warnings for pages
     inside either window, and whether any welded word is inside the
     scene span.

DO NOT propose changes to extract() or find_label() in
scripts\otr_vendor_shakespeare.py -- scoped out, shared with 38 shipping
scenes.

FINISH WITH THREE LISTS, headed exactly:
    STILL BLOCKED BY:   per cell -- D1 / D2 / something new, with the line
    READY WITH THE FOLD: per cell -- yes / no, and the window and label
                        that would vendor it
    REFUTED:            every claim above you could not confirm

Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 39bacab0.

YOUR FOLD TABLES ARE ON MAIN as
docs\2026-09-19-shakespeare-vendoring\GROK_FOLD_TABLES_measured.md --
recorded as measured data, not as a brief. The `(sha256, scene_stem)` key
is adopted and the `SEB` counterexample is what settles it. Your two
label corrections are adopted too: `es/midsummer 3.1` has no period in
its heading, and `es/twelfth_night 1.5` exists only under the coordinate
reader.

NOW THE ARGUMENT AGAINST YOUR OWN TABLES, BECAUSE THE RULING MAKES IT
ASYMMETRIC. An UNBOUND label costs a VOICE and never the dialogue -- the
operator has already agreed to pay that. A WRONG FOLD puts a speech in
another character's mouth, which is the one thing the 2026-09-19 ruling
still refuses. So the two errors are not equal: leaving a form unbound is
cheap and reversible, folding it wrongly is the failure the whole corpus
is built to avoid. Every row that rests on inference rather than on the
page is therefore a row that should probably NOT be folded.

You flagged one yourself: `Min` is "the weakest of those and is still not
'same three letters,' but it is not a photograph."

YOUR JOB IS TO REFUTE YOUR OWN WEAKEST ROWS. Read-only on the repo;
scratch in %TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.
You may render page images with PyMuPDF (`page.get_pixmap(dpi=216)`) and
read them -- another lane did exactly that to find a cue the text layer
had dropped entirely, at about 6 ms a page.

1. THE ONE-OCCURRENCE FOLDS THAT BIND TO A ROSTER NAME. These are the
   dangerous set: a single damaged form, folded onto a real character, on
   evidence that is an argument rather than an image. From your own
   tables:

       LENT -> KENT            REQ -> REGANIA         Gov -> GONERILA
       Min -> MIRANDA          Mır -> MIRANDA         MAJ -> MALVOLIO
       Max -> MARIA            Bur -> BUF (FOOL)      Dorg -> BORGOÑA
       Pes / ÞED -> PED        Boar / BORA -> BORRAS  Han / Ham -> HAMBRON
       Tr -> TITANIA           Chi -> CHICHARILLO     Blen -> ELEN
       HEK / Hek / Hør / UER -> HERMIA
       Lrs / Los -> LISANDRO   Prck / Pock -> PUCK
       Der / DER / Per / DEN -> DEMETRIO

   For EACH: render the page, look at the printed cue, and say what the
   INK says. Three outcomes only -- CONFIRMED BY IMAGE (the glyph damage
   is visible and the name is legible), STILL INFERENCE (the image does
   not settle it), or WRONG (the ink says another name). Report the page
   and the rendered crop's verdict, not a re-argument from PERSONAJES.

2. RE-RANK ON THE ASYMMETRY. For every row that comes back STILL
   INFERENCE, say whether you would now ship it UNBOUND instead. The test
   is not "is it probably right" -- it is "if it is wrong, does a speech
   move to the wrong mouth?" A fold that is probably right and costs a
   wrong-mouth when wrong is a worse trade than an unbound voice.

3. THE ONE THAT IS NOT LIKE THE OTHERS. `Der` / `DER` / `Per` / `DEN` all
   fold to DEMETRIO in midsummer 3.2, and `Per` is one edit from `Pen`
   and `Ped`, which are Don Pedro elsewhere in the corpus. Different
   book, so no collision -- but check the ink on `Per.` specifically. In a
   scene whose roster is LYSANDER, DEMETRIUS, HERMIA, HELENA, OBERON,
   ROBIN, is there any other candidate?

4. THE COUNTS THAT DECIDE THE BUILD, and nobody has written them down.
   Emission is being built now. For each of the eight cells, give the
   EXPECTED OUTPUT so the implementation is checked against a number it
   did not produce: total speeches emitted, speeches per bound roster
   name, and the exact list of labels shipping UNBOUND (upper-cased, as
   `clean_label` will write them). Where your answer to (2) moves a row
   from folded to unbound, use the post-(2) numbers. This is a golden
   expectation written BEFORE the code, which is the only kind that
   cannot be retrofitted.

5. WHAT WOULD MAKE YOU WRONG. Name the single form across all eight
   tables you would most expect a careful reader to overturn, and say
   what evidence would overturn it.

DO NOT propose changes to extract() or find_label().

FINISH WITH THREE LISTS, headed exactly:
    IMAGE VERDICTS:   per form -- CONFIRMED / STILL INFERENCE / WRONG,
                      with the page
    DEMOTE TO UNBOUND: every row you now decline to fold, with the reason
    EXPECTED OUTPUT:  per cell -- total speeches, per-character counts,
                      and the unbound label list

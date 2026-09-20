You are testing whether a ruled design can actually SHIP, in a real repository
at C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
on branch main, near commit 0c437238.

YOUR ONLY JOB IS TO REFUTE, AND TO MEASURE. Ground every claim in a file, a
line, or a number you produced yourself. If you cannot ground a claim, say
"refuted -- could not confirm" rather than agreeing with it. Read-only: do not
edit any file, do not run the render pipeline, do not run any git command that
writes, do not touch the GPU. You MAY write and run a throwaway read-only
Python script; delete it when done.

Python:  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
with $env:PYTHONUTF8=1 set. The scanned PDFs are already cached under
tmp/scan_cache (filename = first 24 hex of sha256 of the source URL), so
nothing needs downloading. scripts/otr_vendor_scan.py has pdf_text(),
strip_running_titles() and a --probe mode you can reuse.

THE QUESTION, AND IT IS THE ONLY ONE THAT MATTERS HERE

Eight Spanish cells are meant to be vendored. The operator ruled that they may
only be written once a set of FAIL-CLOSED gates passes. A gate set that no
real scene can satisfy is not a safe design -- it is option C wearing option
B's clothes, and it would burn the whole implementation before anyone noticed.

So: FOR EACH OF THE EIGHT CELLS, would the gates pass or abort? Measure it.

THE EIGHT CELLS (from config/source_banks/shakespeare/translations/leads.json)

    es king_lear 1.1          Obras dramaticas ... Tomo I (1897), Macpherson
    es midsummer 3.1          same volume
    es midsummer 3.2          same volume
    es tempest 3.1            La tempestad - La noche de Reyes, Jaime Clark
    es twelfth_night 1.5      same volume
    es twelfth_night 2.5      same volume
    es much_ado 2.3           Otelo - Mucho ruido para nada, Jaime Clark
    es much_ado 3.1           same volume

THE GATES, AS RULED

  1. Every detected source cue is either resolved by the exact edition
     registry, or is a documented scene-specific non-speaking exception.
  2. ANY unknown cue-shaped source token FAILS the run. Logging and
     continuing is not acceptable.
  3. Every registry target must validate against the exact scene roster
     sidecar. A label may not bind to a character absent from that scene.
  4. The source-cue inventory must reconcile with the extracted speech
     inventory. A cue missed by extraction fails the run rather than being
     absorbed into the previous speaker.
  5. Every emitted label must have a manifest speaker_map binding unless it
     is a documented intentional unbound label.
  6. The scene boundary must be supplied and verified from the edition's real
     act/scene labels.
  7. A gate failure leaves scene text and manifest byte-for-byte unchanged.

MEASURED FACTS TO START FROM -- CHECK THEM, DO NOT ASSUME THEM

  * With the Clark volume correctly sliced to one play, `es/tempest 3.1` is a
    4,822-character span whose line-initial cues are exactly:
        FER x6, Fer x4, MIR x5, Mir x4, Prós x3, Mır x1, Min x1
    where `Mır` carries a DOTLESS I (U+0131) and `Min` appears to be a misread
    of `Mir`. Three characters, seven forms.
  * `es/twelfth_night 1.5` currently cannot be located at all: extract()
    returns "scene heading 'ESCENA V' not found under ACTO PRIMERO", because
    the volume prints `ESCENA` over `V .` on two lines and _SPLIT_HEADING
    requires the numeral to end its line.
  * The Macpherson volume is 472 pages holding TWO plays plus 228 pages of
    front matter, and its act heading on page 244 reads `AOTO PRIMERO .` --
    the scanner turned a C into an O.

WHAT TO PRODUCE

1. A TABLE, one row per cell: the scene's line-initial cue inventory with
   counts, the number of DISTINCT printed forms, how many of those forms are
   obvious scanner corruptions, and the English scene roster size. This is the
   deliverable; everything else follows from it.

2. PER CELL, A VERDICT: would gate 2 (unknown cue aborts) fire? Name the
   token. A cell where gate 2 fires on a form nobody can confidently map is a
   cell that CANNOT be vendored under this ruling -- say so plainly.

3. THE COUNT THAT DECIDES THE DESIGN: of the eight cells, how many would pass
   all seven gates with a table built ONLY from forms whose meaning the source
   itself proves? If that number is small, say the number and say which cells
   they are. Do not soften it.

4. GATE 4 IS THE SUBTLE ONE. "Reconcile the cue inventory with the extracted
   speech inventory." Read speeches_from_span() in scripts/otr_vendor_scan.py
   and say precisely what "missed by extraction" would mean there -- in
   particular, note what happens to text BEFORE the first recognised mark
   (read the emit loop carefully; it does not do what a reader expects). Then
   say whether gate 4 as worded would catch it.

5. THE OTHER DIRECTION. Is any gate UNSATISFIABLE for a reason that has
   nothing to do with Spanish -- that is, would it also abort the two
   Portuguese scenes already shipping (pt/king_lear 1.1, pt/macbeth 1.3) if
   applied to them? Test it. The ruling requires existing Portuguese behaviour
   to remain intact, so a gate that would abort them must be scoped, and you
   should say which ones those are.

DO NOT propose a redesign. Measure whether the ruled design can ship, and say
what it costs.

FINISH WITH TWO LISTS, headed exactly:

    CANNOT SHIP UNDER THIS RULING:
      one line per cell, with the blocking token or heading named

    REFUTED:
      every claim in this brief you could not confirm against the real files

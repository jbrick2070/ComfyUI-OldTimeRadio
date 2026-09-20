Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 14228ff0.

THIS IS THE ONE ARCHITECTURE QUESTION LEFT ON THE TICKET, and it is a design
question with more than one defensible answer, which is why it goes to you
and not to a measuring lane. Everything below it is measured and shipped;
everything above it is ruled. What is missing is the SHAPE of the thing that
holds the measurements.

YOUR JOB IS TO DESIGN IT, THEN ATTACK YOUR OWN DESIGN. Ground every claim in
a file and a line. Read-only; scratch in %TEMP%; no git writes; no GPU.

WHAT IS SETTLED, SO YOU DO NOT RE-OPEN IT
  * A scene in a scanned volume is addressed by `--pages START-END`, zero
    based, measured off `--probe`. Commit 76f88d8c. All eight Spanish cells
    return complete casts inside their windows.
  * The volume is identified by the PDF's own sha256, never its URL. The
    five hashes are measured (agy, 2026-09-19); the manifest's `raw_sha256`
    signs the STORED TRANSCRIPT and is not a PDF identity.
  * Speaker cues are detected by SHAPE minus a closed refuse-list (Grok's
    spec, 2026-09-19): abbreviation `FER .`/`Fer.`/`Min .`, the Clark
    honorific `D . PED.`, the joint `ALB . Y CORN.`, the title initial
    `R.DEF`. An unresolved cue is EMITTED as an unbound speaker, upper-cased
    on write because `_is_upper_label` in nodes/_otr_passage_selector.py
    refuses `Mır` and `Fer` (measured). Six of eight cells reach zero
    wrong-mouth under it.
  * The operator FORBADE a global Spanish alias table and prefix inference.
    He ALLOWED an edition-scoped exact map, and a dedicated proper-name alias
    for `FERNANDO -> FERDINAND` that is never an entry in PLACE_NAMES.
  * `scripts/otr_vendor_shakespeare.py` is OUT OF SCOPE. It carries
    `EDITION_LABELS`, a `(play_anchor, act_label, scene_label)` tuple per
    HTML scene, and 38 shipping scenes depend on it. Read its comment: a
    label there is "a MEASUREMENT, never an inference." That philosophy is
    the one to extend, not the data structure.

WHAT THE REGISTRY MUST HOLD, per scene of a scanned edition
  * the PDF sha256 (volume identity) and the source URL (provenance only)
  * the page window and the printed scene label, exactly as printed
  * the FOLD table: printed forms that are one person. Grok measured
    `BUF` x18 + `BUR` x4 = one Fool; `Tor/Toe/Tok` = Toby; `Pep/Per/Pes/ÞED`
    = Don Pedro; `ELEN` = Helena, in-roster, so its 4-letter wrap test must
    be skipped. Without the fold, each spelling is a separate unbound voice.
  * the ROSTER target for a fold, when it has one (`BUF -> FOOL`,
    `D . PED -> PRINCE`, `PAJE -> BOY`) -- and NOTHING for a translator's own
    part (`TISBE` when Folger says FLUTE, the Spanish mechanicals), which
    ships unbound by the 2026-09-19 "translator's own choice" ruling.
  * what the vendor run ASSERTS every time: hash matches, label found in
    window, and the one check nobody has named yet -- that a form in the
    fold table actually OCCURS in the scene, so a stale table fails loudly.

DESIGN QUESTIONS, EACH WITH THE STRONGEST ARGUMENT AGAINST YOUR ANSWER
  1. WHERE DOES IT LIVE? Candidates: (a) new fields on the existing rows in
     config/source_banks/shakespeare/translations/leads.json -- read a
     Spanish scan row there first; (b) a new
     config/source_banks/shakespeare/translations/scan_editions.json keyed
     by sha256; (c) a Python dict in scripts/otr_vendor_scan.py beside
     FUNCTION_NAMES, the way EDITION_LABELS lives in the HTML script. Each
     is defensible. Pick one and say what the other two cost. Consider who
     EDITS it -- a window reading `--probe` output by hand -- and who READS
     it -- the vendor script only, never the runtime.
  2. THE FOLD TABLE IS ONE `git diff` AWAY FROM THE FORBIDDEN GLOBAL ALIAS
     TABLE. What structural property keeps it edition-scoped and exact? Is
     keying by sha256 enough, or must each fold entry ALSO name the scene,
     because `SEB` is Sebastian in one Clark play and could be someone else
     in the same book? Grok measured `SEB` as a real cue "later in the same
     Clark book". Decide the scope and defend it.
  3. FERNANDO -> FERDINAND is a PROPER-NAME alias the operator explicitly
     allowed and explicitly kept out of PLACE_NAMES. Is it a fold-table
     entry (edition-scoped) or a new module-level table (language-scoped)?
     The Portuguese Tempestade needs it; nothing else does yet. Say which
     and why the operator's PLACE_NAMES prohibition does or does not reach
     your choice.
  4. THE MANIFEST ROW. Today `otr_vendor_scan.py` writes
     `alignment_confidence: 1.0` and `verdict: READY` as literals, and
     `select_scene` reads confidence against a 0.8 floor (measured). With
     unbound labels now legitimate, what should the row carry so a reader
     can tell a scene with two unbound voices from one with none? A count?
     The list? And should `alignment_confidence` stay a literal, be dropped
     (the runtime reads it), or be derived from something honest?
  5. THE ASSERTION SET. Write the exact list a `--write` run must pass, in
     order, each with what it catches. Then name the one that would have
     caught `horriBANQUO` and the one that would have caught the 78,766
     character cross-play Tempest -- or say plainly that no cheap assertion
     catches the first, because that is likely true.

DO NOT propose changes to extract() or find_label().

FINISH WITH:
    SCHEMA:      the registry shape, as a JSON or Python sketch precise
                 enough to code from, with one real scene filled in
    ASSERTIONS:  the ordered list a write must pass
    AGAINST:     the strongest argument against your own schema
    REFUTED:     every claim above you could not confirm

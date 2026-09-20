# A schema of speaker-cue oddities in historical Shakespeare translations

Source document 0 of the NotebookLM set (the six segments are 1-6). Written
2026-09-19 from what the corpus work measured; every row cites where the
rule or the example lives. Status column: BOUND (resolves to an English
roster name), UNBOUND (ships raw, costs a voice and never a line -- ruling
2026-09-19), REFUSED (the whole cell is not vendored), NOT-A-CUE (removed or
left in the text).

## The object

A **speaker cue** is whatever a printed edition puts before a speech to say
who speaks it. In the English Folger text it is one shape: a name in
capitals. In the translations vendored here it is at least twelve shapes,
and a radio performance needs every one of them reduced to the same thing:
**exactly one owner per line** -- `speaker_map[label].spoken`, the
cast-row name the voice engine is handed [nodes/_otr_verbatim_corpus.py,
the speaker_map contract comment; nodes/OTR_LedgerScriptWriter.py, the
binding site near line 3825].

```
cue
  printed      the label as the edition prints it        "2.ª FEITICEIRA"
  kind         one of the kinds below                     ordinal_function
  edition      translator, year                           Domingos Ramos, 1912
  rule         what resolves it (or refuses it)           ORDINALS x FUNCTION_NAMES
  owner        the roster name, or none                   FIRST WITCH
  status       BOUND | UNBOUND | REFUSED | NOT-A-CUE
  recorded_in  manifest speaker_map | manifest folds | the text (raw) | nowhere
```

## The kinds, as measured

| # | kind | printed example | edition | rule that decides it | owner | status | recorded in |
|---|------|-----------------|---------|----------------------|-------|--------|-------------|
| 1 | name | `MIRANDA` | Ramos 1914 (pt Tempest) | exact match after accent fold [`resolve`, `label_key`] | MIRANDA | BOUND | speaker_map |
| 2 | translated_name | `Edmundo`, `Regane` | Ramos 1919 (pt Lear) | shared stem of 5+ chars, length within 2 [`resolve`, "A TRANSLATED PROPER NOUN"] | EDMUND, REGAN | BOUND | speaker_map |
| 3 | title | `DUQUE DE BORGONHA` | Ramos 1919 | title prefix off, then PLACE_NAMES [`_TITLE_PREFIX`, `PLACE_NAMES`] | BURGUNDY | BOUND | speaker_map |
| 4 | ordinal_function (the witches) | `1.ª FEITICEIRA`, `2. FEITICEIRA`, `BRUJA 1.ª` | Ramos 1912 (pt Macbeth); Macpherson (es) | an ordinal beside a function word, in EITHER order, because the languages disagree about which comes first [`ORDINALS`, `FUNCTION_NAMES`] | FIRST WITCH ... | BOUND | speaker_map |
| 5 | unison | `TODAS TRES` -> ALL; Folger `ALL`; `ALB . Y CORN` ("Deteneos, señor."); `Côro (dispersamente)` | pt Macbeth 1.3; Folger; es Lear 1.1 (Macpherson); pt Tempest 1.2 | `TODAS TRES` resolves through FUNCTION_NAMES to ALL; the joint Albany+Cornwall does not resolve and ships raw; the chorus is not on its own resolvable line and stays inside Ariel's song | ALL / none / (Ariel) | BOUND / UNBOUND / NOT-A-CUE | speaker_map / text / text |
| 6 | song_heading | `Canto de Áriel` | Ramos 1914, page 51 | a song word, a preposition and a name that resolves, on its own line [`_SONG_HEADING`, commit 18ffc540] | ARIEL | BOUND | speaker_map (as ARIEL) |
| 7 | abbreviation | `GLÓS`, `EDM`, `D . PED`, `Min` / `Mır`, `SEB` | Macpherson (es); Clark (es) | prefix of 5+ chars, else refused; `SEB` is TWO Sebastians in one bound volume | GLOUCESTER, EDMUND, DON PEDRO, MIRANDA / none | BOUND or UNBOUND | speaker_map / text |
| 8 | declared_fold | `FERNANDO` = FERDINAND | Ramos 1914 | operator-authorised, scoped to one invocation, refused unless the target is in the scene [`parse_folds`] | FERDINAND | BOUND | manifest `folds` |
| 9 | unbound | `REQ`, `Hek`, `DER`, `ALB . Y CORN` | Macpherson (es) | nothing reaches it and no fold is authorised; ships raw, upper-cased on write | none | UNBOUND | the text only |
| 10 | seated_silent | `AJUSTADO, ebanista` speaking | Macpherson, Midsummer 3.1 | the English gives Snug no speech in 3.1; the translator seated him | none | UNBOUND | the text only |
| 11 | interlude_role | `TISBE`, `PIRAMO`, `Pla`, `Pir` | Macpherson, Midsummer 3.1 | the cast list names the play-within-the-play's roles as personages apart from Flauta and Borras; folding them to FLUTE/BOTTOM would invent a second voice | none | UNBOUND | the text only |
| 12 | renamed_cast (the "Indian improved characters") | nine Hindi cells | Hindi editions found 2026-09-19 | a version that renames the cast is an ADAPTATION, not a translation [docs/OTR_STANDING_RULINGS.md, "ONE MODEL'S OCR COUNTS AS VERBATIM"] | -- | REFUSED (whole cell) | GO_FORWARD_PLAN row 1 |
| 13 | furniture | `MACBETH` at the head of 240 pages; `SCENA II A TEMPESTADE 31`; `SOENA II` (OCR) | Ramos 1912, 1914 | removed by WHERE it sits, never by what it says [`strip_running_titles`, `_edge_folio_parts`] | -- | NOT-A-CUE | nowhere (blanked) |

Sources for rows 7, 9, 10, 11: docs/2026-09-19-shakespeare-vendoring/GROK_FOLD_TABLES_measured.md.
Sources for rows 1-6, 8, 13: scripts/otr_vendor_scan.py (the rule named in
each row is a function or constant in that file, with a comment naming the
volume that forced it) and the manifest rows pt/macbeth 1.3, pt/king_lear
1.1, pt/tempest 1.2, pt/tempest 3.1.

## The in-unison dilemma, stated

Row 5 is the thesis's centre. A line Shakespeare gives to several people at
once is marked four different ways across four editions, and only one of
the four resolves to something the ledger can own (`ALL`). The other three
either ship as a raw label with no voice (`ALB . Y CORN`), stay inside a
neighbouring speech (`Côro`), or would need an invented owner. The
literary convention says "these people, together"; the performance ledger
says "one voice, now". What the runtime does with a roster name `ALL` is
the open measurement Segment 4 is asked to make.

## What the schema does NOT yet hold (proposals, not decisions)

These would be manifest additions and are a design question for the
operator, not something a window may add on its own:

- `speaker_map[label].kind` -- the kind column above, so a reader can tell
  a fold from a stem match from an ordinal without re-deriving it.
- `unbound: [labels]` -- today an unbound label is visible only by reading
  the text; the row says nothing.
- `unison: {label: [members]}` -- `ALB . Y CORN` = [ALBANY, CORNWALL];
  `TODAS TRES` = [FIRST WITCH, SECOND WITCH, THIRD WITCH]. The ledger would
  still pick one voice, but it would know who it was speaking for.
- `songs: {label: singer}` -- row 6 today is recorded as an ordinary ARIEL
  speech; nothing says it was sung.

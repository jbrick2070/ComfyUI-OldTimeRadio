# Manual panel prompts -- the Spanish scanned lane

Written 2026-09-19 for the operator to paste by hand into Grok/Composer, agy and
Codex. Each brief is scoped to a DIFFERENT question and each is briefed to
REFUTE, per the 2026-09-11 contrarian rule. Repo root on every box:

    C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

HEAD at the time of writing: `cc62b2c1`, branch `main`, tree clean.

---

## THE MEASURED FACTS ALL THREE BRIEFS SHARE

`scripts/otr_vendor_scan.py` vendors a Shakespeare scene out of a scanned PDF's
text layer. It is proven on two Portuguese volumes (Domingos Ramos 1912 Macbeth,
1919 Rei Lear) and those print speaker labels as FULL NAMES IN CAPITALS on their
own line -- `MACBETH`, `PROSPERO`, `1.a FEITICEIRA`. The resolver
(`resolve()`, lines ~240-301) anchors every all-caps run against that scene's
ENGLISH Folger roster and refuses anything that does not resolve.

Twelve more cells were filed as ready to run today. Probing every volume first
(2026-09-19, this session) shows they are not one job.

### 1. The two Portuguese Tempestade cells ARE ready

`A Tempestade` (trad. Domingos Ramos, 1914), 216 pages, text layer clean, same
house style as the two proven volumes: `MIRANDA` and `PROSPERO` stand alone in
capitals. Act headings: `ACTO PRIMEIRO` p18, `ACTO SEGUNDO` p62,
`ACTO TERCEIRO` p110, `ACTO QUARTO` p140, `ACTO QUINTO` p162. Running head is
`ACTO I` / `SCENA II` on every page, so `--end-label` is mandatory and the
script's own guard fires without it. No code change needed. NOT what these
briefs are about.

### 2. The eight Spanish cells are blocked, and the block is a DESIGN FORK

Both Spanish editions abbreviate their speaker labels and do not hold their
case. Verbatim from the text layer.

Macpherson, `Obras dramaticas ... Tomo I (1897)`, page 244 (El Rey Lear 1.1):

    AOTO PRIMERO .
    ESCENA PRIMERA .
    Entran KENT,GLOSTER y EDMUNDO
    KENT. - Pensaba que el Rey tenia
    GLOS. - Siempre
    KENT.-No es este joven hijo vuestro?
    Glos. - A mi cargo

Jaime Clark, `La tempestad - La noche de Reyes`, pages 58-59 and 110:

    FER . Juegos penosos hay , cuyas fatigas
    MIR . Por.Dios te ruego, no te afanes tanto !
    Fer. Duena querida , el sol se pondra antes
    Pros. (Aparte.) Oh misero gusano, estas cogido!
    Viol. Cuanto pudiere hare por ablandarla .
    MAR. Si no me dices donde estuviste ...
    BUF. Quemeahorque ...
    Bur. Porque ya no le es posible ver a ninguno.      <-- OCR of BUF.

So a label is (a) an ABBREVIATION of three to six characters, (b) sometimes
capitals and sometimes title case IN THE SAME SCENE, (c) followed by a period
and a dash or an em dash, (d) occasionally misread by the scanner (`Bur.` for
`BUF.`, `AOTO` for `ACTO`).

The current resolver cannot take any of them. Its stem rule (lines ~272-286)
refuses a candidate under five characters outright, with this reason written
into the file: *"putting a speech in the wrong mouth is the failure this corpus
cannot afford, and a near-miss on a short name is exactly how that happens."*
`GLOS` is four. `FER`, `MIR`, `MAR`, `BUF`, `ANT`, `SEB`, `GON` are three.

### 3. Both Spanish volumes hold SEVERAL PLAYS in one file

Measured page ranges: Macpherson Tomo I -- front matter to p227, `EL REY LEAR`
p228-369, `SUENO EN NOCHE DE VERBENA` p370-472. Clark -- `LA TEMPESTAD` p12-97,
`LA NOCHE DE REYES` p98-191. The shared boundary finder
`otr_vendor_shakespeare.extract()` ALREADY takes a `play_label` first argument
for exactly this, and `otr_vendor_scan.py` main() passes `None` into it and
exposes no `--play-label` flag.

### 4. A heading the text layer split is not always rejoined

`_SPLIT_HEADING` requires the numeral to end its line, so `ESCENA` over `V .`
(page 110, with the trailing period) is not rejoined and the scene heading is
invisible.

### 5. The two Portuguese Midsummer cells are NOT ready either

Their `Galeria:` page resolves through the imageinfo API to
`https://upload.wikimedia.org/wikipedia/commons/f/ff/William_Shakespeare_-_Sonho_de_uma_Noite_de_Verao_(trad._Castilho,_1874).pdf`
(the real filename carries an accent on Verao; 248 pages, text layer present).
But Castilho divides the play into QUADROS with scenes numbered continuously
inside each act -- `QUADRO I / SCENA I..VII`, `QUADRO II / SCENA VIII`, and so
on -- which is the same non-alignment already recorded for the French Hugo
Midsummer.

### The rule that governs all of this

From `docs/GO_FORWARD_PLAN.md`: *"A wrong or partial list does not fail loudly;
it moves a speech into another character's mouth, which is the one failure this
corpus cannot afford."*

---

## BRIEF A -- FOR CODEX  (the resolver fork; this is the hard one)

Codex reads a long brief from stdin: run `codex exec -` and paste, rather than
passing it as an argument.

> You are reviewing a design decision in a real repository at
> `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
> **Your job is to REFUTE, not to review.** If you cannot ground a claim in the
> actual files, default to "refuted" and say which file you could not confirm it
> in. Do not edit any file, do not run the render pipeline, do not write to git.
>
> Read `scripts/otr_vendor_scan.py` in full, especially `resolve()`,
> `speeches_from_span()`, `FUNCTION_NAMES`, `ORDINALS`, `PLACE_NAMES` and
> `_CAPS_RUN`. Then read `tests/test_vendor_scan_furniture.py` and
> `nodes/_otr_roster_gender.py::load_roster_characters`.
>
> THE PROBLEM. Two Spanish editions we want to vendor print their speaker
> labels as ABBREVIATIONS in inconsistent case, inline with the dialogue:
> `FER .`, `MIR .`, `Fer.`, `Pros.`, `Viol.`, `MAR.`, `BUF.`, `Bur.` (a scanner
> misread of `BUF.`), `KENT. -`, `GLOS. -`, `Glos. -`, `Gon.`, `ANT.`, `SEB.`.
> The resolver currently accepts only a full name that folds onto the scene's
> English Folger roster, and refuses any candidate under five characters,
> deliberately, because a near-miss puts a speech in the wrong character's
> mouth.
>
> THREE CANDIDATE ANSWERS. Attack all three; do not pick a favourite first.
>   (A) Resolve an abbreviation by prefix against the scene roster and accept it
>       ONLY when exactly one roster name matches, refusing every ambiguous one.
>   (B) Build a per-edition abbreviation table by hand, read off the volume's
>       own PERSONAJES page, and refuse anything not in it.
>   (C) Refuse the Spanish scanned volumes for this lane entirely and leave the
>       eight cells unvendored.
>
> ANSWER THESE, each grounded in a file and a line number:
>
> 1. For (A): find a real scene in `config/source_banks/shakespeare/sources/`
>    whose roster contains TWO names sharing a prefix of three or four
>    characters. Name the file and both characters. If you find one, (A) is not
>    safe as stated -- say what minimum length, if any, makes it safe, and
>    whether that length still admits `FER`, `MIR`, `MAR`, `BUF`.
> 2. For (A): the uniqueness test is against the SCENE roster. Does
>    `roster_for()` return the scene's cast or the play's? Read it. If it is the
>    play's, uniqueness is being tested against a larger set than the scene,
>    which makes it stricter -- or the reverse. Say which, and why it matters.
> 3. A scanner misread (`Bur.` for `BUF.`) resolves to nobody under (A) and (B)
>    alike, and `speeches_from_span` then silently glues that speech onto the
>    previous speaker. Trace that path in the code and say what the reader of
>    the finished file would see. Is there any existing check in this repo that
>    catches it? Grep for it; do not assume.
> 4. `Pros. (Aparte.)` -- does `_LABEL_QUALIFIER` reach a parenthetical that
>    FOLLOWS the label rather than closing the token? Read the regex. Answer yes
>    or no with the anchor that decides it.
> 5. Case. `FER .` and `Fer.` are the same man. `_CAPS_RUN` finds one and the
>    own-line rule finds neither, because neither is alone on its line. What
>    actually has to change for both to be found, and does that change risk
>    matching an ordinary capitalised word that opens a sentence? Give a Spanish
>    sentence from one of these volumes that would false-positive.
> 6. State which of (A), (B), (C) you would ship and the single strongest
>    argument AGAINST your own choice.
>
> Finish with a list headed `MUST-FIX:` -- items that would make the lane ship a
> speech in the wrong character's mouth -- and a separate list headed
> `REFUTED:` naming every claim in this brief you could not confirm.

---

## BRIEF B -- FOR GROK / COMPOSER  (the furniture blast radius)

> You are reviewing a real repository at
> `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
> `main`, HEAD `cc62b2c1`. **Your only job is to REFUTE.** Ground every claim in
> a file and a line. Do not edit files, do not run the render pipeline, do not
> run git commands that write.
>
> Read `tests/test_vendor_scan_furniture.py` FIRST, top to bottom, including the
> two tests at the bottom marked as KNOWN LIMITS. Its header says six of its
> thirteen rules were each broken again by the fix for the next one. Then read
> `strip_running_titles`, `running_titles`, `recurring_headings`,
> `_strip_one_edge` and `_page_edges` in `scripts/otr_vendor_scan.py`.
>
> THE CHANGE UNDER REVIEW IS NOT WRITTEN YET, WHICH IS THE POINT. We intend to
> extend this script to two Spanish scanned volumes. Tell us what those volumes
> will break BEFORE it is written. Measured facts about them:
>
> * `Obras dramaticas de Guillermo Shakespeare - Tomo I (1897)`, 472 pages,
>   holds TWO plays: `EL REY LEAR` pages 228-369 and
>   `SUENO EN NOCHE DE VERBENA` pages 370-472, with 228 pages of biography in
>   front of both.
> * Its running head is the play title -- `EL REY LEAR .` -- and the scanner
>   mangles it differently on almost every page: `EL REY LEAD`,
>   `SUERO EN NOCHE DE VERBENA`, `SUENO EV NOCHE`, `SUERO EN NOCDE DE`,
>   `sueSO EN NOCHE DE VERBENA .`, `SUENO EN NOCHE DE` (truncated).
> * `LEAR` is both the running title's second word and the play's leading
>   character, who speaks on nearly every page of 1.1.
> * The act heading on page 244 reads `AOTO PRIMERO .` -- the scanner turned the
>   C into an O.
>
> ANSWER THESE:
>
> 1. `running_titles` counts a form only when the string REPEATS at a page edge,
>    floor `max(4, len(pages)//5)`. With 472 pages the floor is 94. The running
>    head is mangled into dozens of distinct spellings and each one is a
>    separate key. Does any single spelling reach 94? Reason it out and say what
>    happens to the ones that do not -- name the exact downstream consequence
>    for `LEAR`, citing `resolve()`.
> 2. The floor is a share of the WHOLE FILE, and this file is two plays plus a
>    biography. Is a share-of-volume floor still meaningful here? If you think
>    it is not, say what it should be a share of, and then say what that change
>    would do to each of the thirteen tests in
>    `tests/test_vendor_scan_furniture.py`. Name every test that goes red.
> 3. `test_known_limit_a_speaker_parked_in_the_edge_band_reads_as_furniture`
>    pins the cost of judging by position. On a 472-page volume where one play
>    occupies 142 pages, is that limit more or less likely to fire? Show the
>    arithmetic.
> 4. `_HEADING_SHAPED` matches lines opening with ACTO/ATTO/ACT/SCENA/ESCENA/
>    SCENE. `AOTO PRIMERO .` matches none of them. Walk the consequence through
>    `recurring_headings` and `_strip_one_edge` and say precisely what the
>    reader loses.
> 5. Name any change to this file that would fix an item above AND break one of
>    the thirteen tests. For each, name the test.
>
> Finish with `MUST-FIX:` and `REFUTED:` lists. Put anything you could not
> confirm against the real files under `REFUTED:`, not under `MUST-FIX:`.

---

## BRIEF C -- FOR agy  (the boundary layer; quick QA)

Drive it through the kibitz wrapper rather than a bare `agy -p`:
`python kibitz/scripts/kibitz.py --only agy ...`. If it times out the knob is
`KIBITZ_AGY_PRINT_TIMEOUT` (default 5m; 15m has worked), not
`kibitz.py --timeout`.

> Repository: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.
> **Refute, do not review.** Ground every claim in a file and a line number.
> Read-only: do not edit, do not run git writes.
>
> Read `find_label` and `extract` in `scripts/otr_vendor_shakespeare.py` (around
> lines 768-1016) and `main()` plus `_SPLIT_HEADING` and `_normalise_heading` in
> `scripts/otr_vendor_scan.py`.
>
> FOUR CLAIMS. Confirm or refute each one against the code, and say what a
> reader loses if it is true:
>
> 1. `extract()` accepts a `play_label` as its second positional argument and
>    documents it as REQUIRED on a multi-play volume. `otr_vendor_scan.py`
>    main() passes `None` there and its argparse exposes no `--play-label`. Two
>    Spanish volumes we are about to vendor each hold two plays. Confirm the
>    gap. Then say what happens today if someone asks that script for `ACTO III`
>    in the Clark volume, which holds `LA TEMPESTAD` at pages 12-97 and
>    `LA NOCHE DE REYES` at 98-191 -- which play answers, and does anything
>    downstream notice?
> 2. `_SPLIT_HEADING` rejoins a heading the text layer broke across two lines.
>    The Clark volume prints, on page 110, the two lines `ESCENA` then `V .` --
>    with a space and a full stop after the numeral. Read the regex and say
>    whether it rejoins that. If not, trace what `find_label` then does when
>    asked for `ESCENA V`, and whether the failure is loud or silent.
> 3. `find_label` refuses a prefix match when the next character is
>    alphanumeric, so `SCENA II` cannot match `SCENA III`. Does the same guard
>    hold for `ESCENA PRIMERA` against `ESCENA PRIMERA ,` and
>    `ESCENA PRIMERA .`, both of which occur in these volumes? Say which branch
>    of `find_label` answers, exact or prefix.
> 4. The refusal in `otr_vendor_scan.main()` fires when the caller's
>    `--scene-label` is a recurring head and no `--end-label` was given. The
>    Macpherson volume prints its PLAY TITLE as the running head, not its scene
>    label. Does the guard fire on that volume? If it does not, name the
>    concrete failure it was written to prevent and say whether that failure can
>    still occur here.
>
> Finish with `MUST-FIX:` and `REFUTED:`.

---

## ADDENDUM, MEASURED AFTER THE BRIEFS ABOVE WERE WRITTEN

**The two Portuguese Tempestade cells are NOT ready after all.** Both were run
dry against the real volume and both are unshippable. Correct the claim in
section 1 above; two new defects, both measured, and the second is a fork.

### D1 -- `FERNANDO` resolves to nobody, and he is the scene's lead

Domingos Ramos translates Ferdinand as **Fernando**. The two share only the
stem `FER`, and `resolve()` refuses a stem under five characters, so every one
of his speeches is dropped and its text glued onto whoever spoke last.

    pt/tempest 1.2  ->  123 speeches, 4 labels, `FERNANDO x9` unresolved
    pt/tempest 3.1  ->   14 speeches, 2 labels, `FERNANDO x9` unresolved

Folger 3.1 has three characters in it and Ferdinand opens the scene. Fourteen
speeches from two mouths is not that scene.

### D2 -- a page whose band OPENS with the scene heading keeps its whole band

`_strip_one_edge` walks a page's edge band inward and stops at the first line
that is not furniture. An act line beside a folio is free and does not end the
walk; a SCENE line is deliberately not free, and `running_titles` excludes
every heading-shaped string from `forms` as well. So a band that opens with the
scene heading halts the walk on its very first line and the entire band
survives. Measured on page 112 of the Tempestade, after
`strip_running_titles` has run:

    |SCENA I|
    |95|
    |A TEMPESTADE|
    |FERNANDO|
    |Nao, ente adoravel : prefiro que os tendoes se rachem ...

Page 113, whose band opens with the folio instead, is blanked correctly. The
surviving furniture then joins the caps run behind it, which is why the run
report reads `SCENA II TEMPESTADE x5`, `TEMPESTADE PROSPERO x3` and
`FERNANDO MIRANDA x1` -- each of those consumed a real speech label.

**THIS IS THE FORK, AND IT IS THE ONE I WANT ATTACKED.** Three candidates:

  (i)  Make a scene line free like an act line -- repeated at a page edge AND
       beside a folio -- but ONLY when the caller supplied `--end-label`, since
       that is exactly the flag that stops the boundary depending on scene
       headings. Note the end-label heading itself must still be findable
       afterwards.
  (ii) Leave the furniture walk untouched and drop heading-shaped lines INSIDE
       the extracted span instead, in `speeches_from_span`, which runs after
       both boundaries are already resolved and therefore cannot cost one.
  (iii) Deepen `_EDGE_DEPTH` past three so a four-line band is covered.

### WHAT TO ADD TO EACH BRIEF

**To BRIEF B (Grok / Composer), append:**

> 6. Read `_strip_one_edge` and the comment block above it that explains why a
>    scene line is NOT free while an act line is. Page 112 of the Portuguese
>    `A Tempestade` (1914) has the band `SCENA I` / `95` / `A TEMPESTADE` and
>    survives the strip entirely. Confirm the mechanism against the code and
>    name the line that halts the walk. Then attack all three candidate fixes
>    (i), (ii), (iii) stated in the addendum: for EACH, name every test in
>    `tests/test_vendor_scan_furniture.py` that goes red, and name the real
>    volume behaviour that made the current asymmetry necessary. Say which you
>    would ship and the strongest argument against your own choice.
> 7. For candidate (ii) specifically: `speeches_from_span` receives a span that
>    `extract()` has already bounded. Is there ANY path by which dropping
>    heading-shaped lines there could change a scene boundary? Trace it. If
>    there is none, say so plainly -- that is the claim the candidate rests on.

**To BRIEF C (agy), append:**

> 5. `_EDGE_DEPTH` is 3. The Portuguese `A Tempestade` prints a FOUR line band:
>    folio, `ACTO I`, `A TEMPESTADE`, `SCENA II`. Confirm the depth against the
>    code and say which of the four is never examined. Then say what raising
>    the depth would do to
>    `test_known_limit_a_speaker_parked_in_the_edge_band_reads_as_furniture` and
>    to `test_a_running_head_that_is_also_a_character_loses_the_header_not_the_speaker`.

**To BRIEF A (Codex), append:**

> 7. `PLACE_NAMES` maps a printed place to its Folger name because a fold
>    cannot reach it. Ferdinand is printed `FERNANDO` in Portuguese and
>    `FERNANDO` shares only three characters with `FERDINAND`, so the stem rule
>    refuses him and his speeches are glued onto the previous speaker. Is
>    adding an explicit per-name table entry, in the same shape as
>    `PLACE_NAMES`, the same class of decision as (A)/(B)/(C) above, or a
>    different one? Argue it. Then say whether an explicit table can be audited
>    for completeness against a scene BEFORE the scene is written -- name the
>    check, or say there is none.

---

## WHAT I AM DOING WHILE THESE RUN

Nothing is being vendored. Every one of the twelve cells is now blocked on one
of the questions above, which is the honest state and is a change from the
handoff. No code has been written and nothing has been pushed.

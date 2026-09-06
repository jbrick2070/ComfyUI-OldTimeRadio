# r3 anchor -- the WORSE defect the first two rounds missed

r1 and r2 converged on removing given-name aliases; that is BUILT, green, and it
fixes `COLONEL FITZWILLIAM`. Then a grounded fan-out found a defect neither round
looked for, and the driver REPRODUCED IT LIVE against the shipped fix.

## The live reproduction (driver-run, real sidecar, post-fix code)

`config/source_banks/public_domain_story/sources/pride_prejudice_proposal.provenance.json`
ships exactly TWO rows: `Elizabeth Bennet` (female), `Fitzwilliam Darcy` (male).

    MISS BENNET          -> female   short_form   ('ELIZABETH BENNET',)   correct
    MR BENNET            -> female   short_form   ('ELIZABETH BENNET',)   WRONG
    MR. BENNET           -> female   short_form   ('ELIZABETH BENNET',)   WRONG
    MRS BENNET           -> female   short_form   ('ELIZABETH BENNET',)   WRONG
    LADY BENNET          -> female   short_form   ('ELIZABETH BENNET',)   WRONG
    COLONEL FITZWILLIAM  -> unknown  none         ()                      fixed in r2

**Elizabeth's FATHER is pinned FEMALE, citing his daughter**, with
`gender_confidence` read through as `recalled`.

## Why this is WORSE than the bug we just fixed, and why the r2 fix cannot close it

* `COLONEL FITZWILLIAM` was a wrong CITATION with an accidentally right gender
  (both men). Cosmetic in output.
* `MR BENNET` is a WRONG GENDER **and** a wrong citation. It is strictly worse
  than not joining at all: the 40/40/20 roll gives him 40% male; the join pins
  him female with evidence text.
* Surname-only aliasing cannot fix it. `BENNET` genuinely IS the tail of
  `ELIZABETH BENNET`. Dropping given-name keys does nothing here.
* The two-row `ambiguous_join` abstention (`_verdict_from`, :343) cannot fire:
  Mr. Bennet has NO ROW. Same structural hole as the Colonel.

**The slot is throwing away the strongest evidence it has.** `MR` means male.
`MRS`/`MISS`/`LADY` mean female. `strip_honorifics` (:290) discards that token and
then joins on what is left.

## THE TWO CANDIDATE CLAUSES -- break them

**(A) HEAD-OF-FAMILY GUARD.** In `short_form`, refuse when the slot's residue
after stripping honorifics is a SINGLE token AND the matched candidate form is
MULTI-token. A title plus a bare surname may not bind a row recorded as
given-name + that surname.
* Keeps `SIR TOBY -> TOBY` (row is single-token, guard does not fire).
* Keeps `CAPTAIN AHAB -> Ahab`, `SCROOGE`.
* Blocks `MR BENNET`, `MRS BENNET`, `LADY BENNET` from Elizabeth's row.
* COST: `MISS BENNET` also stops matching. Is that acceptable, given (B)?

**(B) GENDERED-HONORIFIC RUNG (deterministic, render path, NO LLM).** The title
itself yields the gender with NO join and NO citation: MR/SIR/LORD/FATHER ->
male; MRS/MISS/MADAM/LADY/SISTER -> female. Ungendered titles (DR, PROF, CAPTAIN,
COLONEL, REVEREND) yield nothing.
* Recovers `MR BENNET` -> male and `MISS BENNET` -> female CORRECTLY, with no
  wrong-character citation at all.
* Where does it sit in the ladder -- BEFORE the roster join, or only AFTER the
  join abstains? Argue it. A title is weaker evidence than a confirmed roster row
  but stronger than a coin flip.
* What `gender_source` / `gender_confidence` does it stamp? It is not `roster`,
  not `recalled`, not `inferred`.
* Does it violate "never guesses from a name" (module docstring :19)? The driver's
  position: no -- a courtesy title is a STATED fact about the person in the slot,
  not an inference from a name. Attack that.

## Required of your answer

1. Run the REAL resolver against REAL sidecars. Name rows you actually read.
   Reviewer claims that were verified only against invented two-row fixtures have
   already produced one wrong "impossible" proof in this campaign.
2. Find a corpus row where (A) or (B) REGRESSES a currently-correct join. Sweep
   `config/source_banks/**/*provenance*.json` -- 266 rows, 79 sidecars.
3. Say whether (A) and (B) ship together or separately, and what the DONE test is.

Closed, do not reopen: no LLM on the render path; tier 4 declines honestly; the
11 Shakespeare unknowns stay; given-name aliases are already removed.

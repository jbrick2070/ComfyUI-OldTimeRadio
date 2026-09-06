# r2 JUDGMENT -- CONVERGED. Build authorized.

Driver: Claude/Cowork. Panel: Codex (spark, xhigh) + Antigravity (Gemini 3.8 Flash).
Settled by EXECUTION against the real sidecar, not by argument.

## The driver's proposed rule is REJECTED -- by the driver

I proposed refusing a given-name alias match when the slot carried a rank the row
lacks. Antigravity attacked the premise under it ("surname-only regresses
ELIZABETH") and was RIGHT. Ran `resolve_roster_gender` on the real
`pride_prejudice_proposal.provenance.json` rows both ways:

| slot | TODAY | given-name aliases REMOVED |
|---|---|---|
| `ELIZABETH` | female (exact) | **female (qualified)** |
| `ELIZABETH BENNET` | female (exact) | female (exact) |
| `MISS BENNET` | female (short_form) | female (short_form) |
| `MR. DARCY` | male (short_form) | male (short_form) |
| `DARCY` | male (exact) | male (exact) |
| `COLONEL FITZWILLIAM` | **male -- WRONG CHARACTER** | **unknown (none)** |
| `MISS ELIZABETH` | female (short_form) | **unknown (none)** |

`ELIZABETH` survives on the `qualified` tier
(`n.startswith(slot + " ")`, `nodes/_otr_roster_gender.py:410`) with no alias at
all. **My r1/r2 premise was wrong and the honorific rule is unnecessary.** The
simpler fix wins; no honorific special-casing enters the join.

## The one real cost, and it is accepted

`MISS ELIZABETH` stops resolving. **Antigravity raised exactly this case as a
MUST-FIX against MY rule, then proposed a fix that loses it too** -- its CUT #2
claims the negative rule is unnecessary, but surname-only drops `MISS ELIZABETH`
identically. Neither lane caught that; the execution did.

Accepted anyway: an honest `unknown` falls to the 40/40/20 roll, while the status
quo attaches a CONFIDENT WRONG CITATION to permanent provenance. Ranked by damage,
abstention wins. This is documented, not hidden.

Note `FITZWILLIAM` alone still resolves male via `qualified` against
`FITZWILLIAM DARCY`. Correct for Darcy, unfixable for the Colonel while he has no
row, and out of scope.

## WHAT GETS BUILT (both lanes converged on all four)

1. **Render join, read-through filter** -- `_candidate_names`
   (`nodes/_otr_roster_gender.py:318`) drops an alias equal to the GIVEN NAME of a
   multi-token row name. Refinement the driver adds: take the given name from the
   HONORIFIC-STRIPPED row name, so `Dr. Lira Kell` yields `lira`, not `dr`.
   Fixes every existing sidecar with ZERO data migration.
2. **Stamper** -- `_aliases_for`
   (`scripts/otr_stamp_character_genders.py:153`) stops emitting given names, so
   future stamps do not re-introduce it. Antigravity also found it borrows a
   27-title Shakespeare honorific set and so stamped `father` for `Father Brown`;
   use the 61-entry `_HONORIFICS` instead. VERIFY before changing.
3. **The pinned test** -- `test_a_given_name_alias_matches_and_that_is_the_recorded_limit`
   (`tests/test_gender_ladder_recall_and_floor.py:395-403`) currently pins the BUG.
   It is REPLACED with the honest-decline assertion plus companion invariants on
   the same REAL row (`MR. DARCY` male, `ELIZABETH` female via qualified,
   `ELIZABETH BENNET` exact, `MISS BENNET` short_form), and a documented assertion
   that `MISS ELIZABETH` now declines.

## REJECTED / CUT (both lanes agreed; driver concurs)

* **Alias-object schema** `{"alias","kind"}` -- breaks
  `otr_source_provenance_v1`, 79 sidecars and every reader. Unnecessary once given
  names leave the alias list.
* **Stamper-first re-stamp of 79 sidecars** -- pure churn plus a dual-box git
  conflict surface (CLAUDE.md 0B). The read-through filter fixes the live data now.
* **Re-running the LLM stamper for the 11 Shakespeare unknowns** -- antigravity
  grounded it: 2 are group speakers (`ALL`, `_EXCLUDED`), 3 are operator-LOCKED
  (`ARIEL`/`ROBIN`/`PUCK`), 6 are cached `unsure` from the 09-02 run. A re-run
  changes zero rows. Leave them to the roll.
* **Any LLM call in `nodes/`** -- closed in r1.

## Blast radius (CLAUDE.md 0B)

`_candidate_names` is shared render-path code, so it reaches BOTH boxes. It is a
pure function over sidecar rows with no hardware dependence; the proof required
before push is the before/after table above plus a green suite, not a GPU leg.

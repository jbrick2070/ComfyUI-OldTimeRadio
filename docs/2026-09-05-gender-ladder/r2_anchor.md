# r2 anchor -- the surname/given-name alias rule (the ONE open fork)

r1 settled the diagnosis and killed most of the row. Read `r1/final.md` for the
full judgment. The corrected state, all verified against the real files:

* The four-rung ladder ALREADY EXISTS in `scripts/otr_stamp_character_genders.py`
  (VENDOR time). `nodes/_otr_roster_gender.py` is the RENDER path and must never
  call an LLM. That boundary is settled and is not reopened here.
* Tier 4 declines honestly; the 40/40/20 render roll is the floor. Settled.
* Shakespeare has 11 unknown rows, not 32. Measured.

## THE ONLY QUESTION IN THIS ROUND

`_aliases_for` (`scripts/otr_stamp_character_genders.py:153-160`) stamps GIVEN
NAMES as roster join keys. The real committed data:

    pride_prejudice_proposal.provenance.json
      Elizabeth Bennet   female  aliases=['bennet', 'elizabeth']
      Fitzwilliam Darcy  male    aliases=['darcy', 'fitzwilliam']

COLONEL FITZWILLIAM has NO ROW. A slot for him strips the rank "colonel"
(`_HONORIFICS`, `nodes/_otr_roster_gender.py:51`), matches Darcy's `fitzwilliam`
alias through the `short_form` tier (`:401`), and returns MALE citing DARCY. Only
one row matches, so `_verdict_from`'s `ambiguous_join` abstention never fires.

A pinned test already documents the limit:
`tests/test_gender_ladder_recall_and_floor.py:395-403`
(`test_a_given_name_alias_matches_and_that_is_the_recorded_limit`).

**A surname-only rule REGRESSES `ELIZABETH`**, which is a legitimate given-name
join key in the same file. So "stamp only surnames" is rejected as written.

## THE DRIVER'S PROPOSED RULE -- BREAK IT

> A GIVEN-NAME alias match is REFUSED when the slot carried a stripped RANK/TITLE
> honorific that the matched row's own name does not itself carry.

`ELIZABETH` (no honorific stripped) keeps matching Elizabeth Bennet.
`COLONEL FITZWILLIAM` (rank stripped; Darcy's row carries no rank) is refused and
falls through to abstention, then to the render roll.

Attack it on these axes, with real file evidence:

1. **Does it hold on the real corpus?** Check the actual sidecars under
   `config/source_banks/` for slots that legitimately carry an honorific AND
   legitimately match a row by given name (e.g. "SIR TOBY" -> Toby Belch,
   "Miss Mix", "Uncle Silas", "Dr. Watson"). If the rule refuses any of those, it
   is wrong. NAME THE ROWS.
2. **Where does the fix live?** The stamper (`_aliases_for`, which would require
   RE-STAMPING ~79 committed `.provenance.json` sidecars) or the render join
   (`resolve_roster_gender`, local, no data migration) -- or both? State the
   migration cost and who owns it. Note CLAUDE.md 0B: a change reaching both the
   5080 and the 4060 must be measured on both.
3. **Is alias PROVENANCE the better shape?** i.e. stamp aliases as
   `{"alias": "fitzwilliam", "kind": "given"}` vs a flat list, letting the join
   weigh them. Cost: a sidecar schema change across 79 files and every reader.
   Is it worth it, or is it over-engineering for one collision class?
4. **What is the regression test?** Name the exact test that would fail today and
   pass after, using REAL rows (not invented fixtures) -- the admission rule says
   an invented fixture never creates a new bug record.
5. **The 11 unknown Shakespeare rows:** leave them for the roll, or run the
   stamper again? Is there evidence they are genuinely undecidable rather than
   unattempted?

Do NOT re-litigate the render-path/LLM boundary or tier-4 totality. Those are
closed. A proposal that reopens them is invalid.

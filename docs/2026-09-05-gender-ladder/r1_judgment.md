# r1 JUDGMENT -- gender ladder (driver: Claude/Cowork; panel: Codex spark-xhigh, Antigravity Gemini 3.8 Flash)

## The anchor's central claim was WRONG. Both lanes caught it independently.

I claimed `llm_recall` and `name_frequency` "DO NOT EXIST". **False.** I grepped
only `nodes/`. The full four-rung ladder is implemented in
`scripts/otr_stamp_character_genders.py` (842 lines):

| thing | location | verified |
|---|---|---|
| `TIER_ORDER` = roster/pronouns/llm_recall/name_frequency | `:94` | read |
| `GenderIndex` + `INDEX_FILENAME` | `:228`, `:87` | read |
| `Recall` (the LLM rung) | `:276`, returns `llm_recall` at `:308/:334` | read |
| `name_frequency` (curated pool floor) | `:348` | read |
| `decide_all` (all four rungs) | `:372` | read |
| tests | `tests/test_gender_ladder_recall_and_floor.py` | exists |
| persistent indexes | `config/source_banks/{shakespeare,public_domain_story}/character_gender_index.json` | 19 and 84 entries |

**ACCEPTED, and it collapses most of the row.** "Implement tier 3 / keep tier 4"
is DONE work, shipped 2026-09-02. GO_FORWARD 2.1 is stale.

## The architecture boundary I nearly broke

Antigravity MUST-FIX 1 is right and is the most valuable finding of the round:
`nodes/_otr_roster_gender.py` is the RENDER path and its docstring promise ("never
calls an LLM") must STAY TRUE. The LLM rung belongs to the VENDOR-time stamper.
Building tier 3 into `nodes/` -- which my anchor implied -- would have put an LLM
call on the render path and broken node determinism. **Cut entirely.**

## Tier 4 is NOT a total floor, and that is correct

Codex MUST-FIX 2 and Antigravity MUST-FIX 2 agree. Verified at
`scripts/otr_stamp_character_genders.py:372-389`: `decide_all` returns
`("", "", ...)` when every rung declines, and
`tests/test_gender_ladder_recall_and_floor.py:159-163` PINS that decline for
"Sancho Panza". Totality is supplied downstream by the existing 40/40/20 roll,
never by forcing a guess into a permanent sidecar.
**The GO_FORWARD ruling "an LLM failure still yields a gender" is therefore wrong
as written** and is corrected here: the ladder declines honestly; the render roll
is the floor.

## The Shakespeare count is stale: 11 unknown, not 32

Measured over the 14 shakespeare provenance sidecars: 47 male / 27 female /
**11 unknown**. 21 of the original 32 were filled by the 09-02 stamper run.

## The surname fork: diagnosis SETTLED, rule NOT settled -> r2

Antigravity MUST-FIX 4 relocated the root cause and it is PROVEN by the data:

    pride_prejudice_proposal.provenance.json
      Elizabeth Bennet   female  aliases=['bennet', 'elizabeth']
      Fitzwilliam Darcy  male    aliases=['darcy', 'fitzwilliam']

`_aliases_for` (`scripts/otr_stamp_character_genders.py:153-160`) borrows
`mention_forms` and stamps the GIVEN NAME as a join key. COLONEL FITZWILLIAM has
NO ROW, so a slot for him strips "colonel", matches Darcy's `fitzwilliam` alias,
and returns MALE **citing the wrong character** -- with no multi-row ambiguity to
trigger `_verdict_from`'s abstention. My anchor's "both rows match, both male"
trace was wrong; there is only one row.

**Both are male, so the OUTPUT is right and the CITATION is wrong.** This defect
cannot be found by looking at episodes. That part of the anchor stands.

**WHY ANTIGRAVITY'S FIX IS NOT ADOPTED AS WRITTEN.** It proposes stamping only
surnames (final tokens). In the very same sidecar, `Elizabeth Bennet` carries the
given-name alias `elizabeth`, which is a LEGITIMATE and needed join key -- a slot
named ELIZABETH must still match. A surname-only rule regresses it.
**The diagnosis converged; the rule did not. That is what r2 is for.**

## Carried to r2 as the driver's position

An honorific-bearing slot (`COLONEL FITZWILLIAM`) matching a row whose own name
never carries that rank is the signal that the slot denotes a DIFFERENT person.
Proposed rule to be broken in r2: a GIVEN-NAME alias match is refused when the
slot carried a stripped RANK/title honorific that the matched row does not itself
carry. `ELIZABETH` (no honorific) keeps matching; `COLONEL FITZWILLIAM` does not
steal Darcy.

## Cut

* Building any LLM call into `nodes/` -- violates the render-path contract.
* Index locking/concurrency schemes -- render legs never write the index; it is an
  offline vendor artifact. (Both lanes agreed; verified.)
* Removing the `short_form` tier -- breaks SIR TOBY -> TOBY and Miss Mix -> Mix.

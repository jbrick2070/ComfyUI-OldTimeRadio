# R1 LEAN-ARCHITECTURE JUDGMENT -- the 31-plan table is CUT

**Written 2026-07-25 (overnight) by CODER WINDOW A at HEAD `5dd74f93`.
Panel: codex `gpt-5.6-sol` (high) + agy `Gemini 3.6 Flash (High)`, both pins
verified, `--driver claude`. Run:
`kibitz-runs/2026-07-25-still-plans-lean-r1/r1/`. Brief:
`docs/2026-07-25-still-plans-r1-lean-architecture.md`. Claude is the grounded
panelist and sole judge.**

**NOTHING WAS TORN DOWN TONIGHT.** This document is the decision teed up for
the operator, not an executed refactor -- see "Why no code moved" at the end.

## THE ANSWER: both seats, independently, say CUT THE TABLE

- **agy: adopt Option B.** "Delete the 31-plan table and `StillPlanRow` schema
  entirely." Safe to cut "because 27 of 31 registered engines share identical
  4-target fingerprints (3 shape profiles total) ... Replacing 31 declarations
  with 1 capability function reduces code bloat with zero loss in
  functionality."
- **codex: adopt Option C**, and explicitly "Cut the seven-field
  `StillPlanRow`, its closed enums, and copied adapter row declarations after
  the compact capability materializer lands. They encode repeated structural
  outputs rather than independent behavior."

Two seats, two routes, one verdict on the central question. The operator's
"this was over-engineered" read is now confirmed by the panel as well as by
measurement.

## JUDGE CALL: codex's Option C, not agy's Option B

They differ in ONE respect and it matters. agy would delete per-adapter
ownership entirely and put requiredness in "a single capability-keyed function
`engine_requires_still(routing_state)`". codex keeps a MINIMAL per-adapter
descriptor -- `still_mode = scene|mesh|none` plus narrowly named activation
flags and aspect -- feeding ONE pure materializer, with a SEPARATE per-engine
layer-2 prompt hook.

**Option C wins on two grounded arguments:**

1. **agy's central function recreates the disease.** A single function that
   knows every engine's still requirements IS the shape this build exists to
   kill: five modules re-deriving effective engine and seven mechanisms
   deciding requiredness grew precisely because the knowledge lived centrally
   instead of with the adapter. Shrinking 31 declarations into one function
   moves the sprawl, it does not remove it.
2. **The operator's directive REQUIRES per-adapter ownership.** "Ensure that
   each video path has its own customized still operations" is not satisfiable
   by a central table keyed on engine id -- that is what the five capability
   maps already were. codex's compact descriptor keeps ownership where the
   operator asked for it while deleting the schema overhead: from
   31 engines x 4-6 rows x 7 fields down to ~3 fields per engine.

**Also: agy's must-fix #1 is WRONG on two details, discarded.** It says to
retain "the existing 6 geometry constants where they live in
`nodes/_otr_video_engines/render_driver.py`". There are **EIGHT** constants
(codex caught the count too, should-fix #1 -- and my own brief was
self-inconsistent on it), and they live in
`nodes/otr_meta_brief_image_prompt.py` and
`nodes/_otr_story_brief_helpers.py`, NOT in `render_driver.py`. The retention
instruction is right; the location and count are not.

## What BOTH seats agree on beyond the fork

1. **Land the ROUTING FREEZE first, in its own commit, before any still-plan
   replacement** (codex #2, agy #2 and should-fix #3). This is the actual bug
   fix: `otr_video_render_batch.py:322` validates the spine before
   `render_driver.py:2784` applies the override, so with a force map set the
   spine is validated against the PICKED engine and rendered with the FORCED
   one. codex's framing is the one to keep: "This delivers the actual bug fix
   even if the structural simplification needs another iteration."
2. **`style_tail_policy` comes OUT of the structural contract entirely**
   (codex #7, agy #3). Tail selection stays in the prompt composer, where the
   `ltx_radio_mouth` early return already lives safely.
3. **S5's uniqueness metric is premature** (codex #6, agy cut #2). Per-engine
   prompt hooks become simple functions when a model's behaviour actually
   differs; forced uniqueness invites cosmetic drift.
4. **Delete, don't rewire:** the id list at `render_driver.py:635-637`, the
   dispatcher's tri-state basis, the duplicate `_effective_*_for_role`
   helpers, and the stale degrade-chain prose at `eng_humo.py:497`.

## THIS SUPERSEDES the r4b style_tail resolution I adopted hours earlier

The r4b addendum in `docs/2026-07-25-still-plans-r4-judgment.md` locked
`style_tail_policy="full"` on the LTX bookend row plus an S2 composer exemption
for `source="ltx_radio_face"`. **Both seats now say the field should not exist
in the structural contract at all.** Same seats, broader question, better
answer -- so the r4b resolution is superseded, not contradicted: the exemption
was the right answer to "what token?", and "no field" is the right answer to
"should this be structural?". Recorded rather than quietly overwritten.

## NEW grounded findings from this round (I confirmed each)

1. **Freezing the LTX recipe is NOT behaviour-preserving** (codex #3). This is
   the most important new item and it lands on an operator decision.
   `eng_ltx_av.py:402-405` documents the CURRENT contract verbatim: "Read fresh
   every call (an operator flips daily<->hero per beat by swapping
   `OTR_LTX_AV_UNET` / `OTR_LTX_AV_RECIPE`)." So S0b's `ltx_resolved` freeze
   would silently convert a documented PER-BEAT operator capability into an
   episode-scoped one. I read that docstring earlier tonight and did not draw
   the conclusion; codex did.
   **OPERATOR DECISION NEEDED** -- either (a) accept episode-scoped recipe and
   DELETE the contrary per-beat contract from the docstring, or (b) keep
   per-beat switching, which then needs an explicit SHOT-OWNED field rather
   than ambient environment. Default if unruled: **(a)**, because a frozen
   routing state whose recipe can change mid-episode is not frozen -- but this
   removes a capability the code advertises, so it is the operator's call, not
   mine.
2. **Malformed routing config currently FALLS BACK, against the fail-closed
   law** (codex #4). `otr_image_gen_dispatcher.py:377-394` keeps the picked
   engine; `render_driver.py:2784-2799` logs a warning and IGNORES a malformed
   force map (I confirmed: `_LOG.warning(... IGNORED (parse) ...); return
   ledger`). Rejection belongs at the capture boundary, before image
   generation. `IS_CHANGED` must also cover the presence of retired
   `OTR_LTX_AV_SHARP`, which turns `_recipe()` into a hard error.
3. **`+ Add Custom Model` has no defined still contract** (codex #5, and both
   r4 passes raised it). `otr_video_director.py:443-481` permits an unknown
   custom engine id, but neither a closed 31-plan table nor a registry-keyed
   function can know its still requirements. Minimum safe contract: a custom id
   must resolve to a registered adapter carrying the compact descriptor, else
   VideoDirector fails closed. No permissive default profile.
4. **A teardown needs an explicit removal list** (codex #8). S1 and S1b are
   spread across 12 adapters, `nodes/_otr_shared/still_plan_helpers.py`,
   `tests/test_still_plan_audit.py` and
   `tests/test_still_plan_layer2_parity.py`. Every removal and every
   retained-or-replaced invariant must be enumerated, or the old table survives
   as a dormant competing authority. The registry invariant
   `CAPABILITIES == all_engine_names()` (`test_still_plan_audit.py:87-94`)
   is KEPT; only its `valid-plan owners` leg is replaced by
   compact-capability ownership (agy should-fix #2).
5. **`provider_side` should become an explicit required attribute** on every
   registered cloud/BYO engine (agy should-fix #2), retiring the brittle
   three-part rule that catches `cloud_kling_avatar` on its id prefix alone.

## Honest accounting of tonight's S1b against this answer

S1b (`69328cec`) replaced 57 rows' paraphrases with the producer's real
geometry. Under Option C **those 57 row texts get deleted** along with the
table. That is a real cost and it should be stated plainly rather than dressed
up. What SURVIVES and was worth the chunk:

- The **measurement** that the table adds ZERO per-engine differentiation is
  what this R1 rests on. Without S1b's transplant and its dump, the argument
  for cutting the table is an opinion instead of a number.
- The **misdeclared `ltx_audio_in` bookend row** (production emits
  `kind="portrait"` / `source="ltx_radio_face"`, wide geometry, no talking
  flag) is a permanent fact about the system that transfers straight into the
  compact descriptor and the prompt hook.
- The **HuMo aspect split** -- one plan object cannot serve two shipped
  aspects -- transfers as the reason the descriptor carries aspect per engine.
- The **geometry-vs-LOOK boundary** (layer 2 is Python-owned engine safety;
  LOOK is pack-owned) is now written down and machine-checked; it constrains
  the prompt hook under any option.
- S1b also **improved every prompt in the tree immediately** -- the empty
  portrait row on 19 engines and the missing clay-blob clause were live
  degradations, and they are fixed at HEAD tonight regardless of what happens
  to the table.

The ordering lesson, which belongs in the doctrine: **the routing freeze was
always the bug fix, and it should have gone first.** The inherited order put
the table's characterization and declaration ahead of it, so two chunks landed
against a structure the arc then cut. codex #2 says it plainly and it is right.

## Why no code moved tonight

The operator is asleep, having said "do what you can without me". A teardown of
landed, green, pushed code -- 12 adapters, a schema module, two test files --
is hard to unwind and rests on a decision that also carries an unanswered
operator question (the per-beat LTX recipe capability, finding 1 above). Under
the unattended rule the correct move is to do all the preparatory work, state
the decision needed, and stop rather than guess. So tonight ends with the
tree GREEN at `5dd74f93`, S1b landed and improving prompts, and this judgment
plus the consolidated spec as the next window's first job.

## Next window's order

1. **Operator ratifies the cut** (and rules on the per-beat LTX recipe
   question). It changes the plan of record and makes landed code a teardown
   target, so it is a ratification, not a coder call.
2. **ONE consolidated build spec** -- Option C's compact descriptor, the
   materializer, the prompt hook, the explicit teardown list, and every
   accepted r4 / r4b / R1 correction. Both r4 passes and both R1 seats asked
   for this; it is the last doc owed. Mark the locked spec, the corrected plan
   and both judgments history-only.
3. **The ROUTING FREEZE, first and alone**, with the forced-route live proof.
   It ships the real bug fix independent of the table question.
4. Then the descriptor + materializer, then the teardown, then the prompt hook.
